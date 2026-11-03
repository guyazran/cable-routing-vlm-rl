"""Geometric routing evaluator (protocol Section 5).

Insertion is judged from the ordered cable centreline, never from a head
coordinate threshold alone. Walking the centreline from the anchored tail to the
free head, every crossing of a clip's entry and exit planes is located; a
*transit* is an entry-plane crossing inside the aperture immediately followed
(for that clip) by an exit-plane crossing inside the aperture in the same
direction. A cable that passes beside or over a clip crosses the planes outside
the aperture and produces no transit.

The routing state is the tail-to-head sequence of transits. With the cable
inserted head-first through C01, C02, ..., a correct route reads
[C01+, C02+, ..., CN+] from the tail. Points are passed head-first
(``points[0]`` is the free head tip, as the paper's p1), tail last.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clip import Clip

# Tolerance (m) on the aperture cross-section, absorbing the capsule radius
# penetration allowed by the physics contact offset.
APERTURE_TOL = 0.002


@dataclass(frozen=True)
class PlaneCrossing:
    clip_id: str
    plane: str  # "entry" | "exit"
    direction: int  # +1 along clip +x, -1 against
    inside: bool
    s: float  # arc length from the tail


@dataclass(frozen=True)
class Transit:
    clip_id: str
    direction: int
    s: float


@dataclass
class ClipState:
    clip_id: str
    transits: list[Transit] = field(default_factory=list)
    partial: int = 0  # +1 head inside tunnel entered forward, -1 reversed, 0 none
    anomalies: list[str] = field(default_factory=list)


@dataclass
class RoutingState:
    sequence: list[Transit]  # tail -> head
    clips: dict[str, ClipState]
    valid_prefix: int
    complete: bool  # exactly the requested route, nothing else threaded
    errors: list[str]

    def summary(self) -> dict:
        return {
            "sequence": [f"{t.clip_id}{'+' if t.direction > 0 else '-'}" for t in self.sequence],
            "valid_prefix": self.valid_prefix,
            "complete": self.complete,
            "partial": {k: v.partial for k, v in self.clips.items() if v.partial},
            "errors": list(self.errors),
            "anomalies": {k: v.anomalies for k, v in self.clips.items() if v.anomalies},
        }


def _arc_lengths(pts: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def plane_crossings(clip: Clip, tail_to_head: np.ndarray, s: np.ndarray) -> list[PlaneCrossing]:
    local = clip.pose.to_local(tail_to_head)
    out: list[PlaneCrossing] = []
    for plane, X in (("entry", clip.entry_x), ("exit", clip.exit_x)):
        side = local[:, 0] >= X
        idx = np.nonzero(side[:-1] != side[1:])[0]
        for i in idx:
            a, b = local[i], local[i + 1]
            t = (X - a[0]) / (b[0] - a[0])
            q = a + t * (b - a)
            out.append(PlaneCrossing(
                clip.clip_id, plane, 1 if b[0] > a[0] else -1,
                bool(clip.in_aperture(q[1:3], APERTURE_TOL)),
                float(s[i] + t * (s[i + 1] - s[i])),
            ))
    out.sort(key=lambda c: c.s)
    return out


def clip_state(clip: Clip, tail_to_head: np.ndarray, s: np.ndarray) -> ClipState:
    st = ClipState(clip.clip_id)
    inside = [c for c in plane_crossings(clip, tail_to_head, s) if c.inside]
    i = 0
    while i < len(inside):
        c = inside[i]
        nxt = inside[i + 1] if i + 1 < len(inside) else None
        fwd_start = c.plane == "entry" and c.direction > 0
        rev_start = c.plane == "exit" and c.direction < 0
        if nxt is not None and (fwd_start or rev_start):
            want_plane = "exit" if fwd_start else "entry"
            if nxt.plane == want_plane and nxt.direction == c.direction:
                st.transits.append(Transit(clip.clip_id, c.direction, 0.5 * (c.s + nxt.s)))
                i += 2
                continue
            if nxt.plane == c.plane and nxt.direction == -c.direction:
                i += 2  # entered the tunnel and backed out the same side
                continue
            st.anomalies.append(f"unpaired {c.plane}{c.direction:+d} at s={c.s:.3f}")
        elif nxt is None and (fwd_start or rev_start):
            st.partial = c.direction  # head currently inside the tunnel
        else:
            st.anomalies.append(f"unexpected {c.plane}{c.direction:+d} at s={c.s:.3f}")
        i += 1
    return st


def evaluate_routing(clips: list[Clip], points_head_first: np.ndarray, route: list[str]) -> RoutingState:
    """Evaluate threading of ``route`` (ordered clip IDs, all traversed in +x)."""
    pts = np.asarray(points_head_first, dtype=float)[::-1]
    s = _arc_lengths(pts)
    states = {c.clip_id: clip_state(c, pts, s) for c in clips}
    seq = sorted((t for st in states.values() for t in st.transits), key=lambda t: t.s)

    k = 0
    while k < min(len(seq), len(route)) and seq[k].clip_id == route[k] and seq[k].direction > 0:
        k += 1

    errors = []
    for j, t in enumerate(seq):
        if t.clip_id not in route:
            errors.append(f"threaded non-requested clip {t.clip_id}")
        elif t.direction < 0:
            errors.append(f"reverse threading of {t.clip_id}")
        elif j >= len(route) or route[j] != t.clip_id:
            errors.append(f"order violation at position {j + 1}: {t.clip_id}")
    complete = k == len(route) and len(seq) == len(route)
    return RoutingState(seq, states, k, complete, errors)


# --- quantities relative to a single target clip (reward, Insert success) ---

@dataclass(frozen=True)
class HeadMetrics:
    rope_in: bool  # head inside the target clip's tunnel (forward)
    rope_out: bool  # most recent transit (nearest the head) is a forward pass of the target
    d_floor: float  # xy distance head -> entry-face centre
    d_ceil: float  # xy distance head -> exit-face centre
    signed_dist: float  # +d_ceil if rope_out else -d_ceil
    head_local: np.ndarray  # head position in the clip frame


def head_metrics(target: Clip, routing: RoutingState, head_xyz: np.ndarray) -> HeadMetrics:
    st = routing.clips[target.clip_id]
    rope_out = bool(routing.sequence) and routing.sequence[-1].clip_id == target.clip_id \
        and routing.sequence[-1].direction > 0
    head = np.asarray(head_xyz, dtype=float)
    d_floor = float(np.linalg.norm(head[:2] - target.floor_point_world()))
    d_ceil = float(np.linalg.norm(head[:2] - target.ceil_point_world()))
    return HeadMetrics(
        rope_in=st.partial > 0 and not rope_out,
        rope_out=rope_out,
        d_floor=d_floor,
        d_ceil=d_ceil,
        signed_dist=d_ceil if rope_out else -d_ceil,
        head_local=target.pose.to_local(head),
    )


def in_insertion_region(target: Clip, head_xyz: np.ndarray, size: float = 0.16) -> bool:
    """Head inside the size x size square directly in front of the entry face."""
    x, y = target.pose.to_local(np.asarray(head_xyz))[:2]
    return (target.entry_x - size <= x <= target.entry_x) and abs(y) <= size / 2
