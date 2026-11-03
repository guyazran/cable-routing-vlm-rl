"""Task-relative quantities the algorithm derives from a :class:`CableState`.

``head_metrics`` serves the insertion policy (observation, reward, local success)
and the reference controller; ``route_status`` interprets the transit sequence
against a requested route. The environment grades episodes with its own,
independent evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interface import CableState, ClipSpec


@dataclass(frozen=True)
class HeadMetrics:
    rope_in: bool  # head inside the clip's tunnel, entered forward
    rope_out: bool  # the transit nearest the head is a forward pass of the clip
    d_floor: float  # xy distance head -> entry-face centre
    d_ceil: float  # xy distance head -> exit-face centre
    signed_dist: float  # +d_ceil if rope_out else -d_ceil
    head_local: np.ndarray  # head position in the clip frame


def head_metrics(clip: ClipSpec, state: CableState) -> HeadMetrics:
    last = state.transits[-1] if state.transits else None
    rope_out = last is not None and last.clip_id == clip.clip_id and last.direction > 0
    head = np.asarray(state.points[0], dtype=float)
    d_floor = float(np.linalg.norm(head[:2] - clip.floor_point()))
    d_ceil = float(np.linalg.norm(head[:2] - clip.ceil_point()))
    return HeadMetrics(
        rope_in=state.partial.get(clip.clip_id, 0) > 0 and not rope_out,
        rope_out=rope_out,
        d_floor=d_floor,
        d_ceil=d_ceil,
        signed_dist=d_ceil if rope_out else -d_ceil,
        head_local=clip.pose.to_local(head),
    )


def in_insertion_region(clip: ClipSpec, head_xyz: np.ndarray, size: float = 0.16) -> bool:
    """Head inside the size x size square directly in front of the entry face."""
    x, y = clip.pose.to_local(np.asarray(head_xyz))[:2]
    return (clip.entry_x - size <= x <= clip.entry_x) and abs(y) <= size / 2


@dataclass(frozen=True)
class RouteStatus:
    valid_prefix: int  # leading requested clips threaded forward, in order
    complete: bool  # exactly the requested route, nothing else threaded
    extra_threading: bool  # transits beyond the valid prefix (wrong clip, order or direction)


def route_status(state: CableState, route: list[str]) -> RouteStatus:
    seq = state.transits
    k = 0
    while k < min(len(seq), len(route)) and seq[k].clip_id == route[k] and seq[k].direction > 0:
        k += 1
    return RouteStatus(k, k == len(route) and len(seq) == len(route), len(seq) > k)
