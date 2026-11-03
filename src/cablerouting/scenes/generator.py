"""Scene family generator for the horizon test (protocol Section 8).

Each scene places N clips C01..CN along a route inside the region that is both
reachable by a virtual robot base at the origin (facing +x) and visible to the
fixed planner camera. Stage k (clip k -> k+1) is a circular arc with chord l_k
and heading change D_k; each clip's insertion axis is tangent to the route, so
the cable leaves clip k and arrives at clip k+1 along their axes.

Diversity with bounded local difficulty:
* the start position and heading are uniform over the feasible region;
* every stage draws l_k ~ U(stage_length) and |D_k| ~ U(turn_deg) from the
  same distribution at every route position;
* the turn sign follows a per-scene persistence rho ~ U(sign_persistence):
  low rho gives zigzags, high rho gives hooks, U-turns and spirals.
The route is built clip by clip; a candidate stage that breaks a predeclared
check is redrawn, and a dead end rejects the whole attempt (logged).

Counterbalancing: scene index assigns one of four variants (mirror y -> -y,
reverse traversal), so any position bias introduced by the clip-by-clip
construction is cancelled between the start and the end of the route. Both
sides of every clip are kept feasible so a reversed route stays valid.

Stage shapes (v3): ``arc`` is the construction above. ``hermite`` draws the
direction of the next clip (bearing from the current clip's axis, sign following
the persistence) and its heading change independently, with the heading change
on the same side (hook) or the opposite side (S-bend) at random; the cable path is
the cubic Hermite curve leaving and entering along the clip axes, and its tightest
bend must stay above ``min_path_radius``. Clips are then scattered rather than
strung along a trail.

Robot base (v3): with ``base_clearance`` > 0, every stage path and the initial
cable keep that distance from the robot base at the origin, so the arm's pedestal
never sits on the route. ``base_clearance = 0`` reproduces route-family-v2 layouts
exactly (same seeds, same scenes).

Shape motifs (v4): ``motif = "free"`` is the sign-persistence construction above.
Any other motif fixes each stage's base turn from a per-scene plan, and a stage
try only redraws the chord and a small jitter (``turn_jitter_deg``). The motif
level is T ~ U(turn_deg), and s = +-1 is drawn per scene:
* ``arc``, ``circle``: constant turn s*T (gentle bend vs tight bend that closes
  into a loop on long routes);
* ``spiral``: same sign, |turn| linear from T to T1 ~ U(turn_end_deg) along the
  construction order (the reverse variant turns tightening into loosening);
* ``serpentine``: smooth waves; the sign flips every h stages, h ~ motif_period;
* ``square``: corners of T (~90 deg) every m stages, m ~ motif_period, and
  straight stages (turn 0) between them;
* ``zigzag``: as ``square`` but the corners alternate in sign, so straight legs
  meet at sharp alternating corners.
``serpentine`` and ``zigzag`` add a backbone bias ~ +-U(bias_deg) to every stage
so long routes fold into the workspace instead of running straight.
The plan's phase is random, so corners and sign flips do not sit at fixed
route positions. It is drawn among the phases that show the motif's defining
feature on the route: at least one corner (square), two alternating corners
(zigzag), or one sign flip (serpentine), or as many as the route length allows. ``motif = "free"`` reproduces route-family-v3 layouts exactly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..geometry.clip import Clip, ClipDims, clip_label
from ..geometry.frames import Pose2D, wrap_angle

GENERATOR_VERSION = "route-family-v4"

# Disjoint seed ranges for the three layout pools (protocol Section 9). Datasets
# (scenes/dataset.py) pass their own seed base above these.
POOL_SEED_BASE = {"development": 0, "calibration": 1_000_000, "evaluation": 2_000_000}

MOTIFS = ("free", "arc", "circle", "spiral", "zigzag", "serpentine", "square")


@dataclass(frozen=True)
class SceneFamilyConfig:
    name: str = "primary10"
    n_clips: int = 10
    clip_scale: float = 1.0
    stage_length: tuple[float, float] = (0.16, 0.19)
    turn_deg: tuple[float, float] = (5.0, 50.0)  # |heading change| per stage
    sign_persistence: tuple[float, float] = (0.2, 0.8)  # P(next turn has the same sign), per scene
    stage_tries: int = 40
    max_backtracks: int = 30
    # Reachable region of the virtual robot base at the origin (relaxed slightly beyond a
    # Franka's 0.855 m nominal reach so ten-clip routes are not confined to one arc band).
    reach_radius: tuple[float, float] = (0.15, 0.90)
    max_bearing_deg: float = 150.0
    min_reach_margin: float = 0.02
    base_clearance: float = 0.0  # stage paths and initial cable vs the robot base at the origin (0 = off)
    stage_shape: str = "arc"  # "arc" | "hermite" (see module docstring)
    bearing_deg: tuple[float, float] = (0.0, 0.0)  # hermite: |direction of the next clip| from the axis
    min_path_radius: float = 0.0  # hermite: tightest allowed bend of a stage path
    # Shape motif (see module docstring); "free" uses sign_persistence and ignores the rest.
    motif: str = "free"
    turn_jitter_deg: float = 0.0  # per-stage jitter around the motif's base turn
    turn_end_deg: tuple[float, float] = (0.0, 0.0)  # spiral: level at the end of the construction
    motif_period: tuple[int, ...] = (1,)  # serpentine: stages per sign; square/zigzag: stages per side
    bias_deg: tuple[float, float] = (0.0, 0.0)  # zigzag/serpentine: |backbone turn| per stage
    placement_tries: int = 300  # rigid placements tried per drawn shape
    # Spacing and clearance.
    min_clip_separation: float = 0.10
    path_clearance: float = 0.05
    # Fixed camera view (world xy bounds) that every scene must fit inside. The
    # 1.05 m x 1.40 m extent matches a 1200 x 1600 px top-down render (4:3).
    view_x: tuple[float, float] = (-0.25, 0.80)
    view_y: tuple[float, float] = (-0.70, 0.70)
    view_margin: float = 0.0
    # Insertion region (paper: 0.16 m x 0.16 m) in front of each entry face.
    insertion_region: float = 0.16
    # Cable and initial state.
    # The tail is free (no anchor); ``tail_reserve`` of cable must remain behind
    # C01 after the full route so the tail can never be drawn into the route.
    cable_length: float = 2.4
    cable_radius: float = 0.005
    head_allowance: float = 0.12
    tail_reserve: float = 0.30
    initial_head_axial: tuple[float, float] = (0.06, 0.11)  # distance before C01 entry face
    initial_head_lateral: tuple[float, float] = (-0.025, 0.025)  # 5 cm band
    initial_angle_deg: float = 10.0
    initial_lead: float = 0.30  # straight cable behind the head before it bends away
    initial_max_bend_deg: float = 100.0
    initial_clip_clearance: float = 0.04


@dataclass
class StageStats:
    stage: int  # 1-based index of the clip being reached (2..N)
    chord: float
    relative_angle_deg: float
    arc_length: float


@dataclass
class Scene:
    scene_id: str
    pool: str
    seed: int
    variant: dict
    family: str
    generator_version: str
    clips: list[dict]
    stages: list[dict]
    reach_margin: float
    route_arc_length: float
    route_slack: float
    initial_head: list[float]  # x, y, yaw (yaw points from tail towards head)
    initial_tail: list[float]  # x, y of the free tail end
    cable_length: float
    cable_radius: float
    initial_path: list[list[float]] | None = None  # xy waypoints head -> tail; None = straight
    extras: dict = field(default_factory=dict)

    def clip_objects(self, base_dims: ClipDims | None = None) -> list[Clip]:
        return [Clip.from_dict(c, base_dims) for c in self.clips]

    def route(self, n_targets: int) -> list[str]:
        return [c["clip_id"] for c in self.clips[:n_targets]]

    def initial_polyline(self, spacing: float) -> np.ndarray:
        """Nominal cable along ``initial_path``, head first, resampled at ``spacing``."""
        n = int(round(self.cable_length / spacing))
        if self.initial_path:
            path = np.asarray(self.initial_path, dtype=float)
        else:
            hx, hy, yaw = self.initial_head
            path = np.array([[hx, hy], [hx, hy] - self.cable_length * np.array([np.cos(yaw), np.sin(yaw)])])
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        s = np.arange(n + 1) * spacing
        xy = np.column_stack([np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])])
        return np.column_stack([xy, np.full(n + 1, self.cable_radius)])

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @staticmethod
    def from_dict(d: dict) -> "Scene":
        return Scene(**d)


class SceneRejected(Exception):
    pass


def _arc_points(start: Pose2D, chord: float, turn: float, n: int = 24) -> tuple[np.ndarray, float]:
    """Points along a constant-curvature arc from ``start`` turning by ``turn``."""
    if abs(turn) < 1e-6:
        t = np.linspace(0, chord, n)[:, None]
        return start.xy + t * start.heading, chord
    length = chord * (turn / 2) / np.sin(turn / 2)
    radius = length / turn
    phi = np.linspace(0, turn, n)
    local = np.column_stack([radius * np.sin(phi), radius * (1 - np.cos(phi))])
    return start.to_world(local), abs(length)


def _hermite_points(a: Pose2D, b: Pose2D, n: int = 32) -> tuple[np.ndarray, float]:
    """Cubic Hermite path from clip pose a to b along both axes (tangent length = chord)."""
    L = float(np.linalg.norm(b.xy - a.xy))
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = ((2 * t**3 - 3 * t**2 + 1) * a.xy + (t**3 - 2 * t**2 + t) * L * a.heading
           + (-2 * t**3 + 3 * t**2) * b.xy + (t**3 - t**2) * L * b.heading)
    return pts, float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _min_radius(pts: np.ndarray) -> float:
    """Tightest bend of a polyline (circumradius of consecutive point triples)."""
    a, b, c = pts[:-2], pts[1:-1], pts[2:]
    ab, bc, ca = (np.linalg.norm(u, axis=1) for u in (b - a, c - b, a - c))
    area2 = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    with np.errstate(divide="ignore"):
        return float(np.min(np.where(area2 > 1e-12, ab * bc * ca / (2 * area2), np.inf)))


def _stage_path(a: Pose2D, b: Pose2D, cfg: "SceneFamilyConfig") -> tuple[np.ndarray, float]:
    if cfg.stage_shape == "hermite":
        return _hermite_points(a, b)
    return _arc_points(a, float(np.linalg.norm(b.xy - a.xy)), float(wrap_angle(b.yaw - a.yaw)))


def variant_for_index(index: int) -> dict:
    return {"mirror": 1 if index % 2 == 0 else -1, "reverse": bool((index // 2) % 2)}


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def _polyline_distance(p: np.ndarray, poly: np.ndarray) -> float:
    """Distance from point p to a polyline (vectorised over segments)."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
    return float(np.min(np.linalg.norm(p - (a + t[:, None] * ab), axis=1)))


def _polylines_cross(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two polylines properly intersect (vectorised over all segment pairs)."""
    lo_a, hi_a, lo_b, hi_b = a.min(0), a.max(0), b.min(0), b.max(0)
    if np.any(hi_a < lo_b) or np.any(hi_b < lo_a):
        return False
    p1, p2 = a[:-1, None, :], a[1:, None, :]
    q1, q2 = b[None, :-1, :], b[None, 1:, :]

    def orient(u, v, w):
        return np.sign((v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) - (v[..., 1] - u[..., 1]) * (w[..., 0] - u[..., 0]))

    return bool(np.any((orient(p1, p2, q1) != orient(p1, p2, q2)) & (orient(q1, q2, p1) != orient(q1, q2, p2))))


def _reach_margin(p: np.ndarray, cfg: SceneFamilyConfig) -> float:
    r = float(np.linalg.norm(p))
    bearing = abs(np.degrees(np.arctan2(p[1], p[0])))
    radial = min(r - cfg.reach_radius[0], cfg.reach_radius[1] - r)
    angular = np.radians(cfg.max_bearing_deg - bearing) * r
    return float(min(radial, angular))


def _view_margin(p: np.ndarray, cfg: SceneFamilyConfig) -> float:
    return float(min(p[0] - cfg.view_x[0], cfg.view_x[1] - p[0], p[1] - cfg.view_y[0], cfg.view_y[1] - p[1]))


def _clip_margins(c: Clip, cfg: SceneFamilyConfig, side: int = -1) -> tuple[float, float]:
    """(reach margin, view margin) of a clip and the insertion region on one of its sides.

    side = -1 is the entry side (local -x); +1 is the exit side, which becomes the
    entry side once the route is traversed in reverse.
    """
    pts = _clip_check_points(c, cfg, side)
    return min(_reach_margin(q, cfg) for q in pts), min(_view_margin(q, cfg) for q in pts)


def _clip_check_points(c: Clip, cfg: SceneFamilyConfig, side: int) -> list[np.ndarray]:
    """Clip centre and the corners of its insertion region on ``side``."""
    h, r = cfg.insertion_region, cfg.insertion_region / 2
    return [c.pose.xy] + [c.pose.to_world(np.array([side * (c.exit_x + f * h), y]))
                          for f in (0.0, 1.0) for y in (-r, r)]


def _clip_ok(c: Clip, cfg: SceneFamilyConfig, side: int = -1) -> bool:
    reach, view = _clip_margins(c, cfg, side)
    return reach >= cfg.min_reach_margin and view >= cfg.view_margin


def _room_ahead(c: Clip, cfg: SceneFamilyConfig, side: int) -> bool:
    """A next clip at the minimum stage length straight ahead would still be in reach and view."""
    ahead = c.pose.xy - side * cfg.stage_length[0] * c.pose.heading
    return _reach_margin(ahead, cfg) >= 0 and _view_margin(ahead, cfg) >= 0


def _stage_ok(cand: Clip, arc: np.ndarray, poses: list[Pose2D], arcs: list[np.ndarray],
              cfg: SceneFamilyConfig, side: int) -> bool:
    if not _clip_ok(cand, cfg, side):
        return False
    if cfg.base_clearance > 0 and _polyline_distance(np.zeros(2), arc) < cfg.base_clearance:
        return False
    return _layout_ok(cand, arc, poses, arcs, cfg)


def _layout_ok(cand: Clip, arc: np.ndarray, poses: list[Pose2D], arcs: list[np.ndarray],
               cfg: SceneFamilyConfig) -> bool:
    """Placement-independent checks of a new stage: clip spacing, path clearance, no crossing."""
    fp = cand.footprint_radius
    if any(np.linalg.norm(cand.pose.xy - p.xy) < cfg.min_clip_separation for p in poses):
        return False
    if any(_polyline_distance(p.xy, arc) < cfg.path_clearance + fp for p in poses[:-1]):
        return False
    if any(_polyline_distance(cand.pose.xy, a) < cfg.path_clearance + fp for a in arcs):
        return False
    return not any(_polylines_cross(arc, a) for a in arcs[:-1])


def _feature_phase(rng: np.random.Generator, period: int, events, need: int) -> int:
    """Phase drawn uniformly among those that put at least ``need`` defining events (corners,
    sign flips) on the route, or as many as any phase can when the route is too short."""
    counts = [events(ph) for ph in range(period)]
    floor = min(need, max(counts))
    return int(rng.choice([ph for ph, c in enumerate(counts) if c >= floor]))


def _motif_plan(cfg: SceneFamilyConfig, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Base signed turn (rad) of every construction stage, and the per-scene motif parameters."""
    n = cfg.n_clips - 1
    k = np.arange(n)
    sign = 1 if rng.random() < 0.5 else -1
    level = float(rng.uniform(*cfg.turn_deg))
    info: dict = {"motif": cfg.motif, "motif_sign": sign, "motif_level_deg": level}
    if cfg.motif in ("arc", "circle"):
        base = np.full(n, level)
    elif cfg.motif == "spiral":
        end = float(rng.uniform(*cfg.turn_end_deg))
        base = level + (end - level) * k / max(n - 1, 1)
        info["motif_end_level_deg"] = end
    elif cfg.motif == "serpentine":
        half = int(rng.choice(cfg.motif_period))
        # a sign flip sits before stage j when (j + phase) % half == 0
        phase = _feature_phase(rng, half, lambda ph: int(np.sum((k[1:] + ph) % half == 0)), need=1)
        bias = float(rng.uniform(*cfg.bias_deg)) * (1 if rng.random() < 0.5 else -1)
        base = np.where(((k + phase) // half) % 2 == 0, level, -level) + bias
        info.update(motif_half_period=half, motif_phase=phase, motif_bias_deg=bias)
    elif cfg.motif in ("square", "zigzag"):
        side = int(rng.choice(cfg.motif_period))
        need = 2 if cfg.motif == "zigzag" else 1  # a zigzag shows itself by alternating corners
        phase = _feature_phase(rng, side, lambda ph: int(np.sum((k + ph) % side == 0)), need)
        corner = np.where((k + phase) % side == 0, level, 0.0)
        if cfg.motif == "zigzag":
            bias = float(rng.uniform(*cfg.bias_deg)) * (1 if rng.random() < 0.5 else -1)
            base = corner * np.where(((k + phase) // side) % 2 == 0, 1.0, -1.0) + bias
            info["motif_bias_deg"] = bias
        else:
            base = corner
        info.update(motif_side_stages=side, motif_phase=phase)
    else:
        raise ValueError(f"unknown motif {cfg.motif!r}; expected one of {MOTIFS}")
    return sign * np.radians(base), info


def _build_route(cfg: SceneFamilyConfig, rng: np.random.Generator, side: int) -> tuple[list[Pose2D], dict]:
    """Raw route, clip by clip. ``side`` selects which face of each clip will be the entry."""
    if cfg.motif != "free":
        return _build_motif_route(cfg, rng, side)
    rho = float(rng.uniform(*cfg.sign_persistence))
    # The raw route is built in its forward direction; with side = +1 it is traversed
    # backwards later, so "ahead" in construction is behind the eventual entry face.
    fwd = -1 if side == -1 else 1
    for _ in range(500):
        start = Pose2D(float(rng.uniform(*cfg.view_x)), float(rng.uniform(*cfg.view_y)),
                       float(rng.uniform(-np.pi, np.pi)))
        c0 = Clip("tmp", start, cfg.clip_scale)
        if _clip_ok(c0, cfg, side) and _room_ahead(c0, cfg, fwd):
            break
    else:
        raise SceneRejected("no feasible start pose")
    poses, arcs, signs = [start], [], [1 if rng.random() < 0.5 else -1]
    backtracks = 0
    while len(poses) < cfg.n_clips:
        k = len(poses) - 1
        last = k + 1 == cfg.n_clips - 1
        for _ in range(cfg.stage_tries):
            s = signs[-1] if (k == 0 or rng.random() < rho) else -signs[-1]
            turn = s * np.radians(rng.uniform(*cfg.turn_deg))
            chord = float(rng.uniform(*cfg.stage_length))
            if cfg.stage_shape == "hermite":
                p0 = poses[-1]
                bearing = p0.yaw + s * np.radians(rng.uniform(*cfg.bearing_deg))
                if rng.random() < 0.5:
                    turn = -turn  # S-bend instead of a hook
                xy = p0.xy + chord * np.array([np.cos(bearing), np.sin(bearing)])
                end = Pose2D(float(xy[0]), float(xy[1]), float(wrap_angle(p0.yaw + turn)))
                arc, _ = _hermite_points(p0, end)
                if _min_radius(arc) < cfg.min_path_radius:
                    continue
            else:
                arc, _ = _arc_points(poses[-1], chord, turn)
                end = Pose2D(float(arc[-1][0]), float(arc[-1][1]), float(wrap_angle(poses[-1].yaw + turn)))
            cand = Clip("tmp", end, cfg.clip_scale)
            if _stage_ok(cand, arc, poses, arcs, cfg, side) and (last or _room_ahead(cand, cfg, fwd)):
                poses.append(end)
                arcs.append(arc)
                signs.append(s)
                break
        else:
            backtracks += 1
            if backtracks > cfg.max_backtracks or len(poses) < 2:
                raise SceneRejected(f"dead end at stage {k + 2}")
            poses.pop()
            arcs.pop()
            signs.pop()
    return poses, {"sign_persistence": rho}


def _build_motif_route(cfg: SceneFamilyConfig, rng: np.random.Generator, side: int) -> tuple[list[Pose2D], dict]:
    """Raw route of a shape motif: the whole shape in a local frame, then a rigid placement.

    A motif fixes the shape up to jitter, so the start pose decides feasibility; building
    clip by clip from a random start mostly backtracks into dead ends. Instead the shape
    is drawn first (placement-independent checks only), then rotation ~ U(-pi, pi) and a
    translation ~ uniform over the offsets that keep every clip's check points in view
    are drawn until reach (and base clearance) also hold.
    """
    if cfg.stage_shape != "arc":
        raise ValueError("shape motifs are built from arc stages")
    plan, info = _motif_plan(cfg, rng)
    poses, arcs = [Pose2D(0.0, 0.0, 0.0)], []
    for k in range(cfg.n_clips - 1):
        for _ in range(cfg.stage_tries):
            turn = plan[k] + np.radians(rng.uniform(-cfg.turn_jitter_deg, cfg.turn_jitter_deg))
            arc, _ = _arc_points(poses[-1], float(rng.uniform(*cfg.stage_length)), turn)
            end = Pose2D(float(arc[-1][0]), float(arc[-1][1]), float(wrap_angle(poses[-1].yaw + turn)))
            if _layout_ok(Clip("tmp", end, cfg.clip_scale), arc, poses, arcs, cfg):
                poses.append(end)
                arcs.append(arc)
                break
        else:
            raise SceneRejected(f"shape conflicts with itself at stage {k + 2}")
    local = np.array([q for p in poses for q in _clip_check_points(Clip("tmp", p, cfg.clip_scale), cfg, side)])
    path = np.concatenate(arcs)
    for _ in range(cfg.placement_tries):
        th = float(rng.uniform(-np.pi, np.pi))
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pts = local @ rot.T
        lo = np.array([cfg.view_x[0], cfg.view_y[0]]) + cfg.view_margin - pts.min(0)
        hi = np.array([cfg.view_x[1], cfg.view_y[1]]) - cfg.view_margin - pts.max(0)
        if np.any(lo > hi):
            continue
        t = rng.uniform(lo, hi)
        placed = [Pose2D(*map(float, rot @ p.xy + t), float(wrap_angle(p.yaw + th))) for p in poses]
        if not all(_clip_ok(Clip("tmp", p, cfg.clip_scale), cfg, side) for p in placed):
            continue
        if cfg.base_clearance > 0 and _polyline_distance(np.zeros(2), path @ rot.T + t) < cfg.base_clearance:
            continue
        return placed, {**info, "motif_base_turn_deg": np.degrees(plan).tolist()}
    raise SceneRejected("no placement fits reach and view")


def _transform(poses: list[Pose2D], mirror: int, reverse: bool) -> list[Pose2D]:
    if reverse:
        poses = [Pose2D(p.x, p.y, float(wrap_angle(p.yaw + np.pi))) for p in reversed(poses)]
    return [Pose2D(p.x, mirror * p.y, float(wrap_angle(mirror * p.yaw))) for p in poses]


def _initial_cable(cfg: SceneFamilyConfig, clips: list[Clip], arcs: list[np.ndarray],
                   rng: np.random.Generator) -> tuple[np.ndarray, float, dict]:
    """Head in front of C01, a straight lead behind it, then a straight tail towards open space."""
    c1 = clips[0]
    axial = rng.uniform(*cfg.initial_head_axial)
    lateral = rng.uniform(*cfg.initial_head_lateral)
    angle = np.radians(rng.uniform(-cfg.initial_angle_deg, cfg.initial_angle_deg))
    head = c1.pose.to_world(np.array([c1.entry_x - axial, lateral]))
    head_yaw = float(wrap_angle(c1.pose.yaw + angle))
    back = -np.array([np.cos(head_yaw), np.sin(head_yaw)])
    lead_end = head + cfg.initial_lead * back
    tail_len = cfg.cable_length - cfg.initial_lead

    def clearance(a: np.ndarray, b: np.ndarray, skip_c1: bool) -> float:
        seg = np.array([a, b])
        d = [_polyline_distance(c.pose.xy, seg) - c.footprint_radius for c in clips if not (skip_c1 and c is c1)]
        d += [min(_point_segment_distance(q, a, b) for q in arc) for arc in arcs]
        return float(min(d))

    if clearance(head, lead_end, skip_c1=True) < cfg.initial_clip_clearance:
        raise SceneRejected("initial cable lead passes too close to the route")
    origin = np.zeros(2)
    if cfg.base_clearance > 0 and _point_segment_distance(origin, head, lead_end) < cfg.base_clearance:
        raise SceneRejected("initial cable lead passes too close to the robot base")
    best = None
    back_yaw = np.arctan2(back[1], back[0])
    for th in np.radians(np.arange(0, 360, 5)):
        bend = abs(float(wrap_angle(th - back_yaw)))
        if bend > np.radians(cfg.initial_max_bend_deg):
            continue
        tail = lead_end + tail_len * np.array([np.cos(th), np.sin(th)])
        if cfg.base_clearance > 0 and _point_segment_distance(origin, lead_end, tail) < cfg.base_clearance:
            continue
        key = (round(clearance(lead_end, tail, skip_c1=False), 3), -bend)
        if best is None or key > best[0]:
            best = (key, tail, bend)
    if best is None or best[0][0] < cfg.initial_clip_clearance:
        raise SceneRejected("no initial tail direction clears the route")
    (_, tail, bend) = best
    path = np.array([head, lead_end, tail])
    extras = {"initial_head_offset": {"axial": float(axial), "lateral": float(lateral),
                                      "angle_deg": float(np.degrees(angle))},
              "initial_tail_bend_deg": float(np.degrees(bend)),
              "initial_tail_clearance": float(best[0][0])}
    return path, head_yaw, extras


def _seed(pool: str, index: int, attempt: int, seed_base: int | None) -> int:
    return (POOL_SEED_BASE[pool] if seed_base is None else seed_base) + 1000 * index + attempt


def generate_scene(cfg: SceneFamilyConfig, pool: str, index: int, attempt: int = 0, *,
                   seed_base: int | None = None, id_prefix: str | None = None) -> Scene:
    """Generate scene ``index`` of ``pool``; raises SceneRejected with the failed check.

    ``seed_base`` replaces the pool's seed base and ``id_prefix`` the ``<family>-<pool>``
    part of the scene ID (datasets use both to give every cell its own seeds and IDs).
    """
    seed = _seed(pool, index, attempt, seed_base)
    rng = np.random.default_rng(seed)
    var = variant_for_index(index)
    raw, route_info = _build_route(cfg, rng, side=1 if var["reverse"] else -1)
    poses = _transform(raw, var["mirror"], var["reverse"])
    if "motif_base_turn_deg" in route_info:  # construction order -> route order, as _transform does
        base = np.asarray(route_info["motif_base_turn_deg"])
        base = -base[::-1] if var["reverse"] else base
        route_info["motif_base_turn_deg"] = (var["mirror"] * base).tolist()
    clips = [Clip(clip_label(i), p, cfg.clip_scale) for i, p in enumerate(poses)]

    arcs, stages = [], []
    for k in range(len(poses) - 1):
        a, b = poses[k], poses[k + 1]
        turn = float(wrap_angle(b.yaw - a.yaw))
        chord = float(np.linalg.norm(b.xy - a.xy))
        pts, length = _stage_path(a, b, cfg)
        if np.linalg.norm(pts[-1] - b.xy) > 1e-6:  # transforms must preserve the arc construction
            raise AssertionError("route transform broke stage geometry")
        arcs.append(pts)
        stages.append(StageStats(k + 2, chord, float(np.degrees(turn)), float(length)))
    reach_margin = min(_clip_margins(c, cfg)[0] for c in clips)

    path, head_yaw, extras = _initial_cable(cfg, clips, arcs, rng)
    route_arc = float(sum(s.arc_length for s in stages))
    # Cable left over once the head is head_allowance past CN with tail_reserve behind C01.
    slack = cfg.cable_length - (route_arc + cfg.head_allowance + cfg.tail_reserve)
    if slack < 0:
        raise SceneRejected(f"cable too short for the route (slack {slack:.3f})")
    headings = np.unwrap([p.yaw for p in poses])
    extras.update({**route_info, "total_heading_change_deg": float(np.degrees(headings[-1] - headings[0]))})
    if cfg.stage_shape != "arc":
        extras.update({"stage_shape": cfg.stage_shape,
                       "stage_bearing_deg": [float(np.degrees(wrap_angle(np.arctan2(*(b.xy - a.xy)[::-1]) - a.yaw)))
                                             for a, b in zip(poses[:-1], poses[1:])],
                       "stage_min_radius": [_min_radius(a) for a in arcs]})

    return Scene(
        scene_id=f"{id_prefix or f'{cfg.name}-{pool[:3]}'}-{index:04d}",
        pool=pool,
        seed=seed,
        variant=var,
        family=cfg.name,
        generator_version=GENERATOR_VERSION,
        clips=[c.to_dict() for c in clips],
        stages=[asdict(s) for s in stages],
        reach_margin=float(reach_margin),
        route_arc_length=route_arc,
        route_slack=float(slack),
        initial_head=[float(path[0][0]), float(path[0][1]), head_yaw],
        initial_tail=[float(path[-1][0]), float(path[-1][1])],
        cable_length=cfg.cable_length,
        cable_radius=cfg.cable_radius,
        initial_path=path.tolist(),
        extras=extras,
    )


def generate_pool(cfg: SceneFamilyConfig, pool: str, count: int, max_attempts: int = 400, *,
                  seed_base: int | None = None, id_prefix: str | None = None):
    """Generate ``count`` scenes; returns (scenes, rejection log)."""
    scenes, rejected = [], []
    for index in range(count):
        for attempt in range(max_attempts):
            try:
                scenes.append(generate_scene(cfg, pool, index, attempt, seed_base=seed_base, id_prefix=id_prefix))
                break
            except SceneRejected as e:
                rejected.append({"pool": pool, "index": index, "attempt": attempt,
                                 "seed": _seed(pool, index, attempt, seed_base), "reason": str(e)})
        else:
            raise RuntimeError(f"{pool} scene {index}: no feasible layout in {max_attempts} attempts")
    return scenes, rejected


def config_hash(cfg: SceneFamilyConfig) -> str:
    blob = json.dumps(asdict(cfg), sort_keys=True).encode() + GENERATOR_VERSION.encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def write_manifest(path: Path, cfg: SceneFamilyConfig, scenes: list[Scene], rejected: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "family": asdict(cfg),
        "generator_version": GENERATOR_VERSION,
        "config_hash": config_hash(cfg),
        "scenes": [asdict(s) for s in scenes],
        "rejected": rejected,
    }
    path.write_text(json.dumps(payload, indent=1))


def load_manifest(path: Path) -> tuple[dict, list[Scene]]:
    payload = json.loads(Path(path).read_text())
    return payload, [Scene.from_dict(s) for s in payload["scenes"]]
