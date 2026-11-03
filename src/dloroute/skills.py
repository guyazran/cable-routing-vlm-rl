"""Insert / Pull / Flatten skill definitions (protocol Sections 5 and 7).

Every skill takes a target clip and expands into pick-and-place primitives
executed by the world (:meth:`dloroute.interface.World.pick_place`). The same
command has the same semantics for the VLM controller and the ground-truth
reference controller.

Pull(C)    "Pull DLO out from the other side of the clip" (paper Fig. 2), towards target clip C,
           the next clip to thread (the authors' videos: Pull(Center) precedes Insert(Center)).
           Grasp the cable ``pull_grasp_offset`` behind the head tip, lift it to
           ``pull_lift``, carry it to C and lower it so the head lies ``pull_stage`` in
           front of C's entry face on C's axis, pointing along the axis; release. The
           cable is drawn through the clips it already passes (deviation R6b).
Flatten(C) Recovery to an insertion-feasible state in front of clip C, as two
           lifted pick-and-places: the head to ``flatten_head_distance`` in front
           of C's entry face on C's axis, then the point ``flatten_tail_offset``
           behind the head (taken from the cable state after the first move) to
           ``flatten_head_distance + flatten_tail_offset`` in front, so the front of
           the cable lies straight on C's axis. It moves cable physically and never
           resets or edits the cable state.
Collisions: every skill executes its motion as commanded. A gripper-clip collision
           ends the skill and, under the episode's collision rule, the episode, as in
           the paper (Sec. IV-A2; Fig. 6b shows a Flatten collision). No skill checks
           or vetoes its own grasp (deviation R13 was withdrawn on 2026-11-03, R15).
Insert(C)  The frozen RL policy, run closed-loop in C's local frame for up to
           ``local_horizon`` primitives. It stops early on local insertion success
           or a collision, using the cable state only inside the low-level skill.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .decision import Decision
from .geometry import point_at_arc_length, wrap_angle
from .interface import ClipSpec, PickPlace, Waypoint, World
from .rl.observation import ObsSpec, observation
from .rl.policies import Policy
from .rl.reward import RewardParams, insertion_success
from .state import head_metrics


@dataclass(frozen=True)
class SkillParams:
    lift_height: float = 0.03  # grasp-point height for Flatten transport
    pull_grasp_offset: float = 0.0  # arc length from the head tip to the Pull grasp (R6b: the head itself)
    pull_lift: float = 0.10  # grasp-point height while Pull carries the head (the videos show it lifted high)
    pull_stage: float = 0.08  # head distance in front of the target's entry face after a Pull
    flatten_head_distance: float = 0.04  # in front of the entry face
    flatten_tail_offset: float = 0.06
    local_horizon: int = 7  # RL primitives per Insert invocation (paper: up to 7)
    # Insertion action space (paper: 0.16 m x 0.16 m in front of the clip).
    insertion_region: float = 0.16
    max_rel_yaw: float = np.pi / 2
    insert_clearance: float = 0.001  # Insert carries the grasp point this far above resting height


# --- Pull and Flatten -----------------------------------------------------------

def pull_grasp_index(points: np.ndarray, p: SkillParams) -> int:
    return point_at_arc_length(points, p.pull_grasp_offset)


def pull_command(target: ClipSpec, points: np.ndarray, p: SkillParams) -> PickPlace:
    dest = target.pose.to_world(np.array([target.entry_x - p.pull_stage - p.pull_grasp_offset, 0.0]))
    return PickPlace(pull_grasp_index(points, p), (Waypoint(dest[0], dest[1], target.pose.yaw),),
                     p.pull_lift, f"Pull({target.clip_id})")


def flatten_head_command(target: ClipSpec, p: SkillParams) -> PickPlace:
    dest = target.pose.to_world(np.array([target.entry_x - p.flatten_head_distance, 0.0]))
    return PickPlace(0, (Waypoint(dest[0], dest[1], target.pose.yaw),), p.lift_height,
                     f"Flatten({target.clip_id})/head")


def flatten_body_command(target: ClipSpec, points: np.ndarray, p: SkillParams) -> PickPlace:
    """The body grasp: the point at arc length ``flatten_tail_offset`` behind the head, placed
    that far in front of the head's spot on C's axis."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    idx = point_at_arc_length(points, p.flatten_tail_offset)
    dest = target.pose.to_world(np.array([target.entry_x - p.flatten_head_distance - float(cum[idx]), 0.0]))
    return PickPlace(idx, (Waypoint(dest[0], dest[1], target.pose.yaw),), p.lift_height,
                     f"Flatten({target.clip_id})/body")


# --- Insert: the paper's 7-parameter action, in the target clip's frame -------

@dataclass(frozen=True)
class InsertAction:
    grasp_index: int  # index into the sampled observation points
    via_local: np.ndarray  # (2, 3): x, y, yaw of each via point in the clip frame


def decode_insert_action(a: np.ndarray, target: ClipSpec, n_obs_points: int, p: SkillParams) -> InsertAction:
    """Map a in [-1, 1]^7 to (grasp index, two local via points).

    a[0] -> grasp index in {0..K-1}; a[1:4], a[4:7] -> (x, y, yaw) of via 1 and 2:
    x in [entry - region, entry], y in [-region/2, region/2], yaw (heading of the
    held cable tangent relative to the clip's +x axis) in [-max_rel_yaw, max_rel_yaw].
    """
    a = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
    idx = int(np.clip(np.rint((a[0] + 1) / 2 * (n_obs_points - 1)), 0, n_obs_points - 1))
    vias = []
    for j in range(2):
        ax, ay, ayaw = a[1 + 3 * j: 4 + 3 * j]
        x = target.entry_x - p.insertion_region * (1 - (ax + 1) / 2)
        y = ay * p.insertion_region / 2
        vias.append([x, y, ayaw * p.max_rel_yaw])
    return InsertAction(idx, np.array(vias))


def insert_command(action: InsertAction, target: ClipSpec, spec: ObsSpec, transport_height: float) -> PickPlace:
    wps = []
    for x, y, yaw in action.via_local:
        w = target.pose.to_world(np.array([x, y]))
        wps.append(Waypoint(w[0], w[1], float(wrap_angle(target.pose.yaw + yaw))))
    return PickPlace(int(spec.point_indices()[action.grasp_index]), tuple(wps), transport_height,
                     f"Insert({target.clip_id})")


# --- execution --------------------------------------------------------------

@dataclass
class SkillOutcome:
    skill: str
    target: str
    physical_ops: int = 0  # pick-and-place primitives executed
    internal_attempts: int = 0  # RL primitives inside Insert
    collided: bool = False
    collision_clips: list[str] = field(default_factory=list)
    insert_success: bool | None = None  # low-level skill signal; never shown to the VLM
    details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class SkillLibrary:
    """The shared skills: the frozen insertion policy and the scripted Pull and Flatten."""

    def __init__(self, policy: Policy, params: SkillParams = SkillParams(), obs_spec: ObsSpec = ObsSpec(),
                 reward: RewardParams = RewardParams()):
        self.policy = policy
        self.p = params
        self.obs_spec = obs_spec
        self.rp = reward

    def execute(self, d: Decision, world: World) -> SkillOutcome:
        target = world.scene.clip(d.target_clip)
        if d.skill == "Insert":
            return self.insert(target, world)
        out = SkillOutcome(d.skill, d.target_clip)
        p = self.p
        steps = [lambda pts: pull_command(target, pts, p)] if d.skill == "Pull" else \
            [lambda pts: flatten_head_command(target, p),
             lambda pts: flatten_body_command(target, pts, p)]
        for make in steps:
            r = world.pick_place(make(world.state().points))
            out.physical_ops += 1
            out.details.append(r.details)
            if r.collided:
                out.collided, out.collision_clips = True, list(r.collision_clips)
                break
        return out

    def insert(self, target: ClipSpec, world: World) -> SkillOutcome:
        out = SkillOutcome("Insert", target.clip_id, insert_success=False)
        height = world.scene.cable_radius + self.p.insert_clearance
        for t in range(1, self.p.local_horizon + 1):
            st = world.state()
            m = head_metrics(target, st)
            a = np.asarray(self.policy(observation(st.points, target, m.rope_in, self.obs_spec)), dtype=float)
            act = decode_insert_action(a, target, self.obs_spec.n_points, self.p)
            cmd = insert_command(act, target, self.obs_spec, height)
            r = world.pick_place(cmd)
            out.physical_ops = out.internal_attempts = t
            m = head_metrics(target, world.state())
            out.details.append({"t": t, "action": a.tolist(), "grasp_index": act.grasp_index,
                                "via_local": act.via_local.tolist(), "exec": r.details,
                                "signed_dist": m.signed_dist, "rope_in": m.rope_in, "rope_out": m.rope_out})
            if r.collided:
                out.collided, out.collision_clips = True, list(r.collision_clips)
                break
            if insertion_success(m, self.rp):
                out.insert_success = True
                break
        return out
