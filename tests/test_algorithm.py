"""The algorithm against scripted worlds: no simulator and no environment code."""

import numpy as np
import pytest

from dloroute.agent import HierarchicalAgent
from dloroute.controllers.base import DecisionContext
from dloroute.controllers.reference import ReferenceController
from dloroute.controllers.vlm import VLMController
from dloroute.decision import Decision
from dloroute.geometry import Pose2D, point_at_arc_length
from dloroute.interface import CableState, ClipSpec, PrimitiveResult, SceneSpec, Transit, Views
from dloroute.planner.client import QueryRecord
from dloroute.planner.prompts import PromptBuilder, PromptConfig
from dloroute.rl.observation import ObsSpec
from dloroute.rl.training import Curriculum, InsertionEnv, InsertionTask
from dloroute.skills import SkillLibrary, SkillParams, flatten_head_command, pull_command
from dloroute.state import RouteStatus, head_metrics, route_status

R = 0.005
DEPTH = 0.015
C1 = ClipSpec("C01", Pose2D(0.0, 0.0, 0.0), 1.0, DEPTH)
C2 = ClipSpec("C02", Pose2D(0.25, 0.0, 0.0), 1.0, DEPTH)
C3 = ClipSpec("C03", Pose2D(-0.25, 0.0, 0.0), 1.0, DEPTH)
SCENE = SceneSpec((C1, C2, C3), R)


def straight(head_x: float, spacing: float = 0.0125, n: int = 60) -> np.ndarray:
    x = head_x - np.arange(n) * spacing
    return np.column_stack([x, np.zeros(n), np.full(n, R)])


def state(head_x: float, transits=(), partial=None, spacing: float = 0.0125) -> CableState:
    return CableState(straight(head_x, spacing), tuple(Transit(c, d) for c, d in transits), partial or {})


class ScriptedWorld:
    """The cable state advances one scripted step per pick-and-place."""

    def __init__(self, states, collide_on=(), state_allowed=True):
        self.scene = SCENE
        self.states = states
        self.collide_on = set(collide_on)
        self.state_allowed = state_allowed
        self.i = 0
        self.commands = []
        self.viewed = []

    def state(self) -> CableState:
        assert self.state_allowed, "privileged cable state read"
        return self.states[min(self.i, len(self.states) - 1)]

    def views(self, crop_clip: str) -> Views:
        self.viewed.append(crop_clip)
        img = np.zeros((4, 4, 3), np.uint8)
        return Views(img, img, {"clip": crop_clip})

    def pick_place(self, cmd) -> PrimitiveResult:
        self.commands.append(cmd)
        self.i += 1
        hit = self.i in self.collide_on
        return PrimitiveResult(hit, ["C01"] if hit else [], {"label": cmd.label})

    def grasp_collides(self, grasp_index: int) -> bool:
        return False


class RecordingPolicy:
    def __init__(self):
        self.obs = []

    def __call__(self, obs):
        self.obs.append(obs)
        return np.ones(7)  # grasp the last sampled point, both vias at the entry face, aligned


# --- state interpretation -------------------------------------------------------

def test_route_status_and_head_metrics():
    s = state(0.30, [("C01", 1), ("C02", 1)])
    assert route_status(s, ["C01", "C02"]) == RouteStatus(2, True, False)
    assert route_status(s, ["C01"]).extra_threading and not route_status(s, ["C01"]).complete
    assert route_status(state(0.3, [("C01", -1)]), ["C01"]).valid_prefix == 0
    m = head_metrics(C2, s)
    assert m.rope_out and m.signed_dist == pytest.approx(0.30 - (0.25 + DEPTH / 2))
    m = head_metrics(C1, state(0.0, partial={"C01": 1}))
    assert m.rope_in and not m.rope_out and m.signed_dist < 0


# --- skills ---------------------------------------------------------------------

def test_pull_lays_the_head_in_front_of_the_target():
    """R6b: Pull(C) carries the head, lifted, to C and lays it pull_stage in front of C's entry."""
    c = ClipSpec("C05", Pose2D(0.4, 0.1, 0.5), 1.0, DEPTH)
    pts = straight(0.0)
    p = SkillParams()
    cmd = pull_command(c, pts, p)
    assert cmd.grasp_index == point_at_arc_length(pts, p.pull_grasp_offset) == 0
    (wp,) = cmd.waypoints
    grasp_local = c.pose.to_local(np.array([wp.x, wp.y]))
    assert grasp_local == pytest.approx([-DEPTH / 2 - p.pull_stage - p.pull_grasp_offset, 0.0])
    assert wp.yaw == pytest.approx(0.5) and cmd.transport_height == p.pull_lift


def test_flatten_chooses_the_body_grasp_after_the_head_moves():
    world = ScriptedWorld([state(-0.3), state(-0.05, spacing=0.02)])
    out = SkillLibrary(RecordingPolicy()).execute(Decision("Flatten", "C02", False), world)
    head, body = world.commands
    assert head == flatten_head_command(C2, SkillParams()) and head.grasp_index == 0
    assert body.grasp_index == 3  # 6 cm at the post-move 2 cm spacing (5 at the original spacing)
    assert C2.pose.to_local(np.array([body.waypoints[0].x, 0.0]))[0] == pytest.approx(-DEPTH / 2 - 0.10)
    assert (out.physical_ops, out.internal_attempts, out.collided, out.insert_success) == (2, 0, False, None)


def test_scripted_skill_stops_at_a_collision():
    world = ScriptedWorld([state(-0.3)], collide_on={1})
    out = SkillLibrary(RecordingPolicy()).execute(Decision("Flatten", "C01", False), world)
    assert out.collided and out.physical_ops == 1 and out.collision_clips == ["C01"]


def test_insert_runs_the_policy_until_local_success():
    threaded = state(DEPTH / 2 + 0.03, [("C01", 1)])
    world = ScriptedWorld([state(-0.05), state(-0.01, partial={"C01": 1}), threaded])
    policy = RecordingPolicy()
    out = SkillLibrary(policy).execute(Decision("Insert", "C01", False), world)
    assert out.insert_success and out.internal_attempts == out.physical_ops == 2
    assert [o[-1] for o in policy.obs] == [0.0, 1.0]  # rope_in comes from the cable state
    assert all(o.shape == (ObsSpec().dim,) for o in policy.obs)
    cmd = world.commands[0]
    assert cmd.grasp_index == ObsSpec().point_indices()[-1] and len(cmd.waypoints) == 2
    assert cmd.transport_height == pytest.approx(R + SkillParams().insert_clearance)
    assert cmd.waypoints[-1].x == pytest.approx(-DEPTH / 2)


def test_insert_exhausts_the_local_horizon_or_stops_on_collision():
    out = SkillLibrary(RecordingPolicy()).execute(Decision("Insert", "C01", False), ScriptedWorld([state(-0.05)]))
    assert out.internal_attempts == SkillParams().local_horizon and out.insert_success is False
    out = SkillLibrary(RecordingPolicy()).execute(Decision("Insert", "C01", False),
                                                  ScriptedWorld([state(-0.05)], collide_on={3}))
    assert out.collided and out.internal_attempts == 3


# --- controllers ------------------------------------------------------------------

class FakeClient:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []

    def decide(self, messages, clip_ids, seed):
        self.calls.append((messages, clip_ids, seed))
        return QueryRecord(self.decisions.pop(0), None, [{"usage": {"prompt_tokens": 1}}])


def _vlm(client, privileged=False):
    return VLMController(client, PromptBuilder(PromptConfig(allow_missing_examples=True)), ["C01", "C02"],
                         SCENE.clip_ids, episode_seed=3, privileged_progress=privileged)


def test_vlm_controller_sees_only_views_and_its_own_history():
    client = FakeClient([Decision("Insert", "C02", False), Decision(None, None, True)])
    world = ScriptedWorld([state(-0.05)], state_allowed=False)
    agent = HierarchicalAgent(_vlm(client), SkillLibrary(RecordingPolicy(), SkillParams(local_horizon=1)))
    out = agent.decide(world, 0, 0, 20)
    assert out.decision == Decision("Insert", "C02", False) and out.info["crop"]["clip"] == "C01"
    world.state_allowed = True  # the Insert skill itself reads the cable state, as in the paper
    agent.act(out.decision, world)
    world.state_allowed = False
    out = agent.decide(world, 1, 1, 20)
    assert out.decision.done and world.viewed == ["C01", "C02"]  # crop follows the last commanded target
    task = client.calls[1][0][1]["content"][-4]["text"]
    assert "1. Insert(C02)" in task and "of 20" not in task and "privileged" not in task  # no budget shown (R15)
    assert [c[2] for c in client.calls] == [30_000, 30_002] and client.calls[0][1] == SCENE.clip_ids


def test_true_progress_diagnostic_adds_a_labelled_summary():
    client = FakeClient([Decision("Pull", "C01", False)])
    world = ScriptedWorld([state(0.05, [("C01", 1)])])
    out = _vlm(client, privileged=True).decide(DecisionContext(0, 0, 20, world))
    assert "privileged diagnostic" in out.info["task_text"]
    assert "Currently valid ordered prefix: C01." in out.info["task_text"]
    assert "First unsatisfied requested clip: C02." in out.info["task_text"]


def test_reference_controller_on_hand_built_states():
    ref = ReferenceController(SCENE, ["C01", "C02"], SkillParams(), render_images=False)
    assert ref.choose(state(0.30, [("C01", 1), ("C02", 1)]), lambda i: False) == (Decision(None, None, True),
                                                                                 "0-complete")
    ref.reset()
    barely_out = state(DEPTH / 2 + 0.01, [("C01", 1)])
    assert ref.choose(barely_out, lambda i: False) == (Decision("Insert", "C01", False), "1-finish-prev")
    ref.reset()
    out_of_c01 = state(DEPTH / 2 + 0.05, [("C01", 1)])
    assert ref.choose(out_of_c01, lambda i: False) == (Decision("Pull", "C02", False), "3-pull")
    ref.reset()
    assert ref.choose(out_of_c01, lambda i: True) == (Decision("Insert", "C02", False), "3-insert")


# --- training MDP -----------------------------------------------------------------

class FakeInsertionWorld:
    """Head on the target axis; a primitive leaves the head 8 cm ahead of its last waypoint."""

    def __init__(self):
        self.scene = SceneSpec((), R)
        self.scales = []

    def reset_task(self, rng, clip_scale):
        self.scales.append(clip_scale)
        self.scene = SceneSpec((ClipSpec("T", Pose2D(0, 0, 0), clip_scale, DEPTH * clip_scale),), R)
        self.head = -0.08 - rng.uniform(0, 0.02)
        return InsertionTask("T", {"mode": "fake"})

    def state(self):
        t = self.scene.clips[0]
        threaded = self.head > t.exit_x
        return CableState(straight(self.head), (Transit("T", 1),) if threaded else (), {})

    def pick_place(self, cmd):
        self.head = cmd.waypoints[-1].x + 0.08
        return PrimitiveResult(False)


def test_training_env_needs_no_simulator():
    from gymnasium.utils.env_checker import check_env

    world = FakeInsertionWorld()
    env = InsertionEnv(world, seed=0)
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=1)
    assert info["mode"] == "fake" and obs.shape == (ObsSpec().dim,)
    obs, r, term, trunc, info = env.step(np.ones(7, dtype=np.float32))
    assert term and not trunc and info["success"] and r > 0
    lo, hi = Curriculum().scale_initial
    assert all(lo <= s <= hi for s in world.scales)
    env.total_steps = Curriculum().switch_step
    env.reset()
    assert 0.9 <= world.scales[-1] <= 1.5


def test_skills_never_veto_their_own_grasps():
    """R15: no skill queries grasp collisions; a colliding motion is executed and ends the skill."""
    class NoQueryWorld(ScriptedWorld):
        def grasp_collides(self, grasp_index: int) -> bool:
            raise AssertionError("skills must not query grasp collisions")

    for skill in ("Insert", "Pull", "Flatten"):
        world = NoQueryWorld([state(-0.05)], collide_on={1})
        out = SkillLibrary(RecordingPolicy()).execute(Decision(skill, "C01", False), world)
        assert out.collided and out.physical_ops == 1
