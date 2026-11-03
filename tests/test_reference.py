"""Reference controller on geometric cable states: the environment's evaluator supplies the
ground-truth state and the grasp-collision query, as the bridge does in a routing episode."""

from cablerouting.bridge import cable_state, scene_spec
from cablerouting.geometry.clip import Clip
from cablerouting.geometry.frames import Pose2D
from cablerouting.robot.executor import MotionParams, grasp_collides
from dloroute.controllers.reference import ReferenceController
from dloroute.skills import SkillParams

from test_geometry import R, line

C1 = Clip("C01", Pose2D(0.0, 0.0, 0.0))
C2 = Clip("C02", Pose2D(0.25, 0.0, 0.0))
ROUTE = ["C01", "C02"]


def decide(ctrl, pts, clips=(C1, C2)):
    closed = 2 * R + MotionParams().grasp_close_margin
    return ctrl.choose(cable_state(list(clips), pts), lambda i: grasp_collides(pts, i, list(clips), closed))


def make(clips=(C1, C2)):
    return ReferenceController(scene_spec(list(clips), R), ROUTE, SkillParams())


def test_insert_when_ready_then_done():
    ctrl = make()
    d, rule = decide(ctrl, line([(-0.05, 0.0), (-1.0, 0.0)]))
    assert (d.skill, d.target_clip, rule) == ("Insert", "C01", "2b-insert")
    d, rule = decide(ctrl, line([(0.35, 0.0), (-1.0, 0.0)]))
    assert d.done and rule == "0-complete"


def test_barely_out_finishes_the_previous_clip():
    ctrl = make()
    d, rule = decide(ctrl, line([(C1.exit_x + 0.008, 0.0), (-1.0, 0.0)]))
    assert (d.skill, d.target_clip, rule) == ("Insert", "C01", "1-finish-prev")


def test_pull_to_the_next_clip_once_the_head_is_out():
    ctrl = make()
    # C01 threaded, head 5 cm past its exit: out of C02's insertion region -> Pull(C02).
    pts = line([(C1.exit_x + 0.05, 0.0), (-1.0, 0.0)])
    d, rule = decide(ctrl, pts)
    assert (d.skill, d.target_clip, rule) == ("Pull", "C02", "3-pull")


def test_flatten_after_three_failed_inserts():
    ctrl = make()
    pts = line([(-0.05, 0.0), (-1.0, 0.0)])
    rules = [decide(ctrl, pts)[1] for _ in range(5)]
    assert rules == ["2b-insert", "2b-insert", "2b-insert", "2a-retry-flatten", "2b-insert"]


def test_wrong_threading_keeps_the_first_unsatisfied_target():
    clips = (C1, C2, Clip("C03", Pose2D(-0.25, 0.0, 0.0)))
    ctrl = make(clips)
    pts = line([(-0.10, 0.0), (-1.0, 0.0)])  # through C03, which is not requested; head in front of C01
    d, rule = decide(ctrl, pts, clips)
    assert (d.skill, d.target_clip, rule) == ("Insert", "C01", "2b-insert")
