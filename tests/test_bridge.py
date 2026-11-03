"""The bridge: the algorithm's reading of a cable state agrees with the environment's grader."""

import numpy as np
import pytest

from cablerouting.bridge import cable_state, clip_spec, to_plan, to_result
from cablerouting.geometry.clip import Clip
from cablerouting.geometry.frames import Pose2D
from cablerouting.geometry.threading import evaluate_routing, head_metrics as env_head_metrics
from cablerouting.robot.executor import ExecutionResult
from cablerouting.scenes.generator import SceneFamilyConfig
from cablerouting.scenes.insertion import InsertionStartDistribution, sample_insertion_scene
from dloroute.interface import PickPlace, Waypoint
from dloroute.state import head_metrics, route_status

from test_geometry import line

C1 = Clip("C01", Pose2D(0.0, 0.0, 0.0))
C2 = Clip("C02", Pose2D(0.25, 0.0, 0.0), 1.3)
C3 = Clip("C03", Pose2D(0.25, 0.25, np.pi / 2))
CLIPS = [C1, C2, C3]
ROUTE = ["C01", "C02", "C03"]
CASES = {
    "before C01": [(-0.05, 0.0), (-1.0, 0.0)],
    "inside C01": [(0.0, 0.0), (-1.0, 0.0)],
    "through C01": [(0.10, 0.0), (-1.0, 0.0)],
    "C01 C02, head turning": [(0.30, 0.05), (0.30, 0.0), (-1.0, 0.0)],
    "C01 C02 C03": [(0.25, 0.32), (0.25, 0.10), (0.30, 0.0), (-1.0, 0.0)],
    "reverse C02": [(0.10, 0.0), (0.40, 0.0), (0.40, -0.1), (-1.0, -0.1)],
    "beside C01": [(0.10, 0.05), (-1.0, 0.05)],
}


@pytest.mark.parametrize("name", CASES)
def test_algorithm_state_matches_the_grader(name):
    pts = line(CASES[name])
    graded = evaluate_routing(CLIPS, pts, ROUTE)
    st = cable_state(CLIPS, pts)
    rs = route_status(st, ROUTE)
    assert (rs.valid_prefix, rs.complete) == (graded.valid_prefix, graded.complete)
    assert rs.extra_threading == (len(graded.sequence) > graded.valid_prefix)
    for c in CLIPS:
        a, b = head_metrics(clip_spec(c), st), env_head_metrics(c, graded, pts[0])
        assert (a.rope_in, a.rope_out) == (b.rope_in, b.rope_out)
        assert a.signed_dist == pytest.approx(b.signed_dist) and a.d_floor == pytest.approx(b.d_floor)


def test_clip_spec_carries_the_scaled_tunnel():
    s = clip_spec(C2)
    assert (s.entry_x, s.exit_x) == (C2.entry_x, C2.exit_x)
    assert np.allclose(s.floor_point(), C2.floor_point_world()) and np.allclose(s.ceil_point(), C2.ceil_point_world())


def test_command_and_result_conversion():
    cmd = PickPlace(7, (Waypoint(0.1, 0.2, 0.3), Waypoint(0.4, 0.5, 0.6)), 0.006, "Insert(C01)")
    plan = to_plan(cmd)
    assert (plan.grasp_point, plan.transport_height, plan.label) == (7, 0.006, "Insert(C01)")
    assert [(v.x, v.y, v.yaw) for v in plan.vias] == [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)]
    r = to_result(ExecutionResult("x", collided=True, collision_clips=["C02"], collision_phase="via1"))
    assert r.collided and r.collision_clips == ["C02"] and r.details["collision_phase"] == "via1"


def test_insertion_start_states():
    fam, dist = SceneFamilyConfig(), InsertionStartDistribution()
    rng = np.random.default_rng(0)
    modes = set()
    for _ in range(20):
        s = sample_insertion_scene(rng, 1.4, fam, 3.0, 0.0125, 0.0025, dist)
        modes.add(s.mode)
        assert s.target.scale == 1.4 and len(s.polyline) == 241
        head = s.target.pose.to_local(s.polyline[0])
        # Both modes: the head lies in front of the target, where a Pull to it leaves the head (R6b).
        assert s.target.entry_x - 0.13 - 1e-6 <= head[0] <= s.target.entry_x - 0.03 + 1e-6 and abs(head[1]) <= 0.025
        if s.mode == "route_stage":  # the cable runs through the preceding clip
            p = s.clips[0].pose.to_local(s.polyline)
            assert np.any((np.abs(p[:, 0]) < s.clips[0].dims.depth / 2) & (np.abs(p[:, 1]) < 0.005))
    assert modes == {"single", "route_stage"}
