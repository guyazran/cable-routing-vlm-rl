"""Franka kinematics/IK and the v3 layout features (base clearance, Hermite stages)."""

import json

import numpy as np
import pytest

from cablerouting.config import ROOT, load_config
from cablerouting.robot.franka import (Q_HIGH, Q_LOW, Q_READY, FrankaIK, finger_frames, link_frames,
                                       top_down_rotation)
from cablerouting.scenes.generator import _min_radius, _polyline_distance, _stage_path, generate_pool


def test_fk_zero_pose_matches_franka_asset():
    f = link_frames(np.zeros(7))
    # Link origins of the Isaac Sim 6.2 Franka USD in its default (q = 0) pose.
    assert np.allclose(f["panda_link3"][:3, 3], [0.0, 0.0, 0.649], atol=1e-6)
    assert np.allclose(f["panda_link4"][:3, 3], [0.0825, 0.0, 0.649], atol=1e-6)
    assert np.allclose(f["panda_link7"][:3, 3], [0.088, 0.0, 1.033], atol=1e-6)
    assert np.allclose(f["panda_hand"][:3, 3], [0.088, 0.0, 0.926], atol=1e-6)
    assert np.allclose(f["panda_hand"][:3, 2], [0.0, 0.0, -1.0], atol=1e-9)  # fingers point down


def test_top_down_rotation_closing_axis():
    for yaw in np.linspace(-3, 3, 7):
        R = top_down_rotation(yaw)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(R[:, 1], [-np.sin(yaw), np.cos(yaw), 0.0])
        assert np.allclose(R[:, 2], [0.0, 0.0, -1.0])


def test_finger_gap_equals_opening():
    hand = link_frames(Q_READY)["panda_hand"]
    f = finger_frames(hand, 0.03)
    assert np.isclose(np.linalg.norm(f["panda_leftfinger"][:3, 3] - f["panda_rightfinger"][:3, 3]), 0.03)


@pytest.mark.parametrize("r,bearing,z", [(0.3, 0, 0.005), (0.5, 80, 0.03), (0.7, -120, 0.005), (0.6, 170, 0.08)])
def test_ik_reaches_table_targets_top_down(r, bearing, z):
    ik = FrankaIK(point_depth=0.1)
    p = np.array([r * np.cos(np.radians(bearing)), r * np.sin(np.radians(bearing)), z])
    res = ik.solve(p, 0.7, Q_READY)
    assert res.pos_err < 1e-4 and res.rot_err < 1e-3
    assert np.all(res.q >= Q_LOW - 1e-9) and np.all(res.q <= Q_HIGH + 1e-9)
    tip, R, _ = ik.point_and_frames(res.q)
    assert np.allclose(tip, p, atol=1e-4) and R[2, 2] < -0.999


def test_ik_out_of_reach_reports_error():
    res = FrankaIK(point_depth=0.1).solve(np.array([1.0, 0.0, 0.005]), 0.0, Q_READY)
    assert res.pos_err > 0.05


def test_ik_tracking_keeps_wrist_branch():
    ik = FrankaIK(point_depth=0.1)
    q, branch = Q_READY.copy(), None
    for t in np.linspace(0, 1, 60):
        res = ik.solve(np.array([0.35 + 0.2 * t, -0.2 + 0.4 * t, 0.03]), 1.2 * t, q, keep_branch=branch, iters=60)
        if branch is not None:
            assert abs(np.angle(np.exp(1j * (res.yaw_branch - branch)))) < 0.2  # no half-turn flips
            assert np.max(np.abs(res.q - q)) < 0.3  # continuous joint motion
        q, branch = res.q, res.yaw_branch
        assert res.pos_err < 1e-4


def test_v3_reproduces_v2_layouts_when_new_features_are_off():
    pytest.skip("legacy layouts were recorded with setup v1's clip and cable (P1/P5); setup A changed both")
    cfg = load_config()
    old = json.loads((ROOT / "manifests" / "paper5_development.json").read_text())["scenes"][:4]
    new, _ = generate_pool(cfg.family("paper5"), "development", 4)
    for o, n in zip(old, new):
        assert o["clips"] == n.clips and o["initial_path"] == n.initial_path


@pytest.mark.parametrize("family", ["hairpin6", "scatter5", "zigzag5"])
def test_demo_families_respect_robot_base_and_shape(family):
    cfg = load_config()
    fam = cfg.family(family)
    scenes, _ = generate_pool(fam, "development", 3)
    for s in scenes:
        clips = s.clip_objects()
        assert len(clips) == fam.n_clips and s.route_slack >= 0
        for a, b in zip(clips[:-1], clips[1:]):
            path, _ = _stage_path(a.pose, b.pose, fam)
            assert np.allclose(path[0], a.pose.xy) and np.allclose(path[-1], b.pose.xy)
            assert _polyline_distance(np.zeros(2), path) >= fam.base_clearance - 1e-9
            if fam.stage_shape == "hermite":
                assert _min_radius(path) >= fam.min_path_radius
        assert _polyline_distance(np.zeros(2), np.array(s.initial_path)) >= fam.base_clearance - 1e-9
        turns = [abs(st["relative_angle_deg"]) for st in s.stages]
        assert min(turns) >= fam.turn_deg[0] - 1e-6 and max(turns) <= fam.turn_deg[1] + 1e-6
