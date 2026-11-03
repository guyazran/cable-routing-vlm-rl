import numpy as np
import pytest

from cablerouting.geometry.clip import Clip
from cablerouting.geometry.collision import colliding_clips
from cablerouting.geometry.frames import Pose2D
from cablerouting.geometry.threading import evaluate_routing, head_metrics, in_insertion_region
from cablerouting.robot.gripper import gripper_boxes

R = 0.005  # cable radius: centreline height when resting on the table


def line(points_xy, spacing=0.0125, z=R):
    """Head-first polyline through the given xy waypoints (first = head)."""
    pts = [np.asarray(points_xy[0], float)]
    for a, b in zip(points_xy[:-1], points_xy[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        pts += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
    p = np.array(pts)
    return np.column_stack([p, np.full(len(p), z)])


C1 = Clip("C01", Pose2D(0.0, 0.0, 0.0))
C2 = Clip("C02", Pose2D(0.2, 0.0, 0.0))
C3 = Clip("C03", Pose2D(0.2, 0.2, np.pi / 2))


def test_straight_through_one_clip():
    pts = line([(0.05, 0.0), (-1.0, 0.0)])
    r = evaluate_routing([C1, C2], pts, ["C01", "C02"])
    assert [t.clip_id for t in r.sequence] == ["C01"] and r.valid_prefix == 1 and not r.complete
    m = head_metrics(C1, r, pts[0])
    assert m.rope_out and m.signed_dist == pytest.approx(0.05 - C1.exit_x, abs=1e-9)


def test_two_clips_in_order_complete():
    pts = line([(0.30, 0.0), (-1.0, 0.0)])
    r = evaluate_routing([C1, C2], pts, ["C01", "C02"])
    assert r.complete and r.valid_prefix == 2 and r.errors == []


def test_passing_beside_is_not_threading():
    pts = line([(0.05, 0.04), (-1.0, 0.04)])  # 4 cm lateral: outside the 1.1 cm half opening
    r = evaluate_routing([C1], pts, ["C01"])
    assert r.sequence == [] and r.valid_prefix == 0


def test_passing_over_the_bridge_is_not_threading():
    pts = line([(0.05, 0.0), (-1.0, 0.0)], z=C1.top_height + R)
    assert evaluate_routing([C1], pts, ["C01"]).sequence == []


def test_head_beyond_exit_without_passing_through():
    # Head ends beyond the exit plane but the cable came around the side.
    pts = line([(0.05, 0.0), (0.05, 0.05), (-0.5, 0.05)])
    r = evaluate_routing([C1], pts, ["C01"])
    assert r.sequence == [] and not head_metrics(C1, r, pts[0]).rope_out


def test_partial_insertion():
    pts = line([(0.0, 0.0), (-1.0, 0.0)])  # head at the clip centre, inside the tunnel
    r = evaluate_routing([C1], pts, ["C01"])
    assert r.sequence == [] and r.clips["C01"].partial == 1
    assert head_metrics(C1, r, pts[0]).rope_in


def test_reverse_threading_is_an_error():
    pts = line([(-0.05, 0.0), (1.0, 0.0)])  # passes C01 against its insertion direction
    r = evaluate_routing([C1], pts, ["C01"])
    assert r.sequence[0].direction == -1 and r.valid_prefix == 0 and any("reverse" in e for e in r.errors)


def test_wrong_order_and_non_requested():
    pts = line([(0.20, 0.30), (0.20, -0.10), (-1.0, -0.10)])  # through C03 only
    r = evaluate_routing([C1, C2, C3], pts, ["C01", "C02"])
    assert r.valid_prefix == 0 and any("non-requested" in e for e in r.errors)
    pts2 = line([(0.20, 0.30), (0.2, 0.0), (-1.0, 0.0)])  # C01 then C02?: no, C02 is passed sideways
    r2 = evaluate_routing([C1, C2, C3], pts2, ["C01", "C03"])
    assert r2.valid_prefix >= 1


def test_unthreading_regression():
    threaded = line([(0.30, 0.0), (-1.0, 0.0)])
    pulled_back = line([(0.10, 0.0), (-1.2, 0.0)])
    assert evaluate_routing([C1, C2], threaded, ["C01", "C02"]).valid_prefix == 2
    assert evaluate_routing([C1, C2], pulled_back, ["C01", "C02"]).valid_prefix == 1


def test_insertion_region():
    assert in_insertion_region(C1, np.array([-0.05, 0.0, R]))
    assert not in_insertion_region(C1, np.array([0.05, 0.0, R]))
    assert not in_insertion_region(C1, np.array([-0.05, 0.09, R]))


def test_gripper_collision():
    # Fingers straddling the cable right at the clip -> collision; 6 cm before it -> clear.
    assert colliding_clips(gripper_boxes(np.array([0.0, 0.0, R, 0.0]), 0.012), [C1]) == ["C01"]
    assert colliding_clips(gripper_boxes(np.array([-0.06, 0.0, R, 0.0]), 0.012), [C1]) == []
    # Above the clip at safe height -> clear.
    assert colliding_clips(gripper_boxes(np.array([0.0, 0.0, 0.08, 0.0]), 0.08), [C1]) == []


def test_clip_scale_opening():
    assert Clip("X", Pose2D(0, 0, 0), 2.2).dims.opening_width == pytest.approx(0.0484)
