"""Start states for single-target insertion tasks (training and evaluating the insertion policy).

* ``single`` (paper): one clip; head randomised in a 10 cm (axial) x 5 cm
  (lateral) rectangle in front of the entry face, cable angle in [-10, 10] deg.
* ``route_stage`` (reconstruction, deviations R8/R6b): a preceding clip P is
  threaded and the head lies where a Pull to the target leaves it: in the same
  10 cm x 5 cm, +-10 deg region in front of the target as ``single``, with the
  cable curving there from P's exit. P is placed relative to the target with a
  stage chord and turn, so the policy sees the neighbouring clip it meets inside
  a route. The (chord, turn) pair is drawn
  uniformly from ``stages`` (every stage of a scene manifest, e.g. the official
  train split; :func:`stage_pool`) or, if that is empty, from the scene family's
  ranges.

The clip scale is chosen by the caller: the training curriculum is part of the
algorithm (dloroute.rl.training.Curriculum).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geometry.clip import Clip
from ..geometry.frames import Pose2D, wrap_angle
from .generator import SceneFamilyConfig, _arc_points, _hermite_points, load_manifest


@dataclass(frozen=True)
class InsertionStartDistribution:
    route_stage_prob: float = 0.5
    head_axial: tuple[float, float] = (0.03, 0.13)  # distance before the entry face (10 cm range)
    head_lateral: tuple[float, float] = (-0.025, 0.025)  # 5 cm range
    angle_deg: float = 10.0
    stage_tries: int = 50  # resample the head until the curve from P's exit clears both clips
    settle_steps: int = 240
    stages: tuple[tuple[float, float], ...] = ()  # (chord m, signed turn deg); empty = family ranges


@dataclass
class InsertionScene:
    clips: list[Clip]
    target: Clip
    polyline: np.ndarray  # head-first initial cable centreline
    mode: str


STAGE_BEND_OFFSET = 0.01  # route-stage start: where the cable bends, beyond P's exit face (m)


def _along_path(path: np.ndarray, length: float, link_length: float, radius: float) -> np.ndarray:
    """Head-first cable centreline along an xy polyline, resampled at the link length."""
    n = int(round(length / link_length))
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    s = np.arange(n + 1) * link_length
    xy = np.column_stack([np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])])
    return np.column_stack([xy, np.full(n + 1, radius)])


def _straight(head_xy: np.ndarray, yaw: float, length: float, link_length: float, radius: float) -> np.ndarray:
    n = int(round(length / link_length))
    s = np.arange(n + 1) * link_length
    xy = head_xy - s[:, None] * np.array([np.cos(yaw), np.sin(yaw)])
    return np.column_stack([xy, np.full(n + 1, radius)])


def sample_insertion_scene(rng: np.random.Generator, clip_scale: float, family: SceneFamilyConfig,
                           cable_length: float, link_length: float, cable_radius: float,
                           dist: InsertionStartDistribution = InsertionStartDistribution()) -> InsertionScene:
    def straight(head, yaw):
        return _straight(head, yaw, cable_length, link_length, cable_radius)

    target = Clip("T", Pose2D(0.0, 0.0, 0.0), clip_scale)
    if rng.random() >= dist.route_stage_prob:
        head = np.array([target.entry_x - rng.uniform(*dist.head_axial), rng.uniform(*dist.head_lateral)])
        yaw = np.radians(rng.uniform(-dist.angle_deg, dist.angle_deg))
        return InsertionScene([target], target, straight(head, yaw), "single")
    if dist.stages:
        chord, turn_deg = dist.stages[int(rng.integers(len(dist.stages)))]
        turn = np.radians(turn_deg)
    else:
        chord = rng.uniform(*family.stage_length)
        turn = rng.choice([-1, 1]) * np.radians(rng.uniform(*family.turn_deg))
    # Arc P -> T with P at the origin; then express P in T's frame.
    pts, _ = _arc_points(Pose2D(0, 0, 0), chord, turn)
    t_pose = Pose2D(pts[-1][0], pts[-1][1], float(wrap_angle(turn)))
    p_xy = t_pose.to_local(np.zeros(2))
    prev = Clip("P", Pose2D(p_xy[0], p_xy[1], float(wrap_angle(-turn))), clip_scale)
    # As the Pull to T leaves it (R6b): straight through P along its axis to just past the exit face,
    # then a smooth curve to a head pose drawn from the single-clip region in front of T.
    bend = Pose2D(*prev.pose.to_world(np.array([prev.exit_x + STAGE_BEND_OFFSET, 0.0])), prev.pose.yaw)
    for _ in range(dist.stage_tries):
        head = np.array([target.entry_x - rng.uniform(*dist.head_axial), rng.uniform(*dist.head_lateral)])
        yaw = np.radians(rng.uniform(-dist.angle_deg, dist.angle_deg))
        curve, _ = _hermite_points(bend, Pose2D(head[0], head[1], yaw), n=64)
        if _clear_of(curve[1:], [prev, target], cable_radius + 0.005):
            break
    path = np.vstack([curve[::-1], bend.xy - cable_length * prev.pose.heading])
    return InsertionScene([prev, target], target, _along_path(path, cable_length, link_length, cable_radius),
                          "route_stage")


def _clear_of(pts: np.ndarray, clips: list[Clip], margin: float) -> bool:
    """Whether every xy point lies outside each clip's footprint inflated by ``margin``."""
    for c in clips:
        loc = c.pose.to_local(pts)
        inside = (np.abs(loc[:, 0]) <= c.dims.depth / 2 + margin) & (np.abs(loc[:, 1]) <= c.half_length + margin)
        if inside.any():
            return False
    return True


def stage_pool(manifest: str | Path) -> tuple[tuple[float, float], ...]:
    """Every stage (chord, signed turn in degrees) of every scene in a manifest."""
    _, scenes = load_manifest(Path(manifest))
    return tuple((float(st["chord"]), float(st["relative_angle_deg"])) for s in scenes for st in s.stages)
