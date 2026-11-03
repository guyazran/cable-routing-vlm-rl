"""Planar rigid transforms (table frame: z up, poses are (x, y, yaw))."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def rot2(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def heading(self) -> np.ndarray:
        return np.array([np.cos(self.yaw), np.sin(self.yaw)])

    def to_local(self, pts: np.ndarray) -> np.ndarray:
        """World points (..., 2 or 3) -> this frame. z is passed through unchanged."""
        pts = np.asarray(pts, dtype=float)
        out = pts.copy()
        out[..., :2] = (pts[..., :2] - self.xy) @ rot2(self.yaw)
        return out

    def to_world(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        out = pts.copy()
        out[..., :2] = pts[..., :2] @ rot2(self.yaw).T + self.xy
        return out

    def yaw_to_world(self, local_yaw: float | np.ndarray) -> float | np.ndarray:
        return wrap_angle(np.asarray(local_yaw) + self.yaw)

    def yaw_to_local(self, world_yaw: float | np.ndarray) -> float | np.ndarray:
        return wrap_angle(np.asarray(world_yaw) - self.yaw)

    def as_list(self) -> list[float]:
        return [float(self.x), float(self.y), float(self.yaw)]


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_wxyz_to_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
