"""Planar frames and cable arc length (table frame: z up, poses are (x, y, yaw))."""

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


def point_at_arc_length(points: np.ndarray, s: float) -> int:
    """Centreline index (head first) closest to arc length ``s`` from the head tip."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return int(np.argmin(np.abs(cum - s)))


def tangent_yaw(points: np.ndarray, i: int) -> float:
    """Heading of the cable at centreline point i, pointing towards the head."""
    a = points[max(i - 1, 0)]
    b = points[min(i + 1, len(points) - 1)]
    d = a[:2] - b[:2]
    return float(np.arctan2(d[1], d[0]))
