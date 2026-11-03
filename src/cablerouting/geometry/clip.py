"""Omega (pipe-saddle) clip geometry.

Clip-local frame: origin at the clip centre on the table surface, +x is the
insertion axis (entry face at x = -depth/2, exit face at x = +depth/2), +y runs
along the clip's long axis (flange to flange), +z up. The cable passes under the
bridge through the aperture |y| < width/2, 0 < z < height.

``scale`` multiplies every dimension uniformly, so the opening width is
2.2 cm * scale as in the paper's curriculum (scale 2.2 -> 4.84 cm).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import Pose2D


@dataclass(frozen=True)
class ClipDims:
    """Unscaled clip dimensions in metres. ``opening_width`` is the paper's 2.2 cm. Depth and flange
    length match the clip drawn to scale against the 0.16 m action square in the paper's Fig. 3
    (about 7.8 cm flange to flange, 3 cm deep; deviation P5b). The rest are reconstruction choices."""

    opening_width: float = 0.022
    opening_height: float = 0.016
    depth: float = 0.030
    wall: float = 0.003
    flange_length: float = 0.025
    flange_thickness: float = 0.003

    def scaled(self, s: float) -> "ClipDims":
        return ClipDims(*(s * v for v in (
            self.opening_width, self.opening_height, self.depth,
            self.wall, self.flange_length, self.flange_thickness)))


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in the clip-local frame."""

    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class Clip:
    clip_id: str
    pose: Pose2D
    scale: float = 1.0
    base_dims: ClipDims = field(default_factory=ClipDims)

    @property
    def dims(self) -> ClipDims:
        return self.base_dims.scaled(self.scale)

    # --- solid geometry ----------------------------------------------------
    def boxes(self) -> list[Box]:
        d = self.dims
        hw = d.opening_width / 2
        return [
            Box("bridge", (0.0, 0.0, d.opening_height + d.wall / 2),
                (d.depth, d.opening_width + 2 * d.wall, d.wall)),
            Box("wall_pos", (0.0, hw + d.wall / 2, d.opening_height / 2),
                (d.depth, d.wall, d.opening_height)),
            Box("wall_neg", (0.0, -hw - d.wall / 2, d.opening_height / 2),
                (d.depth, d.wall, d.opening_height)),
            Box("flange_pos", (0.0, hw + d.wall + d.flange_length / 2, d.flange_thickness / 2),
                (d.depth, d.flange_length, d.flange_thickness)),
            Box("flange_neg", (0.0, -hw - d.wall - d.flange_length / 2, d.flange_thickness / 2),
                (d.depth, d.flange_length, d.flange_thickness)),
        ]

    @property
    def half_length(self) -> float:
        """Half extent along the long axis (y)."""
        d = self.dims
        return d.opening_width / 2 + d.wall + d.flange_length

    @property
    def top_height(self) -> float:
        d = self.dims
        return d.opening_height + d.wall

    @property
    def footprint_radius(self) -> float:
        d = self.dims
        return float(np.hypot(self.half_length, d.depth / 2))

    # --- reference points used by the reward and the evaluator ---------------
    @property
    def entry_x(self) -> float:
        return -self.dims.depth / 2

    @property
    def exit_x(self) -> float:
        return self.dims.depth / 2

    def floor_point_world(self) -> np.ndarray:
        """Centre of the entry face ("floor" in the paper's Fig. 3), table-plane xy."""
        return self.pose.to_world(np.array([self.entry_x, 0.0]))

    def ceil_point_world(self) -> np.ndarray:
        """Centre of the exit face ("ceiling"), table-plane xy."""
        return self.pose.to_world(np.array([self.exit_x, 0.0]))

    def in_aperture(self, local_yz: np.ndarray, tol: float = 0.0) -> np.ndarray:
        """Whether local (y, z) points lie inside the aperture cross-section."""
        d = self.dims
        y, z = local_yz[..., 0], local_yz[..., 1]
        return (np.abs(y) <= d.opening_width / 2 + tol) & (z <= d.opening_height + tol)

    def to_dict(self) -> dict:
        return {"clip_id": self.clip_id, "pose": self.pose.as_list(), "scale": self.scale}

    @staticmethod
    def from_dict(d: dict, base_dims: ClipDims | None = None) -> "Clip":
        return Clip(d["clip_id"], Pose2D(*d["pose"]), d.get("scale", 1.0), base_dims or ClipDims())


def clip_label(index: int) -> str:
    """0-based route index -> visible label C01, C02, ..."""
    return f"C{index + 1:02d}"
