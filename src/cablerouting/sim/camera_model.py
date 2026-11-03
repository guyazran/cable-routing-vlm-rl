"""Top-down pinhole camera model shared by the renderer, overlays and crop logging.

Cameras look straight down (-z) with image up = world +x and image right =
world -y (the virtual robot base sits at the bottom of the full view). The same
intrinsics are written to the USD camera, so overlays drawn with ``project``
align with the rendered pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HORIZONTAL_APERTURE_MM = 20.955


@dataclass(frozen=True)
class TopDownCamera:
    center_xy: tuple[float, float]
    height: float  # camera height above the table (m)
    view_width: float  # world extent along -y covered at table height (m)
    resolution: tuple[int, int]  # (width, height) pixels

    @property
    def view_height(self) -> float:
        w, h = self.resolution
        return self.view_width * h / w

    @property
    def focal_mm(self) -> float:
        return HORIZONTAL_APERTURE_MM * self.height / self.view_width

    @property
    def vertical_aperture_mm(self) -> float:
        w, h = self.resolution
        return HORIZONTAL_APERTURE_MM * h / w

    @property
    def f_px(self) -> float:
        return self.focal_mm / HORIZONTAL_APERTURE_MM * self.resolution[0]

    @property
    def quat_wxyz(self) -> np.ndarray:
        """Rotation Rz(-90 deg): camera x -> world -y, camera y (image up) -> world +x."""
        return np.array([np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)])

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """World points (..., 3) -> pixel (u right, v down)."""
        p = np.asarray(xyz, dtype=float)
        cx, cy = self.center_xy
        depth = self.height - p[..., 2]
        w, h = self.resolution
        u = w / 2 - self.f_px * (p[..., 1] - cy) / depth
        v = h / 2 - self.f_px * (p[..., 0] - cx) / depth
        return np.stack([u, v], axis=-1)

    def world_bounds(self) -> dict:
        """Table-plane rectangle covered by the image (for crop-bound logging)."""
        cx, cy = self.center_xy
        return {"x": [cx - self.view_height / 2, cx + self.view_height / 2],
                "y": [cy - self.view_width / 2, cy + self.view_width / 2]}


def full_view_camera(view_x=(-0.25, 0.80), view_y=(-0.70, 0.70), resolution=(1600, 1200),
                     height: float = 2.0) -> TopDownCamera:
    return TopDownCamera(((view_x[0] + view_x[1]) / 2, (view_y[0] + view_y[1]) / 2), height,
                         view_y[1] - view_y[0], resolution)


def zoom_camera(center_xy, size: float = 0.24, resolution: int = 768, height: float = 0.6) -> TopDownCamera:
    return TopDownCamera((float(center_xy[0]), float(center_xy[1])), height, size, (resolution, resolution))
