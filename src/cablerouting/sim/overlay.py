"""Visible clip labels and insertion-direction arrows, drawn onto rendered images.

Labels are image annotations placed beside each clip's flange so they never
cover the bridge, where threading has to be judged. They are identical in the
five- and ten-target conditions.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..geometry.clip import Clip
from .camera_model import TopDownCamera

LABEL_OFFSET = 0.022  # beyond the flange end, along the clip's long axis (m)
ARROW_HALF = 0.018  # arrow half-length along the insertion axis (m)
TEXT_HEIGHT = 0.016  # label text height on the table (m)


def annotate(img: np.ndarray, clips: list[Clip], cam: TopDownCamera) -> Image.Image:
    im = Image.fromarray(np.asarray(img)[..., :3]).convert("RGB")
    draw = ImageDraw.Draw(im)
    px_per_m = cam.f_px / cam.height
    font = ImageFont.load_default(size=max(10, int(TEXT_HEIGHT * px_per_m)))
    lw = max(2, int(0.002 * px_per_m))
    for c in clips:
        side = c.half_length + LABEL_OFFSET
        a0 = c.pose.to_world(np.array([-ARROW_HALF, side, 0.0]))
        a1 = c.pose.to_world(np.array([ARROW_HALF, side, 0.0]))
        (u0, v0), (u1, v1) = cam.project(np.array([a0, a1]))
        draw.line([(u0, v0), (u1, v1)], fill=(20, 60, 200), width=lw)
        d = np.array([u1 - u0, v1 - v0])
        d /= max(np.linalg.norm(d), 1e-6)
        n = np.array([-d[1], d[0]])
        head = 0.35 * np.hypot(u1 - u0, v1 - v0)
        tip = np.array([u1, v1])
        draw.polygon([tuple(tip), tuple(tip - head * d + 0.5 * head * n), tuple(tip - head * d - 0.5 * head * n)],
                     fill=(20, 60, 200))
        t = cam.project(c.pose.to_world(np.array([0.0, side + 0.024, 0.0])))
        draw.text((t[0], t[1]), c.clip_id, fill=(0, 0, 0), font=font, anchor="mm",
                  stroke_width=max(1, lw // 2), stroke_fill=(255, 255, 255))
    return im
