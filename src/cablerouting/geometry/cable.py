"""Cable centreline from capsule-chain link transforms (no simulator needed)."""

from __future__ import annotations

import numpy as np


def link_points(link_transforms: np.ndarray, link_length: float) -> np.ndarray:
    """Head-first centreline points (n_links + 1, 3) from link transforms [x, y, z, qx, qy, qz, qw].

    Link 0 is at the head; each link's local +x axis points towards the head.
    """
    tf = np.asarray(link_transforms, dtype=float)
    pos = tf[:, :3]
    x, y, z, w = tf[:, 3:7].T
    axis = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1)
    half = 0.5 * link_length * axis
    return np.vstack([pos[0] + half[0], pos - half])
