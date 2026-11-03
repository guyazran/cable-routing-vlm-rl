"""Shape x length scene dataset: train / validation / test splits and the study's prespecified subsets.

The dataset crosses route shapes (scene families with a ``motif``, e.g. arc,
circle, zigzag, square) with route lengths (number of clips). Every
(shape, n_clips) cell is generated independently, and scene i of a cell takes
the counterbalancing variant of index i (mirror x reverse, period 4).

Train and test (protocol roles: development and evaluation) come from one pool
per cell. Cell c draws seeds from ``seed_base + 100000 c``. Within each cell and
each variant the scenes are permuted with ``split_seed``, and a fixed number go
to train and the rest to test.

Validation (protocol role: the calibration pool, Section 9) is a separate pool
of the same families. It uses its own seed range, ``validation_seed_base + 100000 c``.
Generating it consumes nothing from the train/test pools or their split, so
adding it leaves train and test unchanged.

Every split holds the same number of scenes per cell, balanced over the four
variants. All seed ranges are disjoint from each other and from the
development/calibration/evaluation pools of the generator.

Subsets the protocol runs on are fixed by rule before any episode
(:func:`select_subsets`):

* **screen**: ``screen_per_shape`` ten-clip test scenes per shape;
* **calibration round r**: one validation scene per variant for each shape and
  each calibration length, so 4 per shape and 24 per length.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .generator import (GENERATOR_VERSION, POOL_SEED_BASE, Scene, SceneFamilyConfig, config_hash, generate_pool,
                        variant_for_index)

SPLITS = ("train", "validation", "test")
N_VARIANTS = 4  # mirror x reverse (generator.variant_for_index)
VARIANTS = [tuple(variant_for_index(i).values()) for i in range(N_VARIANTS)]  # fixed variant order
CELL_SEED_STRIDE = 100_000


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "shapes"
    shapes: tuple[str, ...] = ("arc", "circle", "spiral", "serpentine", "zigzag", "square")
    n_clips: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 9, 10)
    train_per_cell: int = 20
    test_per_cell: int = 20
    validation_per_cell: int = 8
    seed_base: int = 10_000_000
    validation_seed_base: int = 20_000_000
    split_seed: int = 20261102
    max_attempts: int = 400
    screen_n_clips: int = 10
    screen_per_shape: int = 5
    calibration_n_clips: tuple[int, ...] = (5, 10)

    @property
    def per_cell(self) -> int:
        """Scenes per cell in the train/test pool."""
        return self.train_per_cell + self.test_per_cell

    @property
    def calibration_rounds(self) -> int:
        return self.validation_per_cell // N_VARIANTS

    def cells(self) -> list[tuple[int, str, int]]:
        """(cell index, shape, n_clips) in seed order."""
        return [(i * len(self.n_clips) + j, shape, n)
                for i, shape in enumerate(self.shapes) for j, n in enumerate(self.n_clips)]

    def cell_seed_base(self, cell: int, split: str = "train") -> int:
        base = self.validation_seed_base if split == "validation" else self.seed_base
        return base + CELL_SEED_STRIDE * cell

    def validate(self, families: dict[str, SceneFamilyConfig]) -> None:
        for k in (self.train_per_cell, self.test_per_cell, self.validation_per_cell):
            if k <= 0 or k % N_VARIANTS:
                raise ValueError(f"per-cell split counts must be positive multiples of {N_VARIANTS}: {k}")
        if 1000 * (max(self.per_cell, self.validation_per_cell) - 1) + self.max_attempts > CELL_SEED_STRIDE:
            raise ValueError("cell seed ranges would overlap")
        if self.seed_base < max(POOL_SEED_BASE.values()) + CELL_SEED_STRIDE * 10:
            raise ValueError("dataset seeds must lie above the protocol pools' seed ranges")
        if abs(self.validation_seed_base - self.seed_base) < CELL_SEED_STRIDE * len(self.cells()):
            raise ValueError("validation and train/test seed ranges overlap")
        missing = [s for s in self.shapes if s not in families]
        if missing:
            raise KeyError(f"dataset shapes without a scene family: {missing}")
        if len(set(self.shapes)) != len(self.shapes) or len(set(self.n_clips)) != len(self.n_clips):
            raise ValueError("duplicate shape or length")
        if min(self.n_clips) < 2:
            raise ValueError("a route needs at least two clips")
        if self.screen_n_clips not in self.n_clips or not set(self.calibration_n_clips) <= set(self.n_clips):
            raise ValueError("screen and calibration lengths must be dataset lengths")
        if not 0 < self.screen_per_shape <= self.test_per_cell:
            raise ValueError("screen_per_shape out of range")


def cell_id(shape: str, n_clips: int) -> str:
    return f"{shape}-n{n_clips:02d}"


def build_dataset(families: dict[str, SceneFamilyConfig], d: DatasetConfig, log=print) -> dict:
    """Generate every cell and split it. Returns scenes, rejection logs and cell records per split."""
    d.validate(families)
    split_rng = np.random.default_rng(d.split_seed)
    k_train = d.train_per_cell // N_VARIANTS
    out = {sp: {"scenes": [], "rejected": []} for sp in SPLITS}
    cells = []
    for cell, shape, n in d.cells():
        fam = dataclasses.replace(families[shape], n_clips=n)
        cid = cell_id(shape, n)
        scenes, rejected = generate_pool(fam, "dataset", d.per_cell, d.max_attempts,
                                         seed_base=d.cell_seed_base(cell), id_prefix=cid)
        split_of: dict[int, str] = {}
        for v in VARIANTS:
            group = [i for i in range(d.per_cell) if _variant_key(i) == v]
            order = split_rng.permutation(group)
            split_of.update({int(i): "train" for i in order[:k_train]})
            split_of.update({int(i): "test" for i in order[k_train:]})
        for i, s in enumerate(scenes):
            s.pool = split_of[i]
        rejected = [{**r, "pool": split_of[r["index"]]} for r in rejected]
        val, val_rejected = generate_pool(fam, "dataset", d.validation_per_cell, d.max_attempts,
                                          seed_base=d.cell_seed_base(cell, "validation"), id_prefix=f"{cid}-val")
        for s in val:
            s.pool = "validation"
        rejected += [{**r, "pool": "validation"} for r in val_rejected]
        for s in scenes + val:
            out[s.pool]["scenes"].append(s)
        for r in rejected:
            out[r["pool"]]["rejected"].append({**r, "cell": cid})
        cells.append({"cell": cell, "cell_id": cid, "shape": shape, "n_clips": n,
                      "seed_base": d.cell_seed_base(cell), "validation_seed_base": d.cell_seed_base(cell, "validation"),
                      "family_hash": config_hash(fam),
                      "accepted": len(scenes) + len(val), "rejected": len(rejected),
                      "rejection_reasons": dict(Counter(r["reason"].split(" (")[0] for r in rejected)),
                      **{sp: sorted(s.scene_id for s in scenes + val if s.pool == sp) for sp in SPLITS}})
        log(f"[dataset] {cid}: {len(scenes)} train/test + {len(val)} validation scenes, "
            f"{len(rejected)} rejected attempts")
    return {"config": d, "families": {s: families[s] for s in d.shapes}, "cells": cells, "splits": out}


def _variant_key(index: int) -> tuple:
    return tuple(variant_for_index(index).values())


def _scene_variant(s: Scene) -> tuple:
    return s.variant["mirror"], s.variant["reverse"]


def _pick(scenes: list[Scene], k: int, offset: int = 0, rank: int = 0) -> list[Scene]:
    """k scenes, cycling over the variants from VARIANTS[offset], lowest scene ID first, from ``rank`` on."""
    groups = [sorted((s for s in scenes if _scene_variant(s) == v), key=lambda s: s.scene_id) for v in VARIANTS]
    picked: list[Scene] = []
    while len(picked) < k:
        row = [groups[(offset + j) % N_VARIANTS][rank] for j in range(N_VARIANTS)
               if rank < len(groups[(offset + j) % N_VARIANTS])]
        if not row:
            raise ValueError(f"cannot pick {k} scenes")
        picked += row[: k - len(picked)]
        rank += 1
    return picked


def select_subsets(ds: dict) -> dict[str, dict]:
    """The protocol's prespecified subsets. They depend only on scene IDs and variants, never on outcomes.

    * screen: for the s-th shape (dataset order), its ``screen_n_clips`` test scenes, lowest ID first per
      variant, one per variant, then a fifth from variant s mod 4. Each shape gets 5, variants 8/8/7/7.
    * calibration_r<r>: for every shape and calibration length, the r-th validation scene (by ID) of each variant.
    """
    d: DatasetConfig = ds["config"]
    by_cell: dict[tuple[str, int, str], list[Scene]] = {}
    for sp in SPLITS:
        for s in ds["splits"][sp]["scenes"]:
            by_cell.setdefault((s.family, len(s.clips), sp), []).append(s)
    subsets = {"screen": {
        "split": "test",
        "rule": f"per shape (in dataset order s = 0..{len(d.shapes) - 1}): the {d.screen_n_clips}-clip test scenes, "
                f"lowest scene ID first within each variant, cycling over variants from variant s mod 4; "
                f"{d.screen_per_shape} per shape",
        "scenes": [s for i, shape in enumerate(d.shapes)
                   for s in _pick(by_cell[(shape, d.screen_n_clips, "test")], d.screen_per_shape, offset=i)]}}
    for r in range(d.calibration_rounds):
        subsets[f"calibration_r{r + 1}"] = {
            "split": "validation",
            "rule": f"for every shape and n_clips in {list(d.calibration_n_clips)}: validation scene of rank {r} "
                    f"(by scene ID) within each variant; one per variant, 4 per shape and length",
            "scenes": [s for shape in d.shapes for n in d.calibration_n_clips
                       for s in _pick(by_cell[(shape, n, "validation")], N_VARIANTS, rank=r)]}
    return subsets


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_dataset(root: Path, ds: dict) -> dict[str, Path]:
    """Writes the following under ``root``:

    * ``<name>_{train,validation,test}.json``: the splits, each loadable by ``load_manifest``;
    * ``<name>_{screen,calibration_r1,...}.json``: the prespecified subsets;
    * ``<name>_split.json``: the split record.
    """
    d: DatasetConfig = ds["config"]
    root.mkdir(parents=True, exist_ok=True)
    header = {"dataset": asdict(d), "generator_version": GENERATOR_VERSION,
              "families": {k: asdict(v) for k, v in ds["families"].items()},
              "note": "each cell uses its shape family with n_clips replaced by the cell's length"}
    paths = {}
    for sp in SPLITS:
        paths[sp] = root / f"{d.name}_{sp}.json"
        payload = {**header, "split": sp,
                   "scenes": [asdict(s) for s in ds["splits"][sp]["scenes"]],
                   "rejected": ds["splits"][sp]["rejected"]}
        paths[sp].write_text(json.dumps(payload, indent=1))
    subsets = select_subsets(ds)
    for name, sub in subsets.items():
        paths[name] = root / f"{d.name}_{name}.json"
        paths[name].write_text(json.dumps({**header, "subset": name, "split": sub["split"], "rule": sub["rule"],
                                           "scenes": [asdict(s) for s in sub["scenes"]]}, indent=1))
    record = {**header, "split_seed": d.split_seed,
              "manifests": {k: {"path": p.name, "sha256": _sha(p),
                                "scenes": len(ds["splits"][k]["scenes"]) if k in SPLITS else len(subsets[k]["scenes"])}
                            for k, p in paths.items()},
              "counts": split_counts([s for sp in SPLITS for s in ds["splits"][sp]["scenes"]]),
              "subsets": {k: {"split": v["split"], "rule": v["rule"], "scene_ids": [s.scene_id for s in v["scenes"]]}
                          for k, v in subsets.items()},
              "cells": ds["cells"]}
    paths["split"] = root / f"{d.name}_split.json"
    paths["split"].write_text(json.dumps(record, indent=1))
    return paths


def split_counts(scenes: list[Scene]) -> dict:
    """{split: {shape: {n_clips: count}}} and per-split variant counts, for checking balance."""
    table: dict = {}
    variants: dict = {}
    for s in scenes:
        n = str(len(s.clips))
        table.setdefault(s.pool, {}).setdefault(s.family, {}).setdefault(n, 0)
        table[s.pool][s.family][n] += 1
        key = f"mirror={s.variant['mirror']},reverse={s.variant['reverse']}"
        variants.setdefault(s.pool, Counter())[key] += 1
    return {"by_shape_and_length": table, "by_variant": {k: dict(v) for k, v in variants.items()}}
