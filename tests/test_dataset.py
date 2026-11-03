"""Shape motifs and the stratified shape x length train/test split."""

import dataclasses
import itertools
import json
from collections import Counter

import numpy as np
import pytest

from cablerouting.config import load_config
from cablerouting.scenes.dataset import DatasetConfig, build_dataset, select_subsets, split_counts, write_dataset
from cablerouting.scenes.generator import (POOL_SEED_BASE, _motif_plan, _polyline_distance, _stage_path,
                                           generate_pool, load_manifest, variant_for_index)

SHAPES = ("arc", "circle", "spiral", "serpentine", "zigzag", "square")


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("n", [3, 10])
def test_stages_follow_the_motif_plan(cfg, shape, n):
    fam = dataclasses.replace(cfg.family(shape), n_clips=n)
    scenes, _ = generate_pool(fam, "development", 4, seed_base=50_000_000)
    for s in scenes:
        clips = s.clip_objects()
        assert len(clips) == n and s.route_slack >= 0 and s.extras["motif"] == shape
        base = np.array(s.extras["motif_base_turn_deg"])
        turns = np.array([st["relative_angle_deg"] for st in s.stages])
        assert len(base) == n - 1
        assert np.all(np.abs(turns - base) <= fam.turn_jitter_deg + 1e-6)  # route order, after mirror/reverse
        assert all(fam.stage_length[0] - 1e-9 <= st["chord"] <= fam.stage_length[1] + 1e-9 for st in s.stages)
        for a, b in zip(clips[:-1], clips[1:]):
            path, _ = _stage_path(a.pose, b.pose, fam)
            assert _polyline_distance(np.zeros(2), path) >= fam.base_clearance - 1e-9
        assert _polyline_distance(np.zeros(2), np.array(s.initial_path)) >= fam.base_clearance - 1e-9


def _plan(fam, seed):
    plan, info = _motif_plan(fam, np.random.default_rng(seed))
    return np.degrees(plan) * info["motif_sign"], info  # sign-normalised: the motif's own turns


@pytest.mark.parametrize("seed", range(5))
def test_motif_patterns(cfg, seed):
    fam = {s: dataclasses.replace(cfg.family(s), n_clips=10) for s in SHAPES}
    for shape in ("arc", "circle"):
        p, info = _plan(fam[shape], seed)
        assert np.allclose(p, info["motif_level_deg"]) and fam[shape].turn_deg[0] <= p[0] <= fam[shape].turn_deg[1]
    p, info = _plan(fam["spiral"], seed)
    assert np.all(np.diff(p) > 0) and p[0] == pytest.approx(info["motif_level_deg"])
    assert p[-1] == pytest.approx(info["motif_end_level_deg"])
    p, info = _plan(fam["serpentine"], seed)
    sides = np.sign(p - info["motif_bias_deg"])
    runs = [len(list(g)) for _, g in itertools.groupby(sides)]
    assert all(r == info["motif_half_period"] for r in runs[1:-1]) and max(runs) <= info["motif_half_period"]
    for shape in ("square", "zigzag"):
        p, info = _plan(fam[shape], seed)
        corners = p - info.get("motif_bias_deg", 0.0)
        idx = np.flatnonzero(np.abs(corners) > 1e-9)
        assert np.all(np.diff(idx) == info["motif_side_stages"])
        assert np.all(np.abs(corners[idx]) == pytest.approx(info["motif_level_deg"]))
        signs = np.sign(corners[idx])
        assert np.all(signs[1:] == (-signs[:-1] if shape == "zigzag" else signs[:-1]))


@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("seed", range(20))
def test_short_routes_keep_the_defining_feature(cfg, n, seed):
    def plan(shape):
        return _plan(dataclasses.replace(cfg.family(shape), n_clips=n), seed)

    p, info = plan("square")
    assert np.sum(np.abs(p) > 1e-9) >= 1  # at least one corner
    p, info = plan("zigzag")
    corners = (p - info["motif_bias_deg"])[np.abs(p - info["motif_bias_deg"]) > 1e-9]
    assert len(corners) >= min(2, (n - 1 + 1) // 2) and np.all(np.sign(corners[1:]) == -np.sign(corners[:-1]))
    p, info = plan("serpentine")
    assert np.any(np.diff(np.sign(p - info["motif_bias_deg"])) != 0)  # at least one sign flip


def test_free_motif_still_reproduces_recorded_layouts(cfg):
    pytest.skip("legacy layouts were recorded with setup v1's clip and cable (P1/P5); setup A changed both")
    from cablerouting.config import ROOT

    old = json.loads((ROOT / "manifests" / "primary10_development.json").read_text())["scenes"][:3]
    new, _ = generate_pool(cfg.family("primary10"), "development", 3)
    for o, n in zip(old, new):
        assert o["clips"] == n.clips and o["initial_path"] == n.initial_path


SMALL = DatasetConfig(name="t", shapes=("circle", "zigzag"), n_clips=(3, 6), train_per_cell=8, test_per_cell=8,
                      validation_per_cell=8, screen_n_clips=6, screen_per_shape=5, calibration_n_clips=(3, 6))
SPLITS = ("train", "validation", "test")


@pytest.fixture(scope="module")
def small(cfg):
    return build_dataset(cfg.families, SMALL, log=lambda *_: None)


def _all(ds):
    return [s for sp in SPLITS for s in ds["splits"][sp]["scenes"]]


def test_splits_are_stratified_by_shape_length_and_variant(small):
    scenes = _all(small)
    cells = Counter((s.pool, s.family, len(s.clips)) for s in scenes)
    assert len(cells) == 3 * 2 * 2 and set(cells.values()) == {8}
    per_variant = Counter((s.pool, s.family, len(s.clips), s.variant["mirror"], s.variant["reverse"]) for s in scenes)
    assert len(per_variant) == 3 * 2 * 2 * 4 and set(per_variant.values()) == {2}
    ids = [s.scene_id for s in scenes]
    assert len(set(ids)) == len(ids)


def test_splits_are_deterministic_and_seeds_are_disjoint(cfg, small):
    again = build_dataset(cfg.families, SMALL, log=lambda *_: None)
    for sp in SPLITS:
        assert [s.to_json() for s in small["splits"][sp]["scenes"]] == [s.to_json() for s in again["splits"][sp]["scenes"]]
    seeds = [s.seed for s in _all(small)] + [r["seed"] for sp in SPLITS for r in small["splits"][sp]["rejected"]]
    assert len(set(seeds)) == len(seeds) and min(seeds) >= SMALL.seed_base > max(POOL_SEED_BASE.values()) + 10**6
    for c in small["cells"]:
        for sp in SPLITS:
            base = c["validation_seed_base"] if sp == "validation" else c["seed_base"]
            assert all(base <= s.seed < base + 100_000 for s in _all(small) if s.scene_id in c[sp])


def test_validation_pool_leaves_train_and_test_unchanged(cfg, small):
    fewer = build_dataset(cfg.families, dataclasses.replace(SMALL, validation_per_cell=4), log=lambda *_: None)
    for sp in ("train", "test"):
        assert [s.to_json() for s in small["splits"][sp]["scenes"]] == [s.to_json() for s in fewer["splits"][sp]["scenes"]]


def test_prespecified_subsets(small):
    subsets = select_subsets(small)
    screen = subsets["screen"]["scenes"]
    assert len(screen) == 2 * 5 and all(s.pool == "test" and len(s.clips) == 6 for s in screen)
    for i, shape in enumerate(SMALL.shapes):
        mine = Counter((s.variant["mirror"], s.variant["reverse"]) for s in screen if s.family == shape)
        extra = tuple(variant_for_index(i % 4).values())
        assert mine[extra] == 2 and sum(mine.values()) == 5 and len(mine) == 4
    rounds = [subsets[f"calibration_r{r}"]["scenes"] for r in (1, 2)]
    assert "calibration_r3" not in subsets and not {s.scene_id for s in rounds[0]} & {s.scene_id for s in rounds[1]}
    for rnd in rounds:
        assert all(s.pool == "validation" for s in rnd)
        per = Counter((s.family, len(s.clips), s.variant["mirror"], s.variant["reverse"]) for s in rnd)
        assert len(per) == 2 * 2 * 4 and set(per.values()) == {1}
    assert {s.scene_id for r in rounds for s in r} == {s.scene_id for s in small["splits"]["validation"]["scenes"]}
    assert [s.scene_id for s in select_subsets(small)["screen"]["scenes"]] == [s.scene_id for s in screen]


def test_written_manifests_load_and_record_the_split(small, tmp_path):
    paths = write_dataset(tmp_path, small)
    for sp in SPLITS:
        payload, scenes = load_manifest(paths[sp])
        assert payload["split"] == sp and all(s.pool == sp for s in scenes)
    record = json.loads(paths["split"].read_text())
    assert record["counts"] == split_counts(_all(small))
    for name in ("screen", "calibration_r1", "calibration_r2"):
        payload, scenes = load_manifest(paths[name])
        assert [s.scene_id for s in scenes] == record["subsets"][name]["scene_ids"]
        assert all(s.pool == payload["split"] for s in scenes)
    assert sum(len(c["train"]) for c in record["cells"]) == record["manifests"]["train"]["scenes"]


def test_dataset_config_rejects_bad_settings(cfg):
    with pytest.raises(ValueError):
        build_dataset(cfg.families, dataclasses.replace(SMALL, test_per_cell=6))
    with pytest.raises(ValueError):
        build_dataset(cfg.families, dataclasses.replace(SMALL, validation_per_cell=6))
    with pytest.raises(ValueError):
        build_dataset(cfg.families, dataclasses.replace(SMALL, screen_n_clips=10))
    with pytest.raises(KeyError):
        build_dataset(cfg.families, dataclasses.replace(SMALL, shapes=("hexagon",)))


# --- insertion training starts sampled from dataset stages; settled-state checks ---------------

def test_route_stage_starts_reproduce_dataset_stages(cfg, small, tmp_path):
    from cablerouting.geometry.frames import wrap_angle
    from cablerouting.geometry.threading import evaluate_routing
    from cablerouting.scenes.insertion import InsertionStartDistribution, sample_insertion_scene, stage_pool

    paths = write_dataset(tmp_path, small)
    pool = stage_pool(paths["train"])
    assert len(pool) == sum(len(s.stages) for s in small["splits"]["train"]["scenes"])
    assert max(abs(t) for _, t in pool) > 55  # zigzag corners: far beyond primary10's 50 deg
    dist = InsertionStartDistribution(route_stage_prob=1.0, stages=pool)
    rng = np.random.default_rng(0)
    for _ in range(40):
        sc = sample_insertion_scene(rng, 1.0, cfg.family("primary10"), 2.4, 0.0125, 0.005, dist)
        p, t = sc.clips[0].pose, sc.target.pose
        chord = float(np.linalg.norm(t.xy - p.xy))
        turn = float(np.degrees(wrap_angle(t.yaw - p.yaw)))
        assert any(abs(chord - c) < 1e-9 and abs(turn - d) < 1e-6 for c, d in pool)
        # the cable runs through P on its axis (inside the aperture), and only P is threaded
        loc = np.array([sc.clips[0].pose.to_local(q[:2]) for q in sc.polyline])
        inside = (loc[:, 0] > sc.clips[0].entry_x - 0.02) & (loc[:, 0] < sc.clips[0].exit_x + 0.005)
        assert inside.any() and np.abs(loc[inside, 1]).max() < 1e-9
        assert evaluate_routing(sc.clips, sc.polyline, []).summary()["sequence"] == ["P+"]
        assert np.allclose(np.linalg.norm(np.diff(sc.polyline[:, :2], axis=0), axis=1), 0.0125, atol=2e-4)


def test_link_points_invert_set_polyline_transforms():
    from cablerouting.geometry.cable import link_points
    from cablerouting.geometry.frames import yaw_to_quat_wxyz

    s = np.linspace(0, 1, 41)
    pts = np.column_stack([0.3 * np.cos(2 * s), 0.3 * np.sin(2 * s), np.full_like(s, 0.005)])
    pts = pts[0] + np.cumsum(np.vstack([[0, 0, 0], 0.0125 * np.diff(pts, axis=0)
                                        / np.linalg.norm(np.diff(pts, axis=0), axis=1)[:, None]]), axis=0)
    centers, fwd = 0.5 * (pts[:-1] + pts[1:]), pts[:-1] - pts[1:]
    q = np.array([yaw_to_quat_wxyz(a) for a in np.arctan2(fwd[:, 1], fwd[:, 0])])
    tf = np.column_stack([centers, q[:, 1:], q[:, :1]])  # backend.set_polyline layout (xyzw)
    assert np.allclose(link_points(tf, 0.0125), pts, atol=1e-9)


def test_settled_state_checks(cfg, small):
    from cablerouting.scenes.state_checks import check_settled

    s = next(x for x in small["splits"]["train"]["scenes"] if len(x.clips) == 6)
    nominal = s.initial_polyline(cfg.cable.link_length)
    ok = check_settled(s, nominal, cfg.cable.link_length, 0.16)
    assert ok["passed"] and ok["max_displacement_m"] == 0.0
    lifted = nominal.copy()
    lifted[50:60, 2] = 0.03
    assert "lies_flat" in check_settled(s, lifted, cfg.cable.link_length, 0.16)["failed"]
    moved = nominal.copy()
    moved[:, 0] += 0.05
    assert "settled_near_nominal" in check_settled(s, moved, cfg.cable.link_length, 0.16)["failed"]
    c2 = s.clip_objects()[1]  # thread the head straight through C02 along its axis
    through = np.array([c2.pose.to_world(np.array([x, 0.0])) for x in np.linspace(0.1, -2.3, len(nominal))])
    bad = check_settled(s, np.column_stack([through, nominal[:, 2]]), cfg.cable.link_length, 0.16)
    assert {"no_threading", "head_in_c01_region", "clear_of_clips"} <= set(bad["failed"])
