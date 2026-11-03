import dataclasses

import numpy as np
import pytest

from cablerouting.config import load_config
from cablerouting.scenes.generator import SceneFamilyConfig, generate_pool, generate_scene
from dloroute.decision import DecisionFormatError, parse_decision
from dloroute.geometry import Pose2D
from dloroute.interface import ClipSpec
from dloroute.planner.prompts import PlannerState, PromptBuilder, PromptConfig
from dloroute.rl.observation import ObsSpec, observation
from dloroute.skills import SkillParams, decode_insert_action

CLIPS = [f"C{i:02d}" for i in range(1, 11)]


def test_parse_valid_and_done():
    d = parse_decision('reasoning...\n```json\n{"skill": "Insert", "target_clip": "C03", "done": false}\n```', CLIPS)
    assert (d.skill, d.target_clip, d.done) == ("Insert", "C03", False)
    assert parse_decision('{"skill": null, "target_clip": null, "done": true}', CLIPS).done


@pytest.mark.parametrize("text", [
    '{"skill": "Insert", "target_clip": "C11", "done": false}',
    '{"skill": "Push", "target_clip": "C01", "done": false}',
    '{"skill": "Insert", "target_clip": "C01", "done": true}',
    '{"skill": "Insert", "done": false}',
    'no json here',
    '{"skill": "Insert", "target_clip": "C01", "done": "no"}',
])
def test_parse_invalid(text):
    with pytest.raises(DecisionFormatError):
        parse_decision(text, CLIPS)


def test_wrong_but_valid_target_is_not_corrected():
    # C07 is not in a five-clip route, but it is a visible clip: passed through as a decision error.
    assert parse_decision('{"skill": "Insert", "target_clip": "C07", "done": false}', CLIPS).target_clip == "C07"


def test_counter_semantics():
    s = PlannerState()
    for skill, tgt in [("Insert", "C01"), ("Insert", "C01"), ("Insert", "C02"), ("Pull", "C02"), ("Insert", "C02")]:
        s.record(skill, tgt)
    assert s.consecutive_inserts == 1 and s.counter_target == "C02" and s.crop_clip == "C02"
    assert [h.render() for h in s.history][:2] == ["1. Insert(C01)", "2. Insert(C01)"]


def test_prompt_mentions_obstacles_for_five_targets():
    pb = PromptBuilder(PromptConfig(allow_missing_examples=True))
    txt = pb.task_text(CLIPS[:5], CLIPS, PlannerState())
    assert "C01 -> C02 -> C03 -> C04 -> C05" in txt and "obstacles" in txt and "C06" in txt
    txt10 = pb.task_text(CLIPS, CLIPS, PlannerState())
    assert "obstacles" not in txt10 and "of 100" not in txt10
    assert "privileged" not in txt10


def test_generator_deterministic_and_balanced():
    cfg = SceneFamilyConfig()
    a, rej_a = generate_pool(cfg, "evaluation", 8)
    b, _ = generate_pool(cfg, "evaluation", 8)
    assert [s.to_json() for s in a] == [s.to_json() for s in b]
    variants = [(s.variant["mirror"], s.variant["reverse"]) for s in a]
    assert len(set(variants)) == 4 and all(variants.count(v) == 2 for v in set(variants))
    assert all(len(s.clips) == 10 and s.route(5) == CLIPS[:5] for s in a)
    assert all(s.route_slack > 0 and s.reach_margin >= cfg.min_reach_margin for s in a)
    for s in a:  # bounded local geometry at every stage
        assert all(cfg.stage_length[0] - 1e-9 <= st["chord"] <= cfg.stage_length[1] + 1e-9 for st in s.stages)
        assert all(cfg.turn_deg[0] - 1e-6 <= abs(st["relative_angle_deg"]) <= cfg.turn_deg[1] + 1e-6 for st in s.stages)


def test_initial_cable_follows_path():
    s = generate_pool(SceneFamilyConfig(), "development", 1)[0][0]
    poly = s.initial_polyline(0.0125)
    assert len(poly) == 193 and np.allclose(poly[0, :2], s.initial_head[:2])
    assert np.allclose(np.linalg.norm(np.diff(poly[:, :2], axis=0), axis=1).sum(), s.cable_length, atol=0.02)


def test_pools_are_disjoint():
    cfg = SceneFamilyConfig()
    seeds = {p: {s.seed for s in generate_pool(cfg, p, 5)[0]} for p in ("development", "calibration", "evaluation")}
    assert not (seeds["development"] & seeds["calibration"]) and not (seeds["calibration"] & seeds["evaluation"])


def test_five_and_ten_share_scene():
    s = generate_pool(SceneFamilyConfig(), "evaluation", 1)[0][0]
    assert s.route(5) == s.route(10)[:5]


def test_insert_action_decoding_bounds():
    c = ClipSpec("C01", Pose2D(0.3, 0.1, 0.5), 1.0, 0.015)
    sp = SkillParams()
    lo = decode_insert_action(-np.ones(7), c, 16, sp)
    hi = decode_insert_action(np.ones(7), c, 16, sp)
    assert lo.grasp_index == 0 and hi.grasp_index == 15
    assert lo.via_local[0, 0] == pytest.approx(c.entry_x - 0.16) and hi.via_local[0, 0] == pytest.approx(c.entry_x)
    assert hi.via_local[0, 1] == pytest.approx(0.08) and hi.via_local[0, 2] == pytest.approx(np.pi / 2)


def test_observation_is_length_independent():
    c = ClipSpec("C01", Pose2D(0, 0, 0), 1.0, 0.015)
    short = np.column_stack([-np.arange(60) * 0.0125, np.zeros(60), np.full(60, 0.005)])
    long = np.column_stack([-np.arange(200) * 0.0125, np.zeros(200), np.full(200, 0.005)])
    assert np.array_equal(observation(short, c, False), observation(long, c, False))
    assert observation(short, c, False).shape == (ObsSpec().dim,)


def test_default_config_loads():
    cfg = load_config()
    assert cfg.family("primary10").n_clips == 10 and cfg.family("paper5").n_clips == 5
    assert cfg.study.per_clip_budget * 10 == 100 and len(cfg.hash()) == 16
    with pytest.raises(KeyError):
        load_config(overrides={"cable.bogus": 1})
