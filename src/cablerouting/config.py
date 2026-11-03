"""Experiment configuration: one YAML file -> typed parameter objects.

Every reconstruction choice lives in configs/*.yaml so the frozen study is
identified by the config hash (protocol Section 9). Unknown keys are errors.
Algorithm parameters (skills, observation, reward, reference, planner, prompts)
are dloroute types; the rest configure the simulated environment.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dloroute.controllers.reference import ReferenceParams
from dloroute.planner.client import GenerationConfig
from dloroute.planner.prompts import PromptConfig
from dloroute.rl.observation import ObsSpec
from dloroute.rl.reward import RewardParams
from dloroute.skills import SkillParams

from .robot.executor import MotionParams
from .scenes.dataset import DatasetConfig
from .scenes.generator import SceneFamilyConfig
from .sim.isaac_backend import CableSpec, PhysicsSpec, RobotSpec
from .sim.routing_env import RenderConfig

ROOT = Path(__file__).resolve().parents[2]


def _build(cls, data: dict | None):
    data = dict(data or {})
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(names)
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    for k, v in data.items():
        if isinstance(v, list):
            data[k] = tuple(tuple(x) if isinstance(x, list) else x for x in v)
    return cls(**data)


@dataclass(frozen=True)
class StudyConfig:
    per_clip_budget: int = 10
    horizons: tuple[int, ...] = (5, 10)
    screen_scenes: int = 30
    eval_dataset: str = "shapes"
    calibration_states: int = 24
    calibration_min_successes: int = 20
    scheduling_seed: int = 20261031


@dataclass(frozen=True)
class Config:
    raw: dict
    families: dict[str, SceneFamilyConfig]
    cable: CableSpec
    physics: PhysicsSpec
    robot: RobotSpec
    motion: MotionParams
    skills: SkillParams
    obs: ObsSpec
    reward: RewardParams
    render: RenderConfig
    reference: ReferenceParams
    generation: GenerationConfig
    prompts: PromptConfig
    study: StudyConfig
    dataset: DatasetConfig

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:16]

    def family(self, name: str) -> SceneFamilyConfig:
        return self.families[name]


def load_config(path: str | Path = ROOT / "configs" / "default.yaml", overrides: dict[str, Any] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    fam = {name: _build(SceneFamilyConfig, {"name": name, **d}) for name, d in raw["scene_families"].items()}
    cfg = Config(
        raw=raw,
        families=fam,
        cable=_build(CableSpec, raw.get("cable")),
        physics=_build(PhysicsSpec, raw.get("physics")),
        robot=_build(RobotSpec, raw.get("robot")),
        motion=_build(MotionParams, raw.get("motion")),
        skills=_build(SkillParams, raw.get("skills")),
        obs=_build(ObsSpec, raw.get("observation")),
        reward=_build(RewardParams, raw.get("reward")),
        render=_build(RenderConfig, raw.get("render")),
        reference=_build(ReferenceParams, raw.get("reference")),
        generation=_build(GenerationConfig, raw.get("planner")),
        prompts=_build(PromptConfig, raw.get("prompts")),
        study=_build(StudyConfig, raw.get("study")),
        dataset=_build(DatasetConfig, raw.get("dataset")),
    )
    for f in cfg.families.values():  # the cable is a single physical object across families
        if abs(f.cable_length - cfg.cable.length) > 1e-9 or abs(f.cable_radius - cfg.cable.radius) > 1e-9:
            raise ValueError(f"family {f.name}: cable must match cable section")
    return cfg
