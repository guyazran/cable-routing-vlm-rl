"""One closed-loop routing episode (protocol Section 10).

Budget B(N) = b * N high-level skill invocations. After the last permitted
invocation the controller gets one completion query; any physical action it
requests then ends the episode as budget exhaustion. Terminal outcomes:
completion declared, collision, interface failure (format / cutoff /
context_overflow / transport), or budget exhaustion. Recoverable routing
errors stay in the episode. The final state is graded independently of the
controller's completion flag.

The runner owns the evaluation protocol; the agent (dloroute) owns every
decision and skill, and reaches the simulator only through the bridge's World.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from dloroute.agent import HierarchicalAgent

from ..bridge import SimWorld
from ..sim.routing_env import RoutingEnv
from .progress import ProgressTracker


@dataclass
class EpisodeConfig:
    n_targets: int
    per_clip_budget: int = 10

    @property
    def budget(self) -> int:
        return self.per_clip_budget * self.n_targets


@dataclass
class EpisodeResult:
    scene_id: str
    controller: str
    n_targets: int
    budget: int
    success: bool = False
    terminal: str = ""  # done | collision | budget_exhausted | format | cutoff | context_overflow | transport | error
    declared_done: bool = False
    premature_done: bool = False
    final_routing: dict = field(default_factory=dict)
    progress: dict = field(default_factory=dict)
    invocations: int = 0
    physical_ops: int = 0
    internal_attempts: int = 0
    recovery_calls: int = 0
    queries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_time_s: float = 0.0
    model_time_s: float = 0.0
    versions: dict = field(default_factory=dict)


def _save_images(info: dict, out: Path, idx: int) -> dict:
    paths = {}
    for k, img in info.pop("images", {}).items():
        p = out / "images" / f"{idx:03d}_{k}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
        paths[k] = str(p.relative_to(out))
    return paths


def run_episode(env: RoutingEnv, agent: HierarchicalAgent, cfg: EpisodeConfig,
                initial_state: dict, out_dir: Path, versions: dict | None = None) -> EpisodeResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    route = env.scene.route(cfg.n_targets)
    res = EpisodeResult(env.scene.scene_id, agent.name, cfg.n_targets, cfg.budget, versions=versions or {})
    tracker = ProgressTracker(cfg.n_targets)
    t_start = time.monotonic()
    env.reset(initial_state)
    agent.reset()
    world = SimWorld(env)
    trace = (out_dir / "trace.jsonl").open("w")
    try:
        decision_index = 0
        while True:
            routing = env.routing(route)
            out = agent.decide(world, decision_index, res.invocations, cfg.budget)
            img_paths = _save_images(out.info, out_dir, decision_index)
            q = out.info.get("query")
            if q is not None:
                res.queries += len(q["attempts"])
                for a in q["attempts"]:
                    u = a.get("usage") or {}
                    res.prompt_tokens += u.get("prompt_tokens", 0) or 0
                    res.completion_tokens += u.get("completion_tokens", 0) or 0
                    res.model_time_s += a.get("latency_s", 0.0) or 0.0
            record = {"decision_index": decision_index, "invocations_before": res.invocations,
                      "routing_before": routing.summary(), "images": img_paths,
                      "decision": out.decision.to_dict() if out.decision else None,
                      "interface_failure": out.failure, "info": out.info}
            decision_index += 1

            if out.failure is not None or out.decision is None:
                res.terminal = out.failure or "error"
                trace.write(json.dumps(record, default=str) + "\n")
                break
            d = out.decision
            if d.done:
                res.declared_done = True
                res.terminal = "done"
                trace.write(json.dumps(record, default=str) + "\n")
                break
            if res.invocations >= cfg.budget:
                res.terminal = "budget_exhausted"
                record["note"] = "physical action requested after the last permitted invocation"
                trace.write(json.dumps(record, default=str) + "\n")
                break

            t0 = time.monotonic()
            skill = agent.act(d, world)
            res.invocations += 1
            res.physical_ops += skill.physical_ops
            res.internal_attempts += skill.internal_attempts
            res.recovery_calls += int(d.skill == "Flatten")
            after = env.routing(route)
            tracker.update(after.valid_prefix, res.invocations, after.errors)
            record.update({"skill_result": skill.to_dict(), "routing_after": after.summary(),
                           "valid_prefix_after": after.valid_prefix, "skill_wall_s": time.monotonic() - t0})
            trace.write(json.dumps(record, default=str) + "\n")
            trace.flush()
            if skill.collided:
                res.terminal = "collision"
                break
    except Exception as e:  # infrastructure failure: keep the attempt, classify it
        res.terminal = "error"
        trace.write(json.dumps({"exception": f"{type(e).__name__}: {e}"}) + "\n")
        raise
    finally:
        final = env.routing(route)
        res.final_routing = final.summary()
        res.progress = tracker.to_dict()
        res.success = res.terminal == "done" and final.complete
        res.premature_done = res.declared_done and not final.complete
        res.wall_time_s = time.monotonic() - t_start
        trace.close()
        (out_dir / "result.json").write_text(json.dumps(res.__dict__, indent=1, default=str))
    return res
