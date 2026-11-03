"""Screen analysis (protocol Sections 11-12): primary paired test, stage curves, figures.

Reads ``<runs>/<scene_id>/<controller>_<N>/result.json``. The primary comparison
is the VLM's five- vs ten-target full-task success; everything else is
diagnostic.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..analysis import stats as S


def load_results(runs: Path) -> dict[tuple[str, int], dict[str, dict]]:
    """{(controller, N): {scene_id: result}} for completed (non-archived) episodes."""
    out: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for f in sorted(runs.glob("*/*/result.json")):
        if ".attempt-" in f.parent.name:
            continue
        r = json.loads(f.read_text())
        out[(r["controller"], int(r["n_targets"]))][r["scene_id"]] = r
    return out


def _paired(results, controller: str) -> tuple[list[str], list[bool], list[bool]]:
    a, b = results.get((controller, 5), {}), results.get((controller, 10), {})
    scenes = sorted(set(a) & set(b))
    return scenes, [a[s]["success"] for s in scenes], [b[s]["success"] for s in scenes]


def condition_summary(rs: dict[str, dict], n_targets: int, b: int) -> dict:
    succ = [r["success"] for r in rs.values()]
    k, n = int(sum(succ)), len(succ)
    first = [r["progress"]["first_attained"] for r in rs.values()]
    surv = S.progress_survival([r["progress"]["max_prefix"] for r in rs.values()], n_targets)
    return {
        "n": n, "successes": k, "rate": k / n if n else None, "wilson95": S.wilson_interval(k, n),
        "survival": surv.tolist(),
        "at_risk": [int(sum(r["progress"]["max_prefix"] >= j for r in rs.values())) for j in range(n_targets + 1)],
        "conditional_failure": S.conditional_stage_failure(surv),
        "prefix_budget_attainment": S.prefix_budget_attainment(first, b, n_targets).tolist(),
        "terminal": dict(Counter(r["terminal"] for r in rs.values())),
        "premature_done": int(sum(r["premature_done"] for r in rs.values())),
        "regressions": int(sum(len(r["progress"]["regressions"]) for r in rs.values())),
        "repaired_regressions": int(sum(sum(x["repaired_at"] is not None for x in r["progress"]["regressions"])
                                        for r in rs.values())),
        "efficiency_mean": {k2: float(np.mean([r[k2] for r in rs.values()])) for k2 in
                            ("invocations", "physical_ops", "internal_attempts", "recovery_calls", "queries",
                             "prompt_tokens", "completion_tokens", "wall_time_s", "model_time_s")},
    }


def analyze_screen(runs: Path, out: Path, per_clip_budget: int, seed: int = 12345) -> dict:
    results = load_results(runs)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"conditions": {}, "paired": {}}
    for (ctrl, n), rs in sorted(results.items()):
        report["conditions"][f"{ctrl}_{n}"] = condition_summary(rs, n, per_clip_budget)
    for ctrl in sorted({c for c, _ in results}):
        scenes, five, ten = _paired(results, ctrl)
        if not scenes:
            continue
        t = S.paired_table(five, ten)
        report["paired"][ctrl] = {
            "n_pairs": t.n, "table": {"n11": t.n11, "n10": t.n10, "n01": t.n01, "n00": t.n00},
            "decrease": t.decrease, "exact_mcnemar_p": S.exact_mcnemar(t),
            "ci95_bonferroni_clopper_pearson": S.discordance_difference_ci(t),
        }
    report["primary"] = report["paired"].get("vlm", {"note": "no paired VLM results"})
    v5, v10 = results.get(("vlm", 5), {}), results.get(("vlm", 10), {})
    r5, r10 = results.get(("reference", 5), {}), results.get(("reference", 10), {})
    common = sorted(set(v5) & set(v10) & set(r5) & set(r10))
    if common:
        bs = S.interaction_bootstrap([v5[s]["success"] for s in common], [v10[s]["success"] for s in common],
                                     [r5[s]["success"] for s in common], [r10[s]["success"] for s in common],
                                     seed=seed)
        report["interaction_I"] = {"n_scenes": len(common), "estimate": bs.estimate, "bootstrap95": bs.ci,
                                   "degenerate": bs.degenerate, "bootstrap_seed": seed, "resamples": 10_000}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    _figures(report, out)
    return report


def _figures(report: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = report["conditions"]
    if not conds:
        return
    names = list(conds)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    rates = [conds[c]["rate"] or 0 for c in names]
    # Clipped at 0: at k = 0 or k = n the bound equals the rate up to rounding.
    lo = [max(0.0, r - conds[c]["wilson95"][0]) for r, c in zip(rates, names)]
    hi = [max(0.0, conds[c]["wilson95"][1] - r) for r, c in zip(rates, names)]
    ax.bar(names, rates, yerr=[lo, hi], capsize=4, color="#4C72B0")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Full-task success (95% Wilson)")
    for i, c in enumerate(names):
        ax.text(i, 0.02, f"{conds[c]['successes']}/{conds[c]['n']}", ha="center", color="white")
    fig.tight_layout()
    fig.savefig(out / "fig1_success.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for c in names:
        k = np.arange(len(conds[c]["survival"]))
        axes[0].plot(k, conds[c]["survival"], marker="o", label=c)
        h = [np.nan if x is None else x for x in conds[c]["conditional_failure"]]
        axes[1].plot(k[1:], h, marker="o", label=c)
    axes[0].set(xlabel="clip k", ylabel="S(k) progress survival", ylim=(0, 1.02))
    axes[1].set(xlabel="stage k", ylabel="h(k) first-attainment failure", ylim=(0, 1.02))
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig2_3_stages.png", dpi=150)
    plt.close(fig)
