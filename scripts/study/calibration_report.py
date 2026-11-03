"""Calibration gate (protocol Section 9; deviation L6): >= 20/24 successes per check and controller.

    python scripts/study/calibration_report.py --runs runs/study/calibration_r1 --out reports/calibration_r1.json

Check 1: five-clip validation scenes, all five requested. Check 2: ten-clip validation scenes, first
five requested. Both controllers on the same 24 + 24 initial states. Unfinished or unscorable
episodes are listed and make the gate undecided, never passed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cablerouting.analysis.stats import wilson_interval
from cablerouting.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config()
    need = cfg.study.calibration_min_successes
    runs = Path(args.runs)
    sched = json.loads((runs / "schedule.json").read_text())
    manifest = json.loads(Path(sched["manifest"]).read_text())
    n_clips = {s["scene_id"]: len(s["clips"]) for s in manifest["scenes"]}
    cells: dict[tuple[int, str], list[dict]] = {}
    missing = []
    for e in sched["episodes"]:
        f = runs / e["scene"] / f"{e['controller']}_{e['targets']}" / "result.json"
        key = (n_clips[e["scene"]], e["controller"])
        if not f.exists():
            missing.append(f"{e['scene']}/{e['controller']}_{e['targets']}")
            cells.setdefault(key, [])
            continue
        r = json.loads(f.read_text())
        cells.setdefault(key, []).append({"scene": e["scene"], "success": r["success"], "terminal": r["terminal"],
                                          "final_prefix": r["final_routing"].get("valid_prefix"),
                                          "invocations": r["invocations"], "premature_done": r["premature_done"]})
    report = {"runs": str(runs), "config_hash": cfg.hash(), "gate": f">= {need} successes of 24 per check and controller",
              "checks": {}, "missing": missing}
    passed = True
    for (nc, ctrl), rows in sorted(cells.items()):
        k, n = sum(r["success"] for r in rows), len(rows)
        check = "check1_five_clip_scenes" if nc == 5 else "check2_ten_clip_scenes_first_five"
        ok = n == 24 and k >= need
        passed &= ok
        report["checks"][f"{check}/{ctrl}"] = {
            "successes": k, "n": n, "rate": k / n if n else None, "wilson95": wilson_interval(k, n),
            "passes_gate": ok, "terminal": dict(Counter(r["terminal"] for r in rows)),
            "premature_done": sum(r["premature_done"] for r in rows),
            "failures": [r for r in rows if not r["success"]]}
    report["gate_passed"] = passed and not missing
    report["gate_status"] = "passed" if report["gate_passed"] else ("incomplete" if missing else "failed")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    for name, c in report["checks"].items():
        print(f"{name:55s} {c['successes']:2d}/{c['n']:2d}  {'PASS' if c['passes_gate'] else 'fail'}  {c['terminal']}")
    print("gate:", report["gate_status"], f"({len(missing)} episodes missing)" if missing else "")


if __name__ == "__main__":
    main()
