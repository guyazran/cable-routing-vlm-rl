"""One-screen status of the study run: Slurm jobs, training progress, queues, worker health.

    python scripts/study/status.py
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "runs" / "study"


def jobs() -> None:
    out = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%i|%j|%T|%M|%C|%b|%N|%R"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    cpus = sum(int(l.split("|")[4]) for l in out if l.split("|")[2] == "RUNNING")
    print(f"jobs ({cpus} cores running):")
    for l in out:
        i, n, st, t, c, g, node, r = l.split("|")
        print(f"  {i:9s} {n:10s} {st:9s} {t:>11s} {c:>4s} cpu {g:12s} {node if st == 'RUNNING' else r}")


def training() -> None:
    for run in sorted((STUDY / "train").glob("sac_*")):
        mons = sorted(run.glob("monitor*.csv"))
        rows = [r for m in mons for r in list(csv.reader(m.open()))[2:]]
        if not rows:
            print(f"  {run.name}: no episodes yet")
            continue
        steps = sum(int(r[1]) for r in rows)
        last = rows[-300:]
        meta = json.loads((run / "train_meta.json").read_text())
        rate = steps / max(1.0, float(rows[-1][2]))
        total = meta.get("env_steps", 0)
        eta = (total - steps) / rate / 60 if rate > 0 else float("nan")
        print(f"  {run.name}: {steps}/{total} env steps ({rate:.1f}/s, ~{eta:.0f} min left); last {len(last)} "
              f"episodes: success {sum(r[3] == 'True' for r in last) / len(last):.2f}, "
              f"collision {sum(r[4] == 'True' for r in last) / len(last):.2f}; "
              f"checkpoints {len(list(run.glob('sac_*_steps.zip')))}{' +final' if (run / 'sac_final.zip').exists() else ''}")


def queues() -> None:
    for q in sorted((STUDY / "queue").glob("*")):
        tasks = sorted(p.stem for p in (q / "tasks").glob("*.json"))
        done = {p.stem for p in (q / "done").glob("*.json")}
        running = [t for t in tasks if (q / "claims" / t).exists() and t not in done]
        print(f"  {q.name}: {len(done)}/{len(tasks)} done, {len(running)} running, "
              f"{len(tasks) - len(done) - len(running)} waiting")


def workers() -> None:
    now = time.time()
    bad = []
    for log in sorted((STUDY / "logs").glob("pod_*/worker_*.log")):
        txt = log.read_text(errors="replace")[-20000:]
        if "Traceback" in txt.split("[worker")[-1]:
            bad.append(f"{log.relative_to(STUDY)} (traceback)")
    print(f"  worker logs with a recent traceback: {bad or 'none'}")


if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    jobs()
    print("training:")
    training()
    print("queues:")
    queues()
    workers()
