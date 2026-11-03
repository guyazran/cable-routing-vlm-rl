"""Shared file-system work queue for running study units on many workers across Slurm jobs.

Layout under a queue directory (on the shared /home, visible to every node)::

    tasks/<task_id>.json     the unit: what to run and where its output goes
    claims/<task_id>/        created with mkdir (atomic on this NFS) by the worker that takes it;
                             holds ``owner`` (worker id) and ``attempts``
    done/<task_id>.json      written once the unit's output is complete

Workers scan tasks in sorted id order (the id prefix sets the priority) and claim the first
unclaimed one. A worker id is ``<SLURM job id>-<worker index>``; a requeued job keeps its id, so on
start-up a worker first takes back its own unfinished claims (Slurm preemption requeues with the
same job id). Claims whose owner job no longer exists are released only by :func:`reap`, which the
operator runs: nothing ever takes over a live worker's claim.

Rerun rule (declared before any study episode; independent of task outcome): an episode that
crashes, loses its worker mid-run (preemption, node failure) or ends on a planner transport error
is unscorable. Its directory is kept as ``<out>.attempt-<k>`` with the cause in ``failure.txt``,
and the episode is rerun from its saved initial state, up to ``MAX_ATTEMPTS`` times.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

MAX_ATTEMPTS = 4


class Queue:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.tasks, self.claims, self.done = (self.root / d for d in ("tasks", "claims", "done"))
        for d in (self.tasks, self.claims, self.done):
            d.mkdir(parents=True, exist_ok=True)

    # --- producer ------------------------------------------------------------
    def add(self, task_id: str, spec: dict) -> bool:
        p = self.tasks / f"{task_id}.json"
        if p.exists():
            return False
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(spec, indent=1))
        tmp.rename(p)
        return True

    # --- consumer ------------------------------------------------------------
    def task_ids(self) -> list[str]:
        return sorted(p.stem for p in self.tasks.glob("*.json"))

    def spec(self, task_id: str) -> dict:
        return json.loads((self.tasks / f"{task_id}.json").read_text())

    def is_done(self, task_id: str) -> bool:
        return (self.done / f"{task_id}.json").exists()

    def owner(self, task_id: str) -> str | None:
        try:
            return (self.claims / task_id / "owner").read_text().strip()
        except OSError:
            return None

    def own_unfinished(self, worker_id: str) -> list[str]:
        return [t for t in sorted(p.name for p in self.claims.iterdir())
                if not self.is_done(t) and self.owner(t) == worker_id]

    def claim_next(self, worker_id: str, accept) -> str | None:
        """Claim the first unclaimed task for which ``accept(spec)`` is true."""
        for t in self.task_ids():
            if self.is_done(t) or (self.claims / t).exists():
                continue
            try:
                spec = self.spec(t)
            except (OSError, json.JSONDecodeError):
                continue
            if not accept(spec):
                continue
            try:
                (self.claims / t).mkdir()
            except FileExistsError:
                continue
            (self.claims / t / "owner").write_text(worker_id)
            return t
        return None

    def bump_attempts(self, task_id: str) -> int:
        p = self.claims / task_id / "attempts"
        n = int(p.read_text()) + 1 if p.exists() else 1
        p.write_text(str(n))
        return n

    def finish(self, task_id: str, summary: dict) -> None:
        tmp = self.done / f"{task_id}.tmp"
        tmp.write_text(json.dumps(summary, default=str))
        tmp.rename(self.done / f"{task_id}.json")

    # --- operator ------------------------------------------------------------
    def status(self) -> dict:
        ids = self.task_ids()
        done = sum(self.is_done(t) for t in ids)
        claimed = sum((self.claims / t).exists() and not self.is_done(t) for t in ids)
        return {"tasks": len(ids), "done": done, "running": claimed, "waiting": len(ids) - done - claimed}


def worker_id(index: int) -> str:
    return f"{os.environ.get('SLURM_JOB_ID', 'local-' + socket.gethostname())}-{index}"


def live_jobs() -> set[str]:
    out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%A"],
                         capture_output=True, text=True).stdout
    return set(out.split())


def reap(q: Queue, dry_run: bool = False) -> list[str]:
    """Release claims whose owner's Slurm job is gone (cancelled, failed, timed out)."""
    jobs = live_jobs()
    released = []
    for c in sorted(q.claims.iterdir()):
        t = c.name
        if q.is_done(t):
            continue
        owner = q.owner(t) or ""
        job = owner.rsplit("-", 1)[0]
        if job.startswith("local-") or job in jobs:
            continue
        released.append(t)
        if not dry_run:
            stale = q.claims / f".released-{t}-{int(time.time())}"
            c.rename(stale)
            shutil.rmtree(stale, ignore_errors=True)
    return released


def archive_attempt(out_dir: Path, cause: str) -> Path | None:
    """Keep an unscorable episode's partial output as ``<out>.attempt-<k>`` with its cause."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return None
    k = 0
    while (dest := out_dir.parent / f"{out_dir.name}.attempt-{k}").exists():
        k += 1
    out_dir.rename(dest)
    (dest / "failure.txt").write_text(cause.rstrip() + "\n")
    return dest
