"""Per-episode progress bookkeeping (protocol Section 11).

``first_attained[k-1]`` is the 1-based skill-invocation count at which a
simultaneously valid ordered prefix of length k first held (after release and
settling), giving S(k), h(k) and equal-prefix-budget attainment. Regressions of
the valid prefix and whether they were later repaired are tracked separately,
as are routing errors such as wrong-clip or out-of-order threading, including
errors that were repaired later.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProgressTracker:
    n_targets: int
    current: int = 0
    best: int = 0
    first_attained: list[int | None] = field(default_factory=list)
    regressions: list[dict] = field(default_factory=list)
    error_events: list[dict] = field(default_factory=list)
    _active_errors: set = field(default_factory=set)

    def __post_init__(self):
        self.first_attained = [None] * self.n_targets

    def update(self, valid_prefix: int, invocation: int, errors: list[str]) -> None:
        k = min(valid_prefix, self.n_targets)
        if k < self.current:
            self.regressions.append({"invocation": invocation, "from": self.current, "to": k, "repaired_at": None})
        for r in self.regressions:
            if r["repaired_at"] is None and k >= r["from"]:
                r["repaired_at"] = invocation
        for j in range(k):
            if self.first_attained[j] is None:
                self.first_attained[j] = invocation
        self.current = k
        self.best = max(self.best, k)
        now = set(errors)
        for e in sorted(now - self._active_errors):
            self.error_events.append({"invocation": invocation, "error": e, "cleared_at": None})
        for ev in self.error_events:
            if ev["cleared_at"] is None and ev["error"] not in now:
                ev["cleared_at"] = invocation
        self._active_errors = now

    def to_dict(self) -> dict:
        return {
            "final_prefix": self.current,
            "max_prefix": self.best,
            "first_attained": self.first_attained,
            "regressions": self.regressions,
            "error_events": self.error_events,
        }
