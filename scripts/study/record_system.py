"""Record the system under test at the start of a calibration round (protocol Section 9).

    python scripts/study/record_system.py --policy checkpoints/sac_v6_s1_14808.zip --out runs/study/calibration_r2/system.json

``freeze --calibration <system.json>`` later refuses to freeze a system whose code, config, prompt or
policy hash differs from the one recorded here.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from cablerouting.cli import _file_hash, code_hash
from cablerouting.config import load_config
from dloroute.planner.prompts import PromptBuilder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    cfg = load_config()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a calibration round's system is recorded once")
    rec = {"recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"), "config_hash": cfg.hash(),
           "prompt_hash": PromptBuilder(cfg.prompts).version_hash(), "policy": a.policy,
           "policy_sha256": _file_hash(Path(a.policy)), "per_clip_budget": cfg.study.per_clip_budget,
           "reference": dataclasses.asdict(cfg.reference), "skills": dataclasses.asdict(cfg.skills),
           "planner": dataclasses.asdict(cfg.generation), "code_sha256": code_hash(), "note": a.note}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
