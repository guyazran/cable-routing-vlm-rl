"""Cache the Isaac Sim 6.2 Franka Panda USD and every layer/texture it uses under assets/.

The renderer then needs no network access. The asset is NVIDIA's (Isaac Sim asset
license) and is not committed: assets/robots/ is git-ignored.

    python scripts/install/fetch_robot_assets.py
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np  # noqa: F401  (loads the env's libstdc++ before torch/Kit)
import torch  # noqa: F401  (the env's CUDA runtime before Kit; see README)

ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2"
FRANKA = "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "assets" / "robots" / "franka_panda"))
    args = ap.parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    from pxr import Usd, UsdUtils

    url = f"{ASSET_ROOT}/{FRANKA}"
    base = url.rsplit("/", 1)[0] + "/"
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(url)
    out = Path(args.out)
    files = []
    for ident in [l.identifier for l in layers] + list(assets):
        if not ident.startswith(base):
            continue  # Kit-provided (e.g. OmniPBR.mdl)
        rel = ident[len(base):]
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            urllib.request.urlretrieve(ident, dst)
        files.append({"path": rel, "bytes": dst.stat().st_size})
    _, _, missing = UsdUtils.ComputeAllDependencies(str(out / "franka.usd"))
    stage = Usd.Stage.Open(str(out / "franka.usd"))
    manifest = {"source": url, "files": files, "unresolved_remote": list(unresolved),
                "unresolved_local": list(missing), "default_prim": str(stage.GetDefaultPrim().GetPath())}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"[assets] {len(files)} files, {sum(f['bytes'] for f in files) / 1e6:.1f} MB -> {out}; "
          f"unresolved local: {missing}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
