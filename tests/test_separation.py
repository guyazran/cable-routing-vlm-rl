"""The algorithm (dloroute) is separate from the simulation/environment code (cablerouting).

* dloroute imports nothing from cablerouting or a simulator, statically or at import time.
* The environment core (geometry, sim, scenes, robot) and the statistics never import dloroute
  or the harness. Only the harness (bridge, eval, cli, config) wires the two together.
"""

import ast
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
SIMULATOR_MODULES = ("isaacsim", "omni", "pxr", "carb")
ENV_CORE = ("geometry", "sim", "scenes", "robot", "analysis")
HARNESS = ("bridge", "eval", "cli", "config")


def _imports(path: Path) -> set[str]:
    """Absolute names of every module imported by ``path`` (relative imports resolved)."""
    pkg = path.relative_to(SRC).with_suffix("").parts[:-1]
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[: len(pkg) - node.level + 1]
                out.add(".".join([*base, *([node.module] if node.module else [])]))
            else:
                out.add(node.module)
    return out


def _files(package: str, sub: tuple[str, ...] | None = None) -> list[Path]:
    root = SRC / package
    files = sorted(root.rglob("*.py"))
    if sub is None:
        return files
    return [f for f in files if f.relative_to(root).parts[0].removesuffix(".py") in sub]


def test_algorithm_never_imports_environment_or_simulator():
    bad = {}
    for f in _files("dloroute"):
        hits = sorted(m for m in _imports(f)
                      if m.split(".")[0] in ("cablerouting", *SIMULATOR_MODULES))
        if hits:
            bad[str(f.relative_to(SRC))] = hits
    assert not bad, bad


def test_environment_core_never_imports_algorithm_or_harness():
    forbidden = {"dloroute", *(f"cablerouting.{h}" for h in HARNESS)}
    bad = {}
    for f in _files("cablerouting", ENV_CORE):
        hits = sorted(m for m in _imports(f)
                      if m.split(".")[0] == "dloroute" or ".".join(m.split(".")[:2]) in forbidden)
        if hits:
            bad[str(f.relative_to(SRC))] = hits
    assert not bad, bad


def test_importing_the_whole_algorithm_loads_no_environment_module():
    code = (
        "import importlib, pkgutil, sys, dloroute\n"
        "for m in pkgutil.walk_packages(dloroute.__path__, 'dloroute.'):\n"
        "    importlib.import_module(m.name)\n"
        f"bad = [m for m in sys.modules if m.split('.')[0] in ('cablerouting',) + {SIMULATOR_MODULES!r}]\n"
        "print(sorted(bad))\n"
        "assert not bad\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
