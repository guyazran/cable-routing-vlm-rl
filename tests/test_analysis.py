import json

from cablerouting.eval.analysis import analyze_screen
from cablerouting.eval.progress import ProgressTracker


def _result(scene, ctrl, n, success, max_prefix):
    t = ProgressTracker(n)
    for k in range(1, max_prefix + 1):
        t.update(k, 3 * k, [])
    return {"scene_id": scene, "controller": ctrl, "n_targets": n, "success": success,
            "terminal": "done" if success else "budget_exhausted", "premature_done": False,
            "progress": t.to_dict(), "invocations": 10, "physical_ops": 20, "internal_attempts": 15,
            "recovery_calls": 1, "queries": 10, "prompt_tokens": 1000, "completion_tokens": 500,
            "wall_time_s": 60.0, "model_time_s": 30.0}


def test_analyze_screen(tmp_path):
    runs = tmp_path / "runs"
    for i in range(6):
        for ctrl in ("vlm", "reference"):
            for n in (5, 10):
                ok = (n == 5) or (i % 2 == 0) or ctrl == "reference"
                d = runs / f"s{i}" / f"{ctrl}_{n}"
                d.mkdir(parents=True)
                (d / "result.json").write_text(json.dumps(_result(f"s{i}", ctrl, n, ok, n if ok else 6)))
    rep = analyze_screen(runs, tmp_path / "out", per_clip_budget=10)
    assert rep["primary"]["table"] == {"n11": 3, "n10": 3, "n01": 0, "n00": 0}
    assert rep["conditions"]["vlm_10"]["survival"][6] == 1.0
    assert rep["interaction_I"]["estimate"] == 0.5
    assert (tmp_path / "out" / "fig1_success.png").exists()


def test_progress_tracker_regression_and_repair():
    t = ProgressTracker(5)
    t.update(2, 4, [])
    t.update(1, 5, ["order violation at position 2: C04"])
    t.update(3, 9, [])
    d = t.to_dict()
    assert d["first_attained"][:3] == [4, 4, 9] and d["max_prefix"] == 3
    assert d["regressions"] == [{"invocation": 5, "from": 2, "to": 1, "repaired_at": 9}]
    assert d["error_events"][0]["cleared_at"] == 9
