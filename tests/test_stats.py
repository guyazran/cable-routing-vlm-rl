import numpy as np
import pytest

from cablerouting.analysis import stats as S


def test_wilson_bounds():
    lo, hi = S.wilson_interval(16, 20)
    assert 0.5 < lo < 0.8 < hi < 1.0


def test_paired_table_and_mcnemar():
    five = [1] * 20 + [0] * 10
    ten = [1] * 12 + [0] * 8 + [0] * 9 + [1]
    t = S.paired_table(five, ten)
    assert (t.n11, t.n10, t.n01, t.n00) == (12, 8, 1, 9)
    assert t.decrease == pytest.approx(7 / 30)
    assert S.exact_mcnemar(t) == pytest.approx(2 * sum(
        __import__("math").comb(9, k) for k in range(2)) / 2**9)
    lo, hi = S.discordance_difference_ci(t)
    assert lo < t.decrease < hi


def test_mcnemar_no_discordance():
    assert S.exact_mcnemar(S.PairedTable(10, 0, 0, 20)) == 1.0


def test_survival_and_hazard():
    surv = S.progress_survival([10, 10, 5, 3, 0], horizon=10)
    assert surv[0] == 1.0 and surv[1] == pytest.approx(0.8) and surv[10] == pytest.approx(0.4)
    h = S.conditional_stage_failure(surv)
    assert h[0] == pytest.approx(0.2)
    assert S.conditional_stage_failure(np.array([1.0, 0.0, 0.0]))[1] is None


def test_prefix_budget():
    runs = [[3, 8, None], [12, 15, 25]]
    out = S.prefix_budget_attainment(runs, per_clip_budget=10, horizon=3)
    assert list(out) == [0.5, 1.0, 0.5]


@pytest.mark.parametrize("n,d,expected", [(30, 0.30, 0.40), (60, 0.30, 0.80), (100, 0.50, 0.79)])
def test_power_matches_protocol_table(n, d, expected):
    assert S.mcnemar_power(n, d) == pytest.approx(expected, abs=0.015)


def test_bootstrap_degenerate_flag():
    ones = [True] * 10
    r = S.interaction_bootstrap(ones, ones, ones, ones, n_resamples=100)
    assert r.degenerate and r.estimate == 0.0
