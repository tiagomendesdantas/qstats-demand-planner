import numpy as np
import pandas as pd
import pytest

from qstats_planner.forecasting.backtest import run_backtest, weekly_metrics
from qstats_planner.forecasting.models import ModelSpec, run_states
from qstats_planner.forecasting.seasonal_prior import SeasonalPrior
from qstats_planner.forecasting.selection import scaled_errors, select
from qstats_planner.forecasting.uncertainty import build_error_table


def spec(name, fam, params=(), c=1):
    return ModelSpec(name, fam, params, c, False)


def test_ses_recursion_matches_hand_calculation():
    y = np.array([[10.0], [20.0], [10.0]])
    s = run_states([spec("ses", "ses", (0.5,))], y, y, np.ones_like(y, bool))
    assert s.level[0, :, 0].tolist() == [10.0, 15.0, 12.5]


def test_invalid_weeks_do_not_update_the_state():
    y = np.array([[10.0], [999.0], [20.0]])
    valid = np.array([[True], [False], [True]])
    s = run_states([spec("ses", "ses", (0.5,))], y, y, valid)
    assert s.level[0, 1, 0] == 10.0 and s.level[0, 2, 0] == 15.0


def test_croston_sba_tsb():
    y = np.array([[0.0], [6.0], [0.0], [0.0], [6.0]])
    v = np.ones_like(y, bool)
    s = run_states([spec("c", "croston", (0.5,)), spec("s", "sba", (0.5,)), spec("t", "tsb", (0.5, 0.5))], y, y, v)
    # Croston after the second demand: size 6, interval 0.5*3 + 0.5*2 = 2.5 -> 2.4
    assert s.level[0, 4, 0] == pytest.approx(6 / 2.5)
    assert s.level[1, 4, 0] == pytest.approx(6 / 2.5 * 0.75)
    # TSB decays while demand is absent (Croston does not)
    assert s.level[2, 3, 0] < s.level[2, 1, 0] and s.level[0, 3, 0] == s.level[0, 2, 0]


def test_moving_average_and_damped_trend():
    y = np.arange(1, 9, dtype=float)[:, None]
    v = np.ones_like(y, bool)
    s = run_states([spec("ma4", "ma", (4,)), spec("h", "holt", (0.5, 0.3, 0.9))], y, y, v)
    assert s.level[0, -1, 0] == pytest.approx(np.mean([5, 6, 7, 8]))
    assert s.trend[1, -1, 0] > 0 and s.phi[1] == 0.9


def test_rolling_origin_has_no_leakage():
    rng = np.random.default_rng(0)
    W, n = 40, 3
    Y = rng.poisson(10, (W, n)).astype(float)
    models = [spec("ses", "ses", (0.3,))]
    ones = np.ones((W, n))
    args = dict(
        imputed_units=np.zeros((W, n)),
        es_plain=ones * 6,
        es_seas=ones * 6,
        exposure=np.full(W, 6.0),
        first_origin=np.zeros(n, int),
        horizons=(4,),
        max_imputed_share=0.1,
    )
    y = Y / 6
    bt1 = run_backtest(models, run_states(models, y, y, ones.astype(bool)), Y, **args)
    Y2 = Y.copy()
    Y2[30:] *= 5  # change the future after origin 25
    y2 = Y2 / 6
    bt2 = run_backtest(models, run_states(models, y2, y2, ones.astype(bool)), Y2, **args)
    assert np.allclose(bt1.forecast[4][:, :26], bt2.forecast[4][:, :26])  # forecasts made up to week 25
    assert np.allclose(bt1.actual[4][10], Y[11:15].sum(axis=0))
    assert not bt1.scored[4][W - 3 :].any()  # incomplete windows are not scored


def test_censored_windows_are_not_scored():
    W, n = 20, 1
    Y = np.full((W, n), 10.0)
    imp = np.zeros((W, n))
    imp[8] = 5.0  # half of week 8 was reconstructed
    models = [spec("naive", "naive")]
    ones = np.ones((W, n))
    bt = run_backtest(
        models, run_states(models, Y, Y, ones.astype(bool)), Y, imp, ones, ones, np.ones(W), np.zeros(n, int), (2,), 0.10
    )
    assert not bt.scored[2][6, 0] and not bt.scored[2][7, 0] and bt.scored[2][8, 0]


def test_selection_picks_the_better_model_and_prefers_the_simpler_one(cfg):
    M, W, n = 3, 30, 4
    models = [spec("naive", "naive", c=0), spec("ses", "ses", (0.3,), c=2), spec("holt", "holt", (0.3, 0.1, 0.9), c=4)]
    err = np.full((M, W, n), np.nan)
    err[0, 5:] = 0.50
    err[1, 5:] = 0.30
    err[2, 5:] = 0.296  # within the 2% parsimony margin of SES
    seg = np.array(["REGULAR"] * n)
    sel = select(models, err, seg, np.full(n, 8), None, cfg, 6)
    assert (sel.champion == 1).all()
    # an incumbent survives a challenger that is better by less than the switch margin
    err[0, 5:] = 0.29
    sel2 = select(models, err, seg, np.full(n, 8), np.ones(n, int), cfg, 6)
    assert (sel2.champion == 1).all() and "incumbent" in sel2.reason[0]


def test_scaled_errors_mask():
    f = np.array([[[10.0, 5.0]]])
    a = np.array([[12.0, 5.0]])
    e = scaled_errors(f, a, np.array([[True, False]]), np.array([[4.0, 4.0]]))
    assert e[0, 0, 0] == 0.5 and np.isnan(e[0, 0, 1])


def test_error_table_and_metrics():
    class BT:
        forecast = {h: np.full((1, 10, 2), 10.0) for h in (1, 3, 6, 11, 17, 24)}
        actual = {h: np.tile(np.array([[8.0, 14.0]]), (10, 1)) for h in (1, 3, 6, 11, 17, 24)}
        scored = {h: np.ones((10, 2), bool) for h in (1, 3, 6, 11, 17, 24)}

    level = np.full((10, 2), 2.0)
    t = build_error_table(BT, np.zeros(2, int), np.array(["REGULAR", "REGULAR"]), level, min_samples=5)
    e = t.errors("REGULAR", 1)
    assert e.min() == pytest.approx(-1.0) and e.max() == pytest.approx(2.0)
    m = weekly_metrics(np.array([10.0, 10.0]), np.array([8.0, 12.0]), np.array([True, True]))
    assert m["mae"] == 2 and m["bias"] == 0 and m["wape"] == pytest.approx(0.2)


def test_seasonal_prior_is_smooth_and_shrunk():
    sp = SeasonalPrior({0: np.r_[np.ones(9) * 0.8, [1.4, 1.6, 1.2]] / 1.0, 1: np.ones(12)}, np.array([0, 1]))
    f = sp.daily_factors(pd.date_range("2024-01-01", "2024-12-31"))
    assert f.shape == (366, 2) and np.allclose(f[:, 1], 1.0)
    assert np.abs(np.diff(f[:, 0])).max() < 0.05  # no jumps at month boundaries
    assert sp.neutral().daily_factors(pd.date_range("2024-11-01", periods=3)).max() == 1.0
