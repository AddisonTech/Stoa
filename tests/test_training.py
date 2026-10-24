"""Labels, purged walk-forward, calibration math, the deflated Sharpe and the promotion gate."""

from __future__ import annotations

import itertools
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stoa import calibration as cal
from stoa import experiments as ex
from stoa.clock import sessions_between
from stoa.config import Settings
from stoa.data.base import Bars, validate_bars
from stoa.data.synthetic import synthetic_bars
from stoa.dataset import FeatureSet, build_dataset, labels, standardize
from stoa.model import ModelStore
from stoa.training import (
    TrainingInputs,
    calibration_report,
    fit_final,
    folds_for,
    log_trials,
    promotion_gate,
    render_calibration,
    run_walkforward,
    save_model,
    slice_test_rows,
    slice_train_rows,
)
from stoa.walkforward import make_folds
from tests.conftest import SMALL_WATCHLIST, make_settings


def tiny_bars() -> Bars:
    days = sessions_between(date(2024, 1, 2), date(2024, 1, 31))
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    n = len(idx)
    spy_open = np.full(n, 100.0)
    spy_close = np.full(n, 100.0)
    a_open = np.arange(n, dtype=float) + 50.0
    a_close = a_open + 0.5

    def df(o: np.ndarray, c: np.ndarray) -> pd.DataFrame:
        return validate_bars(
            "x",
            pd.DataFrame(
                {
                    "open": o,
                    "high": np.maximum(o, c) + 1,
                    "low": np.minimum(o, c) - 1,
                    "close": c,
                    "volume": np.full(n, 1e6),
                },
                index=idx,
            ),
        )

    return {"SPY": df(spy_open, spy_close), "AAA": df(a_open, a_close)}


def test_label_is_next_open_to_close_h_sessions_later_vs_spy() -> None:
    bars = tiny_bars()
    lab = labels(bars, "SPY", horizon=3)
    days = bars["SPY"].index
    t = days[4]
    row = lab.loc[(t, "AAA")]
    entry = bars["AAA"]["open"].iloc[5]  # next open
    exit_ = bars["AAA"]["close"].iloc[7]  # close 3 sessions after t
    assert row["fwd_ret"] == pytest.approx(exit_ / entry - 1)
    assert row["fwd_excess"] == pytest.approx(exit_ / entry - 1)  # SPY flat
    assert row["y"] == 1.0
    assert row["label_end"] == days[7]
    # The last H sessions have no label yet.
    for t_late in days[-3:]:
        assert math.isnan(lab.loc[(t_late, "AAA"), "y"])
    assert lab.loc[(days[-4], "AAA"), "label_end"] == days[-1]


def test_label_with_a_symbol_that_never_traded_in_the_window() -> None:
    # Real universes include names whose bars all fall outside the window.
    bars = tiny_bars()
    empty = bars["AAA"].iloc[0:0]
    bars["ZZZ"] = empty
    lab = labels(bars, "SPY", horizon=3)
    assert lab.xs("ZZZ", level="symbol")["fwd_ret"].isna().all()
    assert lab.loc[(bars["SPY"].index[4], "AAA"), "y"] == 1.0


def test_label_beats_spy_flag() -> None:
    bars = tiny_bars()
    spy = bars["SPY"].copy()
    spy["close"] = spy["close"] * 2  # SPY doubles from open to close every day
    spy["high"] = spy[["open", "close"]].max(axis=1) + 1
    bars["SPY"] = validate_bars("SPY", spy)
    lab = labels(bars, "SPY", horizon=2)
    assert lab.loc[(bars["SPY"].index[3], "AAA"), "y"] == 0.0


def test_standardize_is_cross_sectional_and_clipped() -> None:
    idx = pd.MultiIndex.from_product(
        [[pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")], list("ABCDEFGHIJ")],
        names=["date", "symbol"],
    )
    vals = list(range(10)) + [0] * 9 + [1000]
    df = pd.DataFrame({"f": vals}, index=idx, dtype=float)
    out = standardize(df, ["f"])
    day1 = out.xs(pd.Timestamp("2024-01-02"), level="date")["f"]
    assert day1.mean() == pytest.approx(0.0) and day1.std() == pytest.approx(1.0)
    assert out["f"].max() <= 3.0
    assert not out["f"].isna().any()


# ---- walk-forward ---------------------------------------------------------------------


@pytest.mark.parametrize("window", ["expanding", "rolling"])
@pytest.mark.parametrize("horizon,embargo", [(5, 5), (10, 10), (10, 15)])
def test_purged_walkforward_has_no_overlap_and_respects_embargo(
    window: str, horizon: int, embargo: int
) -> None:
    sessions = sessions_between(date(2018, 1, 2), date(2022, 12, 30))
    folds = make_folds(
        sessions,
        horizon=horizon,
        embargo=embargo,
        test_sessions=63,
        min_train_sessions=300,
        window=window,
        rolling_sessions=400,
    )
    assert folds
    pos = {d: i for i, d in enumerate(sessions)}
    for f in folds:
        assert not set(f.train_sessions) & set(f.test_sessions)
        last_train, first_test = pos[f.train_end], pos[f.test_start]
        # The last training label ends at last_train + horizon, strictly before the test.
        assert last_train + horizon < first_test
        assert first_test - last_train >= embargo
        assert f.train_end < f.test_start
        if window == "rolling":
            assert len(f.train_sessions) <= 400
    # Test slices are consecutive and never overlap each other.
    for a, b in itertools.pairwise(folds):
        assert a.test_end < b.test_start
    with pytest.raises(ValueError):
        make_folds(sessions, horizon=10, embargo=5, test_sessions=63, min_train_sessions=300)


def test_train_rows_are_purged_against_the_test_window(
    small_settings: Settings, bars: Bars
) -> None:
    ds = build_dataset(bars, "SPY", 10, extra_symbols=SMALL_WATCHLIST, min_dollar_volume=0)
    folds = folds_for(make_settings(model__min_train_sessions="300"), ds)
    assert folds
    for f in folds[:5]:
        tr, te = slice_train_rows(ds, f), slice_test_rows(ds, f)
        assert (tr["label_end"] < pd.Timestamp(f.test_start)).all()
        assert tr.index.get_level_values("date").max() <= pd.Timestamp(f.train_end)
        assert te.index.get_level_values("date").min() >= pd.Timestamp(f.test_start)
        assert not set(tr.index) & set(te.index)


def test_dataset_universe_excludes_benchmark_and_short_history(bars: Bars) -> None:
    ds = build_dataset(bars, "SPY", 10, extra_symbols=SMALL_WATCHLIST, min_dollar_volume=0)
    syms = set(ds.panel.index.get_level_values("symbol"))
    assert "SPY" not in syms and syms <= set(SMALL_WATCHLIST)
    first = ds.panel.index.get_level_values("date").min()
    assert (bars["SPY"].index < first).sum() >= 259


def test_features_have_no_lookahead(bars: Bars) -> None:
    """Changing every bar after t must not change a single feature at or before t."""
    t = bars["SPY"].index[800]
    other = {}
    for s, df in bars.items():
        d = df.copy()
        after = d.index > t
        d.loc[after, ["open", "high", "low", "close"]] *= 1.7
        d.loc[after, "volume"] *= 3
        other[s] = validate_bars(s, d)
    kw = {"extra_symbols": SMALL_WATCHLIST, "min_dollar_volume": 0.0}
    a = build_dataset(bars, "SPY", 10, **kw).panel  # type: ignore[arg-type]
    b = build_dataset(other, "SPY", 10, **kw).panel  # type: ignore[arg-type]
    cols = FeatureSet(quality=False).columns()
    pd.testing.assert_frame_equal(a.loc[:t, cols], b.loc[:t, cols])


# ---- calibration math ------------------------------------------------------------------


def test_brier_logloss_auc_known_values() -> None:
    p = np.array([0.9, 0.8, 0.3, 0.1])
    y = np.array([1, 1, 0, 0])
    assert cal.brier(p, y) == pytest.approx((0.01 + 0.04 + 0.09 + 0.01) / 4)
    expect = -np.mean(np.log([0.9, 0.8, 0.7, 0.9]))
    assert cal.log_loss(p, y) == pytest.approx(expect)
    assert cal.auc(p, y) == 1.0
    assert cal.auc(1 - p, y) == 0.0
    assert cal.auc([0.5, 0.5, 0.5, 0.5], y) == 0.5
    assert math.isnan(cal.auc([0.1, 0.2], [1, 1]))


def test_auc_matches_pairwise_definition() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=300).round(2)
    y = (rng.uniform(size=300) < p).astype(int)
    pos, neg = p[y == 1], p[y == 0]
    pairs = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    assert cal.auc(p, y) == pytest.approx(pairs / (len(pos) * len(neg)))


def test_reliability_and_ece() -> None:
    # Perfectly calibrated within each bin: p = 0.25 hits 25%, p = 0.75 hits 75%.
    p = np.array([0.25] * 4 + [0.75] * 4)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    assert cal.ece(p, y) == pytest.approx(0.0)
    bins = cal.reliability(p, y, bins=4)
    assert [b.count for b in bins] == [0, 4, 0, 4]
    # Overconfident: always says 0.9, right half the time.
    assert cal.ece(np.full(10, 0.9), np.array([1, 0] * 5)) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        cal.brier([1.2], [1])
    with pytest.raises(ValueError):
        cal.brier([0.5], [2])


def test_bucket_table_hit_rates() -> None:
    p = np.array([0.40, 0.52, 0.53, 0.58, 0.62, 0.70])
    y = np.array([0, 1, 0, 1, 1, 0])
    rows = {(b.lo, b.hi): b for b in cal.bucket_table(p, y)}
    assert rows[(0.5, 0.55)].count == 2 and rows[(0.5, 0.55)].hit_rate == 0.5
    assert rows[(0.65, 1.0)].count == 1


# ---- the deflated Sharpe ratio -----------------------------------------------------------


def test_deflated_sharpe_matches_the_published_example() -> None:
    """Bailey and Lopez de Prado (2014), section 'A numerical example'.

    Annualized Sharpe 2.5 from 1,250 daily returns (5 years of 250 days), 100
    trials whose annualized Sharpe ratios have variance 0.5, skewness -3,
    kurtosis 10. The paper reports SR0 = 0.1132 (daily) and DSR = 0.9004.
    """
    days = 250
    sr = 2.5 / math.sqrt(days)
    var = 0.5 / days
    dsr, sr0 = ex.deflated_sharpe(sr, 100, var, 1250, skewness=-3.0, kurt=10.0)
    assert sr0 == pytest.approx(0.1132, abs=5e-5)
    assert dsr == pytest.approx(0.9004, abs=5e-5)


def test_more_trials_means_a_bigger_penalty() -> None:
    sr = 1.5 / math.sqrt(252)
    vals = [ex.deflated_sharpe(sr, n, 0.5 / 252, 756)[0] for n in (1, 10, 100, 1000)]
    assert vals == sorted(vals, reverse=True)
    assert ex.deflated_sharpe(sr, 1, 0.5 / 252, 756)[1] == 0.0  # one trial: plain PSR


def test_pbo_flags_pure_noise_and_spares_a_real_edge() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, (800, 30))
    assert ex.pbo(noise) > 0.3
    edge = noise.copy()
    edge[:, 0] += 0.004
    assert ex.pbo(edge) < 0.1
    assert math.isnan(ex.pbo(noise[:, :1]))


def test_registry_is_append_only_and_counts_trials(tmp_path: Path) -> None:
    reg = ex.Registry(tmp_path / "experiments" / "registry.jsonl")
    rets = pd.Series(np.random.default_rng(2).normal(0.0005, 0.01, 300))
    for i in range(5):
        reg.append(
            kind="train",
            name=f"t{i}",
            config_hash="c",
            data_hash="d",
            params={"i": i},
            metrics={"sharpe_daily": 0.01 * i},
            returns=rets + 0.0001 * i,
            sha="abc",
        )
    before = (tmp_path / "experiments" / "registry.jsonl").read_text()
    reg.append(
        kind="train",
        name="t5",
        config_hash="c",
        data_hash="d",
        params={},
        metrics={"sharpe_daily": 0.05},
        sha="abc",
    )
    after = (tmp_path / "experiments" / "registry.jsonl").read_text()
    assert after.startswith(before) and len(after.splitlines()) == 6
    n, var = reg.penalty_inputs()
    assert n == 6 and var == pytest.approx(np.var([0, 0.01, 0.02, 0.03, 0.04, 0.05], ddof=1))
    d = ex.deflate(reg, list(rets))
    assert d["trials"] == 6 and 0 <= d["dsr"] <= 1 and d["dsr"] <= d["psr"]
    assert reg.trial_matrix() is not None


def test_registry_paths_are_portable(tmp_path: Path) -> None:
    reg = ex.Registry(tmp_path / "experiments" / "registry.jsonl")
    rets = pd.Series(np.linspace(-0.01, 0.01, 50))
    for i in range(2):
        e = reg.append(
            kind="train",
            name=f"t{i}",
            config_hash=f"c{i}",
            data_hash="d",
            params={},
            metrics={"sharpe_daily": 0.01},
            returns=rets + 0.001 * i,
            sha="abc",
        )
        assert "\\" not in e.returns_path and e.returns_path.startswith("returns/")
    # A record written on Windows by an older version still resolves.
    path = tmp_path / "experiments" / "registry.jsonl"
    path.write_text(path.read_text().replace("returns/", "returns\\\\"), encoding="utf-8")
    assert "returns\\\\" in path.read_text()
    m = reg.trial_matrix()
    assert m is not None and m.shape[1] == 2


# ---- training end to end (small) -------------------------------------------------------------


def small_model_settings(tmp_path: Path) -> Settings:
    return make_settings(
        watchlist=",".join(SMALL_WATCHLIST),
        state__dir=str(tmp_path / "state"),
        model__min_train_sessions="300",
        model__test_sessions="126",
        model__min_dollar_volume_20="0",
    )


def test_walkforward_end_to_end_and_report(tmp_path: Path, bars: Bars) -> None:
    s = small_model_settings(tmp_path)
    inputs = TrainingInputs(bars=bars, extra_symbols=tuple(SMALL_WATCHLIST))
    wf = run_walkforward(s, inputs, FeatureSet(quality=False))
    assert wf.chosen in ("logistic", "hgb")
    oos = wf.oos()
    assert oos["p"].between(0.01, 0.99).all()
    # Every out-of-sample prediction came from a model trained before its fold.
    for k in wf.kinds:
        for r in wf.results[k]:
            assert r.model.train_end < r.fold.test_start
    rep = calibration_report(wf)
    md = render_calibration(rep)
    assert "Chosen model" in md and "Per fold" in md
    reg = ex.Registry(s.registry_path)
    ids = log_trials(reg, s, wf)
    assert set(ids) == set(wf.kinds) and len(reg.load()) == len(wf.kinds)
    # Final model and artifact round trip.
    from stoa.dataset import build_dataset as bd

    ds = bd(
        bars,
        "SPY",
        s.model.horizon,
        extra_symbols=SMALL_WATCHLIST,
        min_dollar_volume=0,
        features=FeatureSet(quality=False),
    )
    last = ds.sessions[-1]
    final = fit_final(s, ds, wf.chosen, last)
    assert final.train_end <= last
    store = ModelStore(s.artifact_dir)
    meta = save_model(
        store,
        final,
        settings=s,
        sha="abc",
        data_hash=wf.data_hash,
        oos={},
        experiment_id=ids[wf.chosen],
        created="2021-12-31T00:00:00+00:00",
    )
    store.set_current(meta.version, "test")
    loaded, lmeta = store.load(store.current_version() or "")
    x = ds.on(last)
    assert np.allclose(loaded.predict(x), final.predict(x))
    assert lmeta.features == final.features


def test_promotion_gate_rules() -> None:
    rng = np.random.default_rng(4)
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=200), list("ABCDE")], names=["date", "symbol"]
    )
    # A calibrated, informative forecaster: p is drawn first, y happens with probability p.
    p_good = rng.uniform(0.2, 0.8, len(idx))
    y = (rng.uniform(size=len(idx)) < p_good).astype(float)
    good = pd.DataFrame({"p": p_good, "y": y}, index=idx)
    noisy = pd.DataFrame({"p": rng.uniform(0.3, 0.7, len(idx)), "y": y}, index=idx)
    over = pd.DataFrame({"p": np.where(y == 1, 0.99, 0.6), "y": y}, index=idx)
    assert promotion_gate(good, noisy, tolerance=0, max_ece=0.2).promote
    worse = promotion_gate(noisy, good, tolerance=0, max_ece=0.5)
    assert not worse.promote and "worse" in worse.reason
    assert not promotion_gate(over, None, tolerance=0, max_ece=0.05).promote
    assert promotion_gate(good, None, tolerance=0, max_ece=0.2).promote
    assert not promotion_gate(good.iloc[:10], None, tolerance=0, max_ece=0.2).promote


def test_synthetic_bars_fixture_is_long_enough() -> None:
    b = synthetic_bars(["SPY", "AAA"], date(2016, 1, 4), date(2017, 6, 30), seed=1)
    assert len(b["SPY"]) > 300


def test_pipeline_does_not_invent_an_edge_in_noise(tmp_path: Path) -> None:
    """On pure noise the walk-forward must fail; with a planted edge it must find it."""
    from stoa.data.synthetic import synthetic_universe
    from stoa.experiments import Registry as Reg
    from stoa.retrain import judge

    s = small_model_settings(tmp_path)
    results = {}
    for signal in (0.0, 1.5):
        b, _ = synthetic_universe(
            SMALL_WATCHLIST,
            date(2016, 1, 4),
            date(2021, 12, 31),
            seed=3,
            signal=signal,
            extra=40,
            references=(),
        )
        wf = run_walkforward(
            s,
            TrainingInputs(bars=b, extra_symbols=tuple(b)),
            FeatureSet(quality=False),
            kinds=("logistic",),
        )
        o = wf.scored()
        o.attrs["horizon"] = wf.horizon
        from stoa.training import long_short_returns

        rets = long_short_returns(o, 5, 0.55)
        d = ex.deflate(
            Reg(tmp_path / f"r{signal}.jsonl"), list(rets), period=int(rets.attrs["period"])
        )
        results[signal] = (wf.pooled["logistic"], judge(s, wf, d)[0])
    noise, edge = results[0.0], results[1.5]
    assert not noise[1] and noise[0].auc < 0.55
    assert edge[1] and edge[0].auc > 0.56
