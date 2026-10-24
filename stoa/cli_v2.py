"""The v0.2 and later commands: universe, fundamentals, news, analyst, model, experiments,
exam, fallback, approve and dashboard. Registered on the main `stoa` group in cli.py."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import click

from stoa import alerts, clock
from stoa.state import StateStore

Deco = Callable[[Callable[..., Any]], Callable[..., Any]]


def register(main: click.Group, pass_ctx: Deco, friendly_errors: Deco) -> None:
    """Attach every v0.2 command to ``main``."""

    # ---- universe -------------------------------------------------------------------

    @main.group()
    def universe() -> None:
        """The survivorship-free training universe."""

    @universe.command("build")
    @click.option("--start", type=click.DateTime(["%Y-%m-%d"]), default=None)
    @click.option("--end", type=click.DateTime(["%Y-%m-%d"]), default=None)
    @pass_ctx
    @friendly_errors
    def universe_build(ctx: Any, start: Any, end: Any) -> None:
        """Fetch bars for every index member since history_start, dead names included."""
        from stoa.data.factory import make_provider
        from stoa.membership import default_membership, fetch_renamed
        from stoa.pipeline import coverage_report, universe_cache, write_coverage

        s = ctx.settings
        begin = start.date() if start else s.data.history_start
        finish = end.date() if end else clock.last_closed_session(clock.now_utc())
        m = default_membership()
        symbols = sorted(m.ever_members(begin, finish) | set(s.sources.reference_symbols))
        prov = make_provider(s, ctx.secrets)
        cache = universe_cache(s)
        have: set[str] = set()
        for i in range(0, len(symbols), 100):
            chunk = symbols[i : i + 100]
            # A renamed company's history is under its current ticker (BK under BNY).
            bars = fetch_renamed(lambda tickers: prov.fetch(tickers, begin, finish), chunk)
            cache.write(bars)
            have |= set(bars)
            click.echo(
                f"{min(i + 100, len(symbols))}/{len(symbols)} symbols, {len(have)} with bars"
            )
        rep = coverage_report(m, have, begin, finish)
        path = write_coverage(s, rep)
        click.echo(
            f"{rep['with_bars']} of {rep['ever_members']} index members have bars; "
            f"{rep['left_index_with_bars']} of {rep['left_index']} that left the index. "
            f"Coverage report: {path}"
        )

    # ---- fundamentals ------------------------------------------------------------------

    @main.group()
    def fundamentals() -> None:
        """SEC EDGAR fundamentals and earnings release times."""

    @fundamentals.command("fetch")
    @click.option("--refresh", is_flag=True, help="Ignore the disk cache.")
    @click.option("--watchlist-only", is_flag=True, help="Only the watchlist, not the universe.")
    @pass_ctx
    @friendly_errors
    def fundamentals_fetch(ctx: Any, refresh: bool, watchlist_only: bool) -> None:
        """Quality features and 8-K item 2.02 earnings times, stamped point in time."""
        from stoa.data.sec import fetch_fundamentals, make_sec_client
        from stoa.membership import default_membership, ticker_aliases

        s = ctx.settings
        if watchlist_only or s.model.universe == "watchlist":
            symbols = sorted(set(s.watchlist) - {s.benchmark})
        else:
            symbols = sorted(
                default_membership().ever_members(s.data.history_start, date.today())
                | set(s.watchlist)
            )
        sec = make_sec_client(
            s.sources.sec_user_agent, s.sources.sec_cache_dir, s.sources.sec_rate_per_second
        )
        q, e, problems = fetch_fundamentals(sec, symbols, refresh=refresh, aliases=ticker_aliases())
        s.fundamentals_path.parent.mkdir(parents=True, exist_ok=True)
        q.to_parquet(s.fundamentals_path)
        e.to_parquet(s.earnings_path)
        click.echo(
            f"{q['symbol'].nunique() if len(q) else 0} symbols with fundamentals, "
            f"{len(e)} earnings releases. {len(problems)} symbols without SEC data."
        )
        for sym, why in sorted(problems.items())[:20]:
            click.echo(f"  {sym}: {why}")

    # ---- news -------------------------------------------------------------------------------

    @main.group()
    def news() -> None:
        """Historical news from Alpaca (Benzinga)."""

    @news.command("fetch")
    @click.option("--start", type=click.DateTime(["%Y-%m-%d"]), default=None)
    @click.option("--end", type=click.DateTime(["%Y-%m-%d"]), default=None)
    @pass_ctx
    @friendly_errors
    def news_fetch(ctx: Any, start: Any, end: Any) -> None:
        """Download news for the analyst's symbols into the local store."""
        from stoa.data.factory import alpaca_data_client
        from stoa.data.news import AlpacaNewsClient, NewsStore

        s = ctx.settings
        symbols = analyst_symbols(s)
        begin = start.date() if start else s.data.history_start
        finish = end.date() if end else clock.last_closed_session(clock.now_utc())
        client = AlpacaNewsClient(alpaca_data_client(s, ctx.secrets))
        df = client.fetch(symbols, begin, finish)
        merged = NewsStore(s.news_path).merge(df)
        click.echo(f"{len(df)} items fetched, {len(merged)} in the store at {s.news_path}")

    # ---- analyst ----------------------------------------------------------------------------

    @main.group()
    def analyst() -> None:
        """Claude reads news into structured scores (optional)."""

    @analyst.command("backfill")
    @click.option("--batch/--no-batch", default=True, help="Message Batches API at half price.")
    @click.option("--allow-spend", is_flag=True, help="Run even if the estimate is over budget.")
    @click.option("--estimate", is_flag=True, help="Only print the cost estimate.")
    @click.option("--since", type=click.DateTime(["%Y-%m-%d"]), default=None)
    @pass_ctx
    @friendly_errors
    def analyst_backfill(
        ctx: Any, batch: bool, allow_spend: bool, estimate: bool, since: Any
    ) -> None:
        """Score every stored news item that is not scored yet."""
        from stoa import analyst as an
        from stoa.data.news import NewsStore

        s = ctx.settings
        df = NewsStore(s.news_path).load()
        if since is not None and len(df):
            import pandas as pd

            df = df.loc[pd.to_datetime(df["known_at"], utc=True).dt.date >= since.date()]
        items = an.items_from_news(df, s.analyst.text_chars, s.analyst.max_items_per_symbol_per_day)
        a, store, state = make_analyst(ctx)
        todo = a.pending(items)
        est, n = an.estimate_cost(todo, s.analyst, a.price, s.budget, batch=batch)
        click.echo(
            f"{len(items)} items, {len(items) - len(todo)} already scored, {n} to score. "
            f"Estimated cost ${est:.2f} ({'batch, half price' if batch else 'one by one'}). "
            f"Budget left this month ${a.ledger.remaining():.2f} of ${a.ledger.cap_usd:.2f}."
        )
        if estimate or not todo:
            return
        if a.client is None:
            raise click.ClickException("ANTHROPIC_API_KEY is not set")
        try:
            if batch:
                journal = an.BatchJournal(s.shared_dir / "analyst_batches.json")
                rep = a.run_batches(todo, journal, allow_spend=allow_spend, progress=click.echo)
            else:
                a.check_budget(todo, batch=False, allow_spend=allow_spend)
                rep = a.score_now(todo)
        finally:
            store.save(state)
        rows = a.rows(items)
        an.merge_scores(s.analyst_scores_path, rows)
        click.echo(
            f"Scored {rep.scored}, cached {rep.cached}, failed {rep.failed}, "
            f"spent ${rep.spent_usd:.2f}. {len(rows)} scores in {s.analyst_scores_path}."
        )

    # ---- model ------------------------------------------------------------------------------

    @main.group()
    def model() -> None:
        """Train, report and promote the prediction model."""

    @model.command("train")
    @click.option("--synthetic", is_flag=True, help="Seeded synthetic data (offline demo).")
    @click.option("--no-promote", is_flag=True, help="Train and report, keep the current model.")
    @click.option("--force-promote", is_flag=True, help="Promote even if the gate says no.")
    @click.option("--results", type=click.Path(path_type=Path), default=Path("results"))
    @click.option("--repo", type=click.Path(path_type=Path), default=Path("."))
    @pass_ctx
    @friendly_errors
    def model_train(
        ctx: Any,
        synthetic: bool,
        no_promote: bool,
        force_promote: bool,
        results: Path,
        repo: Path,
    ) -> None:
        """Walk-forward train, write the calibration report, gate and promote."""
        out = run_training(ctx, synthetic, not no_promote, force_promote, results, repo)
        click.echo(out)

    @model.command("report")
    @click.option("--synthetic", is_flag=True)
    @click.option(
        "--horizon", type=int, default=None, help="Default: model.horizon and model.test_horizon."
    )
    @click.option("--ablation", is_flag=True, help="Also compare with and without text.")
    @click.option("--results", type=click.Path(path_type=Path), default=Path("results"))
    @click.option("--repo", type=click.Path(path_type=Path), default=Path("."))
    @pass_ctx
    @friendly_errors
    def model_report(
        ctx: Any, synthetic: bool, horizon: int | None, ablation: bool, results: Path, repo: Path
    ) -> None:
        """Calibration and walk-forward report at each horizon (registered as trials)."""
        from stoa import holdout as h
        from stoa.experiments import Registry
        from stoa.pipeline import feature_set, load_inputs
        from stoa.training import ablation as run_ablation
        from stoa.training import (
            calibration_report,
            log_trials,
            render_ablation,
            render_calibration,
            run_walkforward,
            write_report,
        )

        s = ctx.settings
        last = clock.last_closed_session(clock.now_utc())
        cutoff = (
            h.training_cutoff(repo, s, last)
            if not synthetic
            else min(last, clock.previous_session(s.backtest.holdout_start))
        )
        inputs = load_inputs(s, cutoff, synthetic=synthetic, blocked=h.blocked_window(repo, s))
        both = {s.model.horizon, s.model.test_horizon} - {None}
        horizons = [horizon] if horizon else sorted(h for h in both if h is not None)
        reg = Registry(s.registry_path)
        out = results / f"{pipeline_stamp()}-report"
        md = []
        for hz in horizons:
            wf = run_walkforward(
                s, inputs.training, feature_set(s), horizon=hz, end=cutoff, progress=None
            )
            log_trials(reg, s, wf, name_prefix="report")
            rep = calibration_report(wf)
            write_report(rep, out, name=f"calibration-h{hz}")
            md.append(render_calibration(rep))
        if ablation:
            ab = run_ablation(s, inputs.training, cutoff=s.analyst.contamination_cutoff)
            (out / "ablation.json").write_text(json.dumps(ab, indent=2, default=str))
            md.append(render_ablation(ab))
        (out / "report.md").write_text("\n".join(md), encoding="utf-8")
        click.echo("\n".join(md))
        click.echo(f"Written to {out}")

    @model.command("promote")
    @click.argument("version", required=False)
    @click.option("--yes", is_flag=True)
    @pass_ctx
    @friendly_errors
    def model_promote(ctx: Any, version: str | None, yes: bool) -> None:
        """Show models, or make VERSION the live model by hand (bypasses the gate)."""
        from stoa.model import ModelStore

        store = ModelStore(ctx.settings.artifact_dir)
        current = store.current_version()
        if version is None:
            for v in store.versions():
                _, meta = store.load(v)
                mark = "*" if v == current else " "
                ll = meta.oos.get("log_loss", math.nan)
                click.echo(
                    f"{mark} {v}  {meta.kind:8s} trained to {meta.train_end}  "
                    f"oos log loss {ll:.4f}  dsr {meta.oos.get('dsr', math.nan):.3f}"
                )
            if not store.versions():
                click.echo("No models yet. Run `stoa model train`.")
            return
        if version not in store.versions():
            raise click.ClickException(f"no model {version}")
        if not yes:
            click.confirm(
                f"Make {version} the live model, skipping the promotion gate?", abort=True
            )
        store.set_current(version, "promoted by hand")
        click.echo(f"{version} is now the live model.")

    # ---- experiments --------------------------------------------------------------------------

    @main.group()
    def experiments() -> None:
        """The append-only experiment registry."""

    @experiments.command("list")
    @click.option("--last", type=int, default=20)
    @pass_ctx
    def experiments_list(ctx: Any, last: int) -> None:
        """Every recorded trial, newest last, with the multiple-testing penalty."""
        from stoa.experiments import Registry

        reg = Registry(ctx.settings.registry_path)
        rows = reg.load()
        n, var = reg.penalty_inputs()
        click.echo(
            f"{len(rows)} records, {n} distinct configurations tried "
            f"(variance of daily Sharpe across trials {var:.6f})"
        )
        for e in rows[-last:]:
            sr = e.metrics.get("sharpe_daily")
            ll = e.metrics.get("log_loss")
            parts = [f"{e.created[:19]}", f"{e.id}", f"{e.kind:13s}", e.name]
            if isinstance(sr, (int, float)):
                parts.append(f"sharpe/day {sr:.4f}")
            if isinstance(ll, (int, float)):
                parts.append(f"log loss {ll:.4f}")
            parts.append(f"code {e.code_sha[:10]}")
            click.echo("  ".join(parts))

    # ---- exam -------------------------------------------------------------------------------

    @main.group()
    def exam() -> None:
        """Exam order: walk-forward, sealed holdout, paper, then live."""

    @exam.command("status")
    @click.option("--repo", type=click.Path(path_type=Path), default=Path("."))
    @pass_ctx
    def exam_status(ctx: Any, repo: Path) -> None:
        from stoa import exam as ex
        from stoa import holdout as h

        st = ex.status(ctx.settings, h.read_lock(repo))
        click.echo(ex.render(st, clock.last_closed_session(clock.now_utc())))
        for p in st.problems():
            click.echo(f"  - {p}")

    # ---- fallback ---------------------------------------------------------------------------

    @main.group()
    def fallback() -> None:
        """The drift fallback and quit rule (SPY-hold)."""

    @fallback.command("status")
    @pass_ctx
    def fallback_status(ctx: Any) -> None:
        from stoa import tracking

        s = ctx.settings
        t = StateStore(s.state.dir).load().track
        if t.fallback_active:
            click.echo(f"ACTIVE since {t.fallback_since}: {t.fallback_reason}")
        else:
            click.echo("off")
        exp = tracking.load_expectation(s.shared_dir)
        if exp is not None:
            click.echo(
                f"expectation from {exp.source}: rolling {exp.window} session Sharpe lower band "
                f"{exp.sharpe_p05:.2f}, drawdown 95th percentile {exp.drawdown_p95:.1%}"
            )
        else:
            click.echo("no walk-forward expectation yet (`stoa model train`)")
        g = tracking.grade(t.days)
        click.echo(
            f"tracked {g.sessions} sessions: {g.total_return:+.1%} after costs vs SPY "
            f"{g.spy_return:+.1%}, drawdown {g.max_drawdown:.1%}, Sharpe {g.sharpe:.2f}"
        )

    @fallback.command("reset")
    @click.option("--yes", is_flag=True)
    @pass_ctx
    @friendly_errors
    def fallback_reset(ctx: Any, yes: bool) -> None:
        """Leave SPY-hold and let the model trade again. Positions are not changed."""
        from stoa import tracking

        store = StateStore(ctx.settings.state.dir)
        state = store.load()
        if not state.track.fallback_active:
            click.echo("The fallback is not active.")
            return
        if not yes:
            click.confirm(f"Fallback reason: {state.track.fallback_reason}. Reset it?", abort=True)
        tracking.reset(state.track)
        store.save(state)
        click.echo("Fallback reset. The model trades again from the next DECIDE.")

    # ---- dashboard --------------------------------------------------------------------------

    @main.group()
    def dashboard() -> None:
        """The read-only status page for GitHub Pages."""

    @dashboard.command("build")
    @click.option(
        "--out", type=click.Path(path_type=Path), default=Path("site"), help="Output folder."
    )
    @click.option(
        "--repo",
        type=click.Path(path_type=Path),
        default=Path("."),
        help="Repository root, for HOLDOUT.lock.",
    )
    @click.option(
        "--results",
        type=click.Path(path_type=Path),
        default=None,
        help="Model reports folder (default <shared>/results).",
    )
    @pass_ctx
    def dashboard_build(ctx: Any, out: Path, repo: Path, results: Path | None) -> None:
        """Build the static site from committed state only. No network, no broker, no secrets."""
        from stoa import dashboard as dash

        index = dash.build(ctx.settings, out, repo, results=results)
        click.echo(f"Wrote {index} and {index.parent / 'data.json'}")

    # ---- approve ----------------------------------------------------------------------------

    @main.group()
    def approve() -> None:
        """Hand approval of trades with ntfy buttons."""

    @approve.command("test")
    @pass_ctx
    @friendly_errors
    def approve_test(ctx: Any) -> None:
        """Send a test alert with Approve and Skip buttons, wait for the tap, report it."""
        import time

        from stoa import approvals
        from stoa.cli import make_notifier
        from stoa.models import PlannedOrder, Side

        s, sec = ctx.settings, ctx.secrets
        if sec.ntfy_approval_topic is None:
            raise click.ClickException("STOA_NTFY_APPROVAL_TOPIC is not set")
        nonce = approvals.new_nonce()
        plan_id = "test" + nonce[:6].lower()
        created = clock.now_utc()
        acts = approvals.actions(
            s.alerts.server, sec.ntfy_approval_topic.get_secret_value(), plan_id, "TEST", nonce
        )
        alert = alerts.with_actions(
            alerts.Alert(
                title="Stoa approval test", message="Tap Approve or Skip.", priority=alerts.HIGH
            ),
            acts,
            "This is a test. Nothing will trade.",
        )
        notifier = make_notifier(s, sec)
        if not alerts.safe_send(notifier, alert):
            raise click.ClickException("could not send the test alert")
        client = approvals.ApprovalClient(
            s.alerts.server,
            sec.ntfy_approval_topic.get_secret_value(),
            token=sec.ntfy_token.get_secret_value() if sec.ntfy_token else None,
        )
        order = PlannedOrder(
            symbol="TEST", side=Side.BUY, notional=Decimal(1), ref_price=Decimal(1), reason="test"
        )
        click.echo("Sent. Waiting up to 3 minutes for a tap...")
        v = approvals.wait_and_decide(
            client,
            [order],
            plan_id=plan_id,
            nonce=nonce,
            created_at=created,
            deadline=created + timedelta(minutes=3),
            wait_minutes=3,
            poll_seconds=10,
            now=clock.now_utc,
            sleep=time.sleep,
        )["TEST"]
        click.echo(f"Result: {v.reason.replace('_', ' ')}")


def analyst_symbols(s: Any) -> list[str]:
    if s.analyst.symbols == "universe":
        from stoa.membership import default_membership

        return sorted(
            default_membership().ever_members(s.data.history_start, date.today()) | set(s.watchlist)
        )
    return sorted(set(s.watchlist) - {s.benchmark})


def make_analyst(ctx: Any) -> tuple[Any, StateStore, Any]:
    """An Analyst whose spend is recorded in the state's monthly ledger.

    Returns (analyst, store, state). Save the state afterwards so the spend sticks.
    """
    from stoa import analyst as an
    from stoa import spendcap

    s = ctx.settings
    store = StateStore(s.state.dir)
    state = store.load()
    key = ctx.secrets.llm_api_key
    client = an.make_client(key.get_secret_value()) if key is not None else None
    cap = spendcap.cap_for(state, s.budget, spendcap.month_of(clock.now_utc()))
    ledger = an.Ledger(float(cap), state.llm_spend_by_month)
    cache = an.ScoreCache(s.analyst.cache_dir)
    return an.Analyst(s.analyst, s.budget, s.llm.prices, client, cache, ledger), store, state


def pipeline_stamp() -> str:
    from stoa.pipeline import now_iso

    return now_iso()[:19].replace(":", "").replace("-", "")


def run_training(
    ctx: Any, synthetic: bool, promote: bool, force_promote: bool, results: Path, repo: Path
) -> str:
    from stoa import holdout as h
    from stoa.pipeline import load_inputs
    from stoa.retrain import train

    s = ctx.settings
    last = clock.last_closed_session(clock.now_utc())
    if synthetic:
        cutoff = min(last, clock.previous_session(s.backtest.holdout_start))
    else:
        cutoff = h.training_cutoff(repo, s, last)
    inputs = load_inputs(s, cutoff, synthetic=synthetic, blocked=h.blocked_window(repo, s))
    out = train(
        s,
        inputs,
        cutoff=cutoff,
        results_dir=results,
        promote=promote,
        force_promote=force_promote,
        progress=click.echo,
        tags=["synthetic"] if synthetic else [],
    )
    pooled = out.wf.pooled.get(out.wf.chosen)
    lines = [
        f"Trained through {cutoff} ({len(out.wf.folds)} folds). Chosen: {out.wf.chosen}.",
    ]
    if pooled is not None:
        lines.append(
            f"Out of sample: log loss {pooled.log_loss:.4f}, Brier {pooled.brier:.4f}, "
            f"AUC {pooled.auc:.3f}, calibration error {pooled.ece:.3f}."
        )
    lines.append(
        f"Deflated Sharpe {out.deflated.get('dsr', math.nan):.3f} after "
        f"{int(out.deflated.get('trials', 0))} trials."
    )
    lines.append(
        f"Walk-forward {'PASSED' if out.passed else 'did not pass'}"
        + (": " + "; ".join(out.reasons) if out.reasons else ".")
    )
    lines.append(
        f"Model {out.version}: {'promoted' if out.promoted else 'not promoted'} "
        f"({out.gate_reason})."
    )
    lines.append(f"Report: {out.report_dir}/calibration.md")
    return "\n".join(lines)
