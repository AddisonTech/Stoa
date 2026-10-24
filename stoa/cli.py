"""Command line entry point: `stoa ...`."""

from __future__ import annotations

import functools
import json
import logging
import sys
from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import click

from stoa import __version__, alerts, cli_v2, clock, redact
from stoa.config import Secrets, Settings, load_settings
from stoa.jobs import EXIT_OK, Runtime, live_lock_problems
from stoa.jobs import decide as run_decide
from stoa.jobs import execute as run_execute
from stoa.state import StateStore

log = logging.getLogger("stoa")


class Ctx:
    def __init__(self, config: Path | None) -> None:
        self.config = config
        self._settings: Settings | None = None

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings(self.config)
        return self._settings

    @property
    def secrets(self) -> Secrets:
        return Secrets.from_env()


def make_notifier(settings: Settings, secrets: Secrets) -> alerts.Notifier:
    if secrets.ntfy_topic is None:
        return alerts.ConsoleNotifier()
    return alerts.NtfyNotifier(
        settings.alerts.server,
        secrets.ntfy_topic.get_secret_value(),
        token=secrets.ntfy_token.get_secret_value() if secrets.ntfy_token else None,
        default_click=settings.alerts.click_url,
        retries=settings.alerts.retries,
        max_message_bytes=settings.alerts.max_message_bytes,
        timeout=settings.alerts.timeout_seconds,
    )


def runtime(ctx: Ctx) -> Runtime:
    s, sec = ctx.settings, ctx.secrets
    return Runtime(settings=s, secrets=sec, notifier=make_notifier(s, sec))


pass_ctx = click.make_pass_decorator(Ctx)


def friendly_errors[F: Callable[..., Any]](func: F) -> F:
    """Show expected failures (missing data, holdout refusals) as one clear line."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from stoa.broker.base import BrokerError
        from stoa.data.base import DataError
        from stoa.holdout import HoldoutError

        try:
            return func(*args, **kwargs)
        except (DataError, HoldoutError, BrokerError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

    return cast(F, wrapper)


@click.group()
@click.version_option(__version__, prog_name="stoa")
@click.option(
    "--config", type=click.Path(path_type=Path, exists=True), default=None, help="Config YAML."
)
@click.option("-v", "--verbose", is_flag=True, help="Debug logging.")
@click.pass_context
def main(click_ctx: click.Context, config: Path | None, verbose: bool) -> None:
    """Stoa: daily AI trade ideas, a rule-based risk engine, and honest grading."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    redact.install()
    Secrets.from_env()  # registers every secret value with the redacting log filter
    click_ctx.obj = Ctx(config)


# ---- data ------------------------------------------------------------------------


@main.group()
def data() -> None:
    """Market data."""


@data.command("fetch")
@click.option("--start", type=click.DateTime(["%Y-%m-%d"]), default=None)
@click.option("--end", type=click.DateTime(["%Y-%m-%d"]), default=None)
@click.option("--provider", type=click.Choice(["alpaca", "yahoo", "csv"]), default=None)
@pass_ctx
@friendly_errors
def data_fetch(ctx: Ctx, start: object, end: object, provider: str | None) -> None:
    """Download daily bars for the watchlist into the local cache."""
    from stoa.data.base import DataError
    from stoa.data.cache import DEFAULT_NAMESPACE, BarCache
    from stoa.data.factory import make_provider

    s = ctx.settings
    if provider:
        s = s.model_copy(update={"data": s.data.model_copy(update={"provider": provider})})
    begin = start.date() if start else s.data.history_start  # type: ignore[attr-defined]
    finish = end.date() if end else clock.last_closed_session(clock.now_utc())  # type: ignore[attr-defined]
    prov = make_provider(s, ctx.secrets)
    refs = [r for r in s.sources.reference_symbols if r not in s.watchlist]
    bars = prov.fetch([*s.watchlist, *refs], begin, finish)
    missing = sorted(set(s.watchlist) - set(bars))
    if missing:
        raise DataError(f"no bars for watchlist symbols {missing}")
    for r in refs:
        if r not in bars:
            click.echo(f"{r}: no bars; that baseline will be missing from the scoreboard")
    BarCache(s.data.cache_dir, DEFAULT_NAMESPACE).write(bars)
    for sym, df in sorted(bars.items()):
        click.echo(f"{sym:6s} {len(df):5d} bars {df.index[0].date()} to {df.index[-1].date()}")
    source = getattr(prov, "last_feed_used", None) or s.data.provider
    click.echo(f"Saved to {s.data.cache_dir / DEFAULT_NAMESPACE} (source {source}).")


# ---- backtests and grading ---------------------------------------------------------


def _window_dates(s: Settings, window: str) -> tuple[date, date]:
    if window == "holdout":
        raise click.UsageError("the holdout is scored only with `stoa holdout score`")
    if window not in s.backtest.windows:
        raise click.UsageError(f"unknown window {window!r}; known: {', '.join(s.backtest.windows)}")
    w = s.backtest.windows[window]
    return w.start, w.end


@main.command()
@click.option("--strategy", type=click.Choice(["agents", "spy", "sma"]), default="agents")
@click.option("--window", default="tune_a", help="tune_a, tune_b, tune_c or validation")
@click.option("--llm/--no-llm", default=False, help="Use the Claude scout (replay cache first).")
@click.option("--synthetic", is_flag=True, help="Use seeded synthetic bars (offline demo).")
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Write results here.")
@pass_ctx
@friendly_errors
def backtest(
    ctx: Ctx, strategy: str, window: str, llm: bool, synthetic: bool, out: Path | None
) -> None:
    """Run one strategy over one window."""
    from stoa import scoreboard

    s = ctx.settings
    start, end = _window_dates(s, window)
    bars = scoreboard.load_bars(s, end, synthetic=synthetic)
    run = scoreboard.run_one(s, ctx.secrets, bars, strategy, window, start, end, llm=llm)
    m = run.metrics
    click.echo(f"{strategy} {window} {start} to {end} (equity curve starts at the prior close)")
    for key in (
        "final_equity",
        "total_return",
        "cagr",
        "max_drawdown",
        "sharpe",
        "profit_factor",
        "trade_count",
    ):
        click.echo(
            f"  {key:15s} {getattr(m, key):.4f}"
            if isinstance(getattr(m, key), float)
            else f"  {key:15s} {getattr(m, key)}"
        )
    if llm and strategy == "agents":
        click.echo(f"  llm fallbacks   {run.llm_fallbacks} of {run.metrics.sessions + 1} decisions")
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{strategy}-{window}.json").write_text(json.dumps(run.to_dict(), indent=2))


@main.command()
@click.option("--llm/--no-llm", default=False, help="Use the Claude scout for the v1 agents.")
@click.option("--synthetic", is_flag=True, help="Use seeded synthetic bars (offline demo).")
@click.option("--no-model", is_flag=True, help="Skip the model; grade the v1 rule pipeline.")
@click.option("--results", type=click.Path(path_type=Path), default=Path("results"))
@pass_ctx
@friendly_errors
def score(ctx: Ctx, llm: bool, synthetic: bool, no_model: bool, results: Path) -> None:
    """Every strategy on every window, graded on risk against SPY, the 50/200 rule and MTUM."""
    from stoa import scoreboard
    from stoa.experiments import Registry

    s = ctx.settings
    source = None
    if s.strategy.kind == "model" and not no_model:
        from stoa.pipeline import load_inputs, walkforward_model_source

        board = scoreboard.load_scoreboard()
        end = max(s.backtest.windows[w].end for w in [*board.tune_windows, board.evaluation_window])
        inputs = load_inputs(s, end, synthetic=synthetic)
        source, _ = walkforward_model_source(s, inputs, end)
    report = scoreboard.score_all(
        s,
        ctx.secrets,
        llm=llm,
        synthetic=synthetic,
        model=source,
        registry=Registry(s.registry_path),
    )
    path = scoreboard.write_report(report, results)
    click.echo(scoreboard.render_markdown(report))
    click.echo(f"Report written to {path}")


# ---- holdout -------------------------------------------------------------------------


@main.group()
def holdout() -> None:
    """Seal and score the one-shot holdout."""


@holdout.command("seal")
@click.option("--repo", type=click.Path(path_type=Path), default=Path("."))
@pass_ctx
@friendly_errors
def holdout_seal(ctx: Ctx, repo: Path) -> None:
    """Lock code, config and the pass bar before the holdout is ever scored."""
    from stoa import holdout as h

    lock = h.seal(
        repo,
        ctx.config or Path("config/default.yaml"),
        Path("config/scoreboard.yaml"),
        clock.now_utc(),
    )
    click.echo(
        f"Sealed at commit {lock.commit[:12]} on {lock.sealed_at:%Y-%m-%d}. "
        f"Holdout from {lock.holdout_start}."
    )


@holdout.command("score")
@click.option("--repo", type=click.Path(path_type=Path), default=Path("."))
@click.option("--llm/--no-llm", default=False)
@click.option("--synthetic", is_flag=True, help="Synthetic bars, for demos only.")
@pass_ctx
@friendly_errors
def holdout_score(ctx: Ctx, repo: Path, llm: bool, synthetic: bool) -> None:
    """Score the holdout exactly once. Refuses on a dirty tree or changed code or config."""
    from stoa import holdout as h

    now = clock.now_utc()
    source = holdout_model_source(ctx.settings, repo, now, synthetic)
    result = h.score(
        repo,
        ctx.config or Path("config/default.yaml"),
        Path("config/scoreboard.yaml"),
        ctx.settings,
        ctx.secrets,
        now,
        llm=llm,
        synthetic=synthetic,
        model=source,
    )
    click.echo(json.dumps(result, indent=2))


def holdout_model_source(settings: Settings, repo: Path, now: object, synthetic: bool) -> object:
    """The model sealed with the holdout, predicting each holdout session on its own data."""
    from stoa import holdout as h
    from stoa.dataset import FeatureSet, build_dataset
    from stoa.model import ModelStore
    from stoa.model_strategy import ModelPredictor, ModelStrategy
    from stoa.pipeline import load_inputs
    from stoa.scoreboard import ModelSource
    from stoa.universe import Universe

    lock = h.read_lock(repo)
    if settings.strategy.kind != "model" or lock is None or not lock.model_version:
        return None
    model, _ = ModelStore(settings.artifact_dir).load(lock.model_version)
    end = clock.last_closed_session(now)  # type: ignore[arg-type]
    inputs = load_inputs(settings, end, synthetic=synthetic)
    t = inputs.training
    fs = FeatureSet(quality="q_roe" in model.features, text="t_sentiment" in model.features)
    ds = build_dataset(
        t.bars,
        settings.benchmark,
        model.horizon,
        membership=t.membership,
        extra_symbols=t.extra_symbols,
        exclude=t.exclude,
        fundamentals=t.fundamentals,
        text_scores=t.text_scores,
        features=fs,
        min_price=settings.model.min_price,
        min_dollar_volume=settings.model.min_dollar_volume_20,
        start=settings.backtest.holdout_start - timedelta(days=10),
    )
    strategy = ModelStrategy(
        settings, ModelPredictor(model, ds), inputs.calendar, tradable=inputs.tradable
    )
    tradable = tuple(sorted(inputs.tradable | {settings.benchmark}))
    return ModelSource(
        strategy=strategy,
        bars=t.bars,
        universe=Universe.assume_all_fractionable(tradable, settings.benchmark),
    )


# ---- daily jobs --------------------------------------------------------------------------


@main.command()
@click.option("--force", is_flag=True, help="Run outside the window or again for the same session.")
@pass_ctx
def decide(ctx: Ctx, force: bool) -> None:
    """After the close: build the plan for the next open and send the daily summary."""
    sys.exit(run_decide(runtime(ctx), force=force).code)


@main.command()
@click.option("--force", is_flag=True, help="Run outside the window or again for the same session.")
@pass_ctx
def execute(ctx: Ctx, force: bool) -> None:
    """After the open: submit last night's plan."""
    sys.exit(run_execute(runtime(ctx), force=force).code)


@main.command()
@pass_ctx
def status(ctx: Ctx) -> None:
    """Show mode, locks, state and the pending plan."""
    s, sec = ctx.settings, ctx.secrets
    state = StateStore(s.state.dir).load()
    click.echo(f"mode            {s.mode}")
    click.echo(f"paper keys      {'set' if sec.has_paper_keys() else 'missing'}")
    click.echo(f"live keys       {'set' if sec.has_live_keys() else 'missing'}")
    click.echo(f"claude api key  {'set' if sec.llm_api_key else 'missing (rule scout)'}")
    click.echo(f"ntfy topic      {'set' if sec.ntfy_topic else 'missing (console alerts)'}")
    problems = live_lock_problems(s, sec)
    click.echo(
        f"live lock       {'UNLOCKED' if not problems else 'locked: ' + '; '.join(problems)}"
    )
    click.echo(
        "kill switch     "
        + (
            f"ACTIVE since {state.kill_switch.tripped_session}"
            if state.kill_switch.active
            else "off"
        )
    )
    click.echo(f"high-water mark {state.high_water_mark or 'not set'}")
    click.echo(
        f"last decided    {state.decided_sessions[-1] if state.decided_sessions else 'never'}"
    )
    click.echo(
        f"last executed   {state.executed_sessions[-1] if state.executed_sessions else 'never'}"
    )
    if state.pending_plan:
        p = state.pending_plan
        click.echo(f"pending plan    {p.plan_id} for {p.execute_session}: {len(p.orders)} orders")
        for o in p.orders:
            amount = alerts.money(o.notional) if o.notional else f"{alerts.shares(o.qty or 0)} sh"  # type: ignore[arg-type]
            click.echo(f"  {o.side.value:4s} {o.symbol:6s} {amount} ({o.reason})")
    else:
        click.echo("pending plan    none")
    if state.open_orders:
        click.echo(f"open orders     {', '.join(o.client_order_id for o in state.open_orders)}")
    t = state.track
    click.echo(
        "fallback        "
        + (f"ACTIVE since {t.fallback_since}: {t.fallback_reason}" if t.fallback_active else "off")
    )
    click.echo(f"approval topic  {'set' if sec.ntfy_approval_topic else 'missing'}")
    click.echo(f"healthcheck     {'set' if sec.healthcheck_url else 'missing'}")
    from stoa.model import ModelStore

    click.echo(f"live model      {ModelStore(s.artifact_dir).current_version() or 'none'}")


# ---- alerts --------------------------------------------------------------------------------


@main.group()
def alert() -> None:
    """Phone alerts."""


@alert.command("test")
@pass_ctx
def alert_test(ctx: Ctx) -> None:
    """Send a test notification."""
    notifier = make_notifier(ctx.settings, ctx.secrets)
    ok = alerts.safe_send(notifier, alerts.hello_alert())
    if isinstance(notifier, alerts.ConsoleNotifier):
        click.echo("STOA_NTFY_TOPIC is not set, so the alert was printed here instead of sent.")
    click.echo("Sent." if ok else "Failed to send. Check the topic and server.")
    sys.exit(EXIT_OK if ok else 1)


# ---- kill switch -----------------------------------------------------------------------------


@main.group("kill-switch")
def kill_switch() -> None:
    """Inspect or reset the kill switch."""


@kill_switch.command("status")
@pass_ctx
def kill_switch_status(ctx: Ctx) -> None:
    state = StateStore(ctx.settings.state.dir).load()
    ks = state.kill_switch
    if ks.active:
        click.echo(
            f"ACTIVE since {ks.tripped_session}: {ks.reason}. "
            f"High-water mark {state.high_water_mark}."
        )
    else:
        click.echo(f"off. High-water mark {state.high_water_mark}.")


def _current_equity(ctx: Ctx, equity: str | None) -> Decimal:

    from stoa.jobs import make_broker

    if equity is not None:
        return Decimal(equity)
    s = ctx.settings
    if s.mode == "signal":
        state = StateStore(s.state.dir).load()
        if state.last_close_equity is None:
            raise click.UsageError("no recorded equity yet; pass --equity")
        return state.last_close_equity
    return make_broker(runtime(ctx)).account().equity


@kill_switch.command("reset")
@click.option("--equity", default=None, help="Current equity, if the broker cannot be reached.")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
@pass_ctx
@friendly_errors
def kill_switch_reset(ctx: Ctx, equity: str | None, yes: bool) -> None:
    """Clear the kill switch and rebase the high-water mark to current equity."""
    from stoa import money

    store = StateStore(ctx.settings.state.dir)
    state = store.load()
    if not state.kill_switch.active:
        click.echo("The kill switch is not active.")
        return
    eq = _current_equity(ctx, equity)
    if not yes:
        click.confirm(f"Reset the kill switch and set the high-water mark to {eq}?", abort=True)
    old = money.reset_kill_switch(state, eq)
    store.save(state)
    click.echo(f"Kill switch reset. High-water mark {old} -> {eq}.")


# ---- money plan --------------------------------------------------------------------------------


@main.group("money")
def money_group() -> None:
    """The grow-then-withdraw money plan."""


@money_group.command("status")
@pass_ctx
def money_status(ctx: Ctx) -> None:
    from stoa import money

    s = ctx.settings
    state = StateStore(s.state.dir).load()
    year = clock.now_utc().year
    realized = state.realized_for_year(year)
    click.echo(f"deposit               {alerts.money(s.money.deposit)}")
    click.echo(f"withdraw at           {alerts.money(money.milestone(s.money))}")
    click.echo(
        "plan active           "
        + (
            "yes"
            if money.plan_active(state, s.money)
            else "no (no run yet, or size differs from deposit)"
        )
    )
    click.echo(f"milestone alerted     {'yes' if state.milestone_alerted else 'no'}")
    click.echo(f"principal withdrawn   {'yes' if state.principal_withdrawn else 'no'}")
    click.echo(f"last close equity     {state.last_close_equity}")
    click.echo(f"net deposits          {alerts.money(state.net_flows)}")
    click.echo(f"realized {year} (short-term) {alerts.money(realized)}")
    click.echo(
        f"suggested tax reserve {alerts.money(money.tax_reserve(realized, s.money.tax_rate))} "
        f"at {s.money.tax_rate:.0%}"
    )


@money_group.command("confirm-withdrawal")
@click.option(
    "--equity", default=None, help="Equity after the withdrawal, if the broker cannot be reached."
)
@click.option("--yes", is_flag=True)
@pass_ctx
@friendly_errors
def money_confirm(ctx: Ctx, equity: str | None, yes: bool) -> None:
    """Mark the deposit as withdrawn and rebase the high-water mark."""
    from stoa import money

    store = StateStore(ctx.settings.state.dir)
    state = store.load()
    if state.principal_withdrawn:
        click.echo("The withdrawal is already confirmed.")
        return
    deposit = ctx.settings.money.deposit
    if ctx.settings.mode == "signal":
        eq = money.withdraw_shadow(state, deposit)
        click.echo(f"Signal mode: took {alerts.money(deposit)} out of the paper portfolio.")
    else:
        eq = _current_equity(ctx, equity)
    if not yes:
        click.confirm(f"Confirm you withdrew the deposit and equity is now {eq}?", abort=True)
    old = money.confirm_withdrawal(state, eq, deposit)
    state.principal_flow_pending = state.principal_flow_pending and ctx.settings.mode != "signal"
    store.save(state)
    click.echo(f"Withdrawal confirmed. High-water mark {old} -> {eq}. Stoa now trades house money.")


cli_v2.register(main, pass_ctx, friendly_errors)
