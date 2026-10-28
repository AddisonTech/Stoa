"""Typed settings loaded from YAML plus environment overrides, validated at load.

Secrets are never read from YAML. They come only from environment variables and
live in :class:`Secrets`, which never prints its values.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from stoa import redact

Mode = Literal["backtest", "signal", "paper", "live"]

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
DEFAULT_SCOREBOARD_PATH = Path("config/scoreboard.yaml")
ENV_PREFIX = "STOA__"
LIVE_CONFIRM_PHRASE = "I_UNDERSTAND"


def _blank_to_none(v: Any) -> Any:
    """An empty environment override (``STOA__X__Y=``) means unset, not the current folder."""
    return None if isinstance(v, str) and not v.strip() else v


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RiskLimits(_Frozen):
    max_position_pct: Decimal = Decimal("0.20")
    max_positions: int = 5
    max_gross_exposure_pct: Decimal = Decimal("1.00")
    cash_buffer_pct: Decimal = Decimal("0.02")
    min_order_usd: Decimal = Decimal("1.00")
    daily_loss_limit_pct: Decimal = Decimal("0.03")
    kill_switch_drawdown_pct: Decimal = Decimal("0.25")
    max_orders_per_day: int = 10
    price_sanity_pct: Decimal = Decimal("0.08")
    block_entries_when_risk_off: bool = True
    max_sector_pct: Decimal = Decimal("0.40")
    max_pair_correlation: float = 0.8
    correlation_lookback: int = 60
    max_benchmark_pct: Decimal = Decimal("0.98")  # SPY-hold fallback only

    @model_validator(mode="after")
    def _check(self) -> RiskLimits:
        if not Decimal(0) < self.max_position_pct <= Decimal(1):
            raise ValueError("max_position_pct must be in (0, 1]")
        if not Decimal(0) < self.max_gross_exposure_pct <= Decimal(1):
            raise ValueError("max_gross_exposure_pct must be in (0, 1]; Stoa never borrows")
        if not Decimal(0) <= self.cash_buffer_pct < Decimal(1):
            raise ValueError("cash_buffer_pct must be in [0, 1)")
        if self.min_order_usd < Decimal("1.00"):
            raise ValueError("min_order_usd must be at least 1.00 (broker minimum)")
        if not Decimal(0) < self.kill_switch_drawdown_pct < Decimal(1):
            raise ValueError("kill_switch_drawdown_pct must be in (0, 1)")
        if not Decimal(0) < self.daily_loss_limit_pct < Decimal(1):
            raise ValueError("daily_loss_limit_pct must be in (0, 1)")
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least 1")
        if self.max_orders_per_day < self.max_positions:
            raise ValueError(
                "max_orders_per_day must be >= max_positions so exits are never blocked"
            )
        if self.price_sanity_pct <= 0:
            raise ValueError("price_sanity_pct must be positive")
        if not self.max_position_pct <= self.max_sector_pct <= Decimal(1):
            raise ValueError("max_sector_pct must be between max_position_pct and 1")
        if not 0 < self.max_pair_correlation <= 1:
            raise ValueError("max_pair_correlation must be in (0, 1]")
        if self.correlation_lookback < 20:
            raise ValueError("correlation_lookback must be at least 20 sessions")
        if not Decimal(0) < self.max_benchmark_pct <= Decimal(1) - self.cash_buffer_pct:
            raise ValueError("max_benchmark_pct must leave the cash buffer")
        return self


class QuantConfig(_Frozen):
    min_return_60: float = 0.0
    max_volatility_20: float = 0.60
    max_return_5: float = 0.15
    max_rsi_14: float = 85.0
    min_dollar_volume_20: float = 50_000_000.0


class ExitConfig(_Frozen):
    stop_atr_multiple: float = 2.5
    trailing_enabled: bool = True
    trailing_atr_multiple: float = 3.0
    take_profit_atr_multiple: float | None = None
    max_hold_sessions: int = 30
    trend_break_sessions: int = 2

    @model_validator(mode="after")
    def _check(self) -> ExitConfig:
        if self.stop_atr_multiple <= 0 or self.trailing_atr_multiple <= 0:
            raise ValueError("ATR multiples must be positive")
        if self.max_hold_sessions < 1 or self.trend_break_sessions < 1:
            raise ValueError("session counts must be at least 1")
        return self


class ScoutConfig(_Frozen):
    kind: Literal["rule", "llm", "auto"] = "auto"
    max_candidates: int = Field(default=5, ge=1, le=20)
    weight_return_60: float = 0.6
    weight_return_120: float = 0.4


class ModelPrice(_Frozen):
    input_per_mtok: float
    output_per_mtok: float


class LLMConfig(_Frozen):
    model: str = "claude-haiku-5-5"
    max_tokens: int = Field(default=1500, ge=256)
    anonymize: bool = True
    feature_decimals: int = Field(default=3, ge=0, le=6)
    monthly_cost_cap_usd: float = Field(default=5.0, ge=0)
    cache_dir: Path = Path("data/llm_cache")
    recent_trades_in_prompt: int = Field(default=10, ge=0)
    timeout_seconds: float = 60.0
    max_retries: int = 2
    prices: dict[str, ModelPrice] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> LLMConfig:
        if self.model not in self.prices:
            raise ValueError(f"no price configured for model {self.model!r}")
        return self


class CostConfig(_Frozen):
    slippage_bps: float = Field(default=5.0, ge=0)
    sell_fee_bps: float = Field(default=0.3, ge=0)
    min_sell_fee_usd: Decimal = Decimal("0.01")


class JobWindows(_Frozen):
    decide_after_close_minutes: int = Field(default=30, ge=0)
    decide_end: str = "23:59"
    execute_after_open_minutes: int = Field(default=15, ge=0)
    execute_window_minutes: int = Field(default=120, ge=1)
    # The morning sweep (New York time): it may start from sweep_start, reads news
    # published up to news_cutoff, and taps count until approval_deadline.
    sweep_start: str = "08:50"
    news_cutoff: str = "09:00"
    approval_deadline: str = "09:40"
    # The close job sends "Approvals closed" at approval_deadline, then answers late
    # taps with "Too late" until late_replies_until. It starts up to 5 minutes early.
    late_replies_until: str = "10:15"

    @field_validator(
        "decide_end", "sweep_start", "news_cutoff", "approval_deadline", "late_replies_until"
    )
    @classmethod
    def _hhmm(cls, v: str) -> str:
        hh, mm = v.split(":")
        if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59):
            raise ValueError("job window times must be HH:MM")
        return v


class Window(_Frozen):
    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> Window:
        if self.end < self.start:
            raise ValueError("window end before start")
        return self


class BacktestConfig(_Frozen):
    starting_equity: Decimal = Decimal("100")
    warmup_sessions: int = Field(default=260, ge=200)
    holdout_start: date = date(2026, 7, 1)
    holdout_end: date | None = date(2026, 10, 23)  # last holdout session; None means open-ended
    windows: dict[str, Window] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> BacktestConfig:
        if self.starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        if self.holdout_end is not None and self.holdout_end <= self.holdout_start:
            raise ValueError("holdout_end must be after holdout_start")
        for name, w in self.windows.items():
            if self.overlaps_holdout(w.start, w.end):
                raise ValueError(f"window {name} overlaps the sealed holdout")
        return self

    def overlaps_holdout(self, start: date, end: date) -> bool:
        return end >= self.holdout_start and (self.holdout_end is None or start <= self.holdout_end)


class DataConfig(_Frozen):
    provider: Literal["alpaca", "yahoo", "csv", "synthetic"] = "alpaca"
    cache_dir: Path = Path("data/bars")
    csv_dir: Path = Path("data/csv")
    history_start: date = date(2016, 1, 1)
    feed: str = "sip"
    fallback_feed: str = "iex"
    quote_feed: str = "iex"
    live_lookback_days: int = Field(default=450, ge=400)
    timeout_seconds: float = 30.0
    max_retries: int = Field(default=4, ge=0)
    min_fresh_fraction: float = Field(default=0.9, gt=0, le=1)


class AlpacaConfig(_Frozen):
    paper_url: str = "https://paper-api.alpaca.markets"
    live_url: str = "https://api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"
    fill_timeout_seconds: float = 90.0
    poll_interval_seconds: float = 2.0


class AlertConfig(_Frozen):
    server: str = "https://ntfy.sh"
    click_url: str | None = None
    retries: int = Field(default=2, ge=0)
    timeout_seconds: float = 10.0
    max_message_bytes: int = Field(default=3500, le=4096)


class MoneyConfig(_Frozen):
    deposit: Decimal = Decimal("100")
    milestone_multiple: Decimal = Decimal("2")
    tax_rate: Decimal = Decimal("0.25")

    @model_validator(mode="after")
    def _check(self) -> MoneyConfig:
        if self.deposit <= 0 or self.milestone_multiple <= 1:
            raise ValueError("deposit must be positive and milestone_multiple above 1")
        if not Decimal(0) <= self.tax_rate < Decimal(1):
            raise ValueError("tax_rate must be in [0, 1)")
        return self


class DemoConfig(_Frozen):
    """Synthetic data only (tests and the offline demo). Never used for grading real data."""

    signal: float = Field(default=0.0, ge=0)  # planted momentum edge; 0 is pure noise


class StateConfig(_Frozen):
    dir: Path = Path("state")
    shared_dir: Path | None = None  # models, registry, data shared by paper and live

    _blank = field_validator("shared_dir", mode="before")(_blank_to_none)


class MetricsConfig(_Frozen):
    risk_free_rate: float = 0.0
    bootstrap_samples: int = Field(default=1000, ge=10)
    bootstrap_block: int = Field(default=20, ge=1)
    seed: int = 7


class StrategyConfig(_Frozen):
    """Which pipeline DECIDE runs. ``model`` is the v0.2 default, ``rule`` is v1's."""

    kind: Literal["model", "rule"] = "model"
    trade_universe: Literal["watchlist", "universe"] = "watchlist"


class SizingConfig(_Frozen):
    """Bigger when sure and calm, smaller when not. Always clipped by the risk caps."""

    method: Literal["confidence_vol", "equal"] = "confidence_vol"
    edge: Literal["linear", "kelly"] = "kelly"
    kelly_fraction: float = Field(default=0.25, gt=0, le=1)
    vol_measure: Literal["realized", "atr"] = "realized"
    vol_floor: float = Field(default=0.10, gt=0)  # annualized; stops tiny vol from exploding size
    portfolio_vol_target: float = Field(default=0.15, gt=0, le=1)  # annualized
    min_edge: float = Field(default=0.0, ge=0, lt=0.5)


class EarningsConfig(_Frozen):
    enabled: bool = True
    exit_before: bool = True
    holding_window_sessions: int = Field(default=10, ge=1)
    unknown_lookahead_sessions: int = Field(default=15, ge=1)
    unknown_stale_days: int = Field(default=80, ge=1)
    block_when_unknown: bool = True
    confirm_lead_sessions: int = Field(default=10, ge=1)  # backtests: when a date counts as known
    upcoming_source: Literal["auto", "finnhub", "yahoo", "none"] = "auto"
    heuristic_volume_multiple: float = Field(default=2.5, gt=1)


class EventConfig(_Frozen):
    enabled: bool = True
    size_multiplier: float = Field(default=0.5, gt=0, le=1)


class ModelConfig(_Frozen):
    horizon: int = Field(default=10, ge=1, le=60)
    test_horizon: int | None = Field(default=5, ge=1, le=60)  # None: report one horizon only
    train_window: Literal["expanding", "rolling"] = "expanding"
    rolling_sessions: int = Field(default=756, ge=126)
    test_sessions: int = Field(default=63, ge=5)
    min_train_sessions: int = Field(default=504, ge=126)
    embargo_sessions: int | None = None  # None means equal to the horizon
    calibration: Literal["isotonic", "sigmoid"] = "isotonic"
    calibration_fraction: float = Field(default=0.25, gt=0.05, lt=0.6)
    candidates: tuple[Literal["logistic", "hgb"], ...] = ("logistic", "hgb")
    threshold: float = Field(default=0.55, gt=0.5, lt=1)
    max_candidates: int = Field(default=10, ge=1)
    hold_sessions: int | None = None  # None means equal to the horizon
    universe: Literal["sp500", "watchlist"] = "sp500"  # training cross-section
    use_quality: bool = True
    use_text: bool = False
    price_features: tuple[str, ...] | None = None  # None means every price feature
    normalize: Literal["zscore", "rank"] = "zscore"  # how features are scaled across stocks
    rebalance: Literal["daily", "monthly"] = "daily"  # monthly: entries on month starts only
    text_lookback_days: int = Field(default=10, ge=1)
    min_price: float = Field(default=5.0, ge=0)
    min_dollar_volume_20: float = Field(default=20_000_000.0, ge=0)
    gate_folds: int = Field(default=4, ge=1)
    gate_tolerance: float = Field(default=0.0, ge=0)
    max_ece: float = Field(default=0.05, gt=0, lt=1)
    seed: int = 7
    artifact_dir: Path | None = None  # None means <state.dir>/models
    registry_path: Path | None = None  # None means <state.dir>/experiments/registry.jsonl

    @model_validator(mode="after")
    def _check(self) -> ModelConfig:
        if self.embargo_sessions is not None and self.embargo_sessions < self.horizon:
            raise ValueError("embargo_sessions must be at least the horizon")
        if not self.candidates:
            raise ValueError("model.candidates is empty")
        if self.price_features is not None and not self.price_features:
            raise ValueError("model.price_features is empty; use null for every price feature")
        return self

    @property
    def embargo(self) -> int:
        return self.embargo_sessions if self.embargo_sessions is not None else self.horizon

    @property
    def hold(self) -> int:
        return self.hold_sessions if self.hold_sessions is not None else self.horizon


class AnalystConfig(_Frozen):
    """Claude reads text into scores. It never sees prices and never predicts."""

    enabled: bool = True  # used only when ANTHROPIC_API_KEY is set
    model: str = "claude-haiku-5-5"
    max_tokens: int = Field(default=400, ge=128)
    text_chars: int = Field(default=1200, ge=200)
    filing_chars: int = Field(default=6000, ge=500)
    max_items_per_symbol_per_day: int = Field(default=3, ge=1)
    contamination_cutoff: date = date(2026, 7, 1)
    shadow_start: date = date(2026, 10, 26)  # trial 7: only news published from here is scored
    shadow_sessions: int = Field(default=210, ge=1)  # trial 7 evaluates after this many
    symbols: Literal["watchlist", "universe"] = "watchlist"  # whose news is scored
    cache_dir: Path = Path("data/analyst_cache")
    batch_poll_seconds: float = Field(default=60.0, gt=0)
    batch_max_requests: int = Field(default=10000, ge=1, le=100000)
    cache_ttl: Literal["5m", "1h"] = "5m"
    batch_cache_ttl: Literal["5m", "1h"] = "1h"
    est_system_tokens: int = Field(default=950, ge=0)
    est_output_tokens: int = Field(default=120, ge=1)


class BudgetConfig(_Frozen):
    """One monthly dollar cap for every Claude call Stoa makes.

    The analyst warns once a month at ``warn_fraction`` of the cap and pauses at
    the cap. A tap can raise this month's cap by ``raise_step_usd``, never past
    ``hard_ceiling_usd``. Raises end with the month.
    """

    monthly_cap_usd: float = Field(default=25.0, ge=0)
    hard_ceiling_usd: float = Field(default=75.0, ge=0)
    raise_step_usd: float = Field(default=10.0, gt=0)
    warn_fraction: float = Field(default=0.8, gt=0, le=1)
    batch_discount: float = Field(default=0.5, gt=0, le=1)
    cache_write_multiplier_5m: float = 1.25
    cache_write_multiplier_1h: float = 2.0
    cache_read_multiplier: float = 0.1

    @model_validator(mode="after")
    def _ceiling(self) -> BudgetConfig:
        if self.hard_ceiling_usd < self.monthly_cap_usd:
            raise ValueError("budget.hard_ceiling_usd must be at least budget.monthly_cap_usd")
        return self


class DataSourcesConfig(_Frozen):
    sec_user_agent: str = "Stoa research stalor54@yahoo.com"
    sec_rate_per_second: float = Field(default=8.0, gt=0, le=10)
    sec_cache_dir: Path = Path("data/sec")
    news_cache_dir: Path = Path("data/news")
    fundamentals_path: Path | None = None  # None means <state.dir>/fundamentals.parquet
    earnings_path: Path | None = None  # None means <state.dir>/earnings.parquet
    reference_symbols: tuple[str, ...] = ("MTUM",)


class ApprovalConfig(_Frozen):
    mode: Literal["auto", "always", "never"] = "auto"
    # ``buys: mode``: buys follow ``mode`` like sells, so on paper nothing waits for a tap
    # (the owner's decision of 2026-10-28; docs/DECISIONS.md). ``always`` puts the Approve
    # and Skip buttons back on every buy in paper and live. The kill switch never waits.
    buys: Literal["always", "mode"] = "mode"
    live_days: int = Field(default=30, ge=0)
    wait_minutes: float = Field(default=5.0, ge=0, le=20)
    poll_interval_seconds: float = Field(default=30.0, gt=0)
    exempt_reasons: tuple[str, ...] = ("kill_switch",)


class DashboardConfig(_Frozen):
    """The public dashboard on GitHub Pages."""

    # The read-only live positions function (live/, on Vercel). Blank hides the panel.
    live_url: str = ""

    @field_validator("live_url")
    @classmethod
    def _https(cls, v: str) -> str:
        if v and not v.startswith("https://"):
            raise ValueError("dashboard.live_url must be an https:// address")
        return v


class BigPaperConfig(_Frozen):
    """The big paper account: one news brain trial on its own ~$100,000 Alpaca paper account.

    ``strategy`` names the trial: ``trial12`` exactly as registered (its sizing, its rails,
    its SPY core, no overlay), or ``trial13`` (trial 12's picks with trial 13's overlay).
    Kept apart from the $100 account: its own keys (ALPACA_BIG_PAPER_KEY_ID and
    ALPACA_BIG_PAPER_SECRET), its own state folder and its own alerts, tagged with
    ``tag``. It trades only when ``enabled`` is true and both keys are set.
    """

    enabled: bool = False
    strategy: Literal["trial12", "trial13"] = "trial12"
    feed_dir: Path = Path("state/big-paper-feed")  # the news brain's rolling state and pulls
    feed_repo: str = "ASmith91/stoa-feed"  # private Hugging Face dataset that keeps feed_dir
    state_dir: Path = Path("state/big-paper")
    tag: str = "BIG PAPER"
    starting_equity: Decimal = Decimal("100000")
    auto_approve: bool = True  # paper: picks go through without a tap

    @field_validator("tag")
    @classmethod
    def _tag(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("big_paper.tag must not be blank: it tells the two accounts apart")
        return v.strip()


class SweepConfig(_Frozen):
    """The 9:00 am overnight news sweep, read by FinBERT on CPU."""

    enabled: bool = True
    model: str = "ProsusAI/finbert"
    revision: str = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
    batch_size: int = Field(default=16, ge=1)
    max_length: int = Field(default=128, ge=16, le=512)
    max_articles_per_symbol: int = Field(default=20, ge=1)
    max_symbols_per_article: int = Field(default=3, ge=1)
    # Drop a planned buy: an overnight article at least this negative and a mean
    # net tone (positive minus negative) at or below drop_net.
    drop_negative: float = Field(default=0.80, gt=0, le=1)
    drop_net: float = Field(default=-0.30, ge=-1, le=0)
    # Add a reserve pick: an article at least this positive and a mean net tone at
    # or above add_net. Only into a free slot, only names the rules already passed.
    add_positive: float = Field(default=0.90, gt=0, le=1)
    add_net: float = Field(default=0.50, ge=0, le=1)
    max_adds: int = Field(default=1, ge=0)


class CrossCheckConfig(_Frozen):
    enabled: bool = True
    source: Literal["auto", "finnhub", "yahoo"] = "auto"
    tolerance_pct: float = Field(default=0.01, gt=0, lt=0.5)
    on_unavailable: Literal["skip", "proceed"] = "skip"
    timeout_seconds: float = 10.0


class HealthcheckConfig(_Frozen):
    enabled: bool = True
    timeout_seconds: float = 10.0
    retries: int = Field(default=2, ge=0)


class FallbackConfig(_Frozen):
    enabled: bool = True
    sharpe_window: int = Field(default=63, ge=20)
    quit_after_sessions: int = Field(default=126, ge=20)
    min_history_sessions: int = Field(default=63, ge=20)


class ExamConfig(_Frozen):
    paper_min_sessions: int = Field(default=42, ge=1)
    paper_target_sessions: int = Field(default=63, ge=1)
    paper_min_trades: int = Field(default=20, ge=0)
    paper_state_dir: Path | None = None  # where live mode reads the paper record from
    require_walkforward_pass: bool = True
    enforce_live: bool = True  # refuse live mode until the exams pass

    _blank = field_validator("paper_state_dir", mode="before")(_blank_to_none)


class Settings(_Frozen):
    mode: Mode = "paper"
    benchmark: str = "SPY"
    watchlist: tuple[str, ...]
    risk: RiskLimits = RiskLimits()
    quant: QuantConfig = QuantConfig()
    exits: ExitConfig = ExitConfig()
    scout: ScoutConfig = ScoutConfig()
    llm: LLMConfig
    costs: CostConfig = CostConfig()
    job_windows: JobWindows = JobWindows()
    backtest: BacktestConfig = BacktestConfig()
    data: DataConfig = DataConfig()
    alpaca: AlpacaConfig = AlpacaConfig()
    alerts: AlertConfig = AlertConfig()
    money: MoneyConfig = MoneyConfig()
    state: StateConfig = StateConfig()
    metrics: MetricsConfig = MetricsConfig()
    strategy: StrategyConfig = StrategyConfig()
    sizing: SizingConfig = SizingConfig()
    earnings: EarningsConfig = EarningsConfig()
    events: EventConfig = EventConfig()
    model: ModelConfig = ModelConfig()
    analyst: AnalystConfig = AnalystConfig()
    budget: BudgetConfig = BudgetConfig()
    sources: DataSourcesConfig = DataSourcesConfig()
    approval: ApprovalConfig = ApprovalConfig()
    sweep: SweepConfig = SweepConfig()
    dashboard: DashboardConfig = DashboardConfig()
    big_paper: BigPaperConfig = BigPaperConfig()
    crosscheck: CrossCheckConfig = CrossCheckConfig()
    healthcheck: HealthcheckConfig = HealthcheckConfig()
    fallback: FallbackConfig = FallbackConfig()
    exam: ExamConfig = ExamConfig()
    demo: DemoConfig = DemoConfig()

    @field_validator("watchlist", mode="before")
    @classmethod
    def _split(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("watchlist")
    @classmethod
    def _normalize(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        out = tuple(s.strip().upper() for s in v)
        if len(set(out)) != len(out):
            raise ValueError("watchlist has duplicates")
        if not out:
            raise ValueError("watchlist is empty")
        return out

    @model_validator(mode="after")
    def _benchmark_listed(self) -> Settings:
        if self.benchmark.upper() not in self.watchlist:
            raise ValueError("benchmark must be in the watchlist")
        return self

    @property
    def shared_dir(self) -> Path:
        """Where everything paper and live share lives: models, registry, data, exam records."""
        return self.state.shared_dir or self.state.dir

    @property
    def artifact_dir(self) -> Path:
        return self.model.artifact_dir or self.shared_dir / "models"

    @property
    def registry_path(self) -> Path:
        return self.model.registry_path or self.shared_dir / "experiments" / "registry.jsonl"

    @property
    def fundamentals_path(self) -> Path:
        return self.sources.fundamentals_path or self.shared_dir / "fundamentals.parquet"

    @property
    def earnings_path(self) -> Path:
        return self.sources.earnings_path or self.shared_dir / "earnings.parquet"

    @property
    def news_path(self) -> Path:
        return self.sources.news_cache_dir / "news.parquet"

    @property
    def analyst_scores_path(self) -> Path:
        return self.shared_dir / "analyst_scores.parquet"

    @property
    def analyst_cache_dir(self) -> Path:
        return self.analyst.cache_dir

    def strategy_hash(self) -> str:
        """Hash of every setting that changes trading decisions or grading."""
        keys = {
            "benchmark",
            "watchlist",
            "risk",
            "quant",
            "exits",
            "scout",
            "costs",
            "backtest",
            "strategy",
            "sizing",
            "earnings",
            "events",
        }
        payload = self.model_dump(mode="json", include=keys)
        payload["llm"] = self.llm.model_dump(
            mode="json", include={"model", "anonymize", "feature_decimals", "max_tokens"}
        )
        payload["model"] = self.model.model_dump(
            mode="json", exclude={"artifact_dir", "registry_path"}
        )
        payload["analyst"] = self.analyst.model_dump(
            mode="json", include={"model", "max_tokens", "text_chars", "filing_chars"}
        )
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


# Every secret and the environment variable it is read from. Nothing else holds secrets.
SECRET_ENV: dict[str, str] = {
    "alpaca_paper_key_id": "ALPACA_PAPER_KEY_ID",
    "alpaca_paper_secret": "ALPACA_PAPER_SECRET",
    "alpaca_live_key_id": "ALPACA_LIVE_KEY_ID",
    "alpaca_live_secret": "ALPACA_LIVE_SECRET",
    "llm_api_key": "ANTHROPIC_API_KEY",
    "ntfy_topic": "STOA_NTFY_TOPIC",
    "ntfy_token": "STOA_NTFY_TOKEN",
    "live_confirm": "STOA_LIVE_CONFIRM",
    "ntfy_approval_topic": "STOA_NTFY_APPROVAL_TOPIC",
    "healthcheck_url": "STOA_HEALTHCHECK_URL",
    "finnhub_key": "FINNHUB_API_KEY",
    # The big paper account's own keys. Only stoa.bigpaper reads them; the $100 account never does.
    "alpaca_big_paper_key_id": "ALPACA_BIG_PAPER_KEY_ID",
    "alpaca_big_paper_secret": "ALPACA_BIG_PAPER_SECRET",
    # Write access to the big paper feed's private Hugging Face dataset. Read in the process
    # that needs it and passed to each call; never written to disk.
    "hf_token": "HF_TOKEN",
}


class Secrets(BaseModel):
    """Secrets from the environment only. Values never appear in repr or logs."""

    model_config = ConfigDict(frozen=True)

    alpaca_paper_key_id: SecretStr | None = None
    alpaca_paper_secret: SecretStr | None = None
    alpaca_live_key_id: SecretStr | None = None
    alpaca_live_secret: SecretStr | None = None
    llm_api_key: SecretStr | None = None
    ntfy_topic: SecretStr | None = None
    ntfy_token: SecretStr | None = None
    live_confirm: SecretStr | None = None
    ntfy_approval_topic: SecretStr | None = None
    healthcheck_url: SecretStr | None = None
    finnhub_key: SecretStr | None = None
    alpaca_big_paper_key_id: SecretStr | None = None
    alpaca_big_paper_secret: SecretStr | None = None
    hf_token: SecretStr | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Secrets:
        e = os.environ if env is None else env

        def get(name: str) -> SecretStr | None:
            value = e.get(name, "").strip()
            return SecretStr(value) if value else None

        out = cls(**{name: get(env_name) for name, env_name in SECRET_ENV.items()})
        redact.register(out.values())
        return out

    def values(self) -> list[str]:
        """Every secret value that is set, for log and alert redaction."""
        out = []
        for name in type(self).model_fields:
            v = getattr(self, name)
            if isinstance(v, SecretStr) and name != "live_confirm":
                out.append(v.get_secret_value())
        return out

    def has_paper_keys(self) -> bool:
        return self.alpaca_paper_key_id is not None and self.alpaca_paper_secret is not None

    def has_big_paper_keys(self) -> bool:
        return self.alpaca_big_paper_key_id is not None and self.alpaca_big_paper_secret is not None

    def has_live_keys(self) -> bool:
        return self.alpaca_live_key_id is not None and self.alpaca_live_secret is not None

    def live_confirmed(self) -> bool:
        return (
            self.live_confirm is not None
            and self.live_confirm.get_secret_value() == LIVE_CONFIRM_PHRASE
        )


def _set_path(tree: dict[str, Any], path: list[str], value: str) -> None:
    node = tree
    for key in path[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot override {'.'.join(path)}: {key} is not a section")
        node = child
    node[path[-1]] = yaml.safe_load(value) if value.strip() else value


def apply_env_overrides(raw: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """Apply STOA__SECTION__KEY=value overrides plus the STOA_MODE and STOA_STATE_DIR shortcuts."""
    tree: dict[str, Any] = json.loads(json.dumps(raw, default=str))
    for name in sorted(env):
        if name.startswith(ENV_PREFIX):
            path = [p.lower() for p in name[len(ENV_PREFIX) :].split("__") if p]
            if path:
                _set_path(tree, path, env[name])
    if env.get("STOA_MODE"):
        tree["mode"] = env["STOA_MODE"].strip()
    if env.get("STOA_STATE_DIR"):
        tree.setdefault("state", {})["dir"] = env["STOA_STATE_DIR"].strip()
    return tree


def load_settings(path: Path | None = None, env: Mapping[str, str] | None = None) -> Settings:
    e = os.environ if env is None else env
    cfg_path = path or Path(e.get("STOA_CONFIG", str(DEFAULT_CONFIG_PATH)))
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_path} must contain a mapping")
    return Settings.model_validate(apply_env_overrides(raw, e))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
