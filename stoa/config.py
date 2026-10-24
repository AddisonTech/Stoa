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

Mode = Literal["backtest", "signal", "paper", "live"]

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
DEFAULT_SCOREBOARD_PATH = Path("config/scoreboard.yaml")
ENV_PREFIX = "STOA__"
LIVE_CONFIRM_PHRASE = "I_UNDERSTAND"


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

    @field_validator("decide_end")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        hh, mm = v.split(":")
        if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59):
            raise ValueError("decide_end must be HH:MM")
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
    windows: dict[str, Window] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> BacktestConfig:
        if self.starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        for name, w in self.windows.items():
            if w.end >= self.holdout_start:
                raise ValueError(f"window {name} overlaps the sealed holdout")
        return self


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


class StateConfig(_Frozen):
    dir: Path = Path("state")


class MetricsConfig(_Frozen):
    risk_free_rate: float = 0.0
    bootstrap_samples: int = Field(default=1000, ge=10)
    bootstrap_block: int = Field(default=20, ge=1)
    seed: int = 7


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

    def strategy_hash(self) -> str:
        """Hash of every setting that changes trading decisions or grading."""
        keys = {"benchmark", "watchlist", "risk", "quant", "exits", "scout", "costs", "backtest"}
        payload = self.model_dump(mode="json", include=keys)
        payload["llm"] = self.llm.model_dump(
            mode="json", include={"model", "anonymize", "feature_decimals", "max_tokens"}
        )
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


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

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Secrets:
        e = os.environ if env is None else env

        def get(name: str) -> SecretStr | None:
            value = e.get(name, "").strip()
            return SecretStr(value) if value else None

        return cls(
            alpaca_paper_key_id=get("ALPACA_PAPER_KEY_ID"),
            alpaca_paper_secret=get("ALPACA_PAPER_SECRET"),
            alpaca_live_key_id=get("ALPACA_LIVE_KEY_ID"),
            alpaca_live_secret=get("ALPACA_LIVE_SECRET"),
            llm_api_key=get("ANTHROPIC_API_KEY"),
            ntfy_topic=get("STOA_NTFY_TOPIC"),
            ntfy_token=get("STOA_NTFY_TOKEN"),
            live_confirm=get("STOA_LIVE_CONFIRM"),
        )

    def has_paper_keys(self) -> bool:
        return self.alpaca_paper_key_id is not None and self.alpaca_paper_secret is not None

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
