"""The public trial log: every trial Stoa has run, its verdict and one plain line.

The dashboard shows this table so a reader sees every try, not only the one that is live.
Each row comes from docs/DECISIONS.md and the experiment registry on the state branch;
a new trial gets its row when it is graded, in the same commit as its result.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Trial:
    number: str
    name: str
    graded: str  # date graded, or "" if never graded
    verdict: str  # PASS, FAIL, STARVED, or a short status
    result: str  # one plain line


TRIALS: tuple[Trial, ...] = (
    Trial(
        "1 to 5",
        "Price and quality model, 5 and 10 sessions",
        "2026-10-24",
        "FAIL",
        "Walk-forward could not tell winners from losers (out-of-sample AUC 0.496 to 0.498, "
        "0.5 is a coin flip).",
    ),
    Trial(
        "6",
        "Price model, monthly",
        "2026-10-24",
        "FAIL",
        "Walk-forward AUC 0.494; validation -1.3%, Sharpe -1.41.",
    ),
    Trial(
        "7",
        "Claude reads the news (forward test)",
        "",
        "Not graded",
        "Stopped with every trial on 2026-10-29 before it had enough live sessions.",
    ),
    Trial(
        "8",
        "Earnings drift and insider buying",
        "2026-10-25",
        "FAIL",
        "Validation -10.5%, Sharpe -0.04, against SPY's gain.",
    ),
    Trial(
        "9",
        "News brain, daily",
        "2026-10-26",
        "FAIL",
        "Graded 5 session run: -4.6% against SPY +103% (2023 to mid-2026), 6 trades.",
    ),
    Trial(
        "10",
        "News brain, minute timing",
        "2026-10-27",
        "STARVED",
        "6 graded trades; too few to judge.",
    ),
    Trial(
        "11", "News brain, pre-market entry", "2026-10-27", "STARVED", "15 graded trades, -6.5%."
    ),
    Trial(
        "12",
        "News brain, 1 session, idle money in SPY",
        "2026-10-27",
        "FAIL",
        "+169% against SPY +103%, but failed the deflated Sharpe; then failed its one "
        "holdout run on drawdown.",
    ),
    Trial(
        "13",
        "Trial 12 with a risk overlay",
        "2026-10-28",
        "FAIL",
        "+103.9% against SPY +103.1%; drawdown not fixed (15.8%).",
    ),
    Trial(
        "14",
        "Trial 12 with idle money in cash",
        "2026-10-29",
        "FAIL",
        "+54.0% against SPY +103.1%; failed the deflated Sharpe.",
    ),
    Trial(
        "15",
        "The owner's brain, 5 picks a day",
        "",
        "Live on paper",
        "Running on the big paper account since 2026-10-30, graded day by day in its card.",
    ),
    Trial(
        "16",
        "Defense contract awards",
        "2026-11-12",
        "FAIL",
        "680 trades, 42.2% beat SPY after costs, -0.08% a trade.",
    ),
    Trial(
        "17",
        "Filings that barely change (Lazy Prices)",
        "",
        "In progress",
        "Pre-registered rules; waits for the EDGAR pull of every liquid filer.",
    ),
    Trial(
        "18",
        "EIA weekly reports (first run)",
        "",
        "Void",
        "Crashed before any result on pre-2015 EIA history; fixed, rerun as trial 20.",
    ),
    Trial(
        "19",
        "EIA weekly reports (second run)",
        "",
        "Void",
        "Crashed before any result at the news brain's holdout wall; rerun as trial 20.",
    ),
    Trial(
        "20",
        "EIA weekly reports: UNG and USO after a bullish surprise",
        "2026-11-19",
        "FAIL",
        "Next open: -59.8%, 181 trades, 47% won. Same day from 10:35 am: -24.1%, 160 trades, "
        "38% won. Holding both funds: -13.4%.",
    ),
)


def rows() -> list[Trial]:
    return list(TRIALS)
