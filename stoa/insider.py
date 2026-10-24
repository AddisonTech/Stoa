"""Insider purchases from the SEC Insider Transactions Data Sets (Forms 3, 4 and 5).

The SEC publishes the structured content of every Form 3, 4 and 5 as one zip per
quarter (``YYYYqN_form345.zip``) with tab-separated tables. Stoa reads three:

- ``SUBMISSION``: accession, filing date (a date, no time), issuer CIK and
  ticker, form type, and ``AFF10B5ONE`` (the 10b5-1 box, filled in from 2023).
- ``REPORTINGOWNER``: the insider's CIK and relationship (Director, Officer,
  TenPercentOwner, Other, or a combination).
- ``NONDERIV_TRANS``: transaction code, date, shares and price.

The result is one row per open-market purchase (code P, acquired) by an officer
or director. Filing dates carry no time and Form 4s arrive until 10 pm, so a
purchase is known only from the close of the session after its filing date
(see ``known_session``).
"""

from __future__ import annotations

import io
import logging
import zipfile
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import pandas as pd

from stoa import clock
from stoa.data.base import DataError
from stoa.data.sec import SECClient

log = logging.getLogger(__name__)

URL_PATTERNS = (
    "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip",
    "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/"
    "{q}_form345.zip",
)
PURCHASE_COLUMNS = (
    "accession",
    "issuer_cik",
    "symbol",
    "owner_cik",
    "relationship",
    "trans_date",
    "filing_date",
    "shares",
    "price",
    "value",
    "plan_10b5_1",
    "form",
)


def quarters(first: str, last: str) -> list[str]:
    """``2015q3`` to ``2026q3`` inclusive, as SEC file stems."""
    y, q = int(first[:4]), int(first[-1])
    ly, lq = int(last[:4]), int(last[-1])
    out = []
    while (y, q) <= (ly, lq):
        out.append(f"{y}q{q}")
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def download(sec: SECClient, quarter: str, dest: Path, refresh: bool = False) -> Path:
    """Fetch one quarter's zip (throttled, with the SEC User-Agent) unless it is on disk."""
    path = dest / f"{quarter}_form345.zip"
    if path.exists() and not refresh:
        return path
    dest.mkdir(parents=True, exist_ok=True)
    for pattern in URL_PATTERNS:
        sec.limiter.wait()
        resp = sec.client.get(pattern.format(q=quarter))
        if resp.status_code == 200 and resp.content[:2] == b"PK":
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(resp.content)
            tmp.replace(path)
            return path
    raise DataError(f"insider data set {quarter} not found at the SEC")


def _table(z: zipfile.ZipFile, name: str, columns: list[str]) -> pd.DataFrame:
    member = next((n for n in z.namelist() if n.upper().startswith(name)), None)
    if member is None:
        raise DataError(f"{name} missing from the insider data set")
    with z.open(member) as fh:
        head = pd.read_csv(fh, sep="\t", dtype=str, nrows=0)
    use = [c for c in columns if c in head.columns]
    with z.open(member) as fh:
        df = pd.read_csv(
            io.TextIOWrapper(fh, encoding="utf-8", errors="replace"),
            sep="\t",
            dtype=str,
            usecols=use,
            quoting=3,
        )
    for c in columns:
        if c not in df.columns:
            df[c] = None
    return df


def _date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format="%d-%b-%Y", errors="coerce").dt.date


def _flag(values: pd.Series) -> pd.Series:
    """The 10b5-1 box: True, False, or None where the filing has no such field."""
    v = values.fillna("").str.strip().str.lower()
    return v.map({"1": True, "true": True, "0": False, "false": False}).astype(object)


def parse(zip_path: Path) -> pd.DataFrame:
    """Officer and director open-market purchases in one quarter's data set."""
    with zipfile.ZipFile(zip_path) as z:
        sub = _table(
            z,
            "SUBMISSION",
            [
                "ACCESSION_NUMBER",
                "FILING_DATE",
                "ISSUERCIK",
                "ISSUERTRADINGSYMBOL",
                "DOCUMENT_TYPE",
                "AFF10B5ONE",
            ],
        )
        own = _table(
            z, "REPORTINGOWNER", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"]
        )
        tr = _table(
            z,
            "NONDERIV_TRANS",
            [
                "ACCESSION_NUMBER",
                "TRANS_CODE",
                "TRANS_DATE",
                "TRANS_SHARES",
                "TRANS_PRICEPERSHARE",
                "TRANS_ACQUIRED_DISP_CD",
            ],
        )
    tr = tr.loc[(tr["TRANS_CODE"] == "P") & (tr["TRANS_ACQUIRED_DISP_CD"] == "A")]
    rel = own["RPTOWNER_RELATIONSHIP"].fillna("")
    own = own.loc[rel.str.contains("Officer") | rel.str.contains("Director")]
    # A filing can list several owners; each one bought the reported shares jointly. Keep
    # one owner per filing (the first listed) so a joint filing is not counted twice.
    own = own.drop_duplicates("ACCESSION_NUMBER")
    df = tr.merge(own, on="ACCESSION_NUMBER").merge(sub, on="ACCESSION_NUMBER")
    out = pd.DataFrame(
        {
            "accession": df["ACCESSION_NUMBER"],
            "issuer_cik": pd.to_numeric(df["ISSUERCIK"], errors="coerce").astype("Int64"),
            "symbol": df["ISSUERTRADINGSYMBOL"].fillna("").str.strip().str.upper(),
            "owner_cik": pd.to_numeric(df["RPTOWNERCIK"], errors="coerce").astype("Int64"),
            "relationship": df["RPTOWNER_RELATIONSHIP"],
            "trans_date": _date(df["TRANS_DATE"]),
            "filing_date": _date(df["FILING_DATE"]),
            "shares": pd.to_numeric(df["TRANS_SHARES"], errors="coerce"),
            "price": pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce"),
            "plan_10b5_1": _flag(df["AFF10B5ONE"]),
            "form": df["DOCUMENT_TYPE"],
        }
    )
    out["value"] = out["shares"] * out["price"]
    out = out.dropna(subset=["trans_date", "filing_date", "owner_cik", "issuer_cik"])
    out = out.loc[(out["shares"] > 0) & (out["price"] > 0) & (out["symbol"] != "")]
    return out[list(PURCHASE_COLUMNS)].reset_index(drop=True)


def combine(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """All quarters, with amendments that repeat a purchase counted once (the first filing)."""
    parts = [f for f in frames if len(f)]
    if not parts:
        return pd.DataFrame(columns=list(PURCHASE_COLUMNS))
    df = pd.concat(parts, ignore_index=True).sort_values(["filing_date", "accession"])
    return df.drop_duplicates(
        ["issuer_cik", "owner_cik", "trans_date", "shares", "price"], keep="first"
    ).reset_index(drop=True)


def known_session(filing_date: date) -> date:
    """The first close a purchase may be acted on: the session after its filing date.

    The data sets give the filing date without a time, and Form 4s are filed until
    10 pm, so the filing date's own close is too early.
    """
    return clock.next_session(filing_date)
