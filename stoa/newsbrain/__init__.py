"""The news brain: research data, the FinBERT reader and trials 9 and 10.

Everything here is research. Nothing in this package is imported by DECIDE,
the sweep or EXECUTE except the reader itself (stoa.reader). Data lives in the
git-ignored ``data/newsbrain/`` folder as zstd parquet, written in chunks so
memory stays bounded, and every pull can be stopped and restarted.
"""

from pathlib import Path

ROOT = Path("data/newsbrain")
PARQUET_LEVEL = 6  # zstd
