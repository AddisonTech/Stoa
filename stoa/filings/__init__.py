"""Trial 17: what companies change in their 10-K and 10-Q text ("Lazy Prices").

Research only: nothing here is imported by the live jobs. Data lives in the git-ignored
``data/filings/`` folder: ``raw/`` (each filing's primary document as EDGAR served it,
zstd parquet per filer), ``text/`` (the parsed full text and sections), ``scores.parquet``
(the year-on-year change scores), ``tags/`` (the local model's reads, trial 17b) and
``results/``. Every step is resumable and logs to ``logs/``.
"""

from pathlib import Path

ROOT = Path("data/filings")
RAW = ROOT / "raw"
TEXT = ROOT / "text"
LOGS = ROOT / "logs"
PARQUET_LEVEL = 6  # zstd
