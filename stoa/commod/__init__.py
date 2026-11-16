"""Trial 18: commodity funds around the weekly EIA reports.

Research only: nothing here is imported by the live jobs. Data lives in the git-ignored
``data/commod/`` folder: ``eia/`` (the weekly storage and petroleum reports with their
release times), ``noaa/`` (population-weighted degree days), ``minute/`` (the funds' minute
bars after the compacted store ends) and ``results/``.
"""

from pathlib import Path

ROOT = Path("data/commod")
LOGS = ROOT / "logs"
