"""Trial 16: the daily contract announcements (docs/DECISIONS.md, "Trial 16").

The Department of War (defense.gov until 2025, war.gov since) posts every contract
of $7.5 million or more each business day at 5 pm New York time, after the close.
This package pulls that archive, reads each award into a row with fixed rules,
maps the awardee to a ticker as of the announcement date, and grades the idea
that a big award to a small public company moves its stock the next day.

Raw pages and every parsed table stay in the git-ignored ``data/contracts/``.
No hosted language model ever reads a historical page: it would know what
happened after.
"""

from pathlib import Path

ROOT = Path("data/contracts")
