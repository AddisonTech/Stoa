"""``python -m stoa.filings <step>``: the trial 17 pipeline, each step resumable.

Steps: ``pull`` (EDGAR primary documents), ``parse`` (text and sections), ``scores``
(year-on-year change scores), ``tickers`` (filing to symbol, point in time), ``tag``
(the local model reads what changed, trial 17b), ``preregister`` and ``trial``.
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    step, rest = argv[0], argv[1:]
    if step == "pull":
        from stoa.filings import pull

        pull.main(int(rest[0]) if rest else None)
    elif step == "parse":
        from stoa.filings import parse

        parse.main(int(rest[0]) if rest else 12)
    elif step == "scores":
        from stoa.filings import scores

        scores.main(int(rest[0]) if rest else 12)
    elif step == "tickers":
        from stoa.filings import tickers

        tickers.main()
    elif step == "tag":
        from stoa.filings import tag

        tag.main(rest)
    elif step in ("preregister", "trial"):
        from stoa.filings import trial17

        return trial17.main(step, rest)
    else:
        print(f"unknown step {step}\n{__doc__}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
