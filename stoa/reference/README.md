# Vendored reference data

| File | What | Source | License |
| --- | --- | --- | --- |
| `sp500_membership.csv` | S&P 500 membership intervals (symbol, first day in, first day out), intervals open on or after 2015 | Built by `scripts/build_reference.py` from `S&P 500 Historical Components & Changes (Updated).csv` in [fja05680/sp500](https://github.com/fja05680/sp500), commit a2430f2 (2026-09-07), covering changes through 2026-08-18 | MIT, see `LICENSE-sp500.txt` |
| `ticker_aliases.csv` | Membership symbols whose history Alpaca and the SEC file under the company's current ticker (BK to BNY, FI to FISV, JEC to J, WLTW to WTW) | Checked against Alpaca daily bars and company announcements | Stoa's own |
| `sectors.csv` | GICS sector of each current member, plus SPY, QQQ and MTUM | `sp500.csv` in the same repository (scraped from Wikipedia by its author) | MIT |
| `fomc.csv` | FOMC policy statement days 2016 to 2027 | federalreserve.gov press releases (2016 to 2020) and the FOMC meeting calendar page (2021 to 2027), URL per row | US government work, public domain |
| `cpi.csv` | CPI release days 2016 to 2027 | bls.gov CPI release archive (2016 to 2025) and the BLS release schedules for 2026 and 2027, URL per row | US government work, public domain |

Known limits, in plain words:

- The membership file is a community dataset. Its author reconciles it against Wikipedia and news by hand. It can have a wrong day here and there. It is far better than using today's members for the past, which is the bias it exists to remove.
- Sectors are today's GICS sectors, applied to the past. Companies that left the index before the snapshot have no sector and count as `Unknown`, which the sector cap treats as its own bucket.
- A symbol that was renamed shows up as two symbols in the membership file (the old one leaves, the new one joins). Alpaca keeps the whole history under the current ticker only, so `ticker_aliases.csv` fetches the old symbol's bars and filings under the new ticker. Each symbol still counts only on its own membership days. A company that merged into a different one (CCE into CCEP in 2016) stays delisted.
- 2027 FOMC dates are tentative until the Fed confirms each meeting. Run `python scripts/build_reference.py` and update these files at least once a quarter.
