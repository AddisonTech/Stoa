# Trial 17: where I left off (paused 2026-11-19)

Paused by the owner on the Dell to free memory. Every job stopped at a point it resumes from with nothing lost: the pull skips filers already in `data/filings/logs/pull.jsonl` and every raw, text and score file is written whole (no partial files were left).

## Where each job stopped

| Job | State |
|---|---|
| EDGAR pull | 1,658 of 6,689 filers (1,658 of the 5,297 ever-liquid filers 17a needs, which go first); 58,074 filings on disk |
| Parse | 56,663 filings parsed (1,404 filers) |
| Scores | 1,299 filers scored; the rest are picked up on the next cycle |
| Ticker map | not built yet: it runs once the liquid filers are all pulled |
| Daily bars | complete through 2026-11-13; nothing to resume |

## Trials

- **17a**: pre-registered 2026-11-19 (spec `70afd079acf8b56b`, stoa-state `8041182`), not run. It runs once when the supervisor logs `liquid_ready`.
- **17b**: not pre-registered; follows 17a (`trial todo`, `tag`, `preregister b`, `trial b`).
- **20** (trial 18's rules): graded once, 20a and 20b FAIL (DECISIONS.md, "Trial 20"). Trials 18 and 19 are void.
- The owner's decision that trials 17 and 20 read the spent holdout months is in DECISIONS.md.

## Resume

From PowerShell:

```
cd C:\Users\AddisonSmith\Stoa; Start-Process .venv\Scripts\python.exe "scripts\filings_supervise.py" -WindowStyle Hidden -RedirectStandardOutput data\filings\logs\supervise.out -RedirectStandardError data\filings\logs\supervise.err
```

`scripts/filings_supervise.py` restarts the pull and runs parse and scores every 10 minutes (1 to 4 workers by free memory). It keeps free memory above 6.2 GB by stopping parse or scores first, then the pull, and restarts them above 7.5 GB. Once every ever-liquid filer is pulled it stops the pull, runs a last parse and scores, builds the ticker map and logs `liquid_ready` in `data/filings/logs/supervise.jsonl`. Then:

```
.venv\Scripts\python.exe -m stoa.filings trial a
```

At the pause the pull was fetching about 4 to 10 filers a minute, so the liquid filers need roughly 6 to 12 more hours of running.
