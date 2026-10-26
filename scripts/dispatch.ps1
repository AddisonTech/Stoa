# Start one Stoa job on GitHub Actions by workflow_dispatch, on time. The trading
# jobs run in the daily workflow; "close" is the 9:40 approvals note (close.yml)
# and "analyst" is trial 7's shadow scoring.
#
# Windows Task Scheduler runs this on the Dell (see docs/RUNBOOK.md, section 13).
# cron-job.org is the primary clock at the same minute; this is the first backup,
# and GitHub cron, which starts late or not at all, the last one a few minutes
# later. Runs started here are named "via dell" so the trigger check can tell.
# A second start for the same job is harmless: runs share one concurrency group,
# and each job checks its window and runs once per session (the analyst skips a
# session it has already scored).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\dispatch.ps1 -Job sweep

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("decide", "sweep", "close", "execute", "analyst")]
    [string]$Job
)

$repo = "AddisonTech/Stoa"
$log = Join-Path $PSScriptRoot "..\data\dispatch.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

function Write-Log([string]$text) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $log -Value "$stamp $Job $text" -Encoding utf8
}

for ($try = 1; $try -le 5; $try++) {
    if ($Job -eq "analyst") {
        $out = & gh workflow run analyst.yml -R $repo -f source=dell 2>&1
    } elseif ($Job -eq "close") {
        $out = & gh workflow run close.yml -R $repo -f source=dell 2>&1
    } else {
        $out = & gh workflow run daily.yml -R $repo -f "job=$Job" -f source=dell 2>&1
    }
    if ($LASTEXITCODE -eq 0) {
        Write-Log "dispatched ($out)"
        exit 0
    }
    Write-Log "try $try failed: $out"
    Start-Sleep -Seconds 15
}
exit 1
