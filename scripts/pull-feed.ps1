# Download the big paper feed's private Hugging Face dataset (big_paper.feed_repo), the dry
# runs' pulls and checks included, into data\newsbrain\feedcheck\hf, so the Dell can rebuild
# the same sessions from scratch and compare (docs/RUNBOOK.md 13b, dry runs).
#
# Asks for a Hugging Face token (read access is enough), without showing it, and passes it
# only to the download process through its environment. It is not saved to disk, not logged
# in with, and cleared from this session when the download ends.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\pull-feed.ps1

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$secure = Read-Host "Hugging Face token (read access)" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $env:HF_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    & .\.venv\Scripts\python.exe -m stoa.newsbrain feedpull
    $code = $LASTEXITCODE
}
finally {
    Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}
exit $code
