# Upload trial 15's learning base (data\newsbrain\trial15\learn_base.parquet, built on the Dell
# with `python -m stoa.newsbrain learnbase 2026-10-28`) to the big paper feed's private Hugging
# Face dataset, big_paper.feed_repo (ASmith91/stoa-feed), as learn/base.parquet. The nightly
# learner waits for it. Nothing else in the dataset is touched.
#
# Asks for a Hugging Face token with write access, without showing it, and passes it only to
# the upload process through its environment. It is not saved to disk, not logged in with,
# and cleared from this session when the upload ends.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\push-learn-base.ps1

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$secure = Read-Host "Hugging Face token (write access)" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $env:HF_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    & .\.venv\Scripts\python.exe -m stoa.newsbrain learnpush
    $code = $LASTEXITCODE
}
finally {
    Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}
exit $code
