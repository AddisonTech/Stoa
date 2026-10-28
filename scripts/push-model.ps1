# Upload a retrained news brain model (data\newsbrain\models\<version>) and its backfilled
# scoreboard days (data\newsbrain\scoreboard-out\scoreboard\backfill) to the big paper feed's
# private Hugging Face dataset, big_paper.feed_repo (ASmith91/stoa-feed): models/<version>/ and
# scoreboard/backfill/. Nothing else in the dataset is touched.
#
# Asks for a Hugging Face token with write access, without showing it, and passes it only to
# the upload process through its environment. It is not saved to disk, not logged in with,
# and cleared from this session when the upload ends.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\push-model.ps1 t12-all-20261027

param([Parameter(Mandatory = $true)][string]$Version)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$secure = Read-Host "Hugging Face token (write access)" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $env:HF_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    & .\.venv\Scripts\python.exe -m stoa.newsbrain modelpush $Version
    $code = $LASTEXITCODE
}
finally {
    Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}
exit $code
