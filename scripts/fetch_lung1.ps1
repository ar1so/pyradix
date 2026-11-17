# Fetch the Lung1 cohort helper. No bash required.
$ErrorActionPreference = "Stop"
python (Join-Path $PSScriptRoot "fetch_lung1.py") @args
exit $LASTEXITCODE
