# Download the public fixtures used by the test suite. No bash required.
$ErrorActionPreference = "Stop"
python (Join-Path $PSScriptRoot "fetch_validation_data.py") @args
exit $LASTEXITCODE
