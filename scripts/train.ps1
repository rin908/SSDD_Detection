$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    python tools/train.py @args
    if ($LASTEXITCODE -ne 0) { throw "训练入口失败，退出码 $LASTEXITCODE" }
} finally {
    Pop-Location
}
