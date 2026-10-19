# Run from any directory. The web server uses stdlib; workers use the CPU image.
param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    & python -m src.webapp --port $Port
} finally {
    Pop-Location
}
