# Verified offline release launcher. Run after extracting the final source/kit.
param([int]$Port = 8765, [string]$RuntimeArchive = '')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'Python 3.11+ is required for the local console. Install it before the offline demonstration.'
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker Desktop is required. Install and start it before the offline demonstration.'
}
Push-Location -LiteralPath $projectRoot
try {
    & python scripts/verify_release.py
    if ($LASTEXITCODE -ne 0) { throw 'Frozen source or official data verification failed.' }
    $release = Get-Content -LiteralPath (Join-Path $projectRoot 'RELEASE.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & docker image inspect $release.image_id *> $null
        $imageAvailable = ($LASTEXITCODE -eq 0)
    } finally { $ErrorActionPreference = $previousPreference }
    if (-not $imageAvailable) {
        if (-not $RuntimeArchive -or -not (Test-Path -LiteralPath $RuntimeArchive -PathType Leaf)) {
            throw 'Start Docker Desktop and load runtime/star-chart-cpu.tar from the offline kit.'
        }
        & docker load --input $RuntimeArchive
        if ($LASTEXITCODE -ne 0) { throw 'Loading the frozen CPU image failed.' }
    }
    & python scripts/verify_release.py --environment
    if ($LASTEXITCODE -ne 0) { throw 'Runtime verification failed.' }
    Write-Host "Open http://127.0.0.1:$Port in your browser. Press Ctrl+C to stop."
    & python -m src.webapp --port $Port --image $release.image_id
    if ($LASTEXITCODE -ne 0) { throw 'Local console exited with an error.' }
} finally { Pop-Location }
