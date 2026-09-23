# PowerShell equivalent of dr.sh. Example:
# .\scripts\dr.ps1 python examples/01_load.py
$ErrorActionPreference = 'Stop'

if ($args.Count -eq 0) {
    Write-Host 'Usage: .\scripts\dr.ps1 python examples/01_load.py'
    exit 2
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker is not installed or not on PATH.'
}
# Redirect native stderr without turning an expected image miss into a PS error.
function Test-DockerCommand {
    param([string[]]$DockerArgs)
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & docker @DockerArgs *> $null
        return ($LASTEXITCODE -eq 0)
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}
if (-not (Test-DockerCommand @('info'))) {
    throw 'Docker is unavailable. Start Docker Desktop / the Docker daemon and retry.'
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$imageName = if ($env:SC_IMAGE) { $env:SC_IMAGE } else { 'star-chart:cpu' }
if (-not (Test-DockerCommand @('image', 'inspect', $imageName))) {
    if (-not $env:SC_IMAGE -and (Test-DockerCommand @('image', 'inspect', 'star-chart:cpu-interim'))) {
        $imageName = 'star-chart:cpu-interim'
        Write-Host "Using existing demo image: $imageName"
    } else {
        throw "Image not found: $imageName. Build with: docker build -f docker/Dockerfile.cpu -t star-chart:cpu ."
    }
}
& docker run --rm -v "${projectRoot}:/workspace" -w /workspace `
    -e PYTHONPATH=/workspace -e MPLBACKEND=Agg $imageName @args
exit $LASTEXITCODE
