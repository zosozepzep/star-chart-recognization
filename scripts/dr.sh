#!/usr/bin/env bash
# 在项目容器内执行一条命令。用法: bash scripts/dr.sh pytest tests -q
# 覆盖镜像: SC_IMAGE=star-chart:gpu bash scripts/dr.sh python -m src.cli ...
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "Usage: bash scripts/dr.sh python examples/01_load.py" >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not on PATH." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker is unavailable. Start Docker Desktop / the Docker daemon and retry." >&2
  exit 1
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v cygpath >/dev/null 2>&1; then
  host="$(cygpath -w "$here")"
else
  host="$here"
fi

IMAGE="${SC_IMAGE:-star-chart:cpu}"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  if [ -z "${SC_IMAGE:-}" ] && docker image inspect star-chart:cpu-interim >/dev/null 2>&1; then
    IMAGE=star-chart:cpu-interim
    echo "Using existing demo image: $IMAGE" >&2
  else
    echo "Image not found: $IMAGE" >&2
    echo "Build CPU image: docker build -f docker/Dockerfile.cpu -t star-chart:cpu ." >&2
    exit 1
  fi
fi

MSYS_NO_PATHCONV=1 exec docker run --rm \
  -v "${host}:/workspace" \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  -e MPLBACKEND=Agg \
  "${IMAGE}" "$@"
