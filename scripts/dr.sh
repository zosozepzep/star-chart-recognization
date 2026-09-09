#!/usr/bin/env bash
# 在项目容器内执行一条命令。用法: bash scripts/dr.sh pytest tests -q
# 覆盖镜像: SC_IMAGE=star-chart:gpu bash scripts/dr.sh python -m src.cli ...
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v cygpath >/dev/null 2>&1; then
  host="$(cygpath -w "$here")"
else
  host="$here"
fi

IMAGE="${SC_IMAGE:-star-chart:cpu}"

MSYS_NO_PATHCONV=1 exec docker run --rm \
  -v "${host}:/workspace" \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  -e MPLBACKEND=Agg \
  "${IMAGE}" "$@"
