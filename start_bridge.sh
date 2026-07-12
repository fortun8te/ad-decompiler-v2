#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PORT="${BRIDGE_PORT:-8790}"
INBOX="${FIGMA_INBOX:-$HOME/figma-inbox}"
PYTHON="$ROOT/.venv/bin/python"
FULL_SETUP="${FULL_SETUP:-0}"

ensure_venv() {
  if [[ -x "$PYTHON" ]]; then
    return
  fi
  echo ""
  echo "First run — setting up Python environment..."
  if [[ "$FULL_SETUP" == "1" && -f "$ROOT/setup_rtx.ps1" ]]; then
    echo "Run setup_rtx.ps1 on Windows for the full GPU stack."
  fi
  python3 -m venv "$ROOT/.venv"
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install -r requirements.txt
}

bridge_up() {
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1
}

ensure_venv
"$PYTHON" "$ROOT/scripts/stamp_plugin_build.py" --quiet
"$PYTHON" -m src.bridge_bootstrap --config config.yaml --inbox "$INBOX" --port "$PORT"

if bridge_up; then
  echo ""
  echo "================================================"
  echo "  Bridge is already running"
  echo "  http://localhost:${PORT}"
  echo "================================================"
  exit 0
fi

echo ""
echo "================================================"
echo "  Ad Decompiler Bridge"
echo "  http://localhost:${PORT}"
echo "================================================"
echo "Inbox:  $INBOX"
echo "Config: $ROOT/config.yaml"
echo ""
echo "Press Ctrl+C to stop."
echo ""

exec "$PYTHON" -m src.figma_bridge --inbox "$INBOX" --port "$PORT" --config config.yaml --no-bootstrap
