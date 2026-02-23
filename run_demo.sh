#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-web}"
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-1}"

if [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
fi

deps_ok() {
python3 - <<'PY' >/dev/null 2>&1
import flask  # noqa: F401
import langgraph  # noqa: F401
import langchain_anthropic  # noqa: F401
PY
}

install_deps() {
  echo "[RUNNER] Installing dependencies from requirements.txt"
  python3 -m pip install -r requirements.txt
}

if [[ "$MODE" == "install" ]]; then
  install_deps
  exit 0
fi

if ! deps_ok; then
  if [[ "$AUTO_INSTALL_DEPS" == "1" ]]; then
    install_deps
  else
    echo "[RUNNER] Missing dependencies."
    echo "[RUNNER] Run: pip install -r requirements.txt"
    exit 1
  fi
fi

case "$MODE" in
  web)
    echo "[RUNNER] Starting web UI at http://127.0.0.1:8000"
    python3 web_app.py
    ;;
  local)
    echo "[RUNNER] Running Local LangGraph mode"
    python3 local_agent.py
    ;;
  sandbox)
    echo "[RUNNER] Running Sandbox LangGraph mode"
    python3 sandbox_simulator.py
    ;;
  traditional)
    echo "[RUNNER] Running Traditional mode"
    python3 traditional_demo.py
    ;;
  *)
    echo "Usage: bash run_demo.sh [web|local|sandbox|traditional|install]"
    exit 1
    ;;
esac
