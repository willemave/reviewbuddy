#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'USAGE'
Usage:
  scripts/startup.sh "your research query"
  scripts/startup.sh run "your research query"
  scripts/startup.sh interactive [options]
  scripts/startup.sh resume <run_id> [options]

Examples:
  scripts/startup.sh "best dishwasher for quiet apartment"
  scripts/startup.sh interactive --max-urls 50
  scripts/startup.sh resume run_20250101_123456

Notes:
  - Pass-through CLI flags are supported.
  - Set REVIEWBUDDY_SKIP_SYNC=true to skip uv sync.
  - Set REVIEWBUDDY_SKIP_PLAYWRIGHT_INSTALL=true to skip playwright install.
USAGE
}

if [ "$#" -eq 0 ]; then
  usage
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [ -f ".env.example" ] && [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

has_max_urls=false
has_max_agents=false
has_headful_flag=false
has_help_flag=false

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      has_help_flag=true
      ;;
    --max-urls|--max-urls=*)
      has_max_urls=true
      ;;
    --max-agents|--max-agents=*)
      has_max_agents=true
      ;;
    --headful|--no-headful|--headful=*|--no-headful=*)
      has_headful_flag=true
      ;;
  esac
done

args=("$@")
first_arg="${1:-}"
if [ -n "$first_arg" ]; then
  if [[ "$first_arg" != -* ]] && [[ "$first_arg" != "run" ]] && \
     [[ "$first_arg" != "interactive" ]] && [[ "$first_arg" != "resume" ]]; then
    args=("run" "$@")
  fi
fi

if [ "${REVIEWBUDDY_SKIP_SYNC:-false}" != "true" ]; then
  uv sync
fi
if [ "${REVIEWBUDDY_SKIP_PLAYWRIGHT_INSTALL:-false}" != "true" ]; then
  uv run playwright install
fi

if [ "$has_help_flag" = false ]; then
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "OPENAI_API_KEY is not set. Add it to .env or export it before running." >&2
    exit 1
  fi

  if [ -z "${EXA_API_KEY:-}" ]; then
    echo "EXA_API_KEY is not set. Add it to .env or export it before running." >&2
    exit 1
  fi

  if [ "$has_max_urls" = false ]; then
    args+=(--max-urls "${REVIEWBUDDY_MAX_URLS:-100}")
  fi
  if [ "$has_max_agents" = false ]; then
    args+=(--max-agents "${REVIEWBUDDY_MAX_AGENTS:-3}")
  fi
  if [ "$has_headful_flag" = false ]; then
    if [ "${REVIEWBUDDY_HEADFUL:-true}" = "true" ]; then
      args+=(--headful)
    else
      args+=(--no-headful)
    fi
  fi
fi

uv run reviewbuddy "${args[@]}"
