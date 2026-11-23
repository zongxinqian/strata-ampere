#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-$(dirname "$0")/strata-2x3080-20g.example.json}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

if grep -q "CHANGE_ME_API_KEY" "$CONFIG" || grep -q "/path/to/" "$CONFIG"; then
  echo "Please copy $CONFIG and replace placeholder paths and api_key before launching." >&2
  exit 1
fi

exec python3 "$REPO_ROOT/serve/server.py" --config "$CONFIG"
