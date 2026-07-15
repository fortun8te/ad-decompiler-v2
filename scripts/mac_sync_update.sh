#!/bin/sh
# launchd-friendly wrapper: no prompts, no stash/reset, and exits after one safe check.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON=${PYTHON:-python3}

exec "$PYTHON" "$ROOT/scripts/sync_update.py" --repo "$ROOT" --update --notify "$@"
