#!/bin/sh
# Install a small per-user updater. It fast-forwards clean checkouts only.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
LABEL=com.fortun8te.addecompiler.sync
TARGET="$HOME/Library/LaunchAgents/$LABEL.plist"
TEMPLATE="$ROOT/scripts/$LABEL.plist.template"

mkdir -p "$HOME/Library/LaunchAgents"
sed "s|__REPO_ROOT__|$ROOT|g" "$TEMPLATE" > "$TARGET"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$TARGET"
launchctl kickstart -k "gui/$(id -u)/$LABEL"
printf 'Installed %s (on login and hourly). Clean checkouts update automatically; dirty ones are left untouched.\n' "$LABEL"
