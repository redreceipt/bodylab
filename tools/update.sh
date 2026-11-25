#!/bin/sh
# Bring this copy up to what's on GitHub's main and restart BodyLab, if anything changed. tools/autostart.sh runs it
# every 5 minutes, so the live app runs only pushed code (edits in a working copy never reach it).
set -e
cd "$(dirname "$0")/.."
export GIT_CONFIG_GLOBAL=/dev/null  # ~/.gitconfig may live where launchd agents may not read (cloud storage); a fetch needs none of it
git fetch -q origin main
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] && exit 0
git checkout -q --detach origin/main
uv sync -q --inexact --group fit --group vision
launchctl kickstart -k "gui/$(id -u)/local.bodylab"
echo "$(date '+%F %T') BodyLab updated to $(git rev-parse --short HEAD)"
