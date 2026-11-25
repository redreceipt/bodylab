#!/bin/sh
# Start BodyLab at login and keep it running (macOS LaunchAgent), from this copy of the code; and every 5 minutes bring
# it up to what's pushed to GitHub (tools/update.sh). Run it from a copy that's only for running, e.g. a worktree:
#   git worktree add --detach ../bodylab-live origin/main && ln -s "$PWD/data" "$PWD/models" ../bodylab-live/
# Usage: tools/autostart.sh [off]
set -e
label=local.bodylab
plist="$HOME/Library/LaunchAgents/$label.plist"
updater="$HOME/Library/LaunchAgents/$label.update.plist"
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
launchctl bootout "gui/$(id -u)/$label.update" 2>/dev/null || true
if [ "$1" = off ]; then
  rm -f "$plist" "$updater"
  echo "BodyLab will no longer start at login"
  exit
fi
dir=$(cd "$(dirname "$0")/.." && pwd)
uv=$(command -v uv)
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>$uv</string><string>run</string><string>app.py</string></array>
  <key>WorkingDirectory</key><string>$dir</string>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>$PATH</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/BodyLab.log</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/BodyLab.log</string>
</dict>
</plist>
EOF
cat > "$updater" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$label.update</string>
  <key>ProgramArguments</key><array><string>/bin/sh</string><string>$dir/tools/update.sh</string></array>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>$PATH</string></dict>
  <key>StartInterval</key><integer>300</integer>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/BodyLab.log</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/BodyLab.log</string>
</dict>
</plist>
EOF
launchctl bootstrap "gui/$(id -u)" "$plist"
launchctl bootstrap "gui/$(id -u)" "$updater"
echo "BodyLab starts at login from $dir and follows GitHub's main (log: ~/Library/Logs/BodyLab.log). Turn off: tools/autostart.sh off"
