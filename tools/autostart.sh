#!/bin/sh
# Start BodyLab at login and keep it running (macOS LaunchAgent). Usage: tools/autostart.sh [off]
set -e
label=local.bodylab
plist="$HOME/Library/LaunchAgents/$label.plist"
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
if [ "$1" = off ]; then
  rm -f "$plist"
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
launchctl bootstrap "gui/$(id -u)" "$plist"
echo "BodyLab starts at login (log: ~/Library/Logs/BodyLab.log). Turn off: tools/autostart.sh off"
