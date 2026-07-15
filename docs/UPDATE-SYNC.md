# Safe cross-platform updates

The updater compares `HEAD` with a fetched remote branch and only runs `git pull --ff-only` when
the checkout is clean and simply behind. It never stashes, resets, merges, or pulls a dirty or
diverged tree. The terminal says `CURRENT`, `UPDATE_AVAILABLE`, `UPDATED`, `DIRTY`, or `DIVERGED`
so the reason for a skipped update is visible.

For automatic, safe sync at login and then hourly, install once on each machine:

```bash
# macOS
./scripts/install_macos_sync_launchd.sh
```

```powershell
# Windows RTX PC
.\scripts\install_windows_sync_task.ps1
```

These schedulers pull only a clean checkout. If you are editing locally, they leave it alone and
notify you instead of overwriting your work.

## Windows

For a one-off update from the normal launcher:

```powershell
Start Bridge.bat -Update
```

It uses `scripts\sync_update.py` and then continues with the current checkout if an update is
unsafe or unavailable. If a clean fast-forward succeeds, it refreshes the normal RTX setup before
starting the bridge, so new dependency or model-adapter code is not run in an old environment. It
prints terminal status and, in an interactive Windows session, makes a best-effort native `msg.exe`
notification; unavailable/non-interactive notifications are ignored.

For a direct check/update on the RTX machine:

```powershell
.\scripts\windows_sync.ps1                 # safe update
.\scripts\windows_sync.ps1 -CheckOnly      # report only
```

To use Windows Task Scheduler, create an opt-in task with this program/argument pair (for
example, hourly while the workstation is awake):

```text
Program: powershell.exe
Arguments: -NoProfile -ExecutionPolicy Bypass -File C:\src\ad-decompiler-v2\scripts\windows_sync.ps1
```

Use the correct checkout path. A scheduled task logs its result through the normal PowerShell
task history; it will not overwrite a dirty tree.

## macOS / launchd

From a terminal:

```bash
./scripts/mac_sync_update.sh
```

The macOS wrapper gives a desktop notification when possible and always prints terminal status.
It is non-interactive, so it is suitable for a user LaunchAgent. Example plist (replace both
paths and choose the Python that has your dependencies):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.fortun8te.ad-decompiler.update</string>
  <key>ProgramArguments</key><array>
    <string>/bin/sh</string>
    <string>/Users/you/Downloads/ad-decompiler-v2/scripts/mac_sync_update.sh</string>
  </array>
  <key>StartInterval</key><integer>3600</integer>
  <key>EnvironmentVariables</key><dict>
    <key>PYTHON</key><string>/usr/local/bin/python3</string>
  </dict>
  <key>StandardOutPath</key><string>/tmp/ad-decompiler-update.log</string>
  <key>StandardErrorPath</key><string>/tmp/ad-decompiler-update.err</string>
</dict></plist>
```

Save it as `~/Library/LaunchAgents/com.fortun8te.ad-decompiler.update.plist`, then load it with:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fortun8te.ad-decompiler.update.plist
```

## Direct use

The common command is available on either OS:

```bash
python3 scripts/sync_update.py
python3 scripts/sync_update.py --update --remote origin --branch main
```

There is no `--check` flag because checking is the default; omit `--update` to report only.
Use `--json` for an automation-friendly status payload.
