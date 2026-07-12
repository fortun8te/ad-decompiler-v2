@echo off
title Ad Decompiler Bridge
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_bridge.ps1" %*
if errorlevel 1 (
  echo.
  echo Something went wrong. See the message above.
  pause
)
