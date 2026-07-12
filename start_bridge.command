#!/bin/bash
cd "$(dirname "$0")"
chmod +x start_bridge.sh 2>/dev/null || true
exec ./start_bridge.sh
