#!/usr/bin/env python3
"""Pull latest ad-decompiler on the Windows RTX bridge over Tailscale."""
from __future__ import annotations

import argparse
import json
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_BRIDGE = "http://100.74.135.83:8790"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remote git pull on the RTX bridge host")
    parser.add_argument("--bridge", default=DEFAULT_BRIDGE, help="bridge base URL")
    parser.add_argument("--remote", default="newrepo")
    parser.add_argument("--branch", default="main")
    args = parser.parse_args(argv)

    bridge = args.bridge.rstrip("/")
    url = f"{bridge}/repo/update?{urlencode({'remote': args.remote, 'branch': args.branch})}"
    request = Request(url, data=b"", method="POST")
    try:
        with urlopen(request, timeout=300) as response:
            body = response.read().decode("utf-8")
            status = getattr(response, "status", response.getcode())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "If /repo/update is missing, restart Start Bridge.bat on Windows once — "
            "it auto-pulls on start.",
            file=sys.stderr,
        )
        return 1

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        print(body)
        return 1 if status >= 400 else 0

    print(json.dumps(payload, indent=2))
    if not payload.get("ok"):
        return 1
    if payload.get("restart_required"):
        print("\nRestart Start Bridge.bat on Windows to load the new code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
