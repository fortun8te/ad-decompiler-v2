"""Safe console output when stdout uses a limited encoding (e.g. Windows cp1252)."""
from __future__ import annotations

import sys
from typing import Any, TextIO

_CONFIGURED = False


def configure_stdio() -> None:
    """Prefer UTF-8 for stdout/stderr; no-op if already configured."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def safe_print(
    *args: Any,
    sep: str = " ",
    end: str = "\n",
    flush: bool = False,
    file: TextIO | None = None,
) -> None:
    """Print text without crashing on encodings that lack Unicode arrows, etc."""
    configure_stdio()
    out = file or sys.stdout
    text = sep.join(str(a) for a in args) + end
    try:
        out.write(text)
    except UnicodeEncodeError:
        encoding = getattr(out, "encoding", None) or "utf-8"
        if hasattr(out, "buffer"):
            out.buffer.write(text.encode(encoding, errors="replace"))
        else:
            out.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))
    if flush:
        out.flush()
