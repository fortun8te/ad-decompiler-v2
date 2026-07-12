import io
import sys

import src.console_io as console_io


def test_safe_print_handles_unicode_on_cp1252(monkeypatch):
    buf = io.BytesIO()
    wrapper = io.TextIOWrapper(buf, encoding="cp1252", errors="strict", line_buffering=True)
    monkeypatch.setattr(sys, "stdout", wrapper)
    console_io._CONFIGURED = False

    console_io.safe_print("normalize → 1080x1080", flush=True)

    raw = buf.getvalue()
    assert b"normalize" in raw
    assert raw  # did not raise


def test_configure_stdio_is_idempotent():
    console_io._CONFIGURED = False
    console_io.configure_stdio()
    console_io.configure_stdio()
    assert console_io._CONFIGURED
