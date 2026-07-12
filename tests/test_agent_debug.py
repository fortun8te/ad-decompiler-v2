"""Tests for gated agent debug logging."""
import json
import os

from src import agent_debug


def test_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("AD_DEBUG_SESSION", raising=False)
    agent_debug.log("t.py:fn", "msg", run_dir=str(tmp_path))
    assert agent_debug.tail(str(tmp_path)) == []
    assert not list(tmp_path.iterdir())


def test_env_enables_log_and_tail(tmp_path, monkeypatch):
    monkeypatch.setenv("AD_DEBUG_SESSION", "sess-a")
    agent_debug.log("t.py:fn", "hello", data={"n": 1}, run_dir=str(tmp_path))
    entries = agent_debug.tail(str(tmp_path))
    assert len(entries) == 1
    assert entries[0]["sessionId"] == "sess-a"
    assert entries[0]["message"] == "hello"
    assert (tmp_path / "debug-sess-a.jsonl").exists()


def test_cfg_runtime_debug_session(tmp_path, monkeypatch):
    monkeypatch.delenv("AD_DEBUG_SESSION", raising=False)
    cfg = {"runtime": {"debug_session": "cfg-sid"}}
    assert agent_debug.session_id(cfg=cfg) == "cfg-sid"
    agent_debug.log("t.py:fn", "from cfg", cfg=cfg, run_dir=str(tmp_path))
    assert agent_debug.tail(str(tmp_path), cfg=cfg)[0]["sessionId"] == "cfg-sid"


def test_env_overrides_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AD_DEBUG_SESSION", "env-wins")
    cfg = {"runtime": {"debug_session": "cfg-sid"}}
    assert agent_debug.session_id(cfg=cfg) == "env-wins"
