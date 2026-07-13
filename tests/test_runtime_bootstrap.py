from src import runtime_bootstrap


def test_bootstrap_disabled_is_noop():
    assert runtime_bootstrap.ensure_services({}) == {"ok": True, "enabled": False, "checks": []}


def test_bootstrap_starts_enabled_services(monkeypatch):
    monkeypatch.setattr(runtime_bootstrap, "_start_comfy",
                        lambda cfg, timeout: {"name": "comfyui", "ok": True})
    monkeypatch.setattr(runtime_bootstrap, "_start_vlm",
                        lambda cfg, timeout: {"name": "vlm", "ok": True})
    cfg = {
        "runtime": {"autostart": {"enabled": True}},
        "inpaint": {"comfy": {"enabled": True}},
        "vlm": {"enabled": True},
    }
    result = runtime_bootstrap.ensure_services(cfg)
    assert result["ok"] is True
    assert [item["name"] for item in result["checks"]] == ["comfyui", "vlm"]


def test_bootstrap_fails_closed_when_service_does_not_start(monkeypatch):
    monkeypatch.setattr(runtime_bootstrap, "_start_comfy",
                        lambda cfg, timeout: {"name": "comfyui", "ok": False})
    cfg = {"runtime": {"autostart": {"enabled": True}},
           "inpaint": {"comfy": {"enabled": True}}}
    assert runtime_bootstrap.ensure_services(cfg)["ok"] is False
