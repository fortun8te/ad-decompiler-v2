import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import remote_benchmark as rb  # noqa: E402


class FakeHttp:
    def __init__(self, handlers):
        self.handlers = handlers
        self.calls: list[tuple[str, str]] = []

    def get_json(self, url: str):
        self.calls.append(("GET", url))
        handler = self.handlers.get(("GET", url))
        if handler is None:
            for (method, prefix), fn in self.handlers.items():
                if method == "GET" and url.startswith(prefix):
                    return fn(url)
            raise AssertionError(f"unexpected GET {url}")
        return handler(url)

    def post_bytes(self, url: str, data: bytes):
        self.calls.append(("POST", url))
        handler = self.handlers.get(("POST", url))
        if handler is None:
            for (method, prefix), fn in self.handlers.items():
                if method == "POST" and url.startswith(prefix):
                    return fn(url, data)
            raise AssertionError(f"unexpected POST {url}")
        return handler(url, data)


def test_discover_images_filters_supported_extensions(tmp_path):
    (tmp_path / "a.png").write_bytes(b"png")
    (tmp_path / "b.JPG").write_bytes(b"jpg")
    (tmp_path / "c.webp").write_bytes(b"webp")
    (tmp_path / "skip.gif").write_bytes(b"gif")
    (tmp_path / "note.txt").write_text("nope", encoding="utf-8")

    found = rb.discover_images(tmp_path)
    assert [p.name for p in found] == ["a.png", "b.JPG", "c.webp"]


def test_generate_synthetic_creates_five_pngs(tmp_path):
    paths = rb.generate_synthetic_images(tmp_path)
    assert len(paths) == 5
    assert all(path.suffix == ".png" and path.exists() for path in paths)
    names = {path.name for path in paths}
    assert names == {
        "synthetic_text.png",
        "synthetic_button.png",
        "synthetic_icon.png",
        "synthetic_gradient.png",
        "synthetic_photo_composite.png",
    }


def test_submit_process_retries_on_409(tmp_path):
    image = tmp_path / "ad.png"
    image.write_bytes(b"\x89PNG")
    bridge = "http://bridge.test"
    post_url = f"{bridge}/process?filename=ad.png"
    attempts = {"count": 0}

    def post_handler(url, data):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return 409, {"ok": False, "error": "busy"}
        return 202, {"ok": True, "job_id": "job1", "status": "queued"}

    http = FakeHttp({("POST", post_url): post_handler})
    payload = rb.submit_process(bridge, image, http=http, busy_retry_s=0)
    assert payload["job_id"] == "job1"
    assert attempts["count"] == 2


def test_poll_process_waits_until_terminal_state():
    bridge = "http://bridge.test"
    poll_prefix = f"{bridge}/process?"
    states = iter(
        [
            (200, {"ok": True, "status": "running", "job_id": "job1"}),
            (200, {"ok": True, "status": "running", "job_id": "job1", "stage": "ocr"}),
            (
                200,
                {
                    "ok": True,
                    "status": "done",
                    "job_id": "job1",
                    "duration_s": 12.5,
                    "final_qa_ok": True,
                    "harness_rounds": 1,
                    "harness_stopped": "qa_ok",
                    "staged": True,
                },
            ),
        ]
    )

    def poll_handler(url):
        return next(states)

    http = FakeHttp({("GET", poll_prefix): poll_handler})
    status = rb.poll_process(
        bridge,
        "job1",
        http=http,
        poll_interval_s=0,
        timeout_s=5,
    )
    assert status["status"] == "done"
    assert status["harness_rounds"] == 1


def test_benchmark_image_collects_qa_harness_and_duration(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"\x89PNG")
    bridge = "http://bridge.test"
    post_url = f"{bridge}/process?filename=sample.png"
    poll_url = f"{bridge}/process?job_id=abc123"
    summary_url = f"{bridge}/run-summary?job_id=abc123"

    http = FakeHttp(
        {
            ("POST", post_url): lambda url, data: (
                202,
                {"ok": True, "job_id": "abc123", "status": "queued"},
            ),
            ("GET", poll_url): lambda url: (
                200,
                {
                    "ok": True,
                    "status": "done",
                    "job_id": "abc123",
                    "result": {"duration_s": 9.25, "qa_ok": False},
                    "final_qa_ok": False,
                    "harness_rounds": 2,
                    "harness_stopped": "max_rounds",
                    "staged": True,
                    "doc_id": "sample-doc",
                },
            ),
            ("GET", summary_url): lambda url: (
                200,
                {
                    "ok": True,
                    "job_id": "abc123",
                    "status": "done",
                    "staged": True,
                    "doc_id": "sample-doc",
                    "final_qa_ok": True,
                    "harness_rounds": 2,
                    "harness_stopped": "qa_ok",
                    "manifest": {"doc_id": "sample-doc"},
                },
            ),
        }
    )

    row = rb.benchmark_image(bridge, image, http=http, poll_interval_s=0, busy_retry_s=0)
    assert row["job_id"] == "abc123"
    assert row["status"] == "done"
    assert row["qa_ok"] is True  # run-summary wins
    assert row["harness_rounds"] == 2
    assert row["harness_stopped"] == "qa_ok"
    assert row["duration_s"] == 9.25
    assert row["staged"] is True


def test_run_benchmark_writes_json_and_markdown(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "one.png").write_bytes(b"\x89PNG")
    output_dir = tmp_path / "out"
    bridge = "http://bridge.test"

    def make_http_for(filename: str):
        post_url = f"{bridge}/process?filename={filename}"
        poll_url = f"{bridge}/process?job_id=job-{filename}"
        summary_url = f"{bridge}/run-summary?job_id=job-{filename}"
        return FakeHttp(
            {
                ("POST", post_url): lambda url, data, job=f"job-{filename}": (
                    202,
                    {"ok": True, "job_id": job, "status": "queued"},
                ),
                ("GET", poll_url): lambda url, job=f"job-{filename}": (
                    200,
                    {
                        "ok": True,
                        "status": "done",
                        "job_id": job,
                        "duration_s": 4.0,
                        "final_qa_ok": True,
                        "harness_rounds": 0,
                        "harness_stopped": "already_ok",
                        "staged": True,
                    },
                ),
                ("GET", summary_url): lambda url, job=f"job-{filename}": (
                    200,
                    {
                        "ok": True,
                        "job_id": job,
                        "status": "done",
                        "staged": True,
                        "final_qa_ok": True,
                        "harness_rounds": 0,
                        "harness_stopped": "already_ok",
                    },
                ),
            }
        )

    class RotatingHttp:
        def __init__(self):
            self.inner = make_http_for("one.png")

        def get_json(self, url):
            return self.inner.get_json(url)

        def post_bytes(self, url, data):
            return self.inner.post_bytes(url, data)

    report = rb.run_benchmark(
        bridge=bridge,
        input_dir=input_dir,
        output_dir=output_dir,
        http=RotatingHttp(),
        poll_interval_s=0,
        busy_retry_s=0,
    )
    assert report["summary"]["images"] == 1
    assert report["summary"]["done"] == 1
    json_path = output_dir / "benchmark.json"
    md_path = output_dir / "benchmark.md"
    assert json_path.exists() and md_path.exists()
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["mode"] == "remote_bridge"
    assert saved["runs"][0]["filename"] == "one.png"
    assert "Remote bridge benchmark" in md_path.read_text(encoding="utf-8")


def test_run_benchmark_generate_synthetic_when_empty(tmp_path, monkeypatch):
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    synthetic_dir = tmp_path / "synthetic"
    synthetic_dir.mkdir()

    class NoopHttp:
        def post_bytes(self, url, data):
            return 202, {"ok": True, "job_id": "x", "status": "queued"}

        def get_json(self, url):
            if "/run-summary" in url:
                return 200, {"ok": True, "final_qa_ok": True, "harness_rounds": 0, "harness_stopped": "already_ok"}
            return 200, {"ok": True, "status": "done", "duration_s": 1.0, "final_qa_ok": True, "harness_rounds": 0}

    # Only exercise the synthetic path + one successful image without running all five uploads.
    monkeypatch.setattr(rb, "discover_images", lambda path: [])
    monkeypatch.setattr(
        rb,
        "generate_synthetic_images",
        lambda out: [synthetic_dir / "only.png"],
    )
    (synthetic_dir / "only.png").write_bytes(b"\x89PNG")

    report = rb.run_benchmark(
        bridge="http://bridge.test",
        input_dir=input_dir,
        output_dir=output_dir,
        generate_synthetic=True,
        http=NoopHttp(),
        poll_interval_s=0,
        busy_retry_s=0,
    )
    assert report["synthetic"] is True
    assert report["summary"]["images"] == 1


def test_run_benchmark_exits_when_no_images_and_no_synthetic(tmp_path):
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    with pytest.raises(SystemExit):
        rb.run_benchmark(
            bridge="http://bridge.test",
            input_dir=input_dir,
            output_dir=tmp_path / "out",
            generate_synthetic=False,
            http=FakeHttp({}),
        )
