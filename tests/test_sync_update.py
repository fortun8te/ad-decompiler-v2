from scripts import sync_update


class FakeGit:
    def __init__(self, responses):
        self.responses = {tuple(key): list(value) for key, value in responses.items()}
        self.calls = []

    def __call__(self, _repo, *args):
        key = tuple(args)
        self.calls.append(key)
        return self.responses[key].pop(0)


def _base(head="abc1234", dirty=""):
    return {
        ("rev-parse", "--is-inside-work-tree"): [(0, "true", "")],
        ("status", "--porcelain=v1"): [(0, dirty, "")],
        ("rev-parse", "--short", "HEAD"): [(0, head, "")],
    }


def _clean_remote(head="abc1234", remote="def5678", counts="0\t1"):
    responses = _base(head)
    responses.update({
        ("fetch", "--quiet", "origin", "main"): [(0, "", "")],
        ("rev-parse", "--short", "origin/main"): [(0, remote, "")],
        ("rev-list", "--left-right", "--count", "HEAD...origin/main"): [(0, counts, "")],
    })
    return responses


def test_dirty_tree_never_fetches_or_pulls(tmp_path):
    fake = FakeGit(_base(dirty=" M src/inpaint.py\n?? notes.txt"))

    result = sync_update.sync(tmp_path, update=True, runner=fake)

    assert result["action"] == "dirty"
    assert result["dirty"] is True
    assert not any(call[0] in {"fetch", "pull"} for call in fake.calls)


def test_check_reports_remote_revision_without_pulling(tmp_path):
    fake = FakeGit(_clean_remote())

    result = sync_update.sync(tmp_path, runner=fake)

    assert result["action"] == "update_available"
    assert result["local_revision"] == "abc1234"
    assert result["remote_revision"] == "def5678"
    assert not any(call[0] == "pull" for call in fake.calls)


def test_update_fast_forwards_clean_behind_tree_only(tmp_path):
    responses = _clean_remote()
    responses[("pull", "--ff-only", "origin", "main")] = [(0, "Updating", "")]
    responses[("rev-parse", "--short", "HEAD")] = [(0, "abc1234", ""), (0, "def5678", "")]
    fake = FakeGit(responses)

    result = sync_update.sync(tmp_path, update=True, runner=fake)

    assert result["action"] == "updated"
    assert result["local_revision"] == "def5678"
    assert ("pull", "--ff-only", "origin", "main") in fake.calls


def test_update_refuses_diverged_history(tmp_path):
    fake = FakeGit(_clean_remote(counts="1\t1"))

    result = sync_update.sync(tmp_path, update=True, runner=fake)

    assert result["action"] == "diverged"
    assert not any(call[0] == "pull" for call in fake.calls)


def test_windows_notification_is_best_effort_and_non_blocking(monkeypatch):
    calls = []

    monkeypatch.setattr(sync_update.platform, "system", lambda: "Windows")
    monkeypatch.setattr(sync_update.subprocess, "Popen", lambda args, **kwargs: calls.append((args, kwargs)))

    sync_update._notify("updated safely")

    assert calls[0][0] == ["msg.exe", "*", "Ad Decompiler update: updated safely"]
    assert "timeout" not in calls[0][1]
