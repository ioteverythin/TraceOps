"""Tests for trace_ops.github — PRFetcher, PRDiff, PRFile."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from trace_ops.github import PRDiff, PRFetcher, PRFile


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _pr_api_response(
    pr_number: int = 42,
    title: str = "Fix widget rendering",
    body: str = "Closes #41 — Widget was not rendering correctly on Safari.",
    merged_at: str = "2024-01-15T10:30:00Z",
    author: str = "alice",
) -> dict:
    """Minimal GitHub PR API payload."""
    return {
        "number": pr_number,
        "title": title,
        "body": body,
        "merged_at": merged_at,
        "html_url": f"https://github.com/owner/repo/pull/{pr_number}",
        "user": {"login": author},
        "state": "closed",
    }


def _files_api_response(files: list[dict] | None = None) -> list[dict]:
    if files is None:
        files = [
            {
                "filename": "src/widget.py",
                "additions": 25,
                "deletions": 10,
                "status": "modified",
                "patch": "@@ -1,10 +1,25 @@\n-old line\n+new line",
            },
            {
                "filename": "tests/test_widget.py",
                "additions": 15,
                "deletions": 0,
                "status": "added",
                "patch": "@@ -0,0 +1,15 @@\n+def test_widget(): ...",
            },
        ]
    return files


def _prs_list_response(count: int = 5) -> list[dict]:
    return [
        {
            "number": i,
            "title": f"PR {i}",
            "body": "Fixes an issue.",
            "merged_at": "2024-01-10T12:00:00Z",
            "html_url": f"https://github.com/owner/repo/pull/{i}",
            "user": {"login": "dev"},
            "state": "closed",
        }
        for i in range(1, count + 1)
    ]


# ── PRFile tests ───────────────────────────────────────────────────────────────


class TestPRFile:
    def test_creation(self):
        f = PRFile(
            filename="src/widget.py",
            additions=25,
            deletions=10,
            patch="@@ -1,10 +1,25 @@\n-old\n+new",
            status="modified",
        )
        assert f.filename == "src/widget.py"
        assert f.additions == 25
        assert f.deletions == 10
        assert f.status == "modified"

    def test_to_dict(self):
        f = PRFile(
            filename="src/foo.py",
            additions=5,
            deletions=2,
            patch="@@",
            status="modified",
        )
        d = f.to_dict()
        for key in ["filename", "additions", "deletions", "status"]:
            assert key in d
        assert d["filename"] == "src/foo.py"

    def test_patch_optional(self):
        f = PRFile(filename="a.py", additions=0, deletions=0, patch="", status="unchanged")
        assert f.patch == ""

    def test_net_change(self):
        f = PRFile(filename="a.py", additions=30, deletions=5, patch="", status="modified")
        assert f.additions - f.deletions == 25


# ── PRDiff tests ───────────────────────────────────────────────────────────────


class TestPRDiff:
    def _sample_pr(self) -> PRDiff:
        files = [
            PRFile("src/widget.py", 25, 10, "@@ -1,10 +1,25 @@\n-old\n+new", "modified"),
            PRFile("tests/test_widget.py", 15, 0, "@@ -0,0 +1,15 @@\n+test", "added"),
        ]
        return PRDiff(
            url="https://github.com/owner/repo/pull/42",
            pr_number=42,
            title="Fix widget rendering",
            body="Closes #41 — Widget was not rendering correctly on Safari.",
            files=files,
            merged_at="2024-01-15T10:30:00Z",
            author="alice",
        )

    def test_total_additions(self):
        pr = self._sample_pr()
        assert pr.total_additions == 40

    def test_total_deletions(self):
        pr = self._sample_pr()
        assert pr.total_deletions == 10

    def test_diff_text_combines_patches(self):
        pr = self._sample_pr()
        text = pr.diff_text
        assert "src/widget.py" in text
        assert "tests/test_widget.py" in text
        assert "-old" in text
        assert "+new" in text

    def test_extract_task_prompt_returns_string(self):
        pr = self._sample_pr()
        prompt = pr.extract_task_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 10

    def test_extract_task_prompt_uses_title(self):
        pr = self._sample_pr()
        prompt = pr.extract_task_prompt()
        # Should reference the PR title or body content
        assert "widget" in prompt.lower() or "rendering" in prompt.lower() or "Fix" in prompt

    def test_to_dict_keys(self):
        pr = self._sample_pr()
        d = pr.to_dict()
        for key in ["url", "pr_number", "title", "body", "author",
                    "merged_at", "total_additions", "total_deletions", "files"]:
            assert key in d

    def test_to_dict_files_are_dicts(self):
        pr = self._sample_pr()
        d = pr.to_dict()
        assert isinstance(d["files"], list)
        assert all(isinstance(f, dict) for f in d["files"])

    def test_empty_files(self):
        pr = PRDiff(
            url="https://github.com/owner/repo/pull/1",
            pr_number=1,
            title="Empty PR",
            body="",
            files=[],
            merged_at="2024-01-01T00:00:00Z",
            author="bot",
        )
        assert pr.total_additions == 0
        assert pr.total_deletions == 0
        assert pr.diff_text == ""

    def test_extract_task_prompt_with_empty_body(self):
        pr = PRDiff(
            url="https://github.com/owner/repo/pull/2",
            pr_number=2,
            title="Refactor authentication module",
            body="",
            files=[
                PRFile("src/auth.py", 50, 30, "@@ -1,30 +1,50 @@", "modified"),
            ],
            merged_at="2024-01-01T00:00:00Z",
            author="dev",
        )
        prompt = pr.extract_task_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ── PRFetcher tests ─────────────────────────────────────────────────────────────


def _make_mock_response(data: dict | list, status: int = 200):
    """Return a context-manager mock that mimics urllib.request.urlopen."""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestPRFetcher:
    def test_bad_url_raises_value_error(self):
        fetcher = PRFetcher()
        with pytest.raises(ValueError, match="github.com"):
            fetcher.fetch("https://notgithub.com/owner/repo/issues/1")

    def test_non_pr_url_raises_value_error(self):
        fetcher = PRFetcher()
        with pytest.raises(ValueError):
            fetcher.fetch("https://github.com/owner/repo/issues/1")

    def test_malformed_url_raises_value_error(self):
        fetcher = PRFetcher()
        with pytest.raises((ValueError, Exception)):
            fetcher.fetch("not-a-url-at-all")

    @patch("trace_ops.github.pr_fetcher.urllib.request.urlopen")
    def test_fetch_returns_pr_diff(self, mock_urlopen):
        # First call → PR metadata, second call → files list
        mock_urlopen.side_effect = [
            _make_mock_response(_pr_api_response()),
            _make_mock_response(_files_api_response()),
        ]
        fetcher = PRFetcher()
        pr = fetcher.fetch("https://github.com/owner/repo/pull/42")
        assert isinstance(pr, PRDiff)
        assert pr.pr_number == 42
        assert pr.title == "Fix widget rendering"
        assert pr.author == "alice"
        assert len(pr.files) == 2

    @patch("trace_ops.github.pr_fetcher.urllib.request.urlopen")
    def test_fetch_pr_files_correct(self, mock_urlopen):
        mock_urlopen.side_effect = [
            _make_mock_response(_pr_api_response()),
            _make_mock_response(_files_api_response()),
        ]
        fetcher = PRFetcher()
        pr = fetcher.fetch("https://github.com/owner/repo/pull/42")
        filenames = [f.filename for f in pr.files]
        assert "src/widget.py" in filenames
        assert "tests/test_widget.py" in filenames

    @patch("trace_ops.github.pr_fetcher.urllib.request.urlopen")
    def test_fetch_recent_returns_list(self, mock_urlopen):
        # List call → list of PRs; then N pairs of (PR detail, files)
        pr_list = _prs_list_response(count=3)
        pr_detail = _pr_api_response(pr_number=1)
        files = _files_api_response()
        # First call: list endpoint; then 3 × (detail, files)
        mock_urlopen.side_effect = [
            _make_mock_response(pr_list),
            _make_mock_response(_pr_api_response(pr_number=1)),
            _make_mock_response(files),
            _make_mock_response(_pr_api_response(pr_number=2)),
            _make_mock_response(files),
            _make_mock_response(_pr_api_response(pr_number=3)),
            _make_mock_response(files),
        ]
        fetcher = PRFetcher()
        prs = fetcher.fetch_recent("https://github.com/owner/repo", limit=3)
        assert isinstance(prs, list)
        assert len(prs) == 3
        assert all(isinstance(p, PRDiff) for p in prs)

    @patch("trace_ops.github.pr_fetcher.urllib.request.urlopen")
    def test_fetch_recent_respects_limit(self, mock_urlopen):
        pr_list = _prs_list_response(count=10)
        files = _files_api_response()
        side_effects = [_make_mock_response(pr_list)]
        for i in range(1, 6):
            side_effects.append(_make_mock_response(_pr_api_response(pr_number=i)))
            side_effects.append(_make_mock_response(files))
        mock_urlopen.side_effect = side_effects
        fetcher = PRFetcher()
        prs = fetcher.fetch_recent("https://github.com/owner/repo", limit=5)
        assert len(prs) <= 5

    def test_fetcher_token_from_env(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test_token_xyz")
        fetcher = PRFetcher()
        assert fetcher._token == "test_token_xyz"

    def test_fetcher_explicit_token_overrides_env(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "env_token")
        fetcher = PRFetcher(token="explicit_token")
        assert fetcher._token == "explicit_token"

    def test_fetcher_no_token_is_none(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        fetcher = PRFetcher()
        assert fetcher._token is None

    @patch("trace_ops.github.pr_fetcher.urllib.request.urlopen")
    def test_fetch_404_raises_runtime_error(self, mock_urlopen):
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(
            url="https://api.github.com/repos/owner/repo/pulls/9999",
            code=404,
            msg="Not Found",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )
        fetcher = PRFetcher()
        with pytest.raises((RuntimeError, Exception)):
            fetcher.fetch("https://github.com/owner/repo/pull/9999")

    @patch("trace_ops.github.pr_fetcher.urllib.request.urlopen")
    def test_pr_with_no_patch_field(self, mock_urlopen):
        """Files without a 'patch' key (binary files, etc.) should not crash."""
        files_no_patch = [
            {
                "filename": "assets/image.png",
                "additions": 0,
                "deletions": 0,
                "status": "added",
                # No "patch" key
            }
        ]
        mock_urlopen.side_effect = [
            _make_mock_response(_pr_api_response()),
            _make_mock_response(files_no_patch),
        ]
        fetcher = PRFetcher()
        pr = fetcher.fetch("https://github.com/owner/repo/pull/42")
        assert pr.files[0].patch == ""

    @patch("trace_ops.github.pr_fetcher.urllib.request.urlopen")
    def test_fetch_recent_bad_repo_url_raises(self, mock_urlopen):
        fetcher = PRFetcher()
        with pytest.raises((ValueError, Exception)):
            fetcher.fetch_recent("https://notgithub.com/owner/repo")

    @patch("trace_ops.github.pr_fetcher.urllib.request.urlopen")
    def test_merged_at_preserved(self, mock_urlopen):
        mock_urlopen.side_effect = [
            _make_mock_response(_pr_api_response(merged_at="2024-06-01T09:00:00Z")),
            _make_mock_response(_files_api_response()),
        ]
        fetcher = PRFetcher()
        pr = fetcher.fetch("https://github.com/owner/repo/pull/42")
        assert pr.merged_at == "2024-06-01T09:00:00Z"
