"""GitHub PR fetcher — pull PR diffs as golden baselines for gap analysis.

Inspired by agent-pr-replay's pr_finder + repo modules: fetches merged PRs from
any public GitHub repository and reverse-engineers task prompts from their diffs,
giving you human-validated baselines to compare your agent traces against.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PRFile:
    """A single file changed in a pull request."""

    filename: str
    additions: int
    deletions: int
    patch: str = ""
    status: str = ""  # "modified", "added", "removed", "renamed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "additions": self.additions,
            "deletions": self.deletions,
            "status": self.status,
        }


@dataclass
class PRDiff:
    """A GitHub pull request with its file diffs."""

    url: str
    pr_number: int
    title: str
    body: str
    files: list[PRFile] = field(default_factory=list)
    merged_at: str | None = None
    author: str = ""

    @property
    def diff_text(self) -> str:
        """Full diff as a single unified diff string."""
        parts = []
        for f in self.files:
            parts.append(f"--- a/{f.filename}\n+++ b/{f.filename}")
            if f.patch:
                parts.append(f.patch)
        return "\n".join(parts)

    @property
    def total_additions(self) -> int:
        return sum(f.additions for f in self.files)

    @property
    def total_deletions(self) -> int:
        return sum(f.deletions for f in self.files)

    def extract_task_prompt(self) -> str:
        """Reverse-engineer a task description from the PR title, body and diff.

        Inspired by agent-pr-replay's ``generate_human_prompt`` — produces a
        plain-English task description that could be given to an AI agent to
        reproduce the same change, without revealing the solution.
        """
        lines = [f"Task: {self.title}"]
        if self.body:
            body_preview = self.body[:500].strip()
            lines.append(f"\nContext:\n{body_preview}")
        file_list = ", ".join(f.filename for f in self.files[:10])
        if len(self.files) > 10:
            file_list += f" (and {len(self.files) - 10} more)"
        lines.append(f"\nFiles changed: {file_list}")
        lines.append(f"Changes: +{self.total_additions} -{self.total_deletions} lines")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "pr_number": self.pr_number,
            "title": self.title,
            "body": self.body,
            "author": self.author,
            "merged_at": self.merged_at,
            "total_additions": self.total_additions,
            "total_deletions": self.total_deletions,
            "files": [f.to_dict() for f in self.files],
        }


class PRFetcher:
    """Fetch GitHub PR diffs using the public REST API (no CLI required).

    Use PR diffs as "golden baselines" — the human-validated solution — then
    compare them against your agent's recorded traces to find behavioral gaps.
    This is the same core idea as agent-pr-replay, adapted for TraceOps cassettes.

    Example::

        fetcher = PRFetcher(token="ghp_...")
        diff = fetcher.fetch("https://github.com/owner/repo/pull/123")
        print(diff.extract_task_prompt())

        # Fetch recent merged PRs as golden baselines
        recent = fetcher.fetch_recent("https://github.com/owner/repo", limit=5)
    """

    _PR_RE = re.compile(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)")
    _REPO_RE = re.compile(r"github\.com/([^/]+)/([^/]+?)(?:\.git|/|$)")
    API_BASE = "https://api.github.com"

    def __init__(self, token: str | None = None) -> None:
        """
        Args:
            token: Optional GitHub personal access token. Without it you are
                   limited to 60 requests/hour on the public API.
                   Falls back to the ``GITHUB_TOKEN`` environment variable.
        """
        import os
        self._token: str | None = token if token is not None else os.environ.get("GITHUB_TOKEN")

    def _get(self, url: str, accept: str = "application/vnd.github+json") -> Any:
        req = urllib.request.Request(
            url,
            headers={
                "Accept": accept,
                "User-Agent": "TraceOps/0.5.0",
                **({"Authorization": f"Bearer {self._token}"} if self._token else {}),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise RuntimeError(
                f"GitHub API error {e.code} for {url}: {body[:200]}"
            ) from e

    def fetch(self, pr_url: str) -> PRDiff:
        """Fetch a PR diff by its GitHub URL.

        Args:
            pr_url: Full GitHub PR URL, e.g. ``https://github.com/owner/repo/pull/123``.

        Returns:
            A :class:`PRDiff` with file patches and metadata.

        Raises:
            ValueError: If the URL is not a valid GitHub PR URL.
            RuntimeError: On GitHub API errors (rate limit, 404, etc.).
        """
        m = self._PR_RE.search(pr_url)
        if not m:
            raise ValueError(f"Not a valid GitHub PR URL: {pr_url!r}")
        owner, repo, pr_num = m.group(1), m.group(2), int(m.group(3))
        api_url = f"{self.API_BASE}/repos/{owner}/{repo}/pulls/{pr_num}"

        pr_data = self._get(api_url)
        files_data = self._get(f"{api_url}/files")

        files = [
            PRFile(
                filename=f["filename"],
                additions=f.get("additions", 0),
                deletions=f.get("deletions", 0),
                patch=f.get("patch", ""),
                status=f.get("status", ""),
            )
            for f in files_data
        ]
        return PRDiff(
            url=pr_url,
            pr_number=pr_num,
            title=pr_data.get("title", ""),
            body=pr_data.get("body") or "",
            files=files,
            merged_at=pr_data.get("merged_at"),
            author=pr_data.get("user", {}).get("login", ""),
        )

    def fetch_recent(self, repo_url: str, limit: int = 10) -> list[PRDiff]:
        """Fetch the most recently merged PRs from a GitHub repository.

        Args:
            repo_url: GitHub repo URL, e.g. ``https://github.com/owner/repo``.
            limit: Maximum number of merged PRs to return.

        Returns:
            List of :class:`PRDiff` objects, most-recently-merged first.
        """
        m = self._REPO_RE.search(repo_url)
        if not m:
            raise ValueError(f"Not a valid GitHub repo URL: {repo_url!r}")
        owner, repo = m.group(1), m.group(2)
        api_url = (
            f"{self.API_BASE}/repos/{owner}/{repo}/pulls"
            f"?state=closed&sort=updated&direction=desc&per_page={min(limit * 3, 100)}"
        )
        prs_data = self._get(api_url)
        results: list[PRDiff] = []
        for pr in prs_data:
            if not pr.get("merged_at"):
                continue
            try:
                diff = self.fetch(pr["html_url"])
                results.append(diff)
                if len(results) >= limit:
                    break
            except Exception:
                continue
        return results
