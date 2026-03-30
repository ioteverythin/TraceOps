"""TraceOps GitHub integration sub-package.

Fetch merged PRs from any public GitHub repository and use them as
golden baselines for behavioral gap analysis — inspired by agent-pr-replay.
"""

from .pr_fetcher import PRDiff, PRFetcher, PRFile

__all__ = ["PRFetcher", "PRDiff", "PRFile"]
