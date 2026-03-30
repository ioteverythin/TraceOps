"""TraceOps behavioral analysis sub-package.

Provides multi-trace pattern detection, golden-vs-agent gap analysis, and
markdown guidance generation — inspired by agent-pr-replay's approach of
comparing AI agent behavior to human-validated baselines.
"""

from .gap_analyzer import BehavioralGap, GapAnalyzer, GapReport
from .pattern_detector import ModelStat, PatternDetector, PatternReport, ToolPattern
from .skills_generator import SkillsGenerator

__all__ = [
    # Pattern detection
    "PatternDetector",
    "PatternReport",
    "ToolPattern",
    "ModelStat",
    # Gap analysis
    "GapAnalyzer",
    "GapReport",
    "BehavioralGap",
    # Guidance synthesis
    "SkillsGenerator",
]
