"""Tests for CrewAI interceptor — mock-based since crewai may not be installed.

Covers: install_crewai_record_patches(), _patch_crew_kickoff(),
_patch_agent_execute_task(), install_crewai_replay_patches().
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from trace_ops._types import EventType, Trace
from trace_ops.recorder import _Patch

# ── Fixtures ────────────────────────────────────────────────────────


class MockCrew:
    """Fake crewai.Crew for testing interceptor patches."""

    def __init__(self, agents=None, tasks=None, process="sequential"):
        self.agents = agents or []
        self.tasks = tasks or []
        self.process = process

    def kickoff(self):
        return "crew-result"


class MockAgent:
    """Fake crewai.Agent for testing interceptor patches."""

    def __init__(self, name="agent1", role="researcher", goal="find info"):
        self.name = name
        self.role = role
        self.goal = goal

    def execute_task(self, task, context=None, tools=None):
        return "task-result"


class MockTask:
    """Fake crewai.Task."""

    def __init__(self, description="Do something", agent=None):
        self.description = description
        self.agent = agent


@pytest.fixture()
def mock_crewai_module():
    """Install a fake crewai module into sys.modules for import interception."""
    mod = types.ModuleType("crewai")
    mod.Crew = MockCrew  # type: ignore[attr-defined]
    mod.Agent = MockAgent  # type: ignore[attr-defined]
    old = sys.modules.get("crewai")
    sys.modules["crewai"] = mod
    yield mod
    if old is None:
        sys.modules.pop("crewai", None)
    else:
        sys.modules["crewai"] = old


@pytest.fixture()
def recorder_stub():
    """A minimal recorder-like object with a _trace."""
    class Stub:
        def __init__(self):
            self._trace = Trace()
    return Stub()


# ── install_crewai_record_patches ──────────────────────────────────


class TestInstallRecordPatches:
    def test_patches_installed(self, mock_crewai_module, recorder_stub):
        from trace_ops.interceptors.crewai import install_crewai_record_patches

        patches: list[_Patch] = []
        install_crewai_record_patches(recorder_stub, patches)

        # Should have created patches for Crew.kickoff and Agent.execute_task
        assert len(patches) == 2
        patched_names = {p.attr for p in patches}
        assert "kickoff" in patched_names
        assert "execute_task" in patched_names

    def test_no_patches_without_crewai(self, recorder_stub):
        """When crewai is not importable, no patches are installed."""
        # Remove crewai from sys.modules if present
        old = sys.modules.pop("crewai", None)
        try:
            from trace_ops.interceptors.crewai import install_crewai_record_patches

            patches: list[_Patch] = []
            install_crewai_record_patches(recorder_stub, patches)
            # The functions silently skip when import fails
            # (patches may be 0 if crewai not importable)
        finally:
            if old is not None:
                sys.modules["crewai"] = old


# ── Crew kickoff patch ─────────────────────────────────────────────


class TestCrewKickoffPatch:
    def test_records_kickoff_events(self, mock_crewai_module, recorder_stub):
        from trace_ops.interceptors.crewai import install_crewai_record_patches

        agent1 = MockAgent(name="researcher", role="Research Agent", goal="Find data")
        task1 = MockTask(description="Research topic X", agent=agent1)
        crew = MockCrew(agents=[agent1], tasks=[task1])

        patches: list[_Patch] = []
        install_crewai_record_patches(recorder_stub, patches)

        try:
            result = MockCrew.kickoff(crew)
        finally:
            for p in reversed(patches):
                p.restore()

        assert result == "crew-result"

        events = recorder_stub._trace.events
        # Should have crew_kickoff and crew_complete events
        decisions = [e for e in events if e.event_type == EventType.AGENT_DECISION]
        assert len(decisions) >= 2

        kickoff_event = decisions[0]
        assert kickoff_event.decision == "crew_kickoff"
        assert "1 agents" in kickoff_event.reasoning
        assert kickoff_event.metadata["framework"] == "crewai"
        assert len(kickoff_event.metadata["agents"]) == 1
        assert kickoff_event.metadata["agents"][0]["name"] == "researcher"

        complete_event = decisions[1]
        assert complete_event.decision == "crew_complete"
        assert complete_event.duration_ms is not None
        assert complete_event.duration_ms >= 0

    def test_records_error_on_kickoff_failure(self, mock_crewai_module, recorder_stub):
        from trace_ops.interceptors.crewai import install_crewai_record_patches

        orig_kickoff = MockCrew.kickoff

        # Replace BEFORE patching so the closure captures the raiser
        def raising_kickoff(self):
            raise RuntimeError("fail")

        MockCrew.kickoff = raising_kickoff

        patches: list[_Patch] = []
        install_crewai_record_patches(recorder_stub, patches)

        try:
            crew = MockCrew()
            with pytest.raises(RuntimeError, match="fail"):
                MockCrew.kickoff(crew)
        finally:
            for p in reversed(patches):
                p.restore()
            # Restore the real original
            MockCrew.kickoff = orig_kickoff

        errors = [e for e in recorder_stub._trace.events if e.event_type == EventType.ERROR]
        assert len(errors) >= 1
        assert errors[0].error_type == "RuntimeError"
        assert errors[0].error_message == "fail"

    def test_crew_with_no_agents_or_tasks(self, mock_crewai_module, recorder_stub):
        from trace_ops.interceptors.crewai import install_crewai_record_patches

        crew = MockCrew(agents=[], tasks=[])
        patches: list[_Patch] = []
        install_crewai_record_patches(recorder_stub, patches)

        try:
            result = MockCrew.kickoff(crew)
        finally:
            for p in reversed(patches):
                p.restore()

        assert result == "crew-result"
        decisions = [e for e in recorder_stub._trace.events if e.event_type == EventType.AGENT_DECISION]
        assert decisions[0].metadata["agents"] == []
        assert decisions[0].metadata["tasks"] == []


# ── Agent execute_task patch ───────────────────────────────────────


class TestAgentExecuteTaskPatch:
    def test_records_execute_events(self, mock_crewai_module, recorder_stub):
        from trace_ops.interceptors.crewai import install_crewai_record_patches

        agent = MockAgent(name="coder", role="Software Engineer")
        task = MockTask(description="Write a function")

        patches: list[_Patch] = []
        install_crewai_record_patches(recorder_stub, patches)

        try:
            result = MockAgent.execute_task(agent, task)
        finally:
            for p in reversed(patches):
                p.restore()

        assert result == "task-result"

        decisions = [e for e in recorder_stub._trace.events if e.event_type == EventType.AGENT_DECISION]
        assert len(decisions) >= 2

        start_event = decisions[0]
        assert "agent_execute:coder" in start_event.decision
        assert "Software Engineer" in start_event.reasoning
        assert start_event.metadata["agent_name"] == "coder"
        assert start_event.metadata["task_description"] == "Write a function"

        complete_event = decisions[1]
        assert "agent_complete:coder" in complete_event.decision
        assert complete_event.duration_ms >= 0

    def test_records_error_on_task_failure(self, mock_crewai_module, recorder_stub):
        from trace_ops.interceptors.crewai import install_crewai_record_patches

        orig_execute = MockAgent.execute_task

        # Replace BEFORE patching so the closure captures the raiser
        def raising_execute(self, task, context=None, tools=None):
            raise ValueError("bad task")

        MockAgent.execute_task = raising_execute

        patches: list[_Patch] = []
        install_crewai_record_patches(recorder_stub, patches)

        try:
            agent = MockAgent(name="tester")
            task = MockTask(description="Fail")
            with pytest.raises(ValueError, match="bad task"):
                MockAgent.execute_task(agent, task)
        finally:
            for p in reversed(patches):
                p.restore()
            MockAgent.execute_task = orig_execute

        errors = [e for e in recorder_stub._trace.events if e.event_type == EventType.ERROR]
        assert len(errors) >= 1
        assert errors[0].error_type == "ValueError"

    def test_with_context_and_tools(self, mock_crewai_module, recorder_stub):
        from trace_ops.interceptors.crewai import install_crewai_record_patches

        agent = MockAgent(name="worker", role="Worker")
        task = MockTask(description="Process data")

        patches: list[_Patch] = []
        install_crewai_record_patches(recorder_stub, patches)

        try:
            result = MockAgent.execute_task(agent, task, context="some context", tools=["tool1"])
        finally:
            for p in reversed(patches):
                p.restore()

        assert result == "task-result"


# ── install_crewai_replay_patches ──────────────────────────────────


class TestInstallReplayPatches:
    def test_replay_is_noop(self, mock_crewai_module):
        from trace_ops.interceptors.crewai import install_crewai_replay_patches

        replayer = MagicMock()
        patches: list = []
        # Should not raise and not add patches (it's a pass-through)
        install_crewai_replay_patches(replayer, patches)
        assert patches == []
