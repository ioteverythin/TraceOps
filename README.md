# 🔁 TraceOps

**Record and replay LLM agent traces for deterministic regression testing.**

[![PyPI](https://img.shields.io/pypi/v/TraceOps)](https://pypi.org/project/TraceOps/)
[![Python](https://img.shields.io/pypi/pyversions/TraceOps)](https://pypi.org/project/TraceOps/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TraceOps brings the VCR.py pattern to LLM agents — but at the **SDK level**, not the HTTP level. It intercepts `openai.chat.completions.create`, `anthropic.messages.create`, tool calls, and agent decisions, recording the full execution trace. On replay, it injects recorded responses — **zero API calls, millisecond execution, fully deterministic**.

## Why not just use VCR.py?

VCR.py records HTTP traffic. TraceOps records **agent behavior**:

| | VCR.py / Cagent | TraceOps |
|---|---|---|
| **Records at** | HTTP layer | SDK layer |
| **Understands** | Request/response pairs | LLM calls, tool invocations, agent decisions |
| **Trajectory tracking** | ❌ | ✅ "agent called search, then read_file, then responded" |
| **Regression diff** | Binary (match/no-match) | Semantic ("model changed", "new tool used", "extra LLM call") |
| **Framework-agnostic** | ✅ | ✅ (OpenAI, Anthropic, LiteLLM, LangChain, CrewAI) |
| **Cost tracking** | ❌ | ✅ per-call tokens and USD |
| **Async + Streaming** | Varies | ✅ Native support |

## Quick Start

```bash
pip install TraceOps
# With optional provider support:
pip install TraceOps[openai]           # OpenAI
pip install TraceOps[anthropic]        # Anthropic
pip install TraceOps[langchain]        # LangChain/LangGraph
pip install TraceOps[langgraph]        # LangGraph (includes langchain-core)
pip install TraceOps[crewai]           # CrewAI
pip install TraceOps[all]              # Everything
```

### Record an agent run

```python
from trace_ops import Recorder

with Recorder(save_to="cassettes/test_math.yaml") as rec:
    # Your agent code here — all LLM calls are automatically captured
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )
    print(response.choices[0].message.content)

# Trace saved to cassettes/test_math.yaml
print(f"Recorded {rec.trace.total_llm_calls} LLM calls")
```

### Async support (v0.2)

```python
from trace_ops import Recorder

async with Recorder(save_to="cassettes/async_test.yaml") as rec:
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )
```

### Streaming support (v0.2)

Streaming calls are automatically captured and assembled into complete responses in the cassette. On replay, responses are split back into realistic chunks:

```python
with Recorder(save_to="cassettes/stream_test.yaml") as rec:
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a story"}],
        stream=True,
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")
```

### Replay deterministically

```python
from trace_ops import Replayer

with Replayer("cassettes/test_math.yaml"):
    # Same code — but LLM calls return recorded responses
    # Zero API calls, zero cost, millisecond execution
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )
    assert response.choices[0].message.content  # deterministic!
```

### Regression testing with pytest

```python
import pytest
from trace_ops import Recorder, Replayer, assert_trace_unchanged, load_cassette

# First run: record
def test_agent_record():
    with Recorder(save_to="cassettes/test_agent.yaml"):
        agent.run("Summarize the quarterly report")

# Subsequent runs: replay and check for regressions
def test_agent_regression():
    old_trace = load_cassette("cassettes/test_agent.yaml")
    with Recorder() as rec:
        agent.run("Summarize the quarterly report")
    assert_trace_unchanged(old_trace, rec.trace)
```

### Using the pytest plugin (auto record/replay)

```python
# Uses the cassette fixture — automatically records on first run,
# replays on subsequent runs
def test_agent(cassette):
    agent.run("Summarize the quarterly report")
```

```bash
# First run: record cassettes
pytest --record

# Subsequent runs: replay from cassettes
pytest

# Re-record all cassettes
pytest --record-mode=all
```

### Budget assertions (v0.2)

Guard against cost overruns, token bloat, and infinite loops:

```python
import pytest
from trace_ops import Recorder
from trace_ops.assertions import (
    assert_cost_under,
    assert_tokens_under,
    assert_max_llm_calls,
    assert_no_loops,
)

def test_agent_budget():
    with Recorder() as rec:
        agent.run("Summarize the report")

    assert_cost_under(rec.trace, max_usd=0.50)
    assert_tokens_under(rec.trace, max_tokens=10_000)
    assert_max_llm_calls(rec.trace, max_calls=5)
    assert_no_loops(rec.trace, max_consecutive_same_tool=3)
```

Or use the pytest marker:

```python
@pytest.mark.budget(max_usd=0.50, max_tokens=10_000, max_llm_calls=5)
def test_agent(cassette):
    agent.run("Summarize the report")
```

## What Gets Recorded

Every LLM call captures:
- **Provider** (openai, anthropic, litellm, langchain, crewai)
- **Model** (gpt-4o, claude-4-sonnet, etc.)
- **Messages** (full prompt including system message)
- **Response** (full completion response)
- **Tool calls** (function name, arguments, tool_call_id)
- **Tokens** (input/output counts)
- **Cost** (USD per call)
- **Timing** (milliseconds per call)

Plus agent-level events:
- **Tool invocations** (name, input, output)
- **Agent decisions** (delegation, routing, planning)
- **Errors** (exceptions with type and message)

## Supported Providers

| Provider | Auto-intercepted | Sync | Async | Streaming | Package |
|----------|-----------------|------|-------|-----------|---------|
| OpenAI | ✅ | ✅ | ✅ | ✅ | `openai` |
| Anthropic | ✅ | ✅ | ✅ | ✅ | `anthropic` |
| LiteLLM | ✅ | ✅ | ✅ | ✅ | `litellm` |
| LangChain | ✅ | ✅ | ✅ | — | `langchain-core` |
| LangGraph | ✅ | ✅ | ✅ | ✅ | `langgraph` |
| CrewAI | ✅ | ✅ | ✅ | — | `crewai` |
| Any (manual) | Via `rec.record_tool_call()` | ✅ | ✅ | — | — |

## Framework Integrations (v0.3)

TraceOps now captures **graph-level** and **framework-level** events.

### LangGraph

Record Pregel graph execution with node and stream-level events:

```python
from trace_ops import Recorder
from langgraph.graph import StateGraph

with Recorder(save_to="cassettes/graph.yaml", intercept_langgraph=True):
    # Captured: graph_start, graph_stream_start/end, graph_end
    result = graph.invoke({"messages": [...]})
```

### LangChain

Intercept `BaseChatModel` calls in agents, chains, and LCEL:

```python
with Recorder(save_to="cassettes/agent.yaml", intercept_langchain=True):
    result = agent.invoke({"input": "your question"})
```

### Anthropic Tool Use

Full support for tool_use blocks with automatic tool result injection.

## Normalization (v0.2)

Compare responses across providers with normalized, provider-agnostic representations:

```python
from trace_ops.normalize import normalize_response, normalize_for_comparison

# Both produce the same shape for diffing
openai_clean = normalize_for_comparison(openai_resp, "openai")
anthropic_clean = normalize_for_comparison(anthropic_resp, "anthropic")
# Strips volatile fields (IDs, token counts) — focuses on semantics
```

## CLI

```bash
# Inspect a cassette
replay inspect cassettes/test_math.yaml

# Compare two cassettes
replay diff cassettes/old.yaml cassettes/new.yaml

# Export to JSON
replay export cassettes/test.yaml --format json -o trace.json

# Interactive time-travel debugger (v0.2)
replay debug cassettes/test.yaml
replay debug cassettes/v1.yaml --compare cassettes/v2.yaml
replay debug cassettes/test.yaml --tools-only

# Generate HTML report (v0.2)
replay report cassettes/test.yaml -o report.html

# List all cassettes with stats (v0.2)
replay ls cassettes/

# Aggregate stats across cassettes (v0.2)
replay stats cassettes/

# Delete stale cassettes (v0.2)
replay prune cassettes/ --older-than 30d --dry-run

# Validate cassette integrity (v0.2)
replay validate cassettes/test.yaml
```

## Time-Travel Debugger (v0.2)

Step forward and backward through a recorded trace, inspecting prompts, responses, and tool I/O at each step:

```
$ replay debug cassettes/test_math.yaml

╭─ 🔁 TraceOps debugger ──────────────────────╮
│ Trace ID: abc123                                  │
│ Events: 6  (LLM: 2, Tools: 1)                    │
│ Tokens: 450  Cost: $0.0015  Duration: 1200ms     │
╰──────────────────────────────────────────────────╯
  n/Enter = next · p = prev · q = quit · g <N> = go to event N

LLM Response  #2/6  provider=openai  model=gpt-4o  350ms
╭─ Response ─────────────────────────────╮
│ The answer is 4.                       │
╰────────────────────────────────────────╯
  Tokens: in=100, out=25, $0.0005

[step] _
```

## HTML Report (v0.2)

Generate a shareable single-file HTML report with dark theme, stats grid, trajectory visualization, and expandable events:

```python
from trace_ops.reporters.html import generate_html_report
from trace_ops import load_cassette

trace = load_cassette("cassettes/test.yaml")
generate_html_report(trace, "report.html")
```

## GitHub Action (v0.2)

Use the included composite action in your CI pipeline:

```yaml
- uses: ./.github/actions/TraceOps
  with:
    cassette-dir: cassettes
    record-mode: replay
    pytest-args: tests/test_agent.py -v
    post-diff-comment: true
```

## Trace Diffing

The diff engine compares two traces semantically:

```python
from trace_ops import diff_traces, load_cassette

old = load_cassette("cassettes/v1.yaml")
new = load_cassette("cassettes/v2.yaml")
diff = diff_traces(old, new)

print(diff.summary())
# Trace comparison:
#   ⚠ TRAJECTORY CHANGED (agent took a different path)
#     Old: llm_call:gpt-4o → tool:search → llm_call:gpt-4o
#     New: llm_call:gpt-4o → tool:browse → tool:search → llm_call:gpt-4o
#   Tool calls: 1 more
#   New tools used: browse
```

## Roadmap
- **v0.1**: Record/replay for OpenAI, Anthropic, LiteLLM. Pytest plugin. Diff engine. CLI.
- **v0.2**: Async + streaming support, normalized diffing, LangChain/CrewAI interceptors, budget assertions, time-travel debugger, HTML reports, GitHub Action, expanded CLI.
- **v0.3** (current): LangGraph Pregel interceptor, Anthropic tool_use support, example templates, framework integration tests, 58% code coverage (144 tests).
- **v0.4**: VS Code extension with trace visualization, live replay dashboard, web UI for cassette inspection.

## License

MIT
