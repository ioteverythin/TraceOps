# Changelog

All notable changes to **TraceOps** will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.0] — 2025-01-27

### Added
- **Cost Dashboard** — new `replay costs [dir]` CLI command aggregating spend
  across cassettes, with per-model and per-cassette breakdowns.  Also available
  programmatically via `CostDashboard` / `CostSummary`.
- **Enhanced `replay diff`** — `--detailed` flag for per-event side-by-side
  comparison and `--output` flag to write an HTML diff report.
- **Auto-record mode** — `--record-mode auto` (and `AGENT_REPLAY_RECORD` /
  `AGENT_REPLAY_MODE` environment variables) for zero-config "record if
  missing, replay if exists" workflow.
- Exported `CostDashboard` and `CostSummary` from the top-level package.
- 100 new tests (529 total), raising coverage from 75 % → 77 %.

### Changed
- Harmonised version strings across CLI, HTML reporter, types, and
  `pyproject.toml` (all now `0.4.0`).

### Fixed
- `replay diff` no longer silently swallows HTML-output errors.

---

## [0.3.0] — 2025-01-26

### Added
- **Framework interceptors** — first-class LangChain, LangGraph, and CrewAI
  support via `interceptors/` sub-package.
- **Integration test suite** — end-to-end tests for OpenAI, Anthropic,
  LangChain, and LangGraph recording/replay.
- **pytest plugin tests** — full coverage of `--record`, `--record-mode`,
  `--replay-strict`, `cassette` fixture, `trace_snapshot` fixture, and the
  `@pytest.mark.budget` marker.
- Example scripts under `examples/`.
- README refresh with quick-start, architecture diagram, and badge row.

---

## [0.2.0] — 2025-01-25

### Added
- **Async & streaming support** — `async with Recorder()` / `Replayer()`,
  plus transparent streaming-response replay.
- **Normalization engine** — `normalize_response()` and
  `normalize_for_comparison()` for provider-agnostic diffing.
- **Budget assertions** — `assert_cost_under`, `assert_tokens_under`,
  `assert_max_llm_calls`, `assert_no_loops`.
- **HTML trace reports** — `generate_html_report()` with inline CSS/JS,
  dark theme, token bar chart, and expandable event details.
- **Terminal debugger** — interactive time-travel debugger (`replay debug`).
- **CLI commands** — `replay inspect`, `diff`, `export`, `debug`, `report`,
  `ls`, `prune`, `stats`, `validate`.
- **Cassette management** — `save_cassette` / `load_cassette` with YAML
  storage, automatic redaction of API keys.
- **Diff engine** — `diff_traces` / `assert_trace_unchanged` with DeepDiff
  integration.

---

## [0.1.0] — 2025-01-24

### Added
- Initial release.
- `Recorder` context-manager with monkey-patching for OpenAI, Anthropic,
  and LiteLLM (sync only).
- `Replayer` context-manager returning stored responses.
- Core data model: `Trace`, `TraceEvent`, `TraceMetadata`, `EventType`.
- YAML cassette serialization.
- pytest plugin with `--record` flag and `agent_cassette` marker.
