# Changelog

All notable changes to `tuner-livekit-sdk` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.10] - 2026-08-19

### Added

- **OpenTelemetry trace forwarding to Tuner** — LiveKit Agents' own spans are exported to Tuner and shown as a trace tree on the call details page. Enabled by default; pass `traces_enabled=False` to `TunerPlugin` to turn it off. Every span is tagged with the call id so the trace correlates to the call without any wiring on your side.
- `traces` optional extra (`pip install tuner-livekit-sdk[traces]`) for the OpenTelemetry SDK and OTLP HTTP exporter. Not a hard dependency: without it, trace forwarding is a no-op and nothing else changes.

## [0.1.9] - 2026-07-20

### Added

- `eou_delay` field in transcript segment metadata — time (ms) between the end of user speech and the decision to end their turn, read from LiveKit Agents' `end_of_turn_delay` metric.

## [0.1.8] - 2026-07-14

### Changed

- Bumped `tuner-langchain` minimum version to `0.1.1` (republish of 0.1.8).

## [0.1.7] - 2026-06-05

### Added

- Optional `recipient` field for the callee's phone number or SIP URL (#15).

### Changed

- Moved `_ToolMessageFilter` into the LiveKit SDK layer; applied automatically in `wrap_graph()` (#16).

## [0.1.6] - 2026-06-04

### Added

- Support for the `tuner-langchain` package — `wrap_graph()` / `wrap_chain()` for LangGraph/LangChain observability (#13).

## [0.1.5] - 2026-05-08

### Added

- SIP simulation correlation via `sip_call_id`, matching Tuner simulation calls to the production SIP trunk (#11).

## [0.1.4] - 2026-04-20

### Fixed

- User segments missing `started_speaking_at` / `stopped_speaking_at` timestamps (#10).

## [0.1.3] - 2026-03-31

### Added

- Disconnection reason tracking (`disconnection_reason` on the call payload) (#9).

## [0.1.2] - 2026-03-18

### Changed

- Documentation updates, including cost calculator usage (#7).

## [0.1.1] - 2026-03-16

### Added

- Call metadata support and expanded README documentation (#4).

## [0.1.0] - 2026-03-09

### Added

- Initial release: automatic ingestion of LiveKit Agents session data (transcript, tool calls, usage summary, call metadata) into the Tuner observability API.
