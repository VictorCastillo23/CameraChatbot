# CameraChatbot docs

CameraChatbot ingests camera keyframes, runs person detection/re-identification/pose over them, and is in the middle of a security-oriented overhaul adding continuous tracking and zone/event detection. This folder is the map of what exists today, what's active by default, and what's still on the way.

## Scope caveat

The `security-pipeline-overhaul` program (Fase 0 through Fase 6, plus the psycopg2->psycopg v3 migration) is complete and merged to `main` as of PR11 (#18). This folder was originally written against an unmerged PR chain and is otherwise still accurate, but individual module docs may drift as new work lands — check the module reference pages below if something looks off.

Wherever a capability exists as real, tested code but the live pipeline doesn't call it yet, or it's shipped but switched off by default, that's called out explicitly — see [`01-overview-and-capabilities.md`](01-overview-and-capabilities.md) for the full breakdown. Don't assume "documented" means "running."

## Reading order

**Non-technical / evaluating the project's capabilities:** [`01-overview-and-capabilities.md`](01-overview-and-capabilities.md) is written to stand alone — you can stop there.

**Developer, first time running it:** [`05-running-the-pipeline.md`](05-running-the-pipeline.md) → [`02-architecture.md`](02-architecture.md).

**Developer, extending or debugging something:** add [`03-pipeline-reference.md`](03-pipeline-reference.md) or [`04-security-subsystem-reference.md`](04-security-subsystem-reference.md) depending on what you're touching, plus [`06-database-schema.md`](06-database-schema.md) if it involves Postgres, and [`07-testing-and-dev-tools.md`](07-testing-and-dev-tools.md) before writing any verification for your change.

## Index

| File | Covers | Mainly for |
|---|---|---|
| [`01-overview-and-capabilities.md`](01-overview-and-capabilities.md) | What the system does, and a status matrix of every capability (working / not yet built / dormant / out of scope) | Everyone — start here |
| [`02-architecture.md`](02-architecture.md) | End-to-end data flow, one frame's journey from keyframe to database row | Developers |
| [`03-pipeline-reference.md`](03-pipeline-reference.md) | Module-by-module reference for the core detection/identity pipeline | Developers |
| [`04-security-subsystem-reference.md`](04-security-subsystem-reference.md) | Module-by-module reference for tracking, zones, events, camera calibration | Developers |
| [`05-running-the-pipeline.md`](05-running-the-pipeline.md) | Setup, model files, env vars, all 3 entry points, troubleshooting | Developers |
| [`06-database-schema.md`](06-database-schema.md) | Every Postgres table and its current read/write wiring status | Developers |
| [`07-testing-and-dev-tools.md`](07-testing-and-dev-tools.md) | `tests_manual/` scripts and `tools/` dev utilities | Developers |
| [`08-legacy-and-dormant-code.md`](08-legacy-and-dormant-code.md) | `legacy/`'s live-vs-dead files, and flag-gated dormant features | Developers |
| [`glossary.md`](glossary.md) | Terms used throughout (ReID, ByteTrack, homography, etc.) | Everyone |
