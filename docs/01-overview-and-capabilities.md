# Overview and capabilities

## What CameraChatbot does

CameraChatbot takes keyframes captured from a security camera and runs a pipeline over them: it finds every person and a short allowlist of relevant objects (bags, phones, knives) in each frame, generates a numeric "fingerprint" (embedding) for each person so the same individual can be recognized again later, and assigns each person a persistent identity that survives across separate camera sessions — not just within one video. Results are written to a Postgres database as structured records: which objects appeared, where, when, and (for people) who.

The project is mid-way through a security-oriented overhaul. The original version was a general-purpose detection demo (also estimating emotion, age, hand gestures, and monocular depth). The overhaul strips attributes that cost compute but don't serve a security use case, consolidates the machine-learning stack onto a single inference runtime (ONNX Runtime), and adds what a generic detector demo doesn't have: continuous tracking of a person across a whole video (not just within one batch of frames), a zone/rule engine (restricted areas, loitering thresholds), and a security-event model (intrusion, loitering, and — planned — unauthorized person and weapons detection).

## How to read this page

This page is meant to stand alone for a non-technical reader. The tables below are the whole picture: what's genuinely running today, what's built but not yet switched on, what's explicitly not built yet, and what was considered and deliberately left out. If you want the "how," the Developer docs (linked from each row and from [`README.md`](README.md)) go deeper.

## Capability status

Read this section as four separate lists, not one — "exists" and "runs by default" and "is saved to the database" are three different, independently true-or-false facts about any single capability. The callout after the tables explains why that distinction matters.

### Working today

| Capability | What it does | Detail |
|---|---|---|
| Person & object detection | Finds people and a short list of security-relevant objects (bags, phones, knives) in every frame | [`03`](03-pipeline-reference.md) |
| Re-identification (Re-ID) + persistent identity | Generates a fingerprint per person and matches it against everyone seen before, across sessions | [`03`](03-pipeline-reference.md) |
| Grouping people within one batch | Default mode: groups detections of the same person within one processing batch using similarity clustering | [`03`](03-pipeline-reference.md) |
| Continuous tracking across a whole video | Alternate mode: follows a person frame-to-frame through an entire video, including brief occlusions, instead of only within one batch — built and tested, **off by default** | [`04`](04-security-subsystem-reference.md) |
| Sit/stand pose classification | Labels each detected person as sitting or standing | [`03`](03-pipeline-reference.md) |
| Spatial neighborhood relations | Records which people/objects were near each other in a frame | [`03`](03-pipeline-reference.md) |
| Camera calibration | A one-time, manual per-camera setup step that maps image pixels to real-world ground positions | [`04`](04-security-subsystem-reference.md) |
| Zone definition & classification | Restricted/monitored/safe polygon zones with schedules, and the logic to test whether a detection falls inside one — built, tested, and **wired into the live pipeline (PR8b)**, but currently dormant in production (see the callout below) | [`04`](04-security-subsystem-reference.md) |
| Intrusion & loitering rule evaluation | Rules that turn a tracked person's zone history into security events — built, tested, **wired into the live pipeline, and persisted to Postgres (PR8b)**, but currently dormant in production for the same reason as zone classification | [`04`](04-security-subsystem-reference.md) |
| Core results saved to the database | Detections, identities, and spatial relationships are persisted to Postgres on every run | [`06`](06-database-schema.md) |

### Not yet implemented

These have a reserved place in the code (a function stub, a database table) but no logic behind them yet:

| Capability | Status |
|---|---|
| Unauthorized-person detection | No logic yet; a database table (`authorized_identity`) is already reserved for it |
| Weapons detection | No logic yet; a config placeholder (`WEAPONS_DETECTOR`) is already reserved for it — also blocked on getting a fine-tuned detection model, since the current object detector was never trained on weapon classes |

### Explicitly out of scope for this program

Proposed in the original planning document but never adopted into the roadmap — not "coming later," genuinely not planned:

- Abandoned-object detection (an unattended bag with no one nearby for N minutes)
- Crowding/congestion detection

### Present but dormant

Shipped, working code that's switched off by a configuration flag, kept for possible future use:

- Face detection & attention tracking
- Hand/gesture detection
- Emotion classification
- Age estimation

See [`08-legacy-and-dormant-code.md`](08-legacy-and-dormant-code.md) for what it takes to turn any of these back on.

## The three-axis nuance

*Implemented*, *invoked by the live pipeline*, and *saved to the database* are three separate facts, and a capability can be true on one axis and false on another. Continuous tracking is implemented and tested, but the live pipeline doesn't call it by default — flipping one configuration value turns it on. Zone/event logic is implemented, tested, and (as of PR8b) wired into the live pipeline and its Postgres persistence — but every run today still resolves `camera_id=None`, because no entry point threads a real camera id through yet, so the stage always takes its documented degrade path and produces zero events. That's a specific, tracked plumbing gap, not a config flag — see [`02-architecture.md`](02-architecture.md) for exactly where it bites. When in doubt about a specific capability, check the "Detail" link rather than assuming a working demo exists just because the code does.

## Where to go next

- [`02-architecture.md`](02-architecture.md) — how a frame actually flows through the system today
- [`04-security-subsystem-reference.md`](04-security-subsystem-reference.md) — the security-specific modules in depth
- [`glossary.md`](glossary.md) — unfamiliar terms (ReID, ByteTrack, homography, etc.)
