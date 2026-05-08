# Design — ugc-video-generator

> Stub. Will be expanded as the spec-driven development plan is finalized.

## Goals

- From a single product photo, produce multiple UGC-style short videos
  (vertical 9:16, 15–30 s) with different talents, scripts and settings.
- Run locally (`python main.py`) but stay portable to a cloud runner.
- Be resumable on any failure (per-video JSON state).
- Keep cost per finished video under a configurable budget ceiling.

## High-level flow

1. Poll a Drive `input/` folder for new product photos.
2. **Product Analyst** (LLM, vision) — read the photo + brand guidelines,
   produce a structured product brief.
3. **Creative Director** (LLM, text) — turn the brief into N video specs:
   talent, narrative angle, script blocks, first-frame prompts, motion prompts.
4. **First-frame composite** (image model) — combine talent + product +
   setting into a static first frame per clip.
5. **Video generation** (image-to-video model) — animate each first frame,
   with a single soft-retry on safety blocks.
6. **Post-production** (ffmpeg) — trim end-of-clip drift, stitch clips,
   transcribe with faster-whisper, burn in TikTok-style captions.
7. **Report + upload** — write a human-readable report next to the MP4 and
   upload both to Drive `output/`.

## Repository layout (planned)

```
ugc-video-generator/
├── README.md
├── DESIGN.md
├── .env.example
├── pyproject.toml
├── config/                 # only generic configs are versioned
├── src/
│   ├── main.py
│   ├── orchestrator.py
│   ├── steps/
│   ├── state/
│   ├── prompts/
│   └── utils/
├── assets/                 # gitignored, populated locally
├── state/                  # gitignored, runtime state
├── tests/
└── scripts/
```

## Open design questions

Tracked separately during spec-driven development.

## Authoritative specification

The full design is maintained in [`SPEC.md`](./SPEC.md). Implementation tracks
that document; any divergence is a bug in one or the other.
