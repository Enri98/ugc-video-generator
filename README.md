# ugc-video-generator

Automated pipeline that turns a single product photo into multiple short-form
UGC-style videos (vertical 9:16, 15–30 seconds), suitable for Reels / TikTok /
Shorts.

## What it does

Given a product photo dropped into a Google Drive `input/` folder, the pipeline
generates **N UGC videos per product** — each with a different talent, script
and setting — and uploads the finished videos plus a per-video report back to
Drive.

The pipeline is image-first: a static first frame is composed first, then
animated, then post-produced (trim, stitch, burned-in captions). State is
checkpointed to disk so any failure is resumable.

## Stack

- **Gemini 2.5 Pro** — product analysis + creative direction (script, settings)
- **Gemini 3 Flash Image (Nano Banana 2)** — talent renders + first-frame composite
- **Veo 3.1 Fast** — image-to-video generation
- **faster-whisper** — local audio transcription (free)
- **ffmpeg** — trim, concat, caption burn-in
- **Google Drive API** — input polling + output upload

## Status

Early development. See `DESIGN.md` for architecture and the development plan.

## Privacy note

This repository is intentionally generic and brand-agnostic. Brand-specific
configuration, prompts, assets and internal planning documents are kept local
and excluded via `.gitignore`.
