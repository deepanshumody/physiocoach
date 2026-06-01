# Changelog

All notable changes to PhysioCoach are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

PhysioCoach is built on NVIDIA's open-source
[live-vlm-webui](https://github.com/nvidia-ai-iot/live-vlm-webui) (Apache-2.0).
This changelog covers the PhysioCoach physical-therapy layer added on top of
that base; it does not duplicate the upstream project's history.

## [0.1.0] - 2026-03-08

Initial PhysioCoach release, built at the Dell × NVIDIA Hackathon 2026
(Top-8 of 30 teams, NYU CDS).

### Added
- **Pose-based rep counting** — MediaPipe Pose (33 landmarks) with a per-exercise
  joint-angle state machine that counts reps via down/up angle thresholds
  (`pose_detector.py`).
- **Exercise library** — 15 physical-therapy exercises with joint triplets, rep
  thresholds, ROM targets, and per-exercise VLM coaching prompts
  (`exercise_library.py`).
- **Range-of-motion (ROM) measurement** — live joint-angle estimation rendered on
  the video overlay and sidebar.
- **Auto-detect coaching mode** — the VLM identifies the exercise automatically
  when no specific exercise is selected.
- **Dual-camera support** — front + side webcam feeds relayed over WebRTC for
  fuller form feedback.
- **Real-time coaching pipeline** — VLM form analysis and MediaPipe pose tracking
  run in parallel per frame; coaching cues are spoken aloud via browser TTS
  (`video_processor.py`, `vlm_service.py`, `static/index.html`).
- **Session persistence** — SQLite-backed session and rep-count tracking
  (`session_manager.py`).

### Notes
- Selected Qwen2.5-VL-7B (~800ms/frame) after benchmarking against
  `llama3.2-vision:11b/90b` and `qwen2.5vl:32b`.
- Inherited from the upstream base: the WebRTC/VLM streaming server, GPU/system
  monitoring, and RTSP camera support.
