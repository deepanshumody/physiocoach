# PhysioCoach 🏋️

**Real-time AI physical therapy coaching powered by Vision Language Models and MediaPipe — fully local, no cloud required.**

PhysioCoach watches you exercise through your webcam, analyzes your form using a local VLM, counts your reps using pose estimation, and speaks coaching cues aloud in real time. Everything runs on-device.

Built at the **Dell × NVIDIA Hackathon 2026** by NYU students.

---

## What It Does

- 📷 **Streams your webcam** via WebRTC at 30fps from any browser
- 🤖 **Analyzes your form** using Qwen2.5-VL-7B running locally (~800ms per frame)
- 🦴 **Counts your reps** by tracking joint angles with MediaPipe Pose at 30fps
- 🔊 **Speaks coaching cues** aloud via browser Text-to-Speech
- 📐 **Measures ROM** (Range of Motion) angles for clinical tracking
- 📷 **Supports dual cameras** — front and side view simultaneously

---

## Supported Exercises

| Category | Exercises |
|---|---|
| **Lower body** | Squat, Forward Lunge, Calf Raise, Side-Lying Leg Raise, Standing Hip Abduction, Seated Knee Extension, Seated Towel Knee Slide |
| **Upper body** | Wall Push-Up, Shoulder Raise, Bicep Curl, Wall Slide with Towel, Seated Overhead Press, Resistance Band Row, Tennis Ball Squeeze, Ball Wall Press |
| **Stretch** | Neck Rotation |
| **General** | Auto-detect mode — AI identifies exercise automatically |

---

## How It Works

```
Webcam (30fps)
    │
    ▼
WebRTC stream → server.py
    │
    ├──► Every 15 frames → Qwen2.5-VL-7B (local)
    │         └── JSON coaching cue → LLM → natural language → TTS spoken aloud
    │
    └──► Every frame → MediaPipe Pose
              └── 33 landmarks → joint angle → threshold crossing → rep count
```

The VLM and MediaPipe run **in parallel** — pose tracking never waits for VLM inference.

---

## Architecture

```
src/live_vlm_webui/
├── server.py            # WebRTC + WebSocket server (aiohttp + aiortc)
├── video_processor.py   # Frame capture, VLM calls, feedback pipeline
├── vlm_service.py       # OpenAI-compatible VLM API client
├── pose_detector.py     # MediaPipe Pose wrapper, rep counter, skeleton overlay
├── exercise_library.py  # Exercise definitions, joint configs, VLM prompt templates
├── rom_service.py       # Range of Motion angle measurement
├── session_manager.py   # Session state management
├── gpu_monitor.py       # GPU/CPU/RAM monitoring
└── static/index.html    # Browser frontend
```

---

## Setup

### Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) with `qwen2.5vl:7b` pulled
- A webcam accessible from your browser

### Install

```bash
git clone https://github.com/deepanshumody/live-vlm-webui-main.git
cd live-vlm-webui-main

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

### Pull the model

```bash
ollama pull qwen2.5vl:7b
ollama serve
```

### Run

```bash
./scripts/start_server.sh
```

Open **`https://localhost:8090`** in your browser. Accept the self-signed certificate warning (Advanced → Proceed), then grant camera access.

---

## Usage

1. **Select an exercise** from the dropdown (or leave on General for auto-detection)
2. **Click Start** — the AI begins analyzing your form every ~15 frames
3. **Listen to coaching cues** — spoken aloud via your browser
4. **Watch the rep counter** — increments automatically as you move
5. **Check ROM angles** — displayed live on the video overlay

### Dual Camera Mode

For exercises where front and side views both matter:

1. Connect a second webcam (or use a phone as a second camera)
2. Enable **Dual Camera** in the UI
3. The AI receives both feeds and gives form feedback with full 3D context

---

## How Rep Counting Works

MediaPipe Pose detects 33 body landmarks on every frame. For each exercise, a specific 3-joint triplet is tracked:

| Exercise | Joint triplet | Down threshold | Up threshold |
|---|---|---|---|
| Squat | hip → knee → ankle | 100° | 155° |
| Bicep curl | shoulder → elbow → wrist | 50° | 140° |
| Calf raise | knee → ankle → foot | 160° | 172° |
| Neck rotation | left shoulder → nose → right shoulder | 110° | 165° |

A rep completes when the angle crosses the **down** threshold and then recovers past the **up** threshold. Each exercise has its own joint config defined in `exercise_library.py`.

For shoulder and elbow exercises, the active arm is auto-detected each frame by comparing which wrist is raised or which elbow is more bent.

---

## Model Selection

We tested four models before choosing Qwen2.5-VL-7B:

| Model | Latency | Result |
|---|---|---|
| `llama3.2-vision:11b` | 4–8s | Too slow for real-time |
| `llama3.2-vision:90b` | 60s+ | OOM |
| `qwen2.5vl:32b` | — | OOM |
| `qwen2.5vl:7b` | ~800ms | ✅ Used |

The prompt went through three iterations — the final version removes all fallback phrases so the model always comments on what it actually sees.

---

## Dependencies

| Package | Purpose |
|---|---|
| `aiortc` | WebRTC implementation |
| `aiohttp` | Async HTTP + WebSocket server |
| `mediapipe` | Pose landmark detection |
| `opencv-python` | Frame processing, skeleton overlay |
| `openai` | OpenAI-compatible VLM API client |
| `nvidia-ml-py` / `psutil` | GPU + system monitoring |

---

## Branch Guide

| Branch | Status | Description |
|---|---|---|
| `main` | ✅ Active | ROM display, skeleton overlay, dual camera |
| `fixed` | 🔧 Ahead of main | Additional exercise library entries |
| `taruni` | ✅ Merged | ROM + skeleton features (merged into main) |
| `feature/new-features` | 🔄 Behind main | Earlier rep counter work |
| `deep` | 📦 Stale | Dual camera + RehabCoach UI experiments |

---

## Team

Built at the **Dell × NVIDIA GB10 Hackathon, 2026**
NYU Tandon School of Engineering

---

## License

Apache 2.0
