# Tests

This suite is **inherited from the upstream [live-vlm-webui](https://github.com/nvidia-ai-iot/live-vlm-webui)** project and covers the streaming/monitoring **infrastructure** PhysioCoach is built on — not (yet) the PhysioCoach physical-therapy logic (`pose_detector`, `exercise_library`, `session_manager`). Adding coverage for those modules is the top item on the roadmap.

## What's here

```
tests/
├── unit/test_gpu_monitor_real.py          # GPU/CPU/RAM monitoring helpers
├── integration/test_server.py             # aiohttp server endpoints + static files
├── performance/test_gpu_monitor_performance.py  # timing/regression checks (local)
├── e2e/test_real_workflow.py              # full browser workflow (local; needs GPU/Ollama)
├── utils/                                  # performance + regression helpers
└── conftest.py
```

## Running

```bash
pip install -e .
pip install -r requirements-dev.txt

pytest tests/unit tests/integration        # what CI runs
pytest tests/unit -m "not slow"            # fast subset
pytest --cov=live_vlm_webui                # with coverage
```

## Markers

| Marker | Purpose |
|--------|---------|
| `@pytest.mark.performance` | Performance/benchmark test (local) |
| `@pytest.mark.slow` | Slow-running test |
| `@pytest.mark.e2e` | End-to-end browser test (local; needs GPU/Ollama) |
| `@pytest.mark.asyncio` | Async test |

## CI

`.github/workflows/tests.yml` runs unit + integration tests (Python 3.10–3.12), coverage,
and the lint job (`black`, `ruff`, `mypy`) on every push and PR. Performance and e2e
tests require local hardware and are not gated in CI.
