# End-to-End Tests

These tests exercise a full PhysioCoach session against a **real browser, GPU, and
local VLM**, so they run **locally only** (they are not gated in CI).

## What's here

- `test_real_workflow.py` — drives a complete coaching workflow with real video input
  and actual VLM inference. See [`real_workflow_testing.md`](./real_workflow_testing.md)
  for details and prerequisites.

## Prerequisites

```bash
pip install pytest-playwright
playwright install chromium
```

You also need the server running with a local VLM (Ollama + `qwen2.5vl:7b`):

```bash
./scripts/start_server.sh &
```

## Running

```bash
pytest tests/e2e -v          # run the e2e workflow (local; needs GPU/Ollama)
pytest -m "not e2e"          # exclude e2e tests
```
