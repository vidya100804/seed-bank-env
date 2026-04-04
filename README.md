# 🌱 Seed Bank Curator — OpenEnv Environment

An RL environment where an AI agent manages a seed bank during a climate crisis.
The agent must distribute and crossbreed seed varieties to maximize crop yields
across farming villages facing drought, pests, and disease outbreaks.

## Environment Description

The agent receives observations about village conditions (drought level, pest level,
soil quality) and a catalog of available seeds. It must strategically distribute
seeds and crossbreed varieties to ensure food security across all villages.

## Action Space

| Action | Fields | Description |
|--------|--------|-------------|
| `distribute` | `seed_id`, `village_id` | Send a seed to a village |
| `crossbreed` | `seed_a`, `seed_b` | Create a hybrid seed |
| `rest` | — | Advance to next season |

## Observation Space

- `season` — current season number
- `max_seasons` — maximum seasons allowed
- `villages` — list of villages with drought/pest/soil/yield info
- `available_seeds` — seed catalog with quantities
- `message` — feedback from last action

## Tasks

| Task | Description | Difficulty |
|------|-------------|------------|
| easy | 1 village, drought crisis | Easy |
| medium | 2 villages, drought + pests | Medium |
| hard | 5 villages, spreading disease, 3 seasons | Hard |

## Reward Function

- `+0.5 * yield` for each seed distributed
- `+0.3` bonus if yield meets target (0.6)
- `+0.1` for successful crossbreeding
- `-0.05` per rest/wasted turn
- `-0.1` for invalid actions
- Final grading bonus: fraction of villages meeting target yield

## Baseline Scores

| Task | Score |
|------|-------|
| easy | ~0.85 |
| medium | ~0.65 |
| hard | ~0.45 |

## Setup & Usage

```bash
# Install dependencies
pip install -r server/requirements.txt

# Run locally
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run baseline inference
ENV_URL=http://localhost:8000 \
OPENAI_API_KEY=your_key \
MODEL_NAME=gpt-4o-mini \
python inference.py
```

## Docker

```bash
docker build -t seed-bank-env .
docker run -p 8000:8000 seed-bank-env
```

## API Endpoints

- `POST /reset?task_id=easy` — Reset environment
- `POST /step?task_id=easy` — Take action
- `GET  /state?task_id=easy` — Get current state
- `GET  /health` — Health check
