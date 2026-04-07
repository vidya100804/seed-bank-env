"""
Seed Bank Curator - Baseline Inference Script
Follows mandatory stdout format: [START], [STEP], [END]
"""

import json
import os
from urllib import error, parse, request
from typing import Optional

from openai import OpenAI

# --- Config (read from env vars) ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL")
MAX_STEPS = 10
BENCHMARK = "seed_bank_env"

SYSTEM_PROMPT = """You are an expert seed bank manager during a climate crisis.
Distribute seeds to villages and crossbreed varieties to maximize crop yields.

Respond with valid JSON only. Available actions:

1. {"action_type": "distribute", "seed_id": "<id>", "village_id": "<id>"}
2. {"action_type": "crossbreed", "seed_a": "<id>", "seed_b": "<id>"}
3. {"action_type": "rest"}

Strategy:
- High drought_level village -> use high drought_resist seed
- High pest_level village -> use high pest_resist seed
- No good seed available -> crossbreed first
- Respond ONLY with JSON, no explanation.
"""


def require_env(name: str, value: Optional[str]) -> str:
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable: {name}")


client = OpenAI(
    api_key=require_env("HF_TOKEN", API_KEY),
    base_url=API_BASE_URL,
)


# --- Mandatory log functions ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(f"[END] success={success_val} steps={steps} rewards={rewards_str}", flush=True)


def choose_fallback_action(obs: dict) -> dict:
    villages = obs["villages"]
    seeds = obs["available_seeds"]

    def score_seed(seed_id: str, village: dict) -> float:
        drought_weight = village["drought_level"]
        pest_weight = village["pest_level"]
        soil_weight = village["soil_quality"]

        if "wheat_drought" in seed_id:
            return 0.9 * drought_weight + 0.2 * pest_weight + 0.5 * soil_weight
        if "hybrid_b" in seed_id:
            return 0.4 * drought_weight + 0.9 * pest_weight + 0.6 * soil_weight
        if "hybrid_a" in seed_id:
            return 0.7 * drought_weight + 0.8 * pest_weight + 0.6 * soil_weight
        if "millet" in seed_id:
            return 0.8 * drought_weight + 0.5 * pest_weight + 0.5 * soil_weight
        if "ancient_grain" in seed_id:
            return 0.95 * drought_weight + 0.7 * pest_weight + 0.4 * soil_weight
        return soil_weight

    needy_villages = [village for village in villages if village["needs_seed"]]
    if not needy_villages:
        return {"action_type": "rest"}

    available_seed_ids = [seed_id for seed_id, quantity in seeds.items() if quantity > 0]
    best_action = None
    best_score = -1.0
    for village in needy_villages:
        for seed_id in available_seed_ids:
            candidate_score = score_seed(seed_id, village)
            if candidate_score > best_score:
                best_score = candidate_score
                best_action = {
                    "action_type": "distribute",
                    "seed_id": seed_id,
                    "village_id": village["village_id"],
                }

    return best_action or {"action_type": "rest"}


def http_json(method: str, url: str, payload: Optional[dict] = None, timeout: int = 30) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc

    return json.loads(body)


def run_task(task_id: str) -> float:
    log_start(task_id, BENCHMARK, MODEL_NAME)
    env_url = require_env("ENV_URL", ENV_URL)
    reset_url = f"{env_url}/reset?{parse.urlencode({'task_id': task_id})}"
    obs = http_json("POST", reset_url)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards = []
    step = 0

    try:
        for step in range(1, MAX_STEPS + 1):
            user_msg = f"""Season: {obs['season']}/{obs['max_seasons']}
Villages: {json.dumps(obs['villages'])}
Available seeds: {json.dumps(obs['available_seeds'])}
Message: {obs.get('message', '')}
What is your next action?"""

            messages.append({"role": "user", "content": user_msg})

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=150,
                temperature=0.2,
            )
            action_str = response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": action_str})

            error_msg = None
            try:
                action = json.loads(action_str)
            except json.JSONDecodeError:
                action = choose_fallback_action(obs)
                error_msg = "invalid_json"

            step_url = f"{env_url}/step?{parse.urlencode({'task_id': task_id})}"
            result = http_json("POST", step_url, payload=action)
            obs = result["observation"]
            reward = result["reward"]
            done = result["done"]
            rewards.append(reward)

            log_step(step, json.dumps(action), reward, done, error_msg)

            if done:
                break

        state_url = f"{env_url}/state?{parse.urlencode({'task_id': task_id})}"
        final_state = http_json("GET", state_url)
        total = final_state.get("total_reward", sum(rewards))
        success = total >= 0.5

    except Exception:
        log_end(False, step, rewards)
        raise

    log_end(success, step, rewards)
    return sum(rewards)


def main():
    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        scores[task_id] = run_task(task_id)

    print("\n=== FINAL SCORES ===", flush=True)
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.3f}", flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"  average: {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()
