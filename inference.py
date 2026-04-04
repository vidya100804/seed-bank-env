"""
Seed Bank Curator — Baseline Inference Script
Follows mandatory stdout format: [START], [STEP], [END]
"""

import os
import json
import requests
from typing import Optional
from openai import OpenAI

# --- Config (read from env vars) ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:8000")
MAX_STEPS    = 10
BENCHMARK    = "seed_bank_env"

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

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

# --- Mandatory log functions ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(f"[END] success={success_val} steps={steps} rewards={rewards_str}", flush=True)


def run_task(task_id: str) -> float:
    log_start(task_id, BENCHMARK, MODEL_NAME)

    # Reset environment
    resp = requests.post(f"{ENV_URL}/reset?task_id={task_id}")
    obs  = resp.json()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards  = []
    step     = 0
    done     = False

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
                temperature=0.2
            )
            action_str = response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": action_str})

            error_msg = None
            try:
                action = json.loads(action_str)
            except json.JSONDecodeError:
                action    = {"action_type": "rest"}
                error_msg = "invalid_json"

            step_resp = requests.post(
                f"{ENV_URL}/step?task_id={task_id}",
                json=action
            )
            result = step_resp.json()
            obs    = result["observation"]
            reward = result["reward"]
            done   = result["done"]
            rewards.append(reward)

            log_step(step, json.dumps(action), reward, done, error_msg)

            if done:
                break

        state_resp  = requests.get(f"{ENV_URL}/state?task_id={task_id}")
        final_state = state_resp.json()
        total       = final_state.get("total_reward", sum(rewards))
        success     = total >= 0.5

    except Exception as e:
        log_end(False, step, rewards)
        raise e

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
