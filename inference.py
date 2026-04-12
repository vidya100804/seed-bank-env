"""
Seed Bank Curator - AI Inference Script
Mandatory stdout format: [START], [STEP], [END]
ALL rewards strictly between 0.0 and 1.0 (exclusive)
"""

import json
import os
from urllib import error, parse, request
from typing import Optional

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
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
- No good seed -> crossbreed first
Respond ONLY with JSON, no explanation."""


def clamp(value) -> float:
    """Strictly clamp to (0.01, 0.99) — never 0.0 or 1.0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 0.5
    return round(min(0.99, max(0.01, v)), 4)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, err: Optional[str]) -> None:
    r = clamp(reward)
    print(f"[STEP] step={step} action={action} reward={r:.2f} done={str(done).lower()} error={err or 'null'}", flush=True)


def log_end(success: bool, steps: int, rewards: list) -> None:
    safe = [clamp(r) for r in rewards]
    rewards_str = ",".join(f"{r:.2f}" for r in safe)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def can_reach(url: str, timeout: int = 5) -> bool:
    for path in ["/health", "/"]:
        try:
            req = request.Request(f"{url.rstrip('/')}{path}", method="GET")
            with request.urlopen(req, timeout=timeout) as r:
                if 200 <= r.status < 400:
                    return True
        except Exception:
            continue
    return False


def resolve_env_url() -> str:
    candidates = [
        os.getenv("ENV_URL"),
        os.getenv("OPENENV_URL"),
        "http://127.0.0.1:7860",
        "http://localhost:7860",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]
    for c in candidates:
        if c and can_reach(c):
            return c.rstrip("/")
    raise RuntimeError("Cannot reach environment. Set ENV_URL.")


def http_json(method: str, url: str, payload=None, timeout: int = 60) -> dict:
    data, headers = None, {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')}") from e
    except error.URLError as e:
        raise RuntimeError(f"URL error: {e.reason}") from e


def call_llm(messages: list) -> str:
    key = API_KEY
    if not key:
        raise RuntimeError("Missing HF_TOKEN / API_KEY")
    payload = {"model": MODEL_NAME, "messages": messages, "max_tokens": 200, "temperature": 0.2}
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    req = request.Request(f"{API_BASE_URL.rstrip('/')}/chat/completions",
                          data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()
    except error.HTTPError as e:
        raise RuntimeError(f"LLM HTTP {e.code}: {e.read().decode(errors='replace')}") from e


def fallback_action(obs: dict) -> dict:
    """Rule-based fallback when LLM fails."""
    villages = obs.get("villages", [])
    seeds = obs.get("available_seeds", {})
    needy = [v for v in villages if v.get("needs_seed")]
    if not needy:
        return {"action_type": "rest"}
    available = [s for s, q in seeds.items() if q > 0]
    if not available:
        return {"action_type": "rest"}
    v = needy[0]
    # Pick seed based on village stress
    best, best_s = available[0], -1.0
    for s in available:
        score = 0.5
        if "drought" in s:
            score = v.get("drought_level", 0.5)
        elif "hybrid" in s:
            score = (v.get("drought_level", 0) + v.get("pest_level", 0)) / 2
        if score > best_s:
            best_s, best = score, s
    return {"action_type": "distribute", "seed_id": best, "village_id": v["village_id"]}


def run_task(task_id: str, env_url: str) -> float:
    log_start(task_id, BENCHMARK, MODEL_NAME)

    obs = http_json("POST", f"{env_url}/reset?task_id={task_id}")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards = []
    step = 0

    try:
        for step in range(1, MAX_STEPS + 1):
            user_msg = (
                f"Season: {obs['season']}/{obs['max_seasons']}\n"
                f"Villages: {json.dumps(obs['villages'])}\n"
                f"Seeds: {json.dumps(obs['available_seeds'])}\n"
                f"Message: {obs.get('message', '')}\n"
                f"What is your next action?"
            )
            messages.append({"role": "user", "content": user_msg})

            err_msg = None
            try:
                raw = call_llm(messages)
                messages.append({"role": "assistant", "content": raw})
                if "```" in raw:
                    raw = raw.split("```")[-2] if raw.count("```") >= 2 else raw.split("```")[-1]
                    raw = raw.replace("json", "").strip()
                action = json.loads(raw.strip())
            except Exception:
                action = fallback_action(obs)
                err_msg = "parse_error"

            result = http_json("POST", f"{env_url}/step?task_id={task_id}", payload=action)
            obs = result["observation"]
            raw_reward = result.get("reward", 0.5)
            reward = clamp(raw_reward)   # ← ALWAYS clamp
            done = result.get("done", False)
            rewards.append(reward)

            log_step(step, json.dumps(action), reward, done, err_msg)

            if done:
                break

        # Get final score
        state = http_json("GET", f"{env_url}/state?task_id={task_id}")
        total = clamp(state.get("total_reward", sum(rewards) / max(len(rewards), 1)))
        success = total >= 0.5

    except Exception:
        log_end(False, step, rewards if rewards else [0.5])
        raise

    log_end(success, step, rewards)
    return total


def main():
    env_url = resolve_env_url()
    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        scores[task_id] = run_task(task_id, env_url)

    print("\n=== FINAL SCORES ===", flush=True)
    for t, s in scores.items():
        print(f"  {t}: {s:.3f}", flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"  average: {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()
