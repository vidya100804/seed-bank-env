import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from models import SeedBankAction
from environment.env import SeedBankEnv

app = FastAPI(title="Seed Bank Curator - OpenEnv")

# One env per task (simple single-session setup)
envs = {
    "easy":   SeedBankEnv("easy"),
    "medium": SeedBankEnv("medium"),
    "hard":   SeedBankEnv("hard"),
}


@app.get("/")
def root():
    return {"name": "seed_bank_env", "version": "1.0.0",
            "tasks": ["easy", "medium", "hard"]}


@app.post("/reset")
def reset(task_id: str = "easy"):
    if task_id not in envs:
        return JSONResponse({"error": f"Unknown task: {task_id}"}, status_code=400)
    obs = envs[task_id].reset()
    return obs.dict()


@app.post("/step")
def step(action: SeedBankAction, task_id: str = "easy"):
    if task_id not in envs:
        return JSONResponse({"error": f"Unknown task: {task_id}"}, status_code=400)
    obs, reward, done, info = envs[task_id].step(action)
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state")
def state(task_id: str = "easy"):
    if task_id not in envs:
        return JSONResponse({"error": f"Unknown task: {task_id}"}, status_code=400)
    return envs[task_id].state().dict()


@app.get("/health")
def health():
    return {"status": "ok"}
