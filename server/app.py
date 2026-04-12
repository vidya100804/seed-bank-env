import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from models import SeedBankAction
from environment.env import SeedBankEnv
from tasks import grade_task

app = FastAPI(title="Seed Bank Curator - OpenEnv")

# One env per task (simple single-session setup)
envs = {
    "easy":   SeedBankEnv("easy"),
    "medium": SeedBankEnv("medium"),
    "hard":   SeedBankEnv("hard"),
}


def _strict_unit(value: float) -> float:
    return round(min(0.99, max(0.01, value)), 3)


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


@app.get("/grade")
def grade(task_id: str = "easy"):
    if task_id not in envs:
        return JSONResponse({"error": f"Unknown task: {task_id}"}, status_code=400)
    score = grade_task(task_id, envs[task_id].village_yields)
    return {"task_id": task_id, "score": _strict_unit(score)}


@app.get("/health")
def health():
    return {"status": "ok"}


def main():
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
