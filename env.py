import copy
from typing import Dict, Tuple, Any
from models import SeedBankAction, SeedBankObservation, SeedBankState, VillageInfo
from tasks import TASKS, SEEDS, compute_yield, crossbreed, grade_task


class SeedBankEnv:
    def __init__(self, task_id: str = "easy"):
        self.task_id = task_id
        self.task = TASKS[task_id]
        self.season = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.villages = []
        self.available_seeds = {}
        self.village_yields: Dict[str, float] = {}
        self.custom_seeds: Dict[str, dict] = {}

    def reset(self) -> SeedBankObservation:
        self.season = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.village_yields = {}
        self.custom_seeds = {}
        self.villages = [
            VillageInfo(**v) for v in self.task["villages"]
        ]
        self.available_seeds = copy.deepcopy(self.task["available_seeds"])
        return self._get_observation("Environment reset. Choose your action.")

    def step(self, action: SeedBankAction) -> Tuple[SeedBankObservation, float, bool, dict]:
        if self.done:
            return self._get_observation("Episode done."), 0.0, True, {}

        reward = 0.0
        message = ""
        self.step_count += 1

        # --- DISTRIBUTE ---
        if action.action_type == "distribute":
            seed_id = action.seed_id
            village_id = action.village_id

            all_seeds = {**self.available_seeds, **{k: 1 for k in self.custom_seeds}}
            village = self._get_village(village_id)

            if not seed_id or seed_id not in all_seeds or all_seeds.get(seed_id, 0) == 0:
                reward = -0.1
                message = f"Invalid seed {seed_id}."
            elif not village:
                reward = -0.1
                message = f"Invalid village {village_id}."
            else:
                # Use the seed
                if seed_id in self.available_seeds:
                    self.available_seeds[seed_id] -= 1

                # Compute yield
                seed_stats = self.custom_seeds.get(seed_id) or SEEDS.get(seed_id, {})
                yield_val = compute_yield(seed_id, village.__dict__)
                self.village_yields[village_id] = yield_val

                # Update village
                village.current_yield = yield_val
                village.needs_seed = False

                # Reward based on yield
                reward = yield_val * 0.5
                if yield_val >= self.task["target_yield"]:
                    reward += 0.3
                message = f"Distributed {seed_id} to {village_id}. Yield: {yield_val}"

        # --- CROSSBREED ---
        elif action.action_type == "crossbreed":
            seed_a = action.seed_a
            seed_b = action.seed_b
            all_seeds = {**self.available_seeds, **{k: 1 for k in self.custom_seeds}}

            if not seed_a or not seed_b:
                reward = -0.1
                message = "Provide seed_a and seed_b."
            elif seed_a not in all_seeds or seed_b not in all_seeds:
                reward = -0.1
                message = "One or both seeds not available."
            else:
                hybrid = crossbreed(seed_a, seed_b)
                new_id = f"hybrid_{seed_a[:3]}_{seed_b[:3]}"
                self.custom_seeds[new_id] = hybrid
                reward = 0.1
                message = f"Created {new_id}: {hybrid}"

        # --- REST (skip turn) ---
        elif action.action_type == "rest":
            self.season += 1
            reward = -0.05
            message = f"Season advanced to {self.season}."

            # Spread disease each season in hard task
            if self.task_id == "hard":
                for v in self.villages:
                    v.pest_level = min(v.pest_level + 0.1, 1.0)

        else:
            reward = -0.1
            message = f"Unknown action: {action.action_type}"

        self.total_reward += reward

        # Check done
        all_served = all(not v.needs_seed for v in self.villages)
        max_seasons_reached = self.season >= self.task["max_seasons"]
        if all_served or max_seasons_reached:
            self.done = True
            # Final grading bonus
            final_score = grade_task(self.task_id, self.village_yields)
            self.total_reward += final_score
            message += f" | Final score: {final_score}"

        return self._get_observation(message), reward, self.done, {
            "step": self.step_count,
            "total_reward": self.total_reward
        }

    def state(self) -> SeedBankState:
        return SeedBankState(
            task_id=self.task_id,
            season=self.season,
            step_count=self.step_count,
            total_reward=round(self.total_reward, 3),
            done=self.done,
            villages=self.villages,
            available_seeds=self.available_seeds
        )

    def _get_village(self, village_id: str):
        for v in self.villages:
            if v.village_id == village_id:
                return v
        return None

    def _get_observation(self, message: str) -> SeedBankObservation:
        return SeedBankObservation(
            season=self.season,
            max_seasons=self.task["max_seasons"],
            villages=self.villages,
            available_seeds=self.available_seeds,
            task_id=self.task_id,
            message=message
        )
