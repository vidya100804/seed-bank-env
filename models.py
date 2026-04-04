from pydantic import BaseModel
from typing import Optional, List, Dict

# --- ACTION ---
class SeedBankAction(BaseModel):
    action_type: str  # "distribute", "crossbreed", "rest"
    seed_id: Optional[str] = None        # which seed to use
    village_id: Optional[str] = None     # where to send it
    seed_a: Optional[str] = None         # for crossbreeding
    seed_b: Optional[str] = None         # for crossbreeding

# --- OBSERVATION ---
class VillageInfo(BaseModel):
    village_id: str
    drought_level: float      # 0.0 to 1.0
    pest_level: float         # 0.0 to 1.0
    soil_quality: float       # 0.0 to 1.0
    current_yield: float      # 0.0 to 1.0
    needs_seed: bool

class SeedBankObservation(BaseModel):
    season: int
    max_seasons: int
    villages: List[VillageInfo]
    available_seeds: Dict[str, int]   # seed_id -> quantity
    task_id: str
    message: str

# --- STATE ---
class SeedBankState(BaseModel):
    task_id: str
    season: int
    step_count: int
    total_reward: float
    done: bool
    villages: List[VillageInfo]
    available_seeds: Dict[str, int]
