from typing import Dict, Any

# Seed catalog
SEEDS = {
    "wheat_drought": {"drought_resist": 0.9, "pest_resist": 0.3, "yield": 0.7},
    "wheat_normal":  {"drought_resist": 0.4, "pest_resist": 0.4, "yield": 0.9},
    "millet":        {"drought_resist": 0.8, "pest_resist": 0.5, "yield": 0.6},
    "hybrid_a":      {"drought_resist": 0.7, "pest_resist": 0.8, "yield": 0.7},
    "hybrid_b":      {"drought_resist": 0.6, "pest_resist": 0.9, "yield": 0.8},
    "ancient_grain": {"drought_resist": 0.95, "pest_resist": 0.7, "yield": 0.5},
}

TASKS = {
    # EASY: 1 village, drought crisis, pick right seed
    "easy": {
        "description": "One village facing severe drought. Pick the best seed.",
        "max_seasons": 1,
        "villages": [
            {
                "village_id": "v1",
                "drought_level": 0.9,
                "pest_level": 0.1,
                "soil_quality": 0.7,
                "current_yield": 0.2,
                "needs_seed": True
            }
        ],
        "available_seeds": {
            "wheat_drought": 5,
            "wheat_normal": 5,
            "millet": 5
        },
        "target_yield": 0.6
    },

    # MEDIUM: 2 villages, pest outbreak, crossbreed needed
    "medium": {
        "description": "Two villages with drought AND pest outbreak. Crossbreed for resistance.",
        "max_seasons": 2,
        "villages": [
            {
                "village_id": "v1",
                "drought_level": 0.7,
                "pest_level": 0.8,
                "soil_quality": 0.6,
                "current_yield": 0.2,
                "needs_seed": True
            },
            {
                "village_id": "v2",
                "drought_level": 0.6,
                "pest_level": 0.7,
                "soil_quality": 0.5,
                "current_yield": 0.1,
                "needs_seed": True
            }
        ],
        "available_seeds": {
            "wheat_drought": 3,
            "millet": 3,
            "hybrid_a": 2,
            "hybrid_b": 2
        },
        "target_yield": 0.6
    },

    # HARD: 5 villages, disease spreading, multi-season plan
    "hard": {
        "description": "5 villages, disease spreading each season. Plan carefully.",
        "max_seasons": 3,
        "villages": [
            {"village_id": "v1", "drought_level": 0.8, "pest_level": 0.9,
             "soil_quality": 0.5, "current_yield": 0.1, "needs_seed": True},
            {"village_id": "v2", "drought_level": 0.7, "pest_level": 0.6,
             "soil_quality": 0.6, "current_yield": 0.2, "needs_seed": True},
            {"village_id": "v3", "drought_level": 0.9, "pest_level": 0.5,
             "soil_quality": 0.4, "current_yield": 0.1, "needs_seed": True},
            {"village_id": "v4", "drought_level": 0.6, "pest_level": 0.8,
             "soil_quality": 0.7, "current_yield": 0.3, "needs_seed": False},
            {"village_id": "v5", "drought_level": 0.5, "pest_level": 0.7,
             "soil_quality": 0.8, "current_yield": 0.2, "needs_seed": True},
        ],
        "available_seeds": {
            "wheat_drought": 3,
            "millet": 2,
            "hybrid_a": 2,
            "hybrid_b": 1,
            "ancient_grain": 2
        },
        "target_yield": 0.6
    }
}


def compute_yield(seed_id: str, village: Dict) -> float:
    """Simulate crop yield based on seed + village conditions."""
    if seed_id not in SEEDS:
        return 0.0
    seed = SEEDS[seed_id]
    drought_score = seed["drought_resist"] * (1 - village["drought_level"])
    pest_score    = seed["pest_resist"]    * (1 - village["pest_level"])
    base_yield    = seed["yield"] * village["soil_quality"]
    final_yield   = (drought_score + pest_score + base_yield) / 3.0
    return round(min(final_yield, 1.0), 3)


def crossbreed(seed_a: str, seed_b: str) -> Dict:
    """Average traits of two seeds to produce a hybrid."""
    if seed_a not in SEEDS or seed_b not in SEEDS:
        return {}
    a = SEEDS[seed_a]
    b = SEEDS[seed_b]
    return {
        "drought_resist": round((a["drought_resist"] + b["drought_resist"]) / 2, 2),
        "pest_resist":    round((a["pest_resist"]    + b["pest_resist"])    / 2, 2),
        "yield":          round((a["yield"]          + b["yield"])          / 2, 2),
    }


def grade_task(task_id: str, village_yields: Dict[str, float]) -> float:
    """Score 0.0-1.0 based on how many villages hit target yield."""
    target = TASKS[task_id]["target_yield"]
    total  = len(village_yields)
    if total == 0:
        return 0.0
    passed = sum(1 for y in village_yields.values() if y >= target)
    return round(passed / total, 3)
