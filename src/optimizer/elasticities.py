from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.configs.default import ADJUSTABLE_CATEGORIES, POTENTIAL_SAVINGS_PREFIX


def derive_pain_weights(
    df: pd.DataFrame,
    adjustable_categories: List[str] | None = None,
    min_weight: float = 0.2,
    max_weight: float = 3.0,
) -> Dict[str, float]:
    """Derive per-category pain weights from data.

    Uses correlation between category spend and Potential_Savings_* as a proxy for
    ease of savings. Higher potential savings relative to spend implies lower pain.

    If data is missing or weak, falls back to a heuristic ordering.
    """
    categories = adjustable_categories or ADJUSTABLE_CATEGORIES
    weights: Dict[str, float] = {}

    # Try data-driven estimate
    signals: Dict[str, float] = {}
    for cat in categories:
        ps_col = f"{POTENTIAL_SAVINGS_PREFIX}{cat}"
        if cat in df.columns and ps_col in df.columns:
            x = df[cat].astype(float)
            y = df[ps_col].astype(float)
            denom = np.maximum(x.values, 1e-6)
            ratio = (y.values / denom).clip(min=0.0, max=2.0)
            # Use mean ratio as signal of ease to cut (higher ratio => easier => lower weight)
            signals[cat] = float(np.nanmean(ratio))
        else:
            signals[cat] = np.nan

    if any(np.isfinite(list(signals.values()))):
        # Normalize signals to [0,1]
        vals = np.array([signals[c] if np.isfinite(signals[c]) else 0.0 for c in categories])
        if np.nanmax(vals) - np.nanmin(vals) < 1e-8:
            norm = np.ones_like(vals) * 0.5
        else:
            norm = (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals) + 1e-12)
        # Map to weights: higher signal -> lower weight
        inv = 1.0 - norm
        scaled = min_weight + inv * (max_weight - min_weight)
        for i, cat in enumerate(categories):
            weights[cat] = float(scaled[i])
    else:
        # Heuristic fallback: essentials have higher pain
        heuristic_order = {
            "Healthcare": 3.0,
            "Utilities": 2.5,
            "Education": 2.2,
            "Groceries": 2.0,
            "Transport": 1.8,
            "Miscellaneous": 1.2,
            "Entertainment": 1.0,
            "Eating_Out": 0.9,
        }
        for cat in categories:
            w = heuristic_order.get(cat, 1.5)
            weights[cat] = float(np.clip(w, min_weight, max_weight))

    return weights


