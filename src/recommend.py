from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.configs.default import (
    ADJUSTABLE_CATEGORIES,
    FIXED_CATEGORIES,
    CURRENCY,
)
from src.optimizer.elasticities import derive_pain_weights
from src.optimizer.budget_optimizer import optimize_budget


def recommend_budget(
    meta: Dict[str, object],
    spends: Dict[str, float],
    target_amount_inr: float,
    df_for_weights: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    # Sanitize inputs
    current_spend: Dict[str, float] = {}
    for cat in list(ADJUSTABLE_CATEGORIES) + list(FIXED_CATEGORIES):
        val = float(spends.get(cat, 0.0) or 0.0)
        current_spend[cat] = max(0.0, val)

    # Derive pain weights
    if df_for_weights is not None and not df_for_weights.empty:
        weights = derive_pain_weights(df_for_weights)
    else:
        weights = derive_pain_weights(pd.DataFrame())

    # Optimize
    opt = optimize_budget(
        current_spend=current_spend,
        target_amount=float(max(0.0, target_amount_inr)),
        weights=weights,
    )

    total_current = float(sum(current_spend.values()))
    achieved_pct = float(0.0 if total_current <= 1e-9 else (opt.achieved_savings / total_current))

    # Reason codes: top-3 cheapest categories (lowest weight) and any binding caps
    sorted_weights = sorted(((cat, weights.get(cat, 1.0)) for cat in ADJUSTABLE_CATEGORIES), key=lambda t: t[1])
    cheapest = [c for c, _ in sorted_weights[:3]]
    reasons = {
        "cheapest_categories": cheapest,
        "binding_constraints": opt.binding,
        "feasible": opt.feasible,
    }

    return {
        "currency": CURRENCY,
        "meta": meta,
        "current_spend": current_spend,
        "cuts": opt.cuts,
        "new_budget": opt.new_budget,
        "achieved_savings_inr": opt.achieved_savings,
        "requested_savings_inr": opt.requested_savings,
        "achieved_savings_pct": achieved_pct,
        "capacity_inr": opt.capacity,
        "feasible": opt.feasible,
        "reasons": reasons,
    }


