from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

from src.configs.default import (
    ADJUSTABLE_CATEGORIES,
    FIXED_CATEGORIES,
    FLOOR_FRACTIONS,
    CAP_FRACTIONS,
)


@dataclass
class OptimizeResult:
    cuts: Dict[str, float]
    new_budget: Dict[str, float]
    achieved_savings: float
    requested_savings: float
    feasible: bool
    capacity: float
    binding: List[str]


def _bounds_for_category(cat: str, current: float) -> float:
    floor_frac = FLOOR_FRACTIONS.get(cat, 0.0)
    cap_frac = CAP_FRACTIONS.get(cat, 1.0)
    floor_cut = max(0.0, current * (1.0 - floor_frac))
    cap_cut = max(0.0, current * cap_frac)
    return float(min(floor_cut, cap_cut))


def optimize_budget(
    current_spend: Dict[str, float],
    target_amount: float,
    weights: Dict[str, float],
    adjustable_categories: List[str] | None = None,
    fixed_categories: List[str] | None = None,
) -> OptimizeResult:
    cats = adjustable_categories or ADJUSTABLE_CATEGORIES
    fixed = set(fixed_categories or FIXED_CATEGORIES)

    # Build vectors
    x0: List[float] = []
    ub: List[float] = []
    w: List[float] = []
    var_cats: List[str] = []

    for cat in cats:
        spend = float(current_spend.get(cat, 0.0))
        if spend <= 0.0:
            continue
        upper = _bounds_for_category(cat, spend)
        if upper <= 1e-9:
            continue
        x0.append(min(upper * 0.2, upper))
        ub.append(upper)
        w.append(float(max(1e-3, weights.get(cat, 1.0))))
        var_cats.append(cat)

    # Fixed categories set to zero cut implicitly
    capacity = float(np.sum(ub)) if ub else 0.0
    requested = float(max(0.0, target_amount))
    feasible = capacity + 1e-6 >= requested
    target = requested if feasible else capacity

    if capacity <= 1e-9:
        # Nothing to cut
        cuts = {cat: 0.0 for cat in current_spend.keys()}
        new_budget = {cat: float(current_spend.get(cat, 0.0)) for cat in current_spend.keys()}
        return OptimizeResult(cuts, new_budget, 0.0, requested, False, capacity, [])

    # Objective: sum w_i * x_i^2
    def obj(x: np.ndarray) -> float:
        return float(np.sum(np.array(w) * x * x))

    # Gradient (optional for SLSQP)
    def grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * np.array(w) * x

    # Constraint: sum(x) >= target  => -sum(x) <= -target
    cons = [
        {
            "type": "ineq",
            "fun": lambda x: float(np.sum(x) - target),
            "jac": lambda x: np.ones_like(x),
        }
    ]

    bounds = [(0.0, ub_i) for ub_i in ub]

    res = minimize(
        obj,
        x0=np.array(x0, dtype=float),
        method="SLSQP",
        jac=grad,
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-9, "disp": False},
    )

    x_sol = res.x if res.success else np.minimum(np.array(x0), np.array(ub))
    # Project to bounds and enforce target greedily if needed
    x_sol = np.clip(x_sol, 0.0, np.array(ub))
    gap = float(target - float(np.sum(x_sol)))
    if gap > 1e-6:
        # Distribute remaining cut to cheapest (lowest weight / remaining capacity)
        remaining = np.array(ub) - x_sol
        cost = np.array(w)
        order = np.argsort(cost)  # cheapest first
        for idx in order:
            if gap <= 1e-9:
                break
            add = float(min(remaining[idx], gap))
            if add > 0:
                x_sol[idx] += add
                gap -= add

    achieved = float(np.sum(x_sol))
    binding = []
    for i, cat in enumerate(var_cats):
        if abs(x_sol[i] - ub[i]) <= 1e-6:
            binding.append(cat)

    cuts: Dict[str, float] = {cat: 0.0 for cat in current_spend.keys()}
    for i, cat in enumerate(var_cats):
        cuts[cat] = float(max(0.0, x_sol[i]))

    # Fixed categories explicitly zero cut
    for cat in fixed:
        if cat in current_spend:
            cuts[cat] = 0.0

    new_budget: Dict[str, float] = {}
    for cat, spend in current_spend.items():
        new_budget[cat] = float(max(0.0, float(spend) - cuts.get(cat, 0.0)))

    return OptimizeResult(
        cuts=cuts,
        new_budget=new_budget,
        achieved_savings=achieved,
        requested_savings=requested,
        feasible=feasible and achieved + 1e-6 >= requested,
        capacity=capacity,
        binding=binding,
    )


