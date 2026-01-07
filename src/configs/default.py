from __future__ import annotations

# Core category policy and app-wide constants for the monthly saving optimizer.

# Currency label (display only)
CURRENCY = "INR"

# Categories considered fixed (no cuts allowed)
FIXED_CATEGORIES = [
    "Rent",
    "Loan_Repayment",
]

# Categories eligible for optimization (cuts)
ADJUSTABLE_CATEGORIES = [
    "Groceries",
    "Transport",
    "Eating_Out",
    "Entertainment",
    "Utilities",
    "Healthcare",
    "Education",
    "Miscellaneous",
]

# Per-category minimum floor as a fraction of current spend (0-1)
FLOOR_FRACTIONS = {
    "Groceries": 0.60,
    "Transport": 0.60,
    "Eating_Out": 0.40,
    "Entertainment": 0.40,
    "Utilities": 0.70,
    "Healthcare": 0.80,
    "Education": 0.70,
    "Miscellaneous": 0.40,
}

# Per-category maximum cut as a fraction of current spend (0-1)
CAP_FRACTIONS = {
    "Groceries": 0.50,
    "Transport": 0.50,
    "Eating_Out": 0.50,
    "Entertainment": 0.50,
    "Utilities": 0.50,
    "Healthcare": 0.50,
    "Education": 0.50,
    "Miscellaneous": 0.50,
}

# Columns expected from the dataset for demographics / metadata
EXPECTED_META_COLUMNS = [
    "Income",
    "Age",
    "Dependents",
    "Occupation",
    "City_Tier",
]

# Potential savings columns prefix used for elasticities learning
POTENTIAL_SAVINGS_PREFIX = "Potential_Savings_"


