from __future__ import annotations

import os
from typing import Dict

import pandas as pd
from flask import Flask, render_template, request, jsonify

from src.configs.default import (
    ADJUSTABLE_CATEGORIES,
    FIXED_CATEGORIES,
    EXPECTED_META_COLUMNS,
    CURRENCY,
)
from src.recommend import recommend_budget
from src.models.savings_regressor import train_savings_model, SavingsModel


def create_app() -> Flask:
    app = Flask(__name__)

    # Load dataset for weight derivation (optional)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    default_data_path = os.path.join(project_root, "data", "data.csv")
    df = pd.read_csv(default_data_path) if os.path.exists(default_data_path) else pd.DataFrame()

    # Train model if possible
    model: SavingsModel | None = None
    model_metrics = None
    try:
        if not df.empty:
            # Try to load saved model first
            from src.models.savings_regressor import load_trained_model
            model = load_trained_model()
            
            # If no saved model, train new one
            if model is None:
                print("No saved model found. Training new model...")
                model = train_savings_model(
                    df,
                    use_hyperparameter_tuning=True,
                    cv_folds=5,
                    save_model=True,
                )
            
            if model is not None:
                model_metrics = model.metrics
                print("Model loaded/trained successfully!")
    except Exception as e:
        print(f"Error in model training/loading: {e}")
        import traceback
        traceback.print_exc()
        model = None
        model_metrics = None

    @app.route("/", methods=["GET"])
    def index():
        # Infer enums from data if available
        occupations = sorted(df["Occupation"].dropna().unique().tolist()) if "Occupation" in df else ["Student", "Professional", "Self_Employed", "Retired"]
        city_tiers = sorted(df["City_Tier"].dropna().unique().tolist()) if "City_Tier" in df else ["Tier_1", "Tier_2", "Tier_3"]
        return render_template(
            "index.html",
            adjustable=ADJUSTABLE_CATEGORIES,
            fixed=FIXED_CATEGORIES,
            occupations=occupations,
            city_tiers=city_tiers,
            currency=CURRENCY,
            model_metrics=model_metrics,
        )

    @app.route("/optimize", methods=["POST"])
    def optimize():
        form = request.form
        # Meta
        meta: Dict[str, object] = {
            "Income": _to_float(form.get("Income", "0")),
            "Age": int(_to_float(form.get("Age", "0"))),
            "Dependents": int(_to_float(form.get("Dependents", "0"))),
            "Occupation": form.get("Occupation", ""),
            "City_Tier": form.get("City_Tier", ""),
        }
        target_inr = _to_float(form.get("TargetAmountINR", "0"))

        # Spends
        spends: Dict[str, float] = {}
        for cat in list(ADJUSTABLE_CATEGORIES) + list(FIXED_CATEGORIES):
            spends[cat] = _to_float(form.get(cat, "0"))

        result = recommend_budget(meta, spends, target_inr, df_for_weights=df)

        # Model suggested achievable savings (optional)
        model_suggestion = None
        if 'Income' in meta and model is not None:
            try:
                model_suggestion = model.predict_amount(meta, spends)
                # Debug: Log prediction to console
                print(f"Model prediction for Income={meta.get('Income')}, Occupation={meta.get('Occupation')}: {model_suggestion:.2f} INR")
            except Exception as e:
                print(f"Error in model prediction: {e}")
                model_suggestion = None

        if model_suggestion is not None:
            result["model_suggested_savings_inr"] = float(model_suggestion)

        return render_template(
            "result.html",
            result=result,
            adjustable=ADJUSTABLE_CATEGORIES,
            fixed=FIXED_CATEGORIES,
            currency=CURRENCY,
            model_metrics=model_metrics,
        )

    @app.route("/metrics", methods=["GET"])
    def metrics():
        payload = {
            "model_metrics": model_metrics or {},
            "dataset": {
                "loaded": bool(len(df) > 0),
                "rows": int(len(df)) if len(df) > 0 else 0,
                "columns": sorted(df.columns.tolist()) if len(df) > 0 else [],
            },
        }
        return jsonify(payload)

    @app.route("/analytics", methods=["GET"])
    def analytics():
        """Generate spending analysis by occupation and category"""
        if df.empty or "Occupation" not in df.columns:
            return jsonify({"error": "Dataset not available"}), 404

        all_categories = list(ADJUSTABLE_CATEGORIES) + list(FIXED_CATEGORIES)
        
        # Aggregate spending by occupation and category
        occupation_spending = {}
        for occ in df["Occupation"].dropna().unique():
            occ_data = df[df["Occupation"] == occ]
            category_avg = {}
            for cat in all_categories:
                if cat in occ_data.columns:
                    category_avg[cat] = float(occ_data[cat].mean())
            occupation_spending[occ] = category_avg

        # Calculate total spending per occupation
        occupation_totals = {}
        for occ, cats in occupation_spending.items():
            occupation_totals[occ] = float(sum(cats.values()))

        # Calculate category averages across all occupations
        category_totals = {}
        for cat in all_categories:
            if cat in df.columns:
                category_totals[cat] = float(df[cat].mean())

        return jsonify({
            "occupation_spending": occupation_spending,
            "occupation_totals": occupation_totals,
            "category_totals": category_totals,
            "occupations": sorted(df["Occupation"].dropna().unique().tolist()),
            "categories": all_categories,
        })

    return app


def _to_float(x: str) -> float:
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return 0.0


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)


