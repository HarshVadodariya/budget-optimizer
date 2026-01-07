from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    KFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from src.configs.default import (
    ADJUSTABLE_CATEGORIES,
    FIXED_CATEGORIES,
    EXPECTED_META_COLUMNS,
)


TARGET_COLUMN = "Desired_Savings"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


@dataclass
class SavingsModel:
    pipeline: Pipeline
    feature_columns: List[str]
    categorical_columns: List[str]
    numeric_columns: List[str]
    metrics: Dict[str, float]
    best_params: Dict[str, any]

    def predict_amount(self, meta: Dict[str, object], spends: Dict[str, float]) -> float:
        row: Dict[str, object] = {}
        for c in EXPECTED_META_COLUMNS:
            row[c] = meta.get(c)
        for c in list(ADJUSTABLE_CATEGORIES) + list(FIXED_CATEGORIES):
            row[c] = float(spends.get(c, 0.0) or 0.0)
        X = pd.DataFrame([row], columns=self.feature_columns)
        return float(max(0.0, self.pipeline.predict(X)[0]))

    def save(self, filepath: str) -> None:
        """Save model to disk"""
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> "SavingsModel":
        """Load model from disk"""
        return joblib.load(filepath)


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess data"""
    df = df.copy()
    
    # Remove rows with missing target
    if TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[TARGET_COLUMN])
    
    # Handle missing values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != TARGET_COLUMN:
            df[col] = df[col].fillna(df[col].median())
    
    # Handle missing values in categorical columns
    categorical_cols = ["Occupation", "City_Tier"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown")
    
    # Remove outliers (using IQR method for target)
    if TARGET_COLUMN in df.columns:
        Q1 = df[TARGET_COLUMN].quantile(0.25)
        Q3 = df[TARGET_COLUMN].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[TARGET_COLUMN] >= lower_bound) & (df[TARGET_COLUMN] <= upper_bound)]
    
    return df


def _build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Build features with engineering"""
    cats = ["Occupation", "City_Tier"]
    nums = [
        "Income",
        "Age",
        "Dependents",
        "Desired_Savings_Percentage",
    ] + list(ADJUSTABLE_CATEGORIES) + list(FIXED_CATEGORIES)
    
    # Feature engineering
    df = df.copy()
    
    # Ensure all columns exist
    for col in nums:
        if col not in df.columns:
            df[col] = 0.0
    for col in cats:
        if col not in df.columns:
            df[col] = "Unknown"
    
    # Create interaction features
    if "Income" in df.columns and "City_Tier" in df.columns:
        # Income normalized by city tier (implicit via one-hot encoding)
        pass
    
    # Create ratio features
    if "Income" in df.columns:
        total_spend = sum([df[cat].fillna(0) for cat in nums if cat not in ["Income", "Age", "Dependents", "Desired_Savings_Percentage"]])
        df["Spending_Ratio"] = total_spend / (df["Income"] + 1e-6)
        nums.append("Spending_Ratio")
    
    # Log transform for highly skewed features (optional)
    # if "Income" in df.columns:
    #     df["Income_Log"] = np.log1p(df["Income"])
    #     nums.append("Income_Log")
    
    feature_cols = nums + cats
    X = df[feature_cols].copy()
    
    return X, nums, cats


def train_savings_model(
    df: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.2,
    use_hyperparameter_tuning: bool = True,
    cv_folds: int = 5,
    save_model: bool = True,
) -> Optional[SavingsModel]:
    """
    Train savings prediction model with proper hyperparameter tuning and validation.
    
    Args:
        df: Training dataframe
        random_state: Random seed for reproducibility
        test_size: Fraction of data for testing
        use_hyperparameter_tuning: Whether to use GridSearchCV
        cv_folds: Number of cross-validation folds
        save_model: Whether to save model to disk
    
    Returns:
        SavingsModel object with trained pipeline and metrics
    """
    if TARGET_COLUMN not in df.columns:
        print(f"Warning: {TARGET_COLUMN} column not found in dataset")
        return None
    
    if df.empty:
        print("Warning: Empty dataset")
        return None
    
    print(f"Training model on {len(df)} samples...")
    
    # Clean data
    df_clean = _clean_data(df)
    print(f"After cleaning: {len(df_clean)} samples")
    
    if len(df_clean) < 100:
        print("Warning: Insufficient data after cleaning")
        return None
    
    # Build features
    y = df_clean[TARGET_COLUMN].astype(float).values
    X, nums, cats = _build_features(df_clean)
    
    # Preprocessing pipeline
    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True), nums),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
        ],
        remainder="drop"
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Hyperparameter grid
    if use_hyperparameter_tuning:
        param_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.05, 0.1, 0.15],
            "model__max_depth": [3, 4, 5],
            "model__min_samples_split": [2, 5, 10],
            "model__subsample": [0.8, 0.9, 1.0],
        }
        
        base_gbr = GradientBoostingRegressor(
            random_state=random_state,
            loss="squared_error",
            max_features="sqrt",
        )
        
        pipe = Pipeline(steps=[
            ("pre", preproc),
            ("model", base_gbr),
        ])
        
        # Grid search with cross-validation
        print("Performing hyperparameter tuning with cross-validation...")
        grid_search = GridSearchCV(
            pipe,
            param_grid,
            cv=KFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1,
        )
        
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_
        print(f"Best CV Score (RMSE): {np.sqrt(best_score):.2f}")
        print(f"Best Parameters: {best_params}")
        
        pipe = grid_search.best_estimator_
    else:
        # Use default parameters with some optimization
        gbr = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=5,
            subsample=0.9,
            random_state=random_state,
            loss="squared_error",
            max_features="sqrt",
        )
        
        pipe = Pipeline(steps=[
            ("pre", preproc),
            ("model", gbr),
        ])
        
        pipe.fit(X_train, y_train)
        best_params = {}
    
    # Cross-validation on full training set
    print("Performing cross-validation...")
    cv_scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    cv_rmse = np.sqrt(-cv_scores)
    print(f"CV RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std() * 2:.2f})")
    
    # Predictions
    yhat_train = pipe.predict(X_train)
    yhat_test = pipe.predict(X_test)
    
    # Metrics
    def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-6
        denom = np.maximum(np.abs(y_true), eps)
        return float(np.mean(np.abs(y_true - y_pred) / denom))
    
    def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-6
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        denom = np.maximum(denom, eps)
        return float(np.mean(np.abs(y_true - y_pred) / denom))
    
    mape_train = _mape(y_train, yhat_train)
    mape_test = _mape(y_test, yhat_test)
    acc_train = float(np.clip(1.0 - mape_train, 0.0, 1.0))
    acc_test = float(np.clip(1.0 - mape_test, 0.0, 1.0))
    
    smape_train = _smape(y_train, yhat_train)
    smape_test = _smape(y_test, yhat_test)
    acc_smape_train = float(np.clip(1.0 - smape_train, 0.0, 1.0))
    acc_smape_test = float(np.clip(1.0 - smape_test, 0.0, 1.0))
    
    metrics = {
        "train_mae": float(mean_absolute_error(y_train, yhat_train)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, yhat_train))),
        "train_r2": float(r2_score(y_train, yhat_train)),
        "train_mape": mape_train,
        "train_accuracy": acc_train,
        "train_smape": smape_train,
        "train_accuracy_smape": acc_smape_train,
        "test_mae": float(mean_absolute_error(y_test, yhat_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, yhat_test))),
        "test_r2": float(r2_score(y_test, yhat_test)),
        "test_mape": mape_test,
        "test_accuracy": acc_test,
        "test_smape": smape_test,
        "test_accuracy_smape": acc_smape_test,
        "cv_rmse_mean": float(cv_rmse.mean()),
        "cv_rmse_std": float(cv_rmse.std()),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(len(nums) + len(cats)),
    }
    
    print("\n=== Training Results ===")
    print(f"Train R²: {metrics['train_r2']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    print(f"Train MAE: {metrics['train_mae']:.2f} INR")
    print(f"Test MAE: {metrics['test_mae']:.2f} INR")
    print(f"Train Accuracy (sMAPE): {metrics['train_accuracy_smape']*100:.2f}%")
    print(f"Test Accuracy (sMAPE): {metrics['test_accuracy_smape']*100:.2f}%")
    print(f"CV RMSE: {metrics['cv_rmse_mean']:.2f} ± {metrics['cv_rmse_std']:.2f} INR")
    
    model = SavingsModel(
        pipeline=pipe,
        feature_columns=list(X.columns),
        categorical_columns=cats,
        numeric_columns=nums,
        metrics=metrics,
        best_params=best_params,
    )
    
    # Save model
    if save_model:
        model_path = os.path.join(MODEL_DIR, "savings_model.pkl")
        model.save(model_path)
        print(f"\nModel saved to: {model_path}")
    
    return model


def load_trained_model() -> Optional[SavingsModel]:
    """Load pre-trained model from disk"""
    model_path = os.path.join(MODEL_DIR, "savings_model.pkl")
    if os.path.exists(model_path):
        try:
            return SavingsModel.load(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None
