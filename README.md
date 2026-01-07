# Monthly Saving Optimizer 💰

A modern, AI-powered financial dashboard application that helps users optimize their monthly budget to achieve savings goals. The application combines machine learning predictions with quadratic programming optimization to provide personalized budget recommendations.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Model Performance](#model-performance)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

---

## 🎯 Overview

The **Monthly Saving Optimizer** is a web-based financial planning tool that:

1. **Predicts Achievable Savings**: Uses a Gradient Boosting Regressor trained on historical financial data to predict realistic savings amounts based on user demographics and spending patterns.

2. **Optimizes Budget**: Employs Quadratic Programming (QP) to find the optimal budget cuts across expense categories while minimizing "pain" (weighted by category importance).

3. **Provides Insights**: Delivers actionable recommendations on which categories to cut first, respecting category constraints and minimum spending floors.

3. **Modern UI**: Features a beautiful, responsive dashboard with light/dark mode support, progress bars, and real-time visualizations.

---

## ✨ Features

### Core Functionality

- **Machine Learning Model**: Gradient Boosting Regressor predicts `Desired_Savings` from user profile and expenses
- **Quadratic Programming Optimizer**: Minimizes weighted cuts to achieve target savings
- **Data-Driven Elasticities**: Learns category "pain weights" from historical `Potential_Savings_*` columns
- **Constraint Handling**: Enforces minimum floors and maximum caps per category
- **Feasibility Detection**: Automatically detects if target is unachievable and returns best possible plan

### User Interface

- **Modern Dashboard**: Card-based design with soft shadows and rounded corners
- **Light/Dark Mode**: Toggle between themes with persistent preference
- **Progress Visualization**: Progress bars showing savings goal completion
- **Category Icons**: Visual icons for all expense categories
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Real-time Metrics**: Displays model performance metrics (MAE, RMSE, R², Accuracy)
- **Spending Analysis Charts**: Interactive visualizations showing spending patterns by occupation and category

### Technical Features

- **Train/Test Split**: 80/20 split with comprehensive metrics
- **Feature Engineering**: Includes demographics, category spends, and `Desired_Savings_Percentage`
- **Gradient Optimization**: Uses SLSQP solver for smooth, efficient optimization
- **Error Handling**: Graceful fallbacks when data is missing or models fail

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                    Flask Web Application                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Routes     │  │  Templates   │  │   Static     │  │
│  │  (app.py)    │  │   (HTML)     │  │  (CSS/JS)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    Core ML Pipeline                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Model      │  │  Optimizer   │  │  Recommender │  │
│  │  Training    │  │   (QP)       │  │   Engine    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    Data Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   CSV Data   │  │  Configs     │  │  Elasticities│  │
│  │  (data.csv)  │  │  (default.py)│  │  (weights)   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Input** → Profile (Income, Age, Dependents, Occupation, City Tier) + Monthly Expenses
2. **Model Prediction** → Gradient Boosting predicts achievable savings amount
3. **Optimization** → QP solver finds optimal cuts across categories
4. **Results** → Budget breakdown, savings achieved, insights, and recommendations

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step-by-Step Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /Users/harshvadodriya/ml-project
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment**:
   ```bash
   # On macOS/Linux:
   source .venv/bin/activate
   
   # On Windows:
   .venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r web/requirements.txt
   ```

   This installs:
   - Flask (web framework)
   - NumPy (numerical computing)
   - Pandas (data manipulation)
   - SciPy (optimization and scientific computing)
   - scikit-learn (machine learning)

5. **Verify installation**:
   ```bash
   python -c "import flask, numpy, pandas, scipy, sklearn; print('All packages installed successfully!')"
   ```

---

## 🚀 Usage

### Starting the Application

1. **Activate virtual environment** (if not already active):
   ```bash
   source .venv/bin/activate
   ```

2. **Run the Flask application**:
   ```bash
   python -m web.app
   ```

3. **Access the application**:
   - Open your browser and navigate to: `http://127.0.0.1:5000`
   - The application will automatically:
     - Load `data/data.csv` if available
     - Train the machine learning model
     - Display model performance metrics

### Using the Dashboard

#### Step 1: Enter Profile Information
- **Income**: Monthly income in INR
- **Age**: Your age
- **Dependents**: Number of dependents
- **Occupation**: Select from dropdown (Student, Professional, Self_Employed, Retired)
- **City Tier**: Select from dropdown (Tier_1, Tier_2, Tier_3)
- **Target Savings**: Desired savings amount in INR

#### Step 2: Enter Monthly Expenses
Fill in your current monthly spending for:
- **Fixed Categories** (non-optimizable):
  - Rent
  - Loan_Repayment
- **Adjustable Categories** (optimizable):
  - Groceries
  - Transport
  - Eating_Out
  - Entertainment
  - Utilities
  - Healthcare
  - Education
  - Miscellaneous

#### Step 3: Optimize
- Click the **"Optimize Budget"** button
- View results including:
  - Achieved savings amount and percentage
  - Category-wise budget breakdown
  - Recommended cuts
  - Binding constraints
  - Model-suggested savings
  - **Spending Analysis Charts**: Three interactive charts showing:
    - **Average Spending by Occupation & Category**: Stacked bar chart comparing spending across occupations
    - **Total Spending by Occupation**: Bar chart showing total monthly spending per occupation
    - **Category Spending Comparison**: Horizontal bar chart comparing average spending across all categories

### Dark Mode

- Click the **moon/sun icon** in the header to toggle between light and dark themes
- Your preference is saved in browser localStorage

---

## 🔧 Technical Details

### Machine Learning Model

#### Model Type
- **Algorithm**: Gradient Boosting Regressor (scikit-learn)
- **Target Variable**: `Desired_Savings` (INR)
- **Task**: Regression

#### Features

**Numeric Features**:
- `Income`: Monthly income
- `Age`: User age
- `Dependents`: Number of dependents
- `Desired_Savings_Percentage`: Target savings percentage
- All expense categories (Rent, Groceries, Transport, etc.)

**Categorical Features**:
- `Occupation`: One-hot encoded (Student, Professional, Self_Employed, Retired)
- `City_Tier`: One-hot encoded (Tier_1, Tier_2, Tier_3)

#### Preprocessing
- **Data Cleaning**:
  - Removes rows with missing target values
  - Handles missing values (median for numeric, mode for categorical)
  - Removes outliers using IQR method (1.5 × IQR)
- **Feature Engineering**:
  - Creates `Spending_Ratio` = Total Spending / Income
  - Standardizes numeric features
  - One-hot encodes categorical features
- **Pipeline**: ColumnTransformer → GradientBoostingRegressor

#### Training Configuration
- **Train/Test Split**: 80/20
- **Random State**: 42 (for reproducibility)
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Parameter Grid**:
  - `n_estimators`: [100, 200, 300]
  - `learning_rate`: [0.05, 0.1, 0.15]
  - `max_depth`: [3, 4, 5]
  - `min_samples_split`: [2, 5, 10]
  - `subsample`: [0.8, 0.9, 1.0]
- **Cross-Validation**: 5-fold KFold with shuffling
- **Model Persistence**: Trained models saved to `models/savings_model.pkl`

#### Performance Metrics

The model reports:
- **MAE** (Mean Absolute Error): Average prediction error in INR
- **RMSE** (Root Mean Squared Error): Penalizes larger errors
- **R²** (Coefficient of Determination): Proportion of variance explained (0-1, higher is better)
- **sMAPE** (Symmetric Mean Absolute Percentage Error): Percentage-based error
- **Accuracy (sMAPE)**: 1 - sMAPE (0-1, higher is better)
- **CV RMSE**: Cross-validation RMSE with mean ± standard deviation
- **Best Parameters**: Optimal hyperparameters from GridSearchCV

**Example Metrics**:
```
Train: MAE 320.77, RMSE 735.20, R² 0.990, Accuracy (sMAPE) 90.5% (n=16000)
Test:  MAE 407.35, RMSE 2136.29, R² 0.937, Accuracy (sMAPE) 89.2% (n=4000)
```

### Optimization Engine

#### Algorithm
- **Method**: Quadratic Programming (QP)
- **Solver**: SLSQP (Sequential Least Squares Programming) from SciPy
- **Objective**: Minimize weighted sum of squared cuts

#### Mathematical Formulation

**Decision Variables**:
- `x_i`: Cut amount for category `i` (adjustable categories only)

**Objective Function**:
```
Minimize: Σ w_i * x_i²
```
Where `w_i` is the "pain weight" for category `i` (higher = more painful to cut)

**Constraints**:
1. **Target Savings**: `Σ x_i ≥ target_amount`
2. **Upper Bounds**: `0 ≤ x_i ≤ min(current_spend_i, cap_i)`
3. **Lower Bounds**: `x_i ≥ 0` (implicit)
4. **Fixed Categories**: `x_i = 0` for Rent and Loan_Repayment

**Post-Processing**:
- Enforces minimum floors: `new_budget_i ≥ floor_fraction * current_spend_i`
- Handles infeasibility: Returns maximum achievable savings if target is too high

#### Elasticities (Pain Weights)

**Data-Driven Approach**:
- Uses `Potential_Savings_*` columns from dataset
- Computes ratio: `Potential_Savings / Current_Spend`
- Higher ratio → easier to cut → lower weight
- Normalizes to range [0.2, 3.0]

**Heuristic Fallback** (if data unavailable):
- Healthcare: 3.0 (highest pain)
- Utilities: 2.5
- Education: 2.2
- Groceries: 2.0
- Transport: 1.8
- Miscellaneous: 1.2
- Entertainment: 1.0
- Eating_Out: 0.9 (lowest pain)

### Category Constraints

#### Fixed Categories
- **Rent**: Cannot be cut (cut = 0)
- **Loan_Repayment**: Cannot be cut (cut = 0)

#### Adjustable Categories

**Minimum Floors** (as % of current spend):
- Groceries: 60%
- Transport: 60%
- Utilities: 70%
- Healthcare: 80%
- Education: 70%
- Eating_Out: 40%
- Entertainment: 40%
- Miscellaneous: 40%

**Maximum Caps** (as % of current spend):
- All categories: 50% maximum cut

---

## 📁 Project Structure

```
ml-project/
├── data/
│   └── data.csv                    # Training dataset (20,000+ rows)
├── models/
│   └── savings_model.pkl           # Saved trained model (auto-generated)
├── src/
│   ├── configs/
│   │   └── default.py              # Category definitions, floors, caps
│   ├── models/
│   │   └── savings_regressor.py    # ML model training and prediction
│   ├── optimizer/
│   │   ├── elasticities.py          # Pain weight derivation
│   │   └── budget_optimizer.py     # QP optimization engine
│   └── recommend.py                # Recommendation assembly
├── web/
│   ├── app.py                      # Flask application and routes
│   ├── requirements.txt            # Python dependencies
│   ├── static/
│   │   ├── style.css               # Modern CSS with dark mode
│   │   └── script.js               # Dark mode toggle
│   └── templates/
│       ├── index.html              # Input form page
│       └── result.html             # Results display page
├── requirements.txt                # Root requirements (if any)
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

### Key Files Explained

#### `src/configs/default.py`
- Defines fixed vs adjustable categories
- Sets minimum floors and maximum caps
- Currency configuration (INR)
- Expected column names

#### `src/models/savings_regressor.py`
- `train_savings_model()`: Trains Gradient Boosting model with hyperparameter tuning
- `_clean_data()`: Data cleaning and outlier removal
- `_build_features()`: Feature engineering (ratios, interactions)
- `SavingsModel`: Model wrapper with prediction and persistence methods
- `load_trained_model()`: Loads saved model from disk
- Returns train/test metrics, CV scores, and best parameters

#### `src/optimizer/elasticities.py`
- `derive_pain_weights()`: Computes category weights from data
- Falls back to heuristics if data unavailable

#### `src/optimizer/budget_optimizer.py`
- `optimize_budget()`: Main QP optimization function
- Returns `OptimizeResult` with cuts, new budget, feasibility

#### `src/recommend.py`
- `recommend_budget()`: Orchestrates model + optimizer
- Assembles final recommendation with insights

#### `web/app.py`
- Flask application factory
- Routes: `/`, `/optimize`, `/metrics`, `/analytics`
- Loads data and trains model on startup
- Generates spending analysis data for charts

#### `web/static/charts.js`
- Chart.js integration for spending visualizations
- Three chart types: stacked bar, bar, and horizontal bar
- Theme-aware (adapts to light/dark mode)
- Auto-updates when theme changes

---

## 🌐 API Endpoints

### GET `/`
- **Description**: Home page with input form
- **Response**: HTML page with form fields
- **Query Parameters**: None

### POST `/optimize`
- **Description**: Optimize budget based on user input
- **Request Body**: Form data with:
  - `Income`, `Age`, `Dependents`, `Occupation`, `City_Tier`
  - `TargetAmountINR`
  - All category spends
- **Response**: HTML page with results

### GET `/metrics`
- **Description**: JSON API for model metrics and dataset info
- **Response**: JSON object
  ```json
  {
    "model_metrics": {
      "train_mae": 320.77,
      "train_rmse": 735.20,
      "train_r2": 0.990,
      "train_accuracy_smape": 0.905,
      "test_mae": 407.35,
      "test_rmse": 2136.29,
      "test_r2": 0.937,
      "test_accuracy_smape": 0.892,
      "n_train": 16000,
      "n_test": 4000
    },
    "dataset": {
      "loaded": true,
      "rows": 20000,
      "columns": ["Income", "Age", ...]
    }
  }
  ```

### GET `/analytics`
- **Description**: JSON API for spending analysis by occupation and category
- **Response**: JSON object with aggregated spending data
  ```json
  {
    "occupation_spending": {
      "Student": {
        "Groceries": 3500.25,
        "Transport": 2000.50,
        ...
      },
      "Professional": {
        "Groceries": 4500.75,
        "Transport": 3500.00,
        ...
      },
      ...
    },
    "occupation_totals": {
      "Student": 25000.00,
      "Professional": 45000.00,
      ...
    },
    "category_totals": {
      "Groceries": 4000.00,
      "Transport": 2750.00,
      ...
    },
    "occupations": ["Professional", "Retired", "Self_Employed", "Student"],
    "categories": ["Groceries", "Transport", ...]
  }
  ```
- **Use Case**: Powers the spending analysis charts on the results page

---

## 📊 Model Performance

### Current Performance (Example)

**Training Set** (n=16,000):
- MAE: 320.77 INR
- RMSE: 735.20 INR
- R²: 0.990
- Accuracy (sMAPE): 90.5%

**Test Set** (n=4,000):
- MAE: 407.35 INR
- RMSE: 2,136.29 INR
- R²: 0.937
- Accuracy (sMAPE): 89.2%

### Interpretation

- **R² = 0.937**: Model explains 93.7% of variance in desired savings
- **MAE = 407 INR**: Average prediction error is ~407 INR
- **RMSE = 2,136 INR**: Larger errors are penalized (outliers affect this)
- **Accuracy = 89.2%**: Model predictions are within ~11% of actual values on average

### Model Strengths

✅ High R² indicates strong predictive power  
✅ Good accuracy (sMAPE) for financial predictions  
✅ Handles categorical features well  
✅ Generalizes reasonably to test set

### Areas for Improvement

⚠️ **Test RMSE >> Train RMSE**: Suggests some overfitting or outliers  
💡 **Recommendations**:
- Hyperparameter tuning (n_estimators, learning_rate, max_depth)
- Feature engineering (interactions, log transforms)
- Outlier handling or robust loss functions
- Cross-validation for better generalization estimates

---

## ⚙️ Configuration

### Category Settings

Edit `src/configs/default.py` to customize:

```python
# Fixed categories (cannot be cut)
FIXED_CATEGORIES = ["Rent", "Loan_Repayment"]

# Adjustable categories
ADJUSTABLE_CATEGORIES = [
    "Groceries", "Transport", "Eating_Out",
    "Entertainment", "Utilities", "Healthcare",
    "Education", "Miscellaneous"
]

# Minimum floors (% of current spend)
FLOOR_FRACTIONS = {
    "Groceries": 0.60,      # Can cut max 40%
    "Healthcare": 0.80,     # Can cut max 20%
    # ... etc
}

# Maximum caps (% of current spend)
CAP_FRACTIONS = {
    "Groceries": 0.50,      # Can cut max 50%
    # ... etc
}
```

### Model Training

Edit `src/models/savings_regressor.py` to customize:
- Train/test split ratio (default: 0.2)
- Random state (default: 42)
- Hyperparameter grid (in `param_grid` dictionary)
- Cross-validation folds (default: 5)
- Enable/disable hyperparameter tuning (`use_hyperparameter_tuning`)

**Training Process**:
1. Data cleaning (missing values, outliers)
2. Feature engineering (ratios, interactions)
3. Train/test split (80/20)
4. Hyperparameter tuning with GridSearchCV (5-fold CV)
5. Final model training with best parameters
6. Cross-validation on training set
7. Model evaluation on test set
8. Model persistence to disk

### UI Customization

Edit `web/static/style.css` to customize:
- Color palette (CSS variables in `:root`)
- Card styles, shadows, borders
- Dark mode colors (`[data-theme="dark"]`)

---

## 🐛 Troubleshooting

### Issue: Model not training / metrics showing 0%

**Possible Causes**:
1. `data/data.csv` not found or empty
2. Missing `Desired_Savings` column
3. All values in target column are zero
4. Too many outliers removed during cleaning

**Solutions**:
- Verify `data/data.csv` exists and has data
- Check that `Desired_Savings` column exists
- Ensure target column has non-zero values
- Check console output for training progress and warnings
- Verify sufficient data remains after cleaning (should be > 100 rows)

### Issue: Training takes too long

**Solutions**:
- Reduce hyperparameter grid size in `param_grid`
- Reduce number of CV folds (e.g., 3 instead of 5)
- Set `use_hyperparameter_tuning=False` for faster training
- Use saved model if available (model loads from disk automatically)

### Issue: Optimization returns 0 savings

**Possible Causes**:
1. All categories are zero or below floors
2. Target savings is too high
3. Fixed categories consume all income

**Solutions**:
- Check input values (some categories must have spending)
- Reduce target savings amount
- Verify fixed categories aren't consuming entire budget

### Issue: Dark mode not working

**Solutions**:
- Clear browser localStorage: `localStorage.clear()`
- Check browser console for JavaScript errors
- Verify `web/static/script.js` is loaded

### Issue: Port 5000 already in use

**Solutions**:
- Change port in `web/app.py`: `app.run(port=5001)`
- Kill process using port: `lsof -ti:5000 | xargs kill`

### Issue: Import errors

**Solutions**:
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r web/requirements.txt --force-reinstall`
- Check Python version: `python --version` (should be 3.10+)

---

## 🔮 Future Improvements

### Model Enhancements

- [ ] **Hyperparameter Tuning**: Grid search or Bayesian optimization
- [ ] **Feature Engineering**: Income × City_Tier interactions, log transforms
- [ ] **Ensemble Methods**: Combine multiple models (Random Forest, XGBoost, LightGBM)
- [ ] **Time Series**: If temporal data available, add time-based features
- [ ] **Outlier Handling**: Robust loss functions or outlier detection

### Optimization Enhancements

- [ ] **Multi-Objective**: Balance savings vs. quality of life
- [ ] **Stochastic Optimization**: Account for uncertainty in predictions
- [ ] **Constraint Learning**: Learn floors/caps from user feedback
- [ ] **What-If Analysis**: Allow users to adjust constraints interactively

### UI/UX Enhancements

- [x] **Interactive Charts**: Chart.js visualizations for spending analysis ✅
- [ ] **Export Functionality**: Download results as PDF/CSV
- [ ] **User Accounts**: Save preferences and optimization history
- [ ] **Mobile App**: React Native or Flutter mobile version
- [ ] **Real-time Validation**: Validate inputs as user types
- [ ] **More Chart Types**: Pie charts, line charts for trends, heatmaps
- [ ] **Interactive Filters**: Filter charts by city tier, age group, etc.

### Technical Improvements

- [x] **Model Persistence**: Save trained models to disk (joblib) ✅
- [x] **Hyperparameter Tuning**: GridSearchCV with cross-validation ✅
- [x] **Data Cleaning**: Outlier removal and missing value handling ✅
- [x] **Feature Engineering**: Spending ratios and interactions ✅
- [ ] **Caching**: Cache model predictions for similar inputs
- [ ] **API Versioning**: RESTful API with versioning
- [ ] **Unit Tests**: Comprehensive test suite (pytest)
- [ ] **CI/CD**: Automated testing and deployment
- [ ] **Docker**: Containerize application for easy deployment
- [ ] **Database**: Replace CSV with PostgreSQL/SQLite

### Business Features

- [ ] **Goal Tracking**: Track progress over time
- [ ] **Notifications**: Remind users of budget goals
- [ ] **Social Features**: Compare with similar users (anonymized)
- [ ] **Financial Advice**: AI-generated tips based on spending patterns

---

## 📝 License

This project is provided as-is for educational and personal use.

## 👤 Author

Developed as part of the ML Project portfolio.
Harsh Vadodariya
Kush Sojitra

## 🙏 Acknowledgments

- **scikit-learn**: Machine learning framework
- **SciPy**: Optimization and scientific computing
- **Flask**: Web framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing

---

## 📊 Spending Analysis Visualizations

The application includes interactive charts that analyze spending patterns across different occupations and categories. These visualizations appear on the results page after optimization.

### Chart Types

1. **Average Spending by Occupation & Category** (Stacked Bar Chart)
   - Shows how different occupations spend across expense categories
   - Stacked bars allow comparison of total spending and category distribution
   - Helps identify which categories are most significant for each occupation

2. **Total Spending by Occupation** (Bar Chart)
   - Compares total monthly spending across all occupations
   - Useful for understanding overall spending differences between professions
   - Highlights which occupations have higher or lower total expenses

3. **Category Spending Comparison** (Horizontal Bar Chart)
   - Shows average spending across all categories (aggregated across all occupations)
   - Helps identify which expense categories are typically the largest
   - Useful for understanding general spending patterns

### Technical Details

- **Library**: Chart.js 4.4.0 (loaded via CDN)
- **Theme Support**: Charts automatically adapt to light/dark mode
- **Data Source**: Aggregated from `data/data.csv` via `/analytics` endpoint
- **Update Frequency**: Charts load on results page load
- **Responsive**: Charts resize for mobile and tablet devices

### Insights from Charts

The visualizations help users:
- **Compare Spending**: See how their spending compares to others in their occupation
- **Identify Patterns**: Understand which categories are typically higher for their profession
- **Make Informed Decisions**: Use data-driven insights when optimizing their budget
- **Benchmark**: Compare their spending against population averages

---

## 📞 Support

For issues, questions, or contributions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review code comments in relevant files
3. Check model metrics at `/metrics` endpoint
4. View spending analysis data at `/analytics` endpoint

---

**Last Updated**: 2025  
**Version**: 1.0.0  
**Status**: Production Ready ✅

