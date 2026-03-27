
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

