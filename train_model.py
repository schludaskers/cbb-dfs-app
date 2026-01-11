import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("Generating synthetic historical data...")

# 1. Create Dummy Training Data
# In real life, you would load a CSV of last year's game logs
n_samples = 1000
data = {
    'Last_5_Avg_FP': np.random.uniform(10, 50, n_samples),
    'Opp_Def_Rank': np.random.randint(1, 363, n_samples),
    'Is_Home': np.random.randint(0, 2, n_samples),
    'Minutes_Avg': np.random.uniform(15, 38, n_samples),
}

df = pd.DataFrame(data)

# Target: Actual Fantasy Points (correlated with Last 5 Avg + noise)
# The logic: Better recent form (Last 5) + Worse Opponent (Higher Rank #) = More Points
df['Actual_FP'] = (df['Last_5_Avg_FP'] * 0.8) + (df['Opp_Def_Rank'] * 0.05) + np.random.normal(0, 5, n_samples)

# 2. Prepare Features and Target
X = df[['Last_5_Avg_FP', 'Opp_Def_Rank', 'Is_Home', 'Minutes_Avg']]
y = df['Actual_FP']

# 3. Train XGBoost
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train, y_train)

# 4. Evaluate
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Model Trained. RMSE: {rmse:.2f}")

# 5. Save Model
model.save_model("model.json")
print("Model saved to model.json")