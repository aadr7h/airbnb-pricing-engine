"""
Step 2: Train XGBoost price model on clean.csv
Run:  python step2_model.py
Output: model.json, categories.json, charts/8_actual_vs_pred.png, charts/9_baseline_vs_xgb.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("clean.csv")

# ---- categorical columns: XGBoost ko "category" dtype chahiye
cat_cols = ["neighbourhood_cleansed", "room_type", "property_type"]
cats = {c: sorted(df[c].unique().tolist()) for c in cat_cols}
for c in cat_cols:
    df[c] = pd.Categorical(df[c], categories=cats[c])
json.dump(cats, open("categories.json", "w"))   # Streamlit app ko yahi lists chahiye

# ---- X (inputs) aur y (jo predict karna hai)
X = df.drop(columns="price")
y = np.log1p(df["price"])            # log price: lambi dum ka problem kam

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def report(name, pred_log):
    pred, actual = np.expm1(pred_log), np.expm1(y_test)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    r2_log = r2_score(y_test, pred_log)
    print(f"{name:<22} MAE = EUR {mae:6.1f} | R2(price) = {r2:.3f} | R2(log price) = {r2_log:.3f}")
    return mae, r2_log

# ---- Baselines (XGBoost ko inse behtar hona chahiye, warna fayda kya)
Xd_train = pd.get_dummies(X_train, columns=cat_cols).astype(float)
Xd_test = pd.get_dummies(X_test, columns=cat_cols).astype(float).reindex(columns=Xd_train.columns, fill_value=0)
mu, sd = Xd_train.mean(), Xd_train.std().replace(0, 1)

print("\n--- Results on unseen 20% test data ---")
print(f"{'Dumb guess (median)':<22} MAE = EUR {mean_absolute_error(np.expm1(y_test), np.full(len(y_test), np.expm1(y_train.median()))):6.1f}")
m_ridge = Ridge(alpha=5).fit((Xd_train - mu) / sd, y_train)
mae_r, r2_r = report("Linear (Ridge)", m_ridge.predict((Xd_test - mu) / sd))
m_rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42).fit(Xd_train, y_train)
mae_f, r2_f = report("Random Forest", m_rf.predict(Xd_test))

# ---- XGBoost
model = xgb.XGBRegressor(
    n_estimators=2000, learning_rate=0.03, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_lambda=2.0, enable_categorical=True, tree_method="hist",
    early_stopping_rounds=50, random_state=42,
)
# early stopping ke liye train ka 15% alag validation rakha (test ko haath nahi lagaya)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=1)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print(f"(XGBoost ne {model.best_iteration} trees pe rukna theek samjha)")
pred_log = model.predict(X_test)
mae_x, r2_x = report("XGBoost", pred_log)

model.save_model("model.json")

# ---- Chart 1: actual vs predicted
pred, actual = np.expm1(pred_log), np.expm1(y_test)
fig, ax = plt.subplots(figsize=(6, 5.5))
ax.scatter(actual, pred, s=6, alpha=0.4, color="#2a6fdb")
lim = [0, 1200]
ax.plot(lim, lim, "r--", lw=1, label="perfect prediction")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Actual price (EUR)"); ax.set_ylabel("Predicted price (EUR)")
ax.set_title("XGBoost: actual vs predicted (test set)")
ax.legend(); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("charts/8_actual_vs_pred.png", dpi=110); plt.close(fig)

# ---- Chart 2: model comparison
fig, ax = plt.subplots(figsize=(6, 3.3))
names, maes = ["Ridge", "Random Forest", "XGBoost"], [mae_r, mae_f, mae_x]
ax.barh(names, maes, color=["#bbbbbb", "#bbbbbb", "#2a6fdb"])
for i, v in enumerate(maes):
    ax.text(v + 1, i, f"EUR {v:.0f}", va="center")
ax.set_xlabel("Mean absolute error (lower = better)")
ax.set_title("Model comparison")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("charts/9_baseline_vs_xgb.png", dpi=110); plt.close(fig)
print("\nSaved model.json, categories.json, charts 8 & 9")
