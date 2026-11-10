"""
Step 3: SHAP - model ke andar jhaankna
Run:  python step3_shap.py   (step2 ke baad)
Output: charts/10_shap_importance.png, 11_shap_beeswarm.png, 12_shap_single.png, shap_summary.csv
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

df = pd.read_csv("clean.csv")
cats = json.load(open("categories.json"))
for c, vals in cats.items():
    df[c] = pd.Categorical(df[c], categories=vals)

X = df.drop(columns="price")
model = xgb.XGBRegressor(enable_categorical=True)
model.load_model("model.json")

# ---- SHAP values: har listing, har feature ka contribution (log-price me)
explainer = shap.TreeExplainer(model)
sv = explainer(X)                      # sv.values shape = (rows, features)
base_price = np.expm1(sv.base_values[0])
print(f"Base price (average listing): EUR {base_price:.0f}")

# ---- Chart 10: global importance (mean |SHAP|)
imp = pd.Series(np.abs(sv.values).mean(axis=0), index=X.columns).sort_values()
top = imp.tail(15)
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.barh(top.index, top.values, color="#2a6fdb")
ax.set_xlabel("Mean |SHAP| (log-price impact)")
ax.set_title("Kaunsa feature price pe sabse zyada asar daalta hai")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("charts/10_shap_importance.png", dpi=110); plt.close(fig)

# ---- Chart 11: beeswarm (direction bhi dikhata hai)
plt.figure()
shap.plots.beeswarm(sv, max_display=15, show=False)
plt.title("Feature value ka price pe asar (red = high value)")
plt.tight_layout(); plt.savefig("charts/11_shap_beeswarm.png", dpi=110, bbox_inches="tight"); plt.close()

# ---- Chart 12: ek listing ka explanation (waterfall)
i = int((df["price"] - df["price"].median()).abs().idxmin())    # median price wali listing
plt.figure()
shap.plots.waterfall(sv[i], max_display=10, show=False)
plt.tight_layout(); plt.savefig("charts/12_shap_single.png", dpi=110, bbox_inches="tight"); plt.close()

# ---- Euro me translate: feature ka average EUR asar (approx)
# log-space SHAP ko seedha EUR me nahi badal sakte, lekin approx: base_price * (exp(shap) - 1)
eur = pd.DataFrame(base_price * (np.exp(sv.values) - 1), columns=X.columns)
summary = pd.DataFrame({
    "mean_abs_eur_impact": eur.abs().mean().round(1),
}).sort_values("mean_abs_eur_impact", ascending=False)
summary.to_csv("shap_summary.csv")
print("\nTop 10 features (average EUR impact per listing):")
print(summary.head(10).to_string())

# ---- Kuch aam sawaalon ke jawab: ek feature badalne ka asar
def effect(col, mask_true, mask_false):
    return eur.loc[mask_true, col].mean() - eur.loc[mask_false, col].mean()

print("\nBathtub hone par avg EUR asar :", round(eur.loc[df.has_bathtub == 1, "has_bathtub"].mean(), 1))
print("AC hone par avg EUR asar      :", round(eur.loc[df.has_air_conditioning == 1, "has_air_conditioning"].mean(), 1))
print("Superhost hone par avg EUR asar:", round(eur.loc[df.host_is_superhost == 1, "host_is_superhost"].mean(), 1))
print("Washer hone par avg EUR asar  :", round(eur.loc[df.has_washer == 1, "has_washer"].mean(), 1))
