"""
Step 1: Airbnb Amsterdam data - explore, visualise, clean.
Run:  python step1_clean.py
Output: clean.csv + charts in ./charts/
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("charts", exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
BLUE = "#2a6fdb"

# ---------------------------------------------------------------- 1. LOAD
df = pd.read_csv("listings.csv")   # .gz file ho to "listings.csv.gz" likh do
print("Raw shape:", df.shape)

# ---------------------------------------------------------------- 2. KAUNSE COLUMNS CHAHIYE
# 90 columns me se sirf wahi jo price ko affect karte hain (URL, host_name, description etc. bekaar)
keep = [
    "price", "neighbourhood_cleansed", "latitude", "longitude",
    "room_type", "property_type", "accommodates", "bedrooms", "beds",
    "bathrooms_text", "amenities", "number_of_reviews", "review_scores_rating",
    "reviews_per_month", "host_is_superhost", "minimum_nights", "availability_365",
]
df = df[keep].copy()

# ---------------------------------------------------------------- 3. MISSING VALUES DEKHO
missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.barh(missing_pct.index[::-1], missing_pct.values[::-1], color=BLUE)
ax.set_xlabel("% rows missing")
ax.set_title("Missing values per column (raw data)")
for i, v in enumerate(missing_pct.values[::-1]):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
fig.tight_layout(); fig.savefig("charts/1_missing.png"); plt.close(fig)

# ---------------------------------------------------------------- 4. PRICE (TARGET) CLEAN
# "$1,250.00" (string) -> 1250.0 (number)
df["price"] = df["price"].astype(str).str.replace(r"[$,]", "", regex=True)
df["price"] = pd.to_numeric(df["price"], errors="coerce")   # "nan" -> NaN

# Jis listing ka price hi nahi, usse model kuch seekh nahi sakta -> hata do
n_before = len(df)
df = df.dropna(subset=["price"])
print(f"Price missing wali rows hata di: {n_before - len(df)} (bachi: {len(df)})")

# ---------------------------------------------------------------- 5. OUTLIERS
low, high = df["price"].quantile([0.01, 0.99])
print(f"Price 1st pct = {low:.0f}, 99th pct = {high:.0f}, max = {df['price'].max():.0f}")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
axes[0].hist(df["price"], bins=60, color=BLUE)
axes[0].set_title("Price (raw) - long right tail")
axes[0].set_xlabel("EUR / night")
df = df[(df["price"] >= max(low, 20)) & (df["price"] <= high)]
axes[1].hist(df["price"], bins=60, color=BLUE)
axes[1].set_title("Price after outlier removal")
axes[1].set_xlabel("EUR / night")
axes[2].hist(np.log1p(df["price"]), bins=60, color="#1b9e77")
axes[2].set_title("log(price) - almost bell shaped")
axes[2].set_xlabel("log(1 + price)")
fig.tight_layout(); fig.savefig("charts/2_price_distribution.png"); plt.close(fig)

# ---------------------------------------------------------------- 6. BATHROOMS
# "bathrooms" column poori khaali hai; text se number nikalenge
# "1.5 baths" -> 1.5 | "Half-bath" -> 0.5 | "Shared half-bath" -> 0.5
txt = df["bathrooms_text"].fillna("")
num = txt.str.extract(r"(\d+\.?\d*)")[0].astype(float)
is_half = txt.str.contains("half", case=False)
df["bathrooms"] = np.where(num.isna() & is_half, 0.5, num)
df["shared_bath"] = txt.str.contains("shared", case=False).astype(int)
df = df.drop(columns="bathrooms_text")

# ---------------------------------------------------------------- 7. AMENITIES
# amenities ek JSON list-string hoti hai: ["Wifi", "Kitchen", ...]
def parse_amenities(s):
    try:
        return [a.lower() for a in json.loads(s)]
    except Exception:
        return []

am = df["amenities"].apply(parse_amenities)
df["n_amenities"] = am.apply(len)
flags = {
    "wifi": "wifi", "kitchen": "kitchen", "air_conditioning": "air conditioning",
    "washer": "washer", "elevator": "elevator", "bathtub": "bathtub",
    "balcony": "balcony", "free_parking": "free parking", "workspace": "dedicated workspace",
}
for col, key in flags.items():
    df["has_" + col] = am.apply(lambda lst: int(any(key in a for a in lst)))
df = df.drop(columns="amenities")

# ---------------------------------------------------------------- 8. SUPERHOST (t/f -> 1/0)
df["host_is_superhost"] = df["host_is_superhost"].astype(str).str.lower().isin(["t", "true", "1"]).astype(int)

# ---------------------------------------------------------------- 9. MISSING FILL
# bedrooms: studio/room me aksar NaN -> median
df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
# beds: NaN ho to bedrooms ke barabar maan lo (reasonable guess)
df["beds"] = df["beds"].fillna(df["bedrooms"])
df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())
df["minimum_nights"] = df["minimum_nights"].fillna(df["minimum_nights"].median())
# reviews: naye listing ke review hi nahi -> "no_reviews" flag + median score
df["no_reviews"] = df["review_scores_rating"].isna().astype(int)
df["review_scores_rating"] = df["review_scores_rating"].fillna(df["review_scores_rating"].median())
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

# ---------------------------------------------------------------- 10. RARE PROPERTY TYPES
# 60+ property types hain, kai me 2-3 listing -> "Other" me daalo
vc = df["property_type"].value_counts()
rare = vc[vc < 30].index
df["property_type"] = df["property_type"].where(~df["property_type"].isin(rare), "Other")

# ---------------------------------------------------------------- 11. VISUAL EXPLORATION (clean data par)
# (a) room type
order = df.groupby("room_type")["price"].median().sort_values().index
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.boxplot([df.loc[df.room_type == r, "price"] for r in order], tick_labels=order, vert=False, showfliers=False)
ax.set_xlabel("EUR / night"); ax.set_title("Price by room type")
fig.tight_layout(); fig.savefig("charts/3_price_by_room_type.png"); plt.close(fig)

# (b) neighbourhood median
nb = df.groupby("neighbourhood_cleansed")["price"].median().sort_values()
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.barh(nb.index, nb.values, color=BLUE)
ax.set_xlabel("Median EUR / night"); ax.set_title("Median price by neighbourhood")
fig.tight_layout(); fig.savefig("charts/4_price_by_neighbourhood.png"); plt.close(fig)

# (c) map
fig, ax = plt.subplots(figsize=(7, 5.5))
sc = ax.scatter(df.longitude, df.latitude, c=np.log1p(df.price), s=5, cmap="viridis", alpha=0.7)
plt.colorbar(sc, label="log(price)")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title("Listings map (colour = price)")
fig.tight_layout(); fig.savefig("charts/5_map.png"); plt.close(fig)

# (d) accommodates + amenities vs price
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
g = df.groupby(df["accommodates"].clip(upper=8))["price"].median()
axes[0].bar(g.index.astype(str), g.values, color=BLUE)
axes[0].set_title("Median price vs guests (8 = 8+)"); axes[0].set_xlabel("accommodates")
axes[1].scatter(df.n_amenities, df.price, s=4, alpha=0.3, color=BLUE)
axes[1].set_title("Price vs number of amenities"); axes[1].set_xlabel("n_amenities")
fig.tight_layout(); fig.savefig("charts/6_guests_amenities.png"); plt.close(fig)

# (e) correlation with price
num_cols = df.select_dtypes("number").columns
corr = df[num_cols].corr()["price"].drop("price").sort_values()
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(corr.index, corr.values, color=np.where(corr.values > 0, BLUE, "#d95f02"))
ax.set_title("Correlation of each feature with price")
fig.tight_layout(); fig.savefig("charts/7_correlation.png"); plt.close(fig)

# ---------------------------------------------------------------- 12. FINAL CHECK + SAVE
print("\nFinal shape:", df.shape)
print("Missing values left:", int(df.isna().sum().sum()))
print(df.describe().T[["mean", "min", "max"]].round(2).to_string())
df.to_csv("clean.csv", index=False)
print("\nSaved clean.csv")
