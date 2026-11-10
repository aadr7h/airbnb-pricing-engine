"""
Amsterdam Airbnb Pricing Engine - Streamlit app
Run:  streamlit run app.py
Files needed in same folder: model.json, categories.json, clean.csv
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="Airbnb Pricing Engine", page_icon="🏠", layout="wide")


# ------------------------------------------------------------------ LOAD (cached)
@st.cache_resource
def load_all():
    model = xgb.XGBRegressor(enable_categorical=True)
    model.load_model("model.json")
    cats = json.load(open("categories.json"))
    data = pd.read_csv("clean.csv")
    cols = [c for c in data.columns if c != "price"]
    # har neighbourhood ka center (lat/lon) - user ko coordinates nahi puchne
    centers = data.groupby("neighbourhood_cleansed")[["latitude", "longitude"]].median()
    return model, cats, data, cols, centers, shap.TreeExplainer(model)


model, cats, data, COLS, centers, explainer = load_all()

AMENITIES = {
    "has_air_conditioning": "Air conditioning",
    "has_bathtub": "Bathtub",
    "has_washer": "Washer",
    "has_balcony": "Balcony",
    "has_elevator": "Elevator",
    "has_workspace": "Dedicated workspace",
    "has_free_parking": "Free parking",
    "has_kitchen": "Kitchen",
    "has_wifi": "Wifi",
}


def predict(row: dict) -> float:
    """row dict -> EUR price. Training jaisa hi column order + category dtype."""
    X = pd.DataFrame([row])[COLS]
    for c, vals in cats.items():
        X[c] = pd.Categorical(X[c], categories=vals)
    return float(np.expm1(model.predict(X)[0])), X


# ------------------------------------------------------------------ HEADER
st.title("🏠 Amsterdam Airbnb Pricing Engine")
st.caption(
    "Apni listing ki details daalo: model fair nightly price batayega aur ye bhi ki "
    "kaunsi cheez price ko kitna badha ya ghata rahi hai. "
    "Trained on ~5,900 real Amsterdam listings (Inside Airbnb) with XGBoost + SHAP."
)

# ------------------------------------------------------------------ SIDEBAR INPUTS
with st.sidebar:
    st.header("Listing details")
    hood = st.selectbox("Neighbourhood", cats["neighbourhood_cleansed"],
                        index=cats["neighbourhood_cleansed"].index("Centrum-West")
                        if "Centrum-West" in cats["neighbourhood_cleansed"] else 0)
    room = st.selectbox("Room type", cats["room_type"],
                        index=cats["room_type"].index("Entire home/apt"))
    ptype = st.selectbox("Property type", cats["property_type"],
                         index=cats["property_type"].index("Entire rental unit")
                         if "Entire rental unit" in cats["property_type"] else 0)

    c1, c2 = st.columns(2)
    guests = c1.number_input("Guests", 1, 16, 2)
    bedrooms = c2.number_input("Bedrooms", 0, 10, 1)
    beds = c1.number_input("Beds", 0, 16, 1)
    baths = c2.number_input("Bathrooms", 0.0, 6.0, 1.0, step=0.5)
    shared = st.checkbox("Bathroom shared")

    st.subheader("Amenities")
    chosen = {k: st.checkbox(v, value=k in ("has_wifi", "has_kitchen")) for k, v in AMENITIES.items()}
    n_am = st.slider("Total amenities listed", 0, 100, 30)

    st.subheader("Host & booking")
    superhost = st.checkbox("Superhost")
    min_nights = st.number_input("Minimum nights", 1, 30, 2)
    avail = st.slider("Days available per year", 0, 365, 180)

    st.subheader("Reviews")
    new_listing = st.checkbox("New listing (no reviews yet)")
    rating = st.slider("Rating", 1.0, 5.0, 4.8, 0.05, disabled=new_listing)
    n_rev = st.number_input("Number of reviews", 0, 3000, 0 if new_listing else 40, disabled=new_listing)
    rpm = st.number_input("Reviews per month", 0.0, 30.0, 0.0 if new_listing else 1.0, step=0.1, disabled=new_listing)

# ------------------------------------------------------------------ BUILD ROW
row = {
    "neighbourhood_cleansed": hood,
    "latitude": float(centers.loc[hood, "latitude"]),
    "longitude": float(centers.loc[hood, "longitude"]),
    "room_type": room,
    "property_type": ptype,
    "accommodates": guests,
    "bedrooms": bedrooms,
    "beds": beds,
    "bathrooms": baths,
    "shared_bath": int(shared),
    "n_amenities": n_am,
    "number_of_reviews": 0 if new_listing else n_rev,
    "review_scores_rating": float(data["review_scores_rating"].median()) if new_listing else rating,
    "reviews_per_month": 0.0 if new_listing else rpm,
    "no_reviews": int(new_listing),
    "host_is_superhost": int(superhost),
    "minimum_nights": min_nights,
    "availability_365": avail,
    **{k: int(v) for k, v in chosen.items()},
}
price, X = predict(row)

# ------------------------------------------------------------------ RESULT
left, right = st.columns([1, 1.4])

with left:
    low, high = price * 0.8, price * 1.2          # MAE ~ 74 EUR ~ +-20% around typical price
    st.metric("Estimated nightly price", f"€{price:,.0f}")
    st.write(f"Likely range: **€{low:,.0f} – €{high:,.0f}**")
    st.caption(
        "Model ki average galti ~€74 hai, isliye isse guide ki tarah use karo, exact price ki tarah nahi. "
        "Luxury (€600+) listings ke liye model price kam bata sakta hai."
    )

    area_med = data.loc[data.neighbourhood_cleansed == hood, "price"].median()
    st.write(f"Is area ({hood}) ka median price: **€{area_med:,.0f}**")

# ------------------------------------------------------------------ "LEAVING ON THE TABLE"
with right:
    st.subheader("Kya add karne se price badh sakta hai?")
    st.caption("Same listing me ek amenity add karke model dobara chalaya gaya.")
    gains = []
    for k, label in AMENITIES.items():
        if not chosen[k]:
            alt = {**row, k: 1, "n_amenities": row["n_amenities"] + 1}
            p2, _ = predict(alt)
            gains.append((label, p2 - price))
    gains = sorted(gains, key=lambda t: -t[1])
    if gains and gains[0][1] > 0.5:
        df_g = pd.DataFrame(gains, columns=["Add this", "Price change (€/night)"]).round(1)
        st.dataframe(df_g, hide_index=True, width="stretch")
        st.caption("Ye correlation hai, guarantee nahi: aisi listings average me itni mehngi hoti hain.")
    else:
        st.success("Tumhare paas already saari main amenities hain (ya unka asar bahut chhota hai).")

# ------------------------------------------------------------------ SHAP EXPLANATION
st.divider()
st.subheader("Price ka hisaab: kaunsi cheez ne kitna asar daala")
sv = explainer(X)
base = float(np.expm1(sv.base_values[0]))
contrib = pd.Series(sv.values[0], index=COLS)
eur = base * (np.exp(contrib) - 1)                # log-asar -> approx EUR
top = eur.reindex(eur.abs().sort_values(ascending=False).index).head(10)[::-1]

fig, ax = plt.subplots(figsize=(8, 4.2))
colors = ["#1b9e77" if v > 0 else "#d95f02" for v in top.values]
labels = [f"{c.replace('_', ' ')} = {X[c].iloc[0]}" for c in top.index]
ax.barh(labels, top.values, color=colors)
ax.axvline(0, color="grey", lw=0.8)
ax.set_xlabel(f"€ per night vs average listing (€{base:,.0f})")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
st.pyplot(fig)
st.caption("Hara = price badhaya, narangi = price ghataya. Values approx EUR me hain (SHAP log-price se convert ki gayi).")

with st.expander("Is model ke baare me"):
    st.markdown(
        """
- **Data:** Inside Airbnb, Amsterdam, ~5,900 listings (price missing wali aur top/bottom 1% outliers hata di gayi).
- **Model:** XGBoost on log(price). Test MAE ≈ €74, R² ≈ 0.65.
- **Limitations:** Price ek specific check-in date ka quote hai (season ka asar hai). Photos, view, host quality jaisi cheezein data me nahi hain.
- **Explainability:** SHAP (TreeExplainer).
        """
    )
