"""
House Price Predictor — Streamlit App
Author: Sebastián Mayorga Castro
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #F0F4F8; }
    .metric-box {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .price-display {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a56db;
    }
    .range-display { color: #6b7280; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "best_model.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


model_bundle = load_model()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Input form
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🏠 House Details")
    st.caption("Fill in the property characteristics below")

    st.subheader("📐 Size & Layout")
    gr_liv_area   = st.slider("Above-ground Living Area (sq ft)", 500, 5000, 1500, 50)
    total_bsmt_sf = st.slider("Basement Area (sq ft)", 0, 3000, 800, 50)
    garage_area   = st.slider("Garage Area (sq ft)", 0, 1500, 400, 25)
    lot_area      = st.slider("Lot Area (sq ft)", 1000, 50000, 8000, 500)

    st.subheader("🛏 Rooms")
    full_bath   = st.selectbox("Full Bathrooms", [1, 2, 3, 4], index=1)
    half_bath   = st.selectbox("Half Bathrooms", [0, 1, 2], index=0)
    bedrooms    = st.selectbox("Bedrooms above ground", [1, 2, 3, 4, 5, 6], index=2)
    totrms_abv  = st.selectbox("Total Rooms above ground", [3, 4, 5, 6, 7, 8, 9, 10], index=3)

    st.subheader("🏗 Condition & Age")
    year_built   = st.slider("Year Built", 1870, 2010, 1990)
    year_remod   = st.slider("Year Last Remodeled", 1950, 2010, 2000)
    overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 6)
    overall_cond = st.slider("Overall Condition (1–10)", 1, 10, 5)

    st.subheader("🌟 Extras")
    has_pool       = st.checkbox("Has Pool")
    has_fireplace  = st.checkbox("Has Fireplace")
    has_second_fl  = st.checkbox("Has 2nd Floor")

    predict_btn = st.button("🔮 Predict Price", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

st.title("🏠 House Price Predictor")
st.markdown("*Powered by Machine Learning — Ames Housing Dataset (Kaggle)*")

st.divider()

# ── Feature calculation ───────────────────────────────────────────────────────
yr_sold         = 2010
house_age       = yr_sold - year_built
years_remod     = yr_sold - year_remod
total_sf        = gr_liv_area + total_bsmt_sf
total_baths     = full_bath + 0.5 * half_bath

# Build a row matching the expected feature set
# (these are the ~30 most influential features from the Ames dataset)
input_data = {
    "OverallQual":     overall_qual,
    "GrLivArea":       gr_liv_area,
    "TotalSF":         total_sf,
    "GarageCars":      min(round(garage_area / 200), 4),
    "GarageArea":      garage_area,
    "TotalBsmtSF":     total_bsmt_sf,
    "1stFlrSF":        gr_liv_area if not has_second_fl else int(gr_liv_area * 0.6),
    "FullBath":        full_bath,
    "TotalBaths":      total_baths,
    "TotRmsAbvGrd":    totrms_abv,
    "YearBuilt":       year_built,
    "YearRemodAdd":    year_remod,
    "HouseAge":        house_age,
    "YearsSinceRemod": years_remod,
    "LotArea":         lot_area,
    "BedroomAbvGr":    bedrooms,
    "OverallCond":     overall_cond,
    "HasGarage":       int(garage_area > 0),
    "HasPool":         int(has_pool),
    "HasSecondFloor":  int(has_second_fl),
    "Fireplaces":      int(has_fireplace),
    "MasVnrArea":      0,
    "WoodDeckSF":      0,
    "OpenPorchSF":     0,
    "2ndFlrSF":        int(gr_liv_area * 0.4) if has_second_fl else 0,
    "HalfBath":        half_bath,
    "BsmtFullBath":    1 if total_bsmt_sf > 0 else 0,
    "BsmtHalfBath":    0,
    "GarageYrBlt":     year_built,
    "LotFrontage":     70,
}

input_df = pd.DataFrame([input_data])

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn or True:   # show demo even without click
    col1, col2, col3 = st.columns([2, 1, 1])

    if model_bundle is not None:
        model  = model_bundle["model"]
        # Align columns with training data
        try:
            model_cols = model.feature_names_in_ if hasattr(model, "feature_names_in_") else input_df.columns
            X_input    = input_df.reindex(columns=model_cols, fill_value=0)
            prediction = model.predict(X_input)[0]
        except Exception:
            prediction = None
    else:
        # Demo mode — heuristic estimate when no model is trained yet
        prediction = (
            overall_qual * 20_000
            + gr_liv_area * 85
            + total_bsmt_sf * 35
            + garage_area * 55
            - house_age * 400
            + int(has_pool) * 8_000
            + int(has_second_fl) * 15_000
            - 30_000
        )
        prediction = max(50_000, prediction)

    if prediction is not None:
        low  = prediction * 0.92
        high = prediction * 1.08

        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div style="color:#6b7280;font-size:0.9rem;margin-bottom:6px">ESTIMATED PRICE</div>
                <div class="price-display">${prediction:,.0f}</div>
                <div class="range-display">Range: ${low:,.0f} — ${high:,.0f}</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.metric("Total Living Area", f"{total_sf:,} sq ft")
            st.metric("House Age", f"{house_age} years")

        with col3:
            st.metric("Overall Quality", f"{overall_qual}/10")
            st.metric("Total Bathrooms", f"{total_baths:.1f}")

        st.divider()

        # ── Key feature breakdown ─────────────────────────────────────────────
        st.subheader("📊 What's driving this estimate?")

        impacts = {
            "Overall Quality":        overall_qual * 20_000,
            "Above-ground Area":      gr_liv_area  * 85,
            "Basement":               total_bsmt_sf * 35,
            "Garage":                 garage_area   * 55,
            "Age penalty":           -house_age * 400,
            "Pool bonus":             int(has_pool) * 8_000,
            "2nd Floor":              int(has_second_fl) * 15_000,
        }

        imp_df = (pd.DataFrame.from_dict(impacts, orient="index", columns=["Impact ($)"])
                   .sort_values("Impact ($)", ascending=True))

        import plotly.express as px
        fig = px.bar(
            imp_df.reset_index(), x="Impact ($)", y="index",
            orientation="h",
            color="Impact ($)",
            color_continuous_scale=["#C0504D", "#FFFFFF", "#4F81BD"],
            color_continuous_midpoint=0,
            labels={"index": "Feature"},
            height=350,
        )
        fig.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            plot_bgcolor="white",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Could not generate prediction. Please train the model first by running `pipeline.py`.")

# ── Info footer ────────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this model"):
    st.markdown("""
**Dataset**: Ames Housing Dataset from Kaggle ([House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques))

**Pipeline**:
1. EDA — distribution analysis, missing values, correlations
2. Feature Engineering — 7 new features (HouseAge, TotalSF, TotalBaths, HasGarage, HasPool, HasSecondFloor, YearsSinceRemod)
3. Modelling — Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost
4. Evaluation — RMSE, MAE, R², residual plots, feature importance

**Tech Stack**: Python · pandas · scikit-learn · XGBoost · Streamlit · Plotly

*Built by Sebastián Mayorga Castro as part of a DS/ML portfolio project.*
    """)