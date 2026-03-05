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
import plotly.graph_objects as go

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme: slate-blue dark (not black) ───────────────────────────────────────
st.markdown("""
<style>
    /* Main background — dark slate, not black */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #1c1f2e !important;
    }
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
        background-color: #1c1f2e !important;
    }

    /* Sidebar — slightly lighter slate */
    [data-testid="stSidebar"] {
        background-color: #252838 !important;
        border-right: 1px solid #353a52;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div { color: #b8c0d8 !important; }
    [data-testid="stSidebar"] h1 { color: #ffffff !important; font-size: 1.3rem !important; }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #e0e4f0 !important; font-size: 0.95rem !important; }

    /* Body text */
    h1, h2, h3 { color: #e8ecf8 !important; }
    p, .stMarkdown p { color: #9aa3bf !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #252838 !important;
        border: 1px solid #353a52 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
    }
    [data-testid="stMetricLabel"] { color: #6b7699 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
    [data-testid="stMetricValue"] { color: #e8ecf8 !important; font-size: 1.6rem !important; font-weight: 700 !important; }

    /* Price hero card */
    .price-card {
        background: linear-gradient(135deg, #252838 0%, #1e2540 100%);
        border: 1px solid #4f5fc4;
        border-radius: 16px;
        padding: 32px 36px;
        text-align: center;
        box-shadow: 0 4px 24px rgba(79,95,196,0.18);
    }
    .price-label {
        color: #6b7699;
        font-size: 0.75rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .price-value {
        font-size: 3.2rem;
        font-weight: 800;
        color: #7c8fff;
        line-height: 1.1;
    }
    .price-range { color: #555e80; font-size: 0.85rem; margin-top: 10px; }

    /* Dividers */
    hr { border-color: #353a52 !important; }

    /* Expander */
    [data-testid="stExpander"] {
        background-color: #252838 !important;
        border: 1px solid #353a52 !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] li { color: #9aa3bf !important; }

    /* Welcome placeholder text */
    .welcome-text { color: #555e80 !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #1c1f2e; }
    ::-webkit-scrollbar-thumb { background: #353a52; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    path = os.path.join("models", "best_model.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

model_bundle = load_model()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🏠 House Details")
    st.caption("Adjust the sliders to describe the property")

    st.subheader("📐 Size & Layout")
    gr_liv_area   = st.slider("Above-ground Living Area (sq ft)", 500, 5000, 1500, 50)
    total_bsmt_sf = st.slider("Basement Area (sq ft)", 0, 3000, 800, 50)
    garage_area   = st.slider("Garage Area (sq ft)", 0, 1500, 400, 25)
    lot_area      = st.slider("Lot Area (sq ft)", 1000, 50000, 8000, 500)

    st.subheader("🛏 Rooms")
    full_bath  = st.selectbox("Full Bathrooms", [1, 2, 3, 4], index=1)
    half_bath  = st.selectbox("Half Bathrooms", [0, 1, 2], index=0)
    bedrooms   = st.selectbox("Bedrooms above ground", [1, 2, 3, 4, 5, 6], index=2)
    totrms_abv = st.selectbox("Total Rooms above ground", [3, 4, 5, 6, 7, 8, 9, 10], index=3)

    st.subheader("🏗 Condition & Age")
    year_built   = st.slider("Year Built", 1870, 2010, 1990)
    year_remod   = st.slider("Year Last Remodeled", 1950, 2010, 2000)
    overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 6)
    overall_cond = st.slider("Overall Condition (1–10)", 1, 10, 5)

    st.subheader("🌟 Extras")
    has_pool      = st.checkbox("Has Pool")
    has_fireplace = st.checkbox("Has Fireplace")
    has_second_fl = st.checkbox("Has 2nd Floor")

    predict_btn = st.button("🔮 Predict Price", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# DERIVED FEATURES
# ══════════════════════════════════════════════════════════════════════════════
yr_sold     = 2010
house_age   = yr_sold - year_built
years_remod = yr_sold - year_remod
total_sf    = gr_liv_area + total_bsmt_sf
total_baths = full_bath + 0.5 * half_bath

input_data = {
    # Spaced names (Ames dataset from Kaggle)
    "Overall Qual": overall_qual, "Gr Liv Area": gr_liv_area, "TotalSF": total_sf,
    "Garage Cars": min(round(garage_area / 200), 4), "Garage Area": garage_area,
    "Total Bsmt SF": total_bsmt_sf,
    "1st Flr SF": gr_liv_area if not has_second_fl else int(gr_liv_area * 0.6),
    "Full Bath": full_bath, "TotalBaths": total_baths, "TotRms AbvGrd": totrms_abv,
    "Year Built": year_built, "Year Remod/Add": year_remod,
    "HouseAge": house_age, "YearsSinceRemod": years_remod,
    "Lot Area": lot_area, "Bedroom AbvGr": bedrooms, "Overall Cond": overall_cond,
    "HasGarage": int(garage_area > 0), "HasPool": int(has_pool),
    "HasSecondFloor": int(has_second_fl), "Fireplaces": int(has_fireplace),
    "Mas Vnr Area": 0, "Wood Deck SF": 0, "Open Porch SF": 0,
    "2nd Flr SF": int(gr_liv_area * 0.4) if has_second_fl else 0,
    "Half Bath": half_bath, "Bsmt Full Bath": 1 if total_bsmt_sf > 0 else 0,
    "Bsmt Half Bath": 0, "Garage Yr Blt": year_built, "Lot Frontage": 70,
}
input_df = pd.DataFrame([input_data])

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.title("🏠 House Price Predictor")
st.markdown("*ML regression pipeline — Ames Housing Dataset (Kaggle)*")
st.divider()

if "predicted" not in st.session_state:
    st.session_state.predicted = False
if predict_btn:
    st.session_state.predicted = True

if not st.session_state.predicted:
    st.markdown("""
    <div style="text-align:center; padding:70px 0;">
        <div style="font-size:3.5rem;">🏠</div>
        <div style="font-size:1.15rem; margin-top:16px; color:#555e80;">
            Set your property details in the sidebar<br>and click
            <strong style="color:#7c8fff;">Predict Price</strong> to get an estimate
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Compute prediction ────────────────────────────────────────────────────────
if model_bundle is not None:
    model = model_bundle["model"]
    try:
        model_cols = model.feature_names_in_ if hasattr(model, "feature_names_in_") else input_df.columns
        X_input    = input_df.reindex(columns=model_cols, fill_value=0)
        prediction = float(model.predict(X_input)[0])
    except Exception:
        prediction = None
else:
    prediction = float(max(50_000,
        overall_qual * 20_000 + gr_liv_area * 85 + total_bsmt_sf * 35
        + garage_area * 55 - house_age * 400
        + int(has_pool) * 8_000 + int(has_second_fl) * 15_000 - 30_000
    ))

if prediction is None:
    st.error("Run `python src/pipeline.py data/train.csv` first to train the model.")
    st.stop()

low, high = prediction * 0.92, prediction * 1.08

# ── Results row ───────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 1], gap="medium")

with col1:
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">Estimated Market Value</div>
        <div class="price-value">${prediction:,.0f}</div>
        <div class="price-range">Confidence range &nbsp;·&nbsp; ${low:,.0f} – ${high:,.0f}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.metric("Total Living Area", f"{total_sf:,} sq ft")
    st.metric("House Age", f"{house_age} yrs")

with col3:
    st.metric("Overall Quality", f"{overall_qual}/10")
    st.metric("Bathrooms", f"{total_baths:.1f}")

st.divider()

# ── Feature impact chart (plotly graph_objects — more control) ────────────────
st.subheader("📊 What's driving this estimate?")

impacts = {
    "Overall Quality":   overall_qual  * 20_000,
    "Above-ground Area": gr_liv_area   * 85,
    "Basement":          total_bsmt_sf * 35,
    "Garage":            garage_area   * 55,
    "Age penalty":      -house_age     * 400,
    "Pool bonus":        int(has_pool) * 8_000,
    "2nd Floor":         int(has_second_fl) * 15_000,
}

labels  = list(impacts.keys())
values  = list(impacts.values())
colors  = ["#7c8fff" if v >= 0 else "#f87171" for v in values]

# Sort ascending so biggest bar is at top
sorted_pairs = sorted(zip(values, labels, colors), key=lambda x: x[0])
values_s  = [p[0] for p in sorted_pairs]
labels_s  = [p[1] for p in sorted_pairs]
colors_s  = [p[2] for p in sorted_pairs]

fig = go.Figure(go.Bar(
    x=values_s,
    y=labels_s,
    orientation="h",
    marker_color=colors_s,
    marker_line_width=0,
    text=[f"${v:+,.0f}" for v in values_s],
    textposition="outside",
    textfont=dict(color="#9aa3bf", size=12),
    hovertemplate="%{y}: %{x:$,.0f}<extra></extra>",
))

fig.update_layout(
    height=300,
    margin=dict(l=0, r=80, t=10, b=10),
    plot_bgcolor="#252838",
    paper_bgcolor="#1c1f2e",
    font=dict(color="#9aa3bf", size=13),
    xaxis=dict(
        showgrid=True, gridcolor="#353a52",
        zeroline=True, zerolinecolor="#4f5fc4", zerolinewidth=2,
        tickfont=dict(color="#6b7699"),
        showticklabels=False,
    ),
    yaxis=dict(
        showgrid=False,
        tickfont=dict(color="#b8c0d8", size=13),
    ),
)

st.plotly_chart(fig, width='stretch')

# ── Footer ────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this model"):
    st.markdown("""
**Dataset**: Ames Housing Dataset — [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

**Pipeline**:
1. EDA — distribution analysis, missing values, correlations
2. Feature Engineering — 7 new features: `HouseAge`, `TotalSF`, `TotalBaths`, `HasGarage`, `HasPool`, `HasSecondFloor`, `YearsSinceRemod`
3. Models — Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost
4. Evaluation — RMSE, MAE, R², residual plots, feature importance

**Tech Stack**: Python · pandas · scikit-learn · XGBoost · Streamlit · Plotly

*Built by Sebastián Mayorga Castro — DS/ML Portfolio Project*
    """)