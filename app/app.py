import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

#
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
            
html, body [class*="css"] {font-family: 'DM Sans', sans-serif;}
            
/* Background */
.stApp { background-color: #0f0f11; color: #e8e8e8; }
            
/* Sidebar */
section[data-testid="stSidebar"] { background-color: #16161a; border-right: 1px solid #2a2a2e; }
            
/* Metric Cards */
[data-testid="metric-container"] { background-color: #1c1c21; border: 1px solid #2a2a2e; border-radius: 10px; padding: 20px; }
[data-testid="metric-container"] label { color: #888 !important; font-size: 0.75rem !important; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f5f5f5 !important; font-size: 1.6rem !important; font-weight: 600; font-family: 'DM Mono', monospace; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }
            
/* Tabs */
.stTabs [data-baseweb="tab-list"] { background-color: #16161a; border-bottom: 1px solid #2a2a2e; gap: 0; }
.stTabs [data-baseweb="tab"] { color: #777; font-size: 0.85rem; letter-spacing: 0.06em; padding: 12px 24px; text-transform: uppercase; }
.stTabs [aria-selected="true"] { color: #f5f5f5 !important; border-bottom: 2px solid #6ee7b7 !important; background: transparent !important; }

/* Buttons */
.stButton > button {
    background: #6ee7b7; color: #0f0f11; font-weight: 600;
    border: none; border-radius: 8px; padding: 10px 28px;
    font-size: 0.875rem; letter-spacing: 0.05em;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .85; }

/* Inputs */
.stSelectbox > div > div, .stSlider { color: #e8e8e8; }

/* Dataframe */
.stDataFrame { border: 1px solid #2a2a2e; border-radius: 8px; }

/* Section headers */
.section-title {
    font-size: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase;
    color: #6ee7b7; font-weight: 500; margin-bottom: 4px;
}
.page-title {
    font-size: 1.8rem; font-weight: 600; color: #f5f5f5;
    letter-spacing: -0.02em; line-height: 1.2;
}
.divider { border: none; border-top: 1px solid #2a2a2e; margin: 24px 0; }

/* Price tag */
.price-tag {
    background: #1c1c21; border: 1px solid #6ee7b7;
    border-radius: 12px; padding: 20px 28px; text-align: center;
    font-family: 'DM Mono', monospace;
}
.price-tag .label { font-size: 0.7rem; letter-spacing: 0.12em; color: #6ee7b7; text-transform: uppercase; margin-bottom: 6px; }
.price-tag .value { font-size: 2.4rem; font-weight: 500; color: #f5f5f5; }

/* Plot background matching */
div[data-testid="stPlotlyChart"], div[data-testid="stImage"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# Matplotlib and Seaborn style
plt.rcParams.update({
    "figure.facecolor": "#1c1c21",
    "axes.facecolor": "#1c1c21",
    "axes.edgecolor": "#2a2a2e",
    "axes.labelcolor": "#888",
    "xtick.color": "#666",
    "ytick.color": "#666",
    "grid.color": "#2a2a2e",
    "grid.alpha": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
    "font.family": "sans-serif",
})

ACCENT = "#6ee7b7"
AUDI_COL = "#60a5fa"
VW_COL = "#f97316"

# Load data
@st.cache_data(show_spinner=False)
def load_data():
    audi_url = "https://raw.githubusercontent.com/Aayush-Bhuyan/Used-Car-Price-Prediction/refs/heads/main/audi.csv"
    vw_url = "https://raw.githubusercontent.com/Aayush-Bhuyan/Used-Car-Price-Prediction/refs/heads/main/vw.csv"
    try:
        df_audi = pd.read_csv(audi_url)
        df_vw = pd.read_csv(vw_url)
    except Exception:
        if os.path.exists("audi.csv") and os.path.exists("vw.csv"):
            df_audi = pd.read_csv("audi.csv")
            df_vw = pd.read_csv("vw.csv")
        else:
            st.error("Could not load datasets. Upload audi.csv and vw.csv to the data folder.")
            st.stop()

    df_audi["brand"] = "Audi"
    df_vw["brand"] = "Volkswagen"
    df = pd.concat([df_audi, df_vw], ignore_index=True)
    return df

@st.cache_data(show_spinner=False)
def clean_and_engineer(df):
    df = df[(df["engineSize"] != 0) & df["price"].between(500, 100000)].copy()
    df["car_age"] = 2020 - df["year"]
    df["mileage_per_year"] = df["mileage"] / df["car_age"].replace([np.inf, -np.inf], 0).fillna(0)
    return df

@st.cache_data(show_spinner=False)
def train_models(df):
    y = df["price"]
    df_f = df.drop(columns=["price", "model", "year"])
    x = pd.get_dummies(df_f, columns=["transmission", "fuelType", "brand"], drop_first=True, dtype=int)
    X_tr, X_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)

    lr = LinearRegression().fit(X_tr, y_tr)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr, y_tr)

    lr_pred = lr.predict(X_te)
    rf_pred = rf.predict(X_te)

    results = {
        "lr": {
            "model": lr,
            "pred": lr_pred,
            "rmse": np.sqrt(mean_squared_error(y_te, lr_pred)),
            "r2": r2_score(y_te, lr_pred),
            "mae": mean_absolute_error(y_te, lr_pred),
        },
        "rf": {
            "model": rf,
            "pred": rf_pred,
            "rmse": np.sqrt(mean_squared_error(y_te, rf_pred)),
            "r2": r2_score(y_te, rf_pred),
            "mae": mean_absolute_error(y_te, rf_pred),
        },
        "X": X, "y_te": y_te, "X_te": X_te,
    }
    return results

# Load
with st.spinner("Loading datasets..."):
    raw = load_data()
    df = clean_and_engineer(raw)

with st.spinner("Training models..."):
    res = train_models(df)
    X, y_te, X_te = res["X"], res["y_te"], res["X_te"]
    lr, rf = res["lr"]["model"], res["rf"]["model"]

# Sidebar
with st.sidebar:
    st.markdown('<p class="section-title">Dataset</p>', unsafe_allow_html=True)
    st.markdown(f"**{len(df):,}** listings after cleaning")
    st.markdown(f"**{df['brand'].nunique()}** brands: • **{df['model'].nunique()}** models")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Navigation</p>', unsafe_allow_html=True)
    page = st.radio("", ["Overview", "EDA", "Modeling", "Predict"], label_visibility="collapsed")

# Page: Overview
if page == "Overview":
    st.markdown('<p class="section-title">VW & Audi</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-title">Used Car Price Predictor</p>', unsafe_allow_html=True)
    st.markdown("AutoTrader UK scrape • ~24 K listings • 2020 baseline")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Listings", f"{len(df):,}")
    c2.metric("Median Price", f"£{df['price'].median():,.0f}")
    c3.metric("RF R² Score", f"{res['rf']['r2']:.3f}")
    c4.metric("RF MAE", f"£{res['rf']['mae']:,}")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col_a, col_b = st.columns(4)
    with col_a:
        st.markdown('<p class="section-title">Price Distribution</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for brand, col in [("Audi", AUDI_COL), ("Volkswagen", VW_COL)]:
            subset = df[df["brand"] == brand]["price"]
            ax.hist(subset, bins=60, alpha=0.7, label=brand, color=col, edgecolor="none")
        ax.set_xlabel("Price (£)")
        ax.set_ylabel("Count")
        ax.legend(frameon=False, fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1000:.0f}k"))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown('<p class="section-title">Year vs Price</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for brand, col in [("Audi", AUDI_COL), ("Volkswagen", VW_COL)]:
            sub = df[df["brand"] == brand]
            ax.scatter(sub["year"], sub["price"], alpha=0.15, s=8, label=brand, color=col)
        ax.set_xlabel("Year")
        ax.set_ylabel("Price (£)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1000:.0f}k"))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

# Page: EDA
elif page == "EDA":
    st.markdown('<p class="section-title">Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-title">What drives used car prices?</p>', unsafe_allow_html=True)
    st.markdown('hr class=="divider">', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.labs(["DISTRIBUTIONS", "CORRELATIONS", "BRAND BREAKDOWN"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1: 
            st.markdown('<p class="section-title">Mileage</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 3))
            ax.hist(df["mileage"], bins=60, color=ACCENT, alpha=0.8, edgecolor="none")
            ax.set_xlabel("Mileage (miles)")
            ax.axis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
            fig.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            st.markdown('<p class="section-title">Engine Size</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 3))
            ax.hist(df["engineSize"], bins=25, color=ACCENT, alpha=0.8, edgecolor="none")
            ax.set_xlabel("Engine Size (L)")
            fig.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<p class="section-title">Fuel Type Mix</p>', unsafe_allow_html=True)
        fuel_counts = df["fuelType"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 2.5))
        bars = ax.barh(fuel_counts.index, fuel_counts.values, color=[ACCENT, AUDI_COL, VW_COL, "#a78bfa", "#f472b6"][:len(fuel_counts)], edgecolor="none", height=0.55)
        for bar, val in zip(bars, fuel_counts.values):
            ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2, f"{val:,}", va="center", fontsize=8, color="#aaa")
        ax.set_xlabel("Count")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        numeric_df = df.select_dtypes(include="number")
        corr = numeric_df.corr()

        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown('<p class="section-title">Correlation Matrix</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            cmap = sns.diverging_palette(220, 20, as_cmap=True)
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5, linecolor="#2a2a2e", ax=ax, annot_kws={"size": 8}, vmin=-1, vmax=1, cbar_kws={"shrink": 0.7})
            ax.tick_params(labelsize=8)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with c2:
            st.markdown('<p class="section-title">Correlation with Price</p>', unsafe_allow_html=True)
            price_corr = corr["price"].drop("price").sort_values()
            colors = [ACCENT if v > 0 else "#f87171" for v in price_corr.values]
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.barh(price_corr.index, price_corr.values, color=colors, edgecolor="none", height=0.6)
            ax.axvline(0, color="#444", linewidth=0.8)
            ax.set_xlabel("Pearson r")
            ax.tick_params(labelsize=8)
            fig.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        brand_stats = df.groupby("brand")["price"].agg(["median", "mean", "std", "count"]).round(0)
        brand_stats.columns = ["Median", "Mean", "Std Dev", "Count"]
        st.markdown('<p class="section-title">Summary Statistics</p>', unsafe_allow_html=True)
        st.dataframe(brand_stats.style.format("£{:,.0f}", subset=["Median", "Mean", "Std Dev"])
                                        .format("{:,.0f}", subset=["Count"])
                                        .set_properties(**{"background-color": "#1c1c21", "color": "#e8e8e8"}),
                     use_container_width=True)

        st.markdown('<p class="section-title">Price Boxplot</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        bp = ax.boxplot(
            [df[df["brand"] == b]["price"].values for b in ["Audi", "Volkswagen"]],
            labels=["Audi", "Volkswagen"],
            patch_artist=True,
            medianprops=dict(color="#0f0f11", linewidth=2),
            flierprops=dict(marker=".", color="#555", markersize=3),
        )
        for patch, color in zip(bp["boxes"], [AUDI_COL, VW_COL]):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"£{y/1000:.0f}k"))
        fig.tight_layout(); st.pyplot(fig); plt.close()

# Page: Models
elif page == "Models":
    st.markdown('<p class="section-title">Model Evaluation</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-title">Linear Regression vs Random Forest</p>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Linear Regression")
        st.metric("R² Score", f"{res['lr']['r2']:.4f}", delta=f"{res['lr']['r2']*100:.1f}% variance explained")
        st.metric("MAE",  f"£{res['lr']['mae']:,.0f}")
        st.metric("RMSE", f"£{res['lr']['rmse']:,.0f}")
    with c2:
        st.markdown("#### Random Forest")
        st.metric("R² Score", f"{res['rf']['r2']:.4f}", delta=f"+{(res['rf']['r2']-res['lr']['r2'])*100:.1f}pp vs LR")
        st.metric("MAE",  f"£{res['rf']['mae']:,.0f}", delta=f"-{(1-res['rf']['mae']/res['lr']['mae'])*100:.0f}% vs LR")
        st.metric("RMSE", f"£{res['rf']['rmse']:,.0f}", delta=f"-{(1-res['rf']['rmse']/res['lr']['rmse'])*100:.0f}% vs LR")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["ACTUAL vs PREDICTED", "FEATURE IMPORTANCE"])

    with tab1:
        c1, c2 = st.columns(2)
        for col, key, label, color in [(c1, "lr", "Linear Regression", AUDI_COL), (c2, "rf", "Random Forest", ACCENT)]:
            with col:
                st.markdown(f'<p class="section-title">{label}</p>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5, 4.5))
                lim = (0, 80000)
                ax.scatter(y_te, res[key]["pred"], alpha=0.15, s=6, color=color)
                ax.plot(lim, lim, "--", color="#555", linewidth=1)
                ax.set_xlim(lim); ax.set_ylim(lim)
                ax.set_xlabel("Actual Price (£)"); ax.set_ylabel("Predicted Price (£)")
                fmt = mticker.FuncFormatter(lambda v, _: f"£{v/1000:.0f}k")
                ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
                fig.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        imp_df = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
        top_rf = imp_df.sort_values("Importance", ascending=False).head(12).sort_values("Importance")

        coef_df = pd.DataFrame({"Feature": X.columns, "Coef": lr.coef_})
        top_lr  = coef_df.reindex(coef_df["Coef"].abs().sort_values().tail(12).index)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="section-title">LR — Top Coefficients</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            colors = [ACCENT if v > 0 else "#f87171" for v in top_lr["Coef"]]
            ax.barh(top_lr["Feature"], top_lr["Coef"], color=colors, edgecolor="none", height=0.65)
            ax.axvline(0, color="#444", linewidth=0.8)
            ax.tick_params(labelsize=8)
            fig.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            st.markdown('<p class="section-title">RF — Feature Importance</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.barh(top_rf["Feature"], top_rf["Importance"], color=ACCENT, edgecolor="none", height=0.65)
            ax.tick_params(labelsize=8)
            fig.tight_layout(); st.pyplot(fig); plt.close()

# Page: Predict
elif page == "Predict":
    st.markdown('<p class="section-title">Price Estimator</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-title">Predict a car\'s market value</p>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.markdown('<p class="section_title">Vehicle Specification</p>', unsafe_allow_html=True)

        brand = st.selectbox("Brand", ["Volkswagen", "Audi"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Semi-Auto"])
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric", "Other"])
        engine_size = st.slider("Engine Size (L)", 0.9, 5.0, 2.0, 0.1)
        year = st.slider("Year", 2000, 2020, 2018)
        mileage = st.slider("Mileage", 0, 200000, 30000, 1000)

        predict_btn = st.button("Predict Price")

    with col_out:
        st.markdown('<p class="section-title">Estimated Market Value</p>', unsafe_allow_html=True)

        if predict_btn:
            car_age = max(2020 - year, 1)
            mileage_per_year = mileage / car_age

            row = {col: 0 for col in X.columns}
            row["engineSize"] = engine_size
            row["mileage"] = mileage
            row["car_age"] = car_age
            row["mileage_per_year"] = mileage_per_year

            if transmission == "Manual" and "transmission_Manual" in row: row["transmission_Manual"] = 1
            if transmission == "Semi-Auto" and "transmission_Semi-Auto" in row: row["transmission_Semi-Auto"] = 1
            if fuel == "Petrol" and "fuelType_Petrol" in row: row["fuelType_Petrol"] = 1
            if fuel == "Diesel" and "fuelType_Diesel" in row: row["fuelType_Diesel"] = 1
            if fuel == "Hybrid" and "fuelType_Hybrid" in row: row["fuelType_Hybrid"] = 1
            if fuel == "Electric" and "fuelType_Electric" in row: row["fuelType_Electric"] = 1
            if brand == "Volkswagen" and "brand_Volkswagen" in row: row["brand_Volkswagen"] = 1

            X_input = pd.DataFrame([row])[X.columns]
            rf_price = rf.predict(X_input)[0]
            lr_price = lr.predict(X_input)[0]

            st.markdown(f"""
            <div class="price-tag" style="border-color:#60a5fa;">
                <div class="label" style="color:#60a5fa">Linear Regression Estimate</div>
                <div class="value">£{lr_price:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:20px; background:#1c1c21; border:1px solid #2a2a2e; border-radius:10px; padding:16px 20px;">
                <p class="section-title">Input Summary</p>
                <p style="font-size:0.82rem; color:#aaa; margin:0; line-height:1.8em;">
                    {brand} • {transmission} • {fuel}<br>
                    {engine_size:.1f}L • {year} • {mileage:,} miles<br>
                    Car Age: {car_age} years • Mileage/Year: {mileage_per_year:,.0f} mi/yr
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="height:260px; display:flex; align-items:center; justify-content:center;
                        border:1px dashed #2a2a2e; border-radius:12px; color:#444; font-size:0.85rem; letter-spacing:0.1em;">
                CONFIGURE VEHICLE → ESTIMATE
            </div>
            """, unsafe_allow_html=True)