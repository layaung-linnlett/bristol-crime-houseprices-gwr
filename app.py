"""
Bristol Crime & House Prices — Interactive Demo
================================================
Streamlit dashboard — no geopandas dependency.
GeoJSON loaded directly with json module.
Plotly Choroplethmapbox handles geometry natively.
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bristol Crime & House Prices",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header { font-size:2rem; font-weight:700; color:#1f4e79; margin-bottom:0.2rem; }
.sub-header  { font-size:1rem; color:#666; margin-bottom:1.5rem; }
.metric-card { background:#f0f4f8; border-radius:10px; padding:1rem 1.2rem;
               text-align:center; border-left:4px solid #1f4e79; }
.metric-value { font-size:1.8rem; font-weight:700; color:#1f4e79; }
.metric-label { font-size:0.85rem; color:#555; margin-top:0.2rem; }
.finding-box  { background:#e8f4e8; border-radius:8px; padding:0.8rem 1rem;
                border-left:4px solid #2e7d32; margin-bottom:0.5rem; }
.warning-box  { background:#fff3e0; border-radius:8px; padding:0.8rem 1rem;
                border-left:4px solid #e65100; margin-bottom:0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_regression_data():
    path = DATA_DIR / "regression_dataset.geojson"
    if not path.exists():
        return None, None
    with open(path, "r") as f:
        geojson = json.load(f)
    rows = []
    for feature in geojson["features"]:
        row = feature["properties"].copy()
        rows.append(row)
    df = pd.DataFrame(rows)
    return df, geojson

@st.cache_data
def load_summary_stats():
    path = DATA_DIR / "summary_statistics.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

@st.cache_data
def load_house_clean():
    # Try CSV first, then ZIP
    csv_path = DATA_DIR / "house_prices_bristol_clean.csv"
    zip_path = DATA_DIR / "house_prices_bristol_clean.zip"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    elif zip_path.exists():
        return pd.read_csv(zip_path, compression="zip")
    return None

@st.cache_data
def load_crime_clean():
    # Try CSV first, then ZIP
    csv_path = DATA_DIR / "crime_bristol_clean.csv"
    zip_path = DATA_DIR / "crime_bristol_clean.zip"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif zip_path.exists():
        df = pd.read_csv(zip_path, compression="zip")
    else:
        return None
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df["year"]  = df["Month"].dt.year
    return df

reg_df, geojson = load_regression_data()
stats           = load_summary_stats()
house_df        = load_house_clean()
crime_df        = load_crime_clean()
DATA_AVAILABLE  = reg_df is not None

# ── Map helper ─────────────────────────────────────────────────────────────────
BRISTOL_CENTER = dict(lat=51.4545, lon=-2.5879)

def make_choropleth(geojson, locations, z, colorscale,
                    title, colorbar_title, zmin=None, zmax=None,
                    hover_template=None, height=460):
    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson,
        locations=locations,
        z=z,
        featureidkey="properties.lsoa_code",
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        colorbar=dict(title=colorbar_title, thickness=12),
        marker=dict(opacity=0.78, line_width=0.3),
        hovertemplate=hover_template or
            "<b>%{location}</b><br>Value: %{z}<extra></extra>",
    ))
    # Calculate centre from data
    lats = [f["geometry"]["coordinates"][0][0][1]
            if f["geometry"]["type"] == "Polygon"
            else f["geometry"]["coordinates"][0][0][0][1]
            for f in geojson["features"]
            if f["geometry"] is not None]
    lons = [f["geometry"]["coordinates"][0][0][0]
            if f["geometry"]["type"] == "Polygon"
            else f["geometry"]["coordinates"][0][0][0][0]
            for f in geojson["features"]
            if f["geometry"] is not None]
    center_lat = np.mean(lats) if lats else 51.4545
    center_lon = np.mean(lons) if lons else -2.5879

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10.5),
        margin=dict(l=0, r=0, t=35, b=0),
        height=height,
        title=dict(text=title, font=dict(size=13), x=0.5),
    )
    return fig

def show_data_warning():
    st.warning(
        "⚠️ Data files not found in `data/`. "
        "Copy processed outputs from Google Drive to the `data/` folder. "
        "See the README for instructions."
    )

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Bristol Crime\n## & House Prices")
    st.markdown("*Spatial Analysis Using GWR*")
    st.markdown("---")
    page = st.radio("Navigate", options=[
        "🏠  Project Overview",
        "📊  Exploratory Analysis",
        "📈  OLS Baseline Model",
        "🗺️  GWR Results",
        "🔍  Key Findings",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown(f"- {stats.get('n_lsoas', 182):,} LSOAs")
    st.markdown(f"- {stats.get('n_house_sales', 34543):,} house sales")
    st.markdown(f"- {stats.get('n_crime_records', 159666):,} crime records")
    st.markdown("- Period: 2021–2025")
    st.markdown("---")
    st.markdown("**Author:** La Yaung Linn Lett")
    st.markdown("**UWE Bristol, 2026**")
    st.markdown(
        "[📂 GitHub](https://github.com/layaung-linnlett/"
        "bristol-crime-houseprices-gwr)"
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Project Overview":
    st.markdown(
        '<div class="main-header">Bristol Crime & House Prices</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">A Spatial Analysis Using Geographically '
        'Weighted Regression (GWR)</div>',
        unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        ["182", "34,543", "159,666", "2021–2025"],
        ["Bristol LSOAs", "House transactions", "Crime records", "Study period"]
    ):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        st.markdown("### Research question")
        st.markdown("""
        > *Does crime affect house prices in Bristol — and does this relationship
        > vary spatially across neighbourhoods?*

        A global OLS model treats every neighbourhood identically.
        **GWR allows the crime-price relationship to differ** between
        neighbourhoods — revealing that the effect ranges from strongly
        negative to near-zero depending on local amenities.
        """)
        st.markdown("### Pipeline")
        for step, detail in {
            "1. Data collection":     "HM Land Registry, Police.uk, ONS, schools, bus stops",
            "2. Cleaning":            "Filter Bristol, IQR outlier removal, LSOA summaries",
            "3. VIF diagnostic":      "Collinearity detected → removed prop_leasehold (VIF=34.7)",
            "4. OLS baseline":        "statsmodels — full p-values, CIs, F-statistic",
            "5. Moran's I":           "I=0.4775 (p=0.001) → spatial autocorrelation confirmed",
            "6. GWR":                 "Adaptive bisquare kernel, AICc bandwidth selection",
            "7. Comparison":          "R² 0.108 → 0.716 using same 5 predictors",
        }.items():
            st.markdown(f"**{step}** — {detail}")

    with col_r:
        st.markdown("### Model comparison")
        st.dataframe(pd.DataFrame({
            "Metric":     ["R²", "Adjusted R²", "AICc"],
            "OLS":        ["0.108", "0.083", "−84.17"],
            "GWR":        ["0.716", "0.636", "−199.38"],
            "GWR better": ["✓", "✓", "✓ (lower)"],
        }), use_container_width=True, hide_index=True)

        st.markdown("### Final predictors")
        st.dataframe(pd.DataFrame({
            "Variable":         ["log(Crime)", "Prop. Flats",
                                 "Dist. to Centre", "Schools Count",
                                 "Dist. to Bus"],
            "What it captures": ["Safety", "Housing type",
                                 "Accessibility", "Amenities", "Transport"],
        }), use_container_width=True, hide_index=True)

        st.markdown("### Data sources")
        for k, v in {
            "🏠 Prices":  "HM Land Registry (price-paid)",
            "🚨 Crime":   "Police.uk (monthly)",
            "🗺️ Geo":    "ONS LSOA boundaries 2021",
            "🏫 Schools": "West of England Combined Authority",
            "🚌 Buses":   "Bristol Open Data",
        }.items():
            st.markdown(f"- **{k}**: {v}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATORY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Exploratory Analysis":
    st.markdown(
        '<div class="main-header">Exploratory Analysis</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Understanding the data before modelling</div>',
        unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        ["Price distribution", "Crime trends", "Spatial maps"])

    with tab1:
        st.markdown("#### House sale price distribution")
        st.markdown(
            "Right-skew confirms why **log transformation** is applied before "
            "regression. The mean sits above the median due to high-value properties.")
        if house_df is not None:
            med = house_df["price"].median()
            avg = house_df["price"].mean()
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=house_df["price"], nbinsx=50,
                marker_color="#4c72b0", opacity=0.85))
            fig.add_vline(x=med, line_dash="dash", line_color="crimson",
                          annotation_text=f"Median: £{med:,.0f}",
                          annotation_position="top right")
            fig.add_vline(x=avg, line_dash="dot", line_color="darkorange",
                          annotation_text=f"Mean: £{avg:,.0f}",
                          annotation_position="top left")
            fig.update_layout(
                xaxis_title="Sale Price (£)", yaxis_title="Number of Sales",
                title="Distribution of House Sale Prices — Bristol (2021–2025)",
                plot_bgcolor="white", height=400)
            fig.update_xaxes(tickformat="£,.0f")
            st.plotly_chart(fig, use_container_width=True)
        else:
            show_data_warning()

    with tab2:
        st.markdown("#### Crime trends by type (2021–2025)")
        st.markdown(
            "Violence and sexual offences dominates throughout. "
            "Total crime remains broadly stable (~31–33k per year).")
        if crime_df is not None:
            crime_pivot  = crime_df.pivot_table(
                index="year", columns="Crime type",
                aggfunc="size", fill_value=0)
            crime_totals = crime_df.groupby("year").size()
            top_types    = crime_df["Crime type"].value_counts().head(5).index
            colours      = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]

            fig = go.Figure()
            for ct, colour in zip(top_types, colours):
                if ct in crime_pivot.columns:
                    fig.add_trace(go.Scatter(
                        x=crime_pivot.index, y=crime_pivot[ct],
                        mode="lines+markers", name=ct,
                        line=dict(color=colour, width=2),
                        marker=dict(size=6)))
            fig.add_trace(go.Scatter(
                x=crime_totals.index, y=crime_totals.values,
                mode="lines+markers", name="Total (all types)",
                line=dict(color="black", width=2.5, dash="dash"),
                marker=dict(symbol="square", size=7)))
            fig.update_layout(
                xaxis_title="Year", yaxis_title="Recorded Crimes",
                title="Crime Trends by Type — Bristol (2021–2025)",
                plot_bgcolor="white", height=400,
                xaxis=dict(tickmode="linear", tick0=2021, dtick=1),
                legend=dict(x=1.02, y=1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            show_data_warning()

    with tab3:
        st.markdown("#### Price and crime — spatial patterns")
        st.markdown(
            "Comparing both maps reveals where the two variables **diverge** — "
            "high-crime, high-price areas near the city centre are the key anomaly "
            "that motivates GWR.")
        if DATA_AVAILABLE:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(make_choropleth(
                    geojson=geojson,
                    locations=reg_df["lsoa_code"],
                    z=reg_df["median_price"],
                    colorscale="YlGnBu",
                    title="Median House Price by LSOA",
                    colorbar_title="Price (£)",
                    hover_template=(
                        "<b>%{location}</b><br>"
                        "Median price: £%{z:,.0f}<extra></extra>"),
                ), use_container_width=True)
            with col2:
                st.plotly_chart(make_choropleth(
                    geojson=geojson,
                    locations=reg_df["lsoa_code"],
                    z=reg_df["total_crimes"],
                    colorscale="Reds",
                    title="Total Crime Count by LSOA",
                    colorbar_title="Crimes",
                    hover_template=(
                        "<b>%{location}</b><br>"
                        "Total crimes: %{z:,.0f}<extra></extra>"),
                ), use_container_width=True)
        else:
            show_data_warning()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — OLS BASELINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  OLS Baseline Model":
    st.markdown(
        '<div class="main-header">Global OLS Baseline Model</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">A single average relationship across all 182 LSOAs</div>',
        unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1.2])

    with col_l:
        st.markdown("### Why VIF mattered")
        st.markdown(
            "Initial check with 6 predictors revealed severe collinearity:")
        st.dataframe(pd.DataFrame({
            "Predictor":  ["log(Crime)", "Prop. Flats",
                           "⚠️ Prop. Leasehold",
                           "Dist. to Centre", "Schools", "Dist. to Bus"],
            "VIF":        [11.54, 29.64, 34.73, 7.80, 1.65, 3.86],
            "Action":     ["Keep", "Keep", "Remove",
                           "Keep", "Keep", "Keep"],
        }), use_container_width=True, hide_index=True)
        st.markdown(
            "In the UK, flats are almost always leasehold — the two variables "
            "capture nearly identical information. After removing `prop_leasehold`, "
            "all remaining VIF values fell below 5.")

        st.markdown("### OLS fit metrics")
        ols_r2     = stats.get("ols_r2",     0.1084)
        ols_adj_r2 = stats.get("ols_adj_r2", 0.0831)
        ols_aicc   = stats.get("ols_aicc",  -84.17)
        m1, m2, m3 = st.columns(3)
        m1.metric("R²",          f"{ols_r2:.3f}")
        m2.metric("Adjusted R²", f"{ols_adj_r2:.3f}")
        m3.metric("AICc",        f"{ols_aicc:.1f}")
        st.markdown(
            '<div class="warning-box">⚠️ <b>OLS explains only 10.8% of price '
            'variation.</b> A single global model cannot capture Bristol\'s '
            'spatial complexity.</div>', unsafe_allow_html=True)

        st.markdown("### Moran's I on OLS residuals")
        mi   = stats.get("morans_I",       0.4775)
        mi_p = stats.get("morans_pvalue",  0.001)
        a1, a2 = st.columns(2)
        a1.metric("Moran's I", f"{mi:.4f}")
        a2.metric("p-value",   f"{mi_p:.3f}")
        st.markdown(
            '<div class="warning-box">⚠️ <b>Strong spatial autocorrelation '
            'in OLS residuals (p = 0.001).</b> The global model violates the '
            'independence assumption → GWR is needed.</div>',
            unsafe_allow_html=True)

    with col_r:
        st.markdown("### OLS coefficient plot")
        st.markdown(
            "Blue = significant (p < 0.05). Grey = not significant. "
            "Only crime and distance to centre matter globally.")
        coefs   = [-0.0778, -0.013, -0.021, -0.025,  0.045]
        pvals   = [ 0.001,   0.894,  0.018,  0.106,  0.534]
        labels  = ["log(Crime)", "Prop. Flats", "Dist. to Centre",
                   "Schools Count", "Dist. to Bus Stop"]
        ci_half = [ 0.023,   0.095,  0.009,  0.016,  0.071]
        colours = ["#4c72b0" if p < 0.05 else "#aaaaaa" for p in pvals]

        fig = go.Figure()
        for i, (label, coef, p, ci, colour) in enumerate(
            zip(labels, coefs, pvals, ci_half, colours)
        ):
            fig.add_trace(go.Scatter(
                x=[coef], y=[label], mode="markers",
                marker=dict(size=11, color=colour),
                error_x=dict(type="data", array=[ci], arrayminus=[ci],
                             visible=True, color=colour, thickness=2),
                showlegend=False,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"Coef: {coef:.4f}<br>"
                    f"p = {p:.3f}<extra></extra>"),
            ))
        fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.4)
        fig.update_layout(
            xaxis_title="Coefficient (effect on log median price)",
            plot_bgcolor="white", height=300,
            margin=dict(l=160, r=20, t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)

        if DATA_AVAILABLE and "ols_residual" in reg_df.columns:
            st.markdown("### OLS residual map")
            st.markdown(
                "Spatial clustering of similar colours confirms Moran's I. "
                "Red = under-predicted. Blue = over-predicted.")
            max_abs = reg_df["ols_residual"].abs().max()
            st.plotly_chart(make_choropleth(
                geojson=geojson,
                locations=reg_df["lsoa_code"],
                z=reg_df["ols_residual"],
                colorscale=[[0, "#2166ac"], [0.5, "#f7f7f7"], [1, "#d6604d"]],
                title="OLS Residuals by LSOA",
                colorbar_title="Residual",
                zmin=-max_abs, zmax=max_abs,
                hover_template=(
                    "<b>%{location}</b><br>"
                    "Residual: %{z:.3f}<extra></extra>"),
                height=380,
            ), use_container_width=True)
        elif DATA_AVAILABLE:
            st.info(
                "OLS residual column not found. Re-run the save cell in your "
                "notebook to include `ols_residual` in the GeoJSON.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — GWR RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️  GWR Results":
    st.markdown(
        '<div class="main-header">GWR Results</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Local crime-price coefficients across '
        'all 182 Bristol LSOAs</div>',
        unsafe_allow_html=True)

    gwr_r2  = stats.get("gwr_r2",             0.7155)
    ols_r2  = stats.get("ols_r2",             0.1084)
    gwr_adj = stats.get("gwr_adj_r2",         0.6359)
    gwr_aic = stats.get("gwr_aicc",         -199.38)
    ols_aic = stats.get("ols_aicc",          -84.17)
    gwr_min = stats.get("gwr_crime_coef_min", -0.2442)
    gwr_max = stats.get("gwr_crime_coef_max",  0.0203)
    gwr_med = stats.get("gwr_crime_coef_med", -0.1191)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("GWR R²",         f"{gwr_r2:.3f}",
              delta=f"+{gwr_r2 - ols_r2:.3f} vs OLS")
    c2.metric("Adjusted R²",    f"{gwr_adj:.3f}")
    c3.metric("AICc",           f"{gwr_aic:.1f}",
              delta=f"{gwr_aic - ols_aic:.1f} vs OLS")
    c4.metric("Crime coef min", f"{gwr_min:.4f}")
    c5.metric("Crime coef max", f"+{gwr_max:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(
        ["Crime coefficient map", "Coefficient distribution"])

    with tab1:
        st.markdown("""
        **How to read this map:** Each LSOA is coloured by its local GWR crime
        coefficient — the estimated effect of crime on house prices *in that
        specific neighbourhood*. **Red** = crime strongly lowers prices.
        **Blue** = crime has little or no negative effect (often city-centre
        LSOAs with strong amenity access).
        """)
        if DATA_AVAILABLE and "gwr_crime_coef" in reg_df.columns:
            max_abs = max(abs(reg_df["gwr_crime_coef"].min()),
                         abs(reg_df["gwr_crime_coef"].max()))
            st.plotly_chart(make_choropleth(
                geojson=geojson,
                locations=reg_df["lsoa_code"],
                z=reg_df["gwr_crime_coef"],
                colorscale=[
                    [0.0, "#d6604d"], [0.25, "#f4a582"],
                    [0.5,  "#f7f7f7"],
                    [0.75, "#92c5de"], [1.0,  "#2166ac"]],
                title="Local GWR Crime Coefficient by LSOA — Bristol",
                colorbar_title="Crime coefficient",
                zmin=-max_abs, zmax=max_abs,
                hover_template=(
                    "<b>%{location}</b><br>"
                    "Crime coef: %{z:.4f}<extra></extra>"),
                height=560,
            ), use_container_width=True)
            pct_neg = (
                (reg_df["gwr_crime_coef"] < -0.05).sum() / len(reg_df) * 100
            )
            st.info(
                f"**{pct_neg:.0f}% of LSOAs** show a meaningfully negative "
                f"crime effect (coefficient < −0.05). The remaining "
                f"{100 - pct_neg:.0f}% — concentrated near the city centre — "
                f"show near-zero or positive effects.")
        elif DATA_AVAILABLE:
            st.warning(
                "GWR coefficient column not found. Re-run the save cell in "
                "your notebook to include `gwr_crime_coef`.")
        else:
            show_data_warning()

    with tab2:
        st.markdown("#### Distribution of local crime coefficients")
        st.markdown(
            "The global OLS estimate (−0.078) sits near the centre but "
            "conceals the full range from −0.244 to +0.020.")
        if DATA_AVAILABLE and "gwr_crime_coef" in reg_df.columns:
            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(
                x=reg_df["gwr_crime_coef"], nbinsx=30,
                marker_color="#4c72b0", opacity=0.8))
            fig_h.add_vline(x=-0.0778, line_dash="dash",
                            line_color="crimson",
                            annotation_text="OLS global: −0.078")
            fig_h.add_vline(x=gwr_med, line_dash="dot",
                            line_color="darkblue",
                            annotation_text=f"GWR median: {gwr_med:.3f}")
            fig_h.update_layout(
                xaxis_title="Local crime coefficient",
                yaxis_title="Number of LSOAs",
                title="Distribution of GWR Crime Coefficients — 182 Bristol LSOAs",
                plot_bgcolor="white", height=380)
            st.plotly_chart(fig_h, use_container_width=True)
            a, b, c = st.columns(3)
            a.metric("Minimum", f"{gwr_min:.4f}", "Strongest negative")
            b.metric("Median",  f"{gwr_med:.4f}")
            c.metric("Maximum", f"+{gwr_max:.4f}", "Weakest / positive")
        else:
            show_data_warning()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — KEY FINDINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Key Findings":
    st.markdown(
        '<div class="main-header">Key Findings</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">What the analysis reveals about Bristol\'s '
        'housing market</div>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="finding-box">✅ <b>Finding 1 — Significant negative correlation.</b>
    Crime and house prices are significantly negatively correlated across Bristol
    LSOAs (Pearson r = −0.23, p &lt; 0.05). High-crime neighbourhoods have
    statistically significantly lower median prices than low-crime areas.</div>

    <div class="finding-box">✅ <b>Finding 2 — Global OLS is insufficient.</b>
    OLS explains only 10.8% of price variation. Moran's I = 0.4775 (p = 0.001)
    confirms strong spatial autocorrelation in residuals — violating the OLS
    independence assumption. A spatially varying model is needed.</div>

    <div class="finding-box">✅ <b>Finding 3 — GWR substantially outperforms OLS.</b>
    R² improves from 0.108 to 0.716 (+0.607) using the same 5 predictors.
    AICc improves by 115 points. The improvement comes entirely from allowing
    coefficients to vary spatially.</div>

    <div class="finding-box">✅ <b>Finding 4 — Strong spatial heterogeneity.</b>
    The local crime coefficient ranges from −0.244 to +0.020 — more than 3× the
    global OLS estimate of −0.078. 79% of LSOAs show a meaningfully negative
    effect; ~21% near the city centre show near-zero or positive effects.</div>

    <div class="finding-box">✅ <b>Finding 5 — Amenities offset crime.</b>
    High-crime, high-price hotspot LSOAs have nearly double the school density
    (1.20 vs 0.69 per LSOA) and better transport access — explaining why crime
    does not suppress prices in those areas.</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Summary comparison")
        st.dataframe(pd.DataFrame({
            "Metric":      ["R²", "Adjusted R²", "AICc", "Crime coefficient"],
            "OLS":         ["0.108", "0.083", "−84.17", "−0.078 (global)"],
            "GWR":         ["0.716", "0.636", "−199.38",
                            "−0.244 to +0.020 (local)"],
            "GWR better?": ["✓", "✓", "✓ (lower)", "✓ (richer)"],
        }), use_container_width=True, hide_index=True)

        st.markdown("### Hotspot amenities")
        st.dataframe(pd.DataFrame({
            "Indicator":     ["Dist. to centre (km)",
                              "Schools per LSOA",
                              "Dist. to bus (km)"],
            "Hotspot LSOAs": ["3.151", "1.20", "0.162"],
            "Other LSOAs":   ["3.217", "0.69", "0.182"],
            "Difference":    ["−0.066", "+0.51 ✓", "−0.020"],
        }), use_container_width=True, hide_index=True)
        st.caption(
            "Hotspots = LSOAs in upper quartile for BOTH crime AND price")

    with col2:
        st.markdown("### Policy implications")
        st.markdown("""
        **For policymakers and urban planners:**

        Crime reduction will have the **largest positive effect on house prices
        in peripheral, amenity-poor neighbourhoods** — those showing the
        strongest negative GWR coefficients (dark red on the map).

        In **central, well-served areas** where the GWR coefficient is near
        zero, investment in schools, transport, and local services may be a
        more effective lever than crime reduction alone.

        A **spatially differentiated** policy approach is both empirically
        justified and practically necessary.
        """)

        st.markdown("### Limitations")
        st.markdown("""
        - Cross-sectional — cannot establish causality
        - Total crime aggregates all offence types equally
        - Omitted variables: deprivation, property size, age
        - GWR sensitive to bandwidth choice
        """)

        st.markdown("### Resources")
        st.markdown("""
        - [📂 GitHub Repository](https://github.com/layaung-linnlett/bristol-crime-houseprices-gwr)
        - [🏠 HM Land Registry](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads)
        - [🚨 Police.uk](https://data.police.uk/data/)
        - [🗺️ ONS Geoportal](https://geoportal.statistics.gov.uk/)
        """)
