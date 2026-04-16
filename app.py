"""
Bristol Crime & House Prices — Interactive Demo
================================================
Streamlit dashboard for the GWR spatial analysis project.

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub → connect to Streamlit Community Cloud
"""

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
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

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        border-left: 4px solid #1f4e79;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f4e79;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #555;
        margin-top: 0.2rem;
    }
    .finding-box {
        background: #e8f4e8;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        border-left: 4px solid #2e7d32;
        margin-bottom: 0.5rem;
    }
    .warning-box {
        background: #fff3e0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        border-left: 4px solid #e65100;
        margin-bottom: 0.5rem;
    }
    .section-divider {
        border-top: 2px solid #e0e0e0;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────
DATA_DIR = Path("data")

@st.cache_data
def load_regression_data():
    """Load pre-computed regression dataset from GeoJSON."""
    path = DATA_DIR / "regression_dataset.geojson"
    if not path.exists():
        return None
    gdf = gpd.read_file(path)
    return gdf.to_crs("EPSG:4326")

@st.cache_data
def load_summary_stats():
    """Load pre-computed summary statistics."""
    path = DATA_DIR / "summary_statistics.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

@st.cache_data
def load_house_clean():
    """Load cleaned house price data."""
    path = DATA_DIR / "house_prices_bristol_clean.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data
def load_crime_clean():
    """Load cleaned crime data."""
    path = DATA_DIR / "crime_bristol_clean.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df["year"]  = df["Month"].dt.year
    return df

reg_gdf  = load_regression_data()
stats    = load_summary_stats()
house_df = load_house_clean()
crime_df = load_crime_clean()

DATA_AVAILABLE = reg_gdf is not None

# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Bristol_coat_of_arms.svg/150px-Bristol_coat_of_arms.svg.png",
             width=60)
    st.markdown("## Bristol Crime & House Prices")
    st.markdown("*Spatial Analysis Using GWR*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        options=[
            "🏠  Project Overview",
            "📊  Exploratory Analysis",
            "📈  OLS Baseline Model",
            "🗺️  GWR Results",
            "🔍  Key Findings",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Dataset**")
    if DATA_AVAILABLE and stats:
        st.markdown(f"- {stats.get('n_lsoas', 182):,} LSOAs")
        st.markdown(f"- {stats.get('n_house_sales', 34543):,} house sales")
        st.markdown(f"- {stats.get('n_crime_records', 159666):,} crime records")
        st.markdown(f"- Period: 2021–2025")
    else:
        st.markdown("- 182 LSOAs")
        st.markdown("- 34,543 house sales")
        st.markdown("- 159,666 crime records")
        st.markdown("- Period: 2021–2025")

    st.markdown("---")
    st.markdown("**Author:** La Yaung Linn Lett")
    st.markdown("**University:** UWE Bristol, 2026")
    st.markdown("[📂 GitHub](https://github.com/layaung-linnlett/bristol-crime-houseprices-gwr)")

# ── Helper: data not available banner ─────────────────────────────────────────
def show_data_warning():
    st.warning(
        "⚠️ Pre-computed data files not found in `data/`. "
        "Copy your processed outputs into the `data/` folder to enable interactive maps. "
        "See the README for instructions.",
        icon="⚠️",
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Project Overview":
    st.markdown('<div class="main-header">Bristol Crime & House Prices</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">A Spatial Analysis Using Geographically Weighted Regression (GWR)</div>',
                unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">182</div>
            <div class="metric-label">Bristol LSOAs analysed</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">34,543</div>
            <div class="metric-label">House transactions</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">159,666</div>
            <div class="metric-label">Crime records</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">2021–2025</div>
            <div class="metric-label">Study period</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### Research question")
        st.markdown("""
        > *Does crime affect house prices in Bristol, and does this relationship
        > vary spatially across neighbourhoods?*

        This project investigates whether higher crime levels are associated with
        lower house prices across Bristol's Lower Layer Super Output Areas (LSOAs),
        and — crucially — **whether this relationship varies across the city**.

        A global OLS model treats every neighbourhood identically. GWR allows
        the crime-price relationship to differ between, say, Clifton and Hartcliffe.
        """)

        st.markdown("### Methodology")
        steps = {
            "1. Data collection": "HM Land Registry, Police.uk, ONS, schools, bus stops",
            "2. Cleaning & aggregation": "Filter to Bristol, IQR outlier removal, LSOA-level summaries",
            "3. VIF diagnostic": "Detected collinearity → removed prop_leasehold (VIF = 34.7)",
            "4. OLS baseline": "Global model with statsmodels — full p-values and CIs",
            "5. Moran's I": "I = 0.4775 (p = 0.001) → spatial autocorrelation confirmed",
            "6. GWR model": "Adaptive bisquare kernel, AICc bandwidth selection",
            "7. Comparison": "R² improved from 0.108 → 0.716 using same 5 predictors",
        }
        for step, detail in steps.items():
            st.markdown(f"**{step}** — {detail}")

    with col_right:
        st.markdown("### Model comparison")
        comparison = pd.DataFrame({
            "Metric":     ["R²", "Adjusted R²", "AICc"],
            "OLS":        ["0.108", "0.083", "−84.17"],
            "GWR":        ["0.716", "0.636", "−199.38"],
            "GWR better": ["✓ Yes", "✓ Yes", "✓ Yes (lower = better)"],
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        st.markdown("### Predictors (final model)")
        predictors = pd.DataFrame({
            "Variable":        ["log(Crime)", "Prop. Flats", "Dist. to Centre",
                               "Schools Count", "Dist. to Bus Stop"],
            "What it captures": ["Neighbourhood safety", "Housing stock type",
                                 "Location / accessibility", "Family amenities",
                                 "Transport connectivity"],
        })
        st.dataframe(predictors, use_container_width=True, hide_index=True)

        st.markdown("### Data sources")
        sources = {
            "🏠 House prices": "HM Land Registry (price-paid)",
            "🚨 Crime": "Police.uk (monthly records)",
            "🗺️ Boundaries": "ONS LSOA (2021)",
            "🏫 Schools": "West of England Combined Authority",
            "🚌 Bus stops": "Bristol Open Data",
        }
        for icon_label, source in sources.items():
            st.markdown(f"- **{icon_label}**: {source}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: EXPLORATORY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Exploratory Analysis":
    st.markdown('<div class="main-header">Exploratory Analysis</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Understanding the data before modelling</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Price distribution", "Crime trends", "Spatial patterns"])

    # Tab 1 — Price distribution
    with tab1:
        st.markdown("#### House sale price distribution")
        st.markdown("""
        The distribution of cleaned house prices shows the right-skew that
        motivates **log transformation** before regression. The median (£316,100)
        sits below the mean (£331,188), confirming the rightward pull of
        high-value properties.
        """)
        if house_df is not None:
            median_p = house_df["price"].median()
            mean_p   = house_df["price"].mean()

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=house_df["price"], nbinsx=50,
                                       marker_color="#4c72b0", opacity=0.85,
                                       name="Sales"))
            fig.add_vline(x=median_p, line_dash="dash", line_color="crimson",
                          annotation_text=f"Median: £{median_p:,.0f}",
                          annotation_position="top right")
            fig.add_vline(x=mean_p, line_dash="dot", line_color="darkorange",
                          annotation_text=f"Mean: £{mean_p:,.0f}",
                          annotation_position="top left")
            fig.update_layout(
                xaxis_title="Sale Price (£)", yaxis_title="Number of Sales",
                title="Distribution of House Sale Prices in Bristol (2021–2025)",
                plot_bgcolor="white", height=420,
            )
            fig.update_xaxes(tickformat="£,.0f")
            st.plotly_chart(fig, use_container_width=True)
        else:
            show_data_warning()
            st.image("outputs/price_distribution.png",
                     caption="House price distribution",
                     use_container_width=True)

    # Tab 2 — Crime trends
    with tab2:
        st.markdown("#### Crime trends by type (2021–2025)")
        st.markdown("""
        Violence and sexual offences is the dominant crime category throughout,
        consistently accounting for around one third of all incidents. Total
        crime remains broadly stable across the study period (~31,000–33,000 per year),
        supporting the use of a single aggregated crime count per LSOA.
        """)
        if crime_df is not None:
            crime_pivot  = crime_df.pivot_table(
                index="year", columns="Crime type", aggfunc="size", fill_value=0
            )
            crime_totals = crime_df.groupby("year").size()
            top_types    = crime_df["Crime type"].value_counts().head(5).index
            colours      = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]

            fig = go.Figure()
            for crime_type, colour in zip(top_types, colours):
                if crime_type in crime_pivot.columns:
                    fig.add_trace(go.Scatter(
                        x=crime_pivot.index, y=crime_pivot[crime_type],
                        mode="lines+markers", name=crime_type,
                        line=dict(color=colour, width=2),
                        marker=dict(size=6),
                    ))
            fig.add_trace(go.Scatter(
                x=crime_totals.index, y=crime_totals.values,
                mode="lines+markers", name="Total (all types)",
                line=dict(color="black", width=2.5, dash="dash"),
                marker=dict(symbol="square", size=7),
            ))
            fig.update_layout(
                xaxis_title="Year", yaxis_title="Number of Recorded Crimes",
                title="Crime Trends by Type in Bristol (2021–2025)",
                plot_bgcolor="white", height=420,
                legend=dict(x=1.02, y=1, bgcolor="rgba(255,255,255,0.8)"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            show_data_warning()

    # Tab 3 — Spatial patterns
    with tab3:
        st.markdown("#### Spatial distribution — prices and crime")
        st.markdown("""
        The maps below show how median house prices and crime counts vary
        across Bristol's 182 LSOAs. Comparing both maps reveals areas where
        the two variables diverge — high-crime but high-price neighbourhoods
        near the city centre motivate the use of GWR over a global model.
        """)
        if DATA_AVAILABLE:
            import json as json_lib
            geojson = json_lib.loads(reg_gdf[["lsoa_code", "geometry"]].to_json())

            col1, col2 = st.columns(2)
            with col1:
                fig_p = go.Figure(go.Choroplethmapbox(
                    geojson=geojson, locations=reg_gdf["lsoa_code"],
                    z=reg_gdf["median_price"],
                    featureidkey="properties.lsoa_code",
                    colorscale="YlGnBu",
                    colorbar=dict(title="Price (£)", thickness=12),
                    marker=dict(opacity=0.75, line_width=0.3),
                    hovertemplate="<b>%{location}</b><br>Median price: £%{z:,.0f}<extra></extra>",
                ))
                fig_p.update_layout(
                    mapbox=dict(style="carto-positron",
                                center=dict(lat=51.4545, lon=-2.5879), zoom=10),
                    margin=dict(l=0, r=0, t=30, b=0), height=420,
                    title="Median House Price by LSOA",
                )
                st.plotly_chart(fig_p, use_container_width=True)

            with col2:
                fig_c = go.Figure(go.Choroplethmapbox(
                    geojson=geojson, locations=reg_gdf["lsoa_code"],
                    z=reg_gdf["total_crimes"],
                    featureidkey="properties.lsoa_code",
                    colorscale="Reds",
                    colorbar=dict(title="Total crimes", thickness=12),
                    marker=dict(opacity=0.75, line_width=0.3),
                    hovertemplate="<b>%{location}</b><br>Total crimes: %{z:,.0f}<extra></extra>",
                ))
                fig_c.update_layout(
                    mapbox=dict(style="carto-positron",
                                center=dict(lat=51.4545, lon=-2.5879), zoom=10),
                    margin=dict(l=0, r=0, t=30, b=0), height=420,
                    title="Total Crime Count by LSOA",
                )
                st.plotly_chart(fig_c, use_container_width=True)
        else:
            show_data_warning()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: OLS BASELINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  OLS Baseline Model":
    st.markdown('<div class="main-header">Global OLS Baseline Model</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">A single average relationship across all 182 LSOAs</div>',
                unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1.2])

    with col_l:
        st.markdown("### Model specification")
        st.latex(r"\log(\text{price}) = \beta_0 + \beta_1 \log(\text{crime}) + \beta_2 \text{prop\_flats} + \beta_3 \text{dist\_centre} + \beta_4 \text{schools} + \beta_5 \text{bus\_dist} + \varepsilon")

        st.markdown("### Why VIF mattered")
        st.markdown("""
        An initial check with 6 predictors revealed severe collinearity:
        """)
        vif_data = pd.DataFrame({
            "Predictor":         ["log(Crime)", "Prop. Flats", "**Prop. Leasehold**",
                                  "Dist. to Centre", "Schools Count", "Dist. to Bus"],
            "VIF":               [11.54, 29.64, "**34.73 ⚠️**", 7.80, 1.65, 3.86],
            "Action":            ["Keep", "Keep", "**Remove**", "Keep", "Keep", "Keep"],
        })
        st.dataframe(vif_data, use_container_width=True, hide_index=True)
        st.markdown("""
        In the UK, flats are almost always leasehold — the two variables
        capture nearly identical information. Removing `prop_leasehold`
        brought all VIF values below 5.
        """)

        st.markdown("### OLS fit metrics")
        ols_r2     = stats.get("ols_r2", 0.1084) if stats else 0.1084
        ols_adj_r2 = stats.get("ols_adj_r2", 0.0831) if stats else 0.0831
        ols_aicc   = stats.get("ols_aicc", -84.17) if stats else -84.17

        m1, m2, m3 = st.columns(3)
        m1.metric("R²", f"{ols_r2:.3f}")
        m2.metric("Adjusted R²", f"{ols_adj_r2:.3f}")
        m3.metric("AICc", f"{ols_aicc:.1f}")

        st.markdown("""
        <div class="warning-box">
        ⚠️ <b>OLS explains only 10.8% of price variation.</b> A global model
        that treats all of Bristol identically is clearly insufficient.
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("### Coefficient estimates")
        st.markdown("""
        Only **log(Crime)** and **Distance to Centre** are statistically
        significant at the 5% level. Blue dots = significant, grey = not significant.
        """)

        coefs = pd.DataFrame({
            "Predictor":  ["log(Crime)", "Prop. Flats", "Dist. to Centre",
                           "Schools Count", "Dist. to Bus Stop"],
            "Coef":       [-0.0778, -0.013, -0.021, -0.025, 0.045],
            "p-value":    [0.001, 0.894, 0.018, 0.106, 0.534],
            "Significant": [True, False, True, False, False],
        })

        fig = go.Figure()
        ci_half = [0.023, 0.095, 0.009, 0.016, 0.071]
        for i, row in coefs.iterrows():
            colour = "#4c72b0" if row["Significant"] else "#aaaaaa"
            fig.add_trace(go.Scatter(
                x=[row["Coef"]], y=[row["Predictor"]],
                mode="markers",
                marker=dict(size=10, color=colour),
                error_x=dict(type="data", array=[ci_half[i]],
                             arrayminus=[ci_half[i]], visible=True,
                             color=colour, thickness=2),
                name=row["Predictor"],
                showlegend=False,
                hovertemplate=f"<b>{row['Predictor']}</b><br>"
                              f"Coef: {row['Coef']:.4f}<br>"
                              f"p = {row['p-value']:.3f}<extra></extra>",
            ))

        fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
        fig.update_layout(
            xaxis_title="Coefficient (effect on log median price)",
            title="OLS Coefficients with 95% Confidence Intervals",
            plot_bgcolor="white", height=320,
            margin=dict(l=160, r=20, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Moran's I — spatial autocorrelation")
        morans_i   = stats.get("morans_I", 0.4775) if stats else 0.4775
        morans_p   = stats.get("morans_pvalue", 0.001) if stats else 0.001
        morans_sig = stats.get("morans_significant", True) if stats else True

        mi1, mi2, mi3 = st.columns(3)
        mi1.metric("Moran's I", f"{morans_i:.4f}")
        mi2.metric("p-value", f"{morans_p:.3f}")
        mi3.metric("Significant", "Yes ✓" if morans_sig else "No ✗")

        st.markdown("""
        <div class="warning-box">
        ⚠️ <b>Strong spatial autocorrelation in OLS residuals.</b>
        Neighbouring LSOAs have similar prediction errors — the global model
        violates the independence assumption. <b>GWR is needed.</b>
        </div>
        """, unsafe_allow_html=True)

        if DATA_AVAILABLE and "ols_residual" in reg_gdf.columns:
            st.markdown("#### OLS residual map")
            import json as json_lib
            geojson  = json_lib.loads(reg_gdf[["lsoa_code", "geometry"]].to_json())
            max_abs  = reg_gdf["ols_residual"].abs().max()
            fig_resid = go.Figure(go.Choroplethmapbox(
                geojson=geojson, locations=reg_gdf["lsoa_code"],
                z=reg_gdf["ols_residual"],
                featureidkey="properties.lsoa_code",
                colorscale=[[0, "#2166ac"], [0.5, "#f7f7f7"], [1, "#d6604d"]],
                zmin=-max_abs, zmax=max_abs,
                colorbar=dict(title="Residual", thickness=10),
                marker=dict(opacity=0.8, line_width=0.3),
                hovertemplate="<b>%{location}</b><br>Residual: %{z:.3f}<extra></extra>",
            ))
            fig_resid.update_layout(
                mapbox=dict(style="carto-positron",
                            center=dict(lat=51.4545, lon=-2.5879), zoom=10),
                margin=dict(l=0, r=0, t=10, b=0), height=300,
            )
            st.plotly_chart(fig_resid, use_container_width=True)
            st.caption("Red = model under-predicts price. Blue = over-predicts. "
                       "Spatial clustering confirms Moran's I result.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: GWR RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️  GWR Results":
    st.markdown('<div class="main-header">GWR Results</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Local crime-price coefficients across Bristol LSOAs</div>',
                unsafe_allow_html=True)

    gwr_r2     = stats.get("gwr_r2", 0.7155) if stats else 0.7155
    gwr_adj_r2 = stats.get("gwr_adj_r2", 0.6359) if stats else 0.6359
    gwr_aicc   = stats.get("gwr_aicc", -199.38) if stats else -199.38
    gwr_min    = stats.get("gwr_crime_coef_min", -0.2442) if stats else -0.2442
    gwr_max    = stats.get("gwr_crime_coef_max", 0.0203) if stats else 0.0203
    gwr_med    = stats.get("gwr_crime_coef_med", -0.1191) if stats else -0.1191

    # Top metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("GWR R²", f"{gwr_r2:.3f}", delta=f"+{gwr_r2 - (stats.get('ols_r2', 0.108) if stats else 0.108):.3f} vs OLS")
    c2.metric("Adjusted R²", f"{gwr_adj_r2:.3f}")
    c3.metric("AICc", f"{gwr_aicc:.1f}", delta=f"{gwr_aicc - (stats.get('ols_aicc', -84.17) if stats else -84.17):.1f} vs OLS")
    c4.metric("Crime coef min", f"{gwr_min:.4f}")
    c5.metric("Crime coef max", f"+{gwr_max:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Crime coefficient map", "Distribution of local effects"])

    with tab1:
        st.markdown("""
        **How to read this map:** Each LSOA is coloured by its local GWR crime
        coefficient — the estimated effect of crime on house prices *in that
        specific neighbourhood*. Red = crime strongly lowers prices.
        Blue = crime has little or no negative effect.
        """)
        if DATA_AVAILABLE and "gwr_crime_coef" in reg_gdf.columns:
            import json as json_lib
            geojson  = json_lib.loads(reg_gdf[["lsoa_code", "geometry"]].to_json())
            max_abs  = max(abs(reg_gdf["gwr_crime_coef"].min()),
                          abs(reg_gdf["gwr_crime_coef"].max()))

            fig_gwr = go.Figure(go.Choroplethmapbox(
                geojson=geojson,
                locations=reg_gdf["lsoa_code"],
                z=reg_gdf["gwr_crime_coef"],
                featureidkey="properties.lsoa_code",
                colorscale=[[0.0, "#d6604d"], [0.25, "#f4a582"],
                            [0.5, "#f7f7f7"],
                            [0.75, "#92c5de"], [1.0, "#2166ac"]],
                zmin=-max_abs, zmax=max_abs,
                colorbar=dict(
                    title="Crime coefficient",
                    thickness=14,
                    tickvals=[-0.2, -0.1, 0, 0.1],
                    ticktext=["−0.20<br>(strong negative)", "−0.10", "0", "+0.10<br>(positive)"],
                ),
                marker=dict(opacity=0.8, line_width=0.3),
                hovertemplate=(
                    "<b>%{location}</b><br>"
                    "Crime coefficient: %{z:.4f}<br>"
                    "<extra></extra>"
                ),
            ))
            fig_gwr.update_layout(
                mapbox=dict(style="carto-positron",
                            center=dict(lat=51.4545, lon=-2.5879), zoom=11),
                margin=dict(l=0, r=0, t=10, b=0), height=550,
            )
            st.plotly_chart(fig_gwr, use_container_width=True)

            pct_neg = (reg_gdf["gwr_crime_coef"] < -0.05).sum() / len(reg_gdf) * 100
            st.info(f"**{pct_neg:.0f}% of LSOAs** show a meaningfully negative crime effect "
                    f"(coefficient < −0.05). The remaining {100 - pct_neg:.0f}% — concentrated "
                    f"near the city centre — show near-zero or positive effects.")
        else:
            show_data_warning()
            st.markdown("""
            *GWR coefficient data not available. Add `gwr_crime_coef` column to
            `regression_dataset.geojson` by running the full analysis notebook first.*
            """)

    with tab2:
        st.markdown("#### Distribution of local crime coefficients")
        st.markdown("""
        The histogram below shows how the GWR crime coefficient varies across
        all 182 LSOAs. The global OLS estimate (−0.078) is shown as a dashed
        line — it sits near the centre of the distribution but conceals the
        full range from −0.244 to +0.020.
        """)
        if DATA_AVAILABLE and "gwr_crime_coef" in reg_gdf.columns:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=reg_gdf["gwr_crime_coef"], nbinsx=30,
                marker_color="#4c72b0", opacity=0.8,
                name="GWR local coefficients",
            ))
            fig_hist.add_vline(x=-0.0778, line_dash="dash", line_color="crimson",
                               annotation_text="OLS global: −0.078",
                               annotation_position="top right")
            fig_hist.add_vline(x=gwr_med, line_dash="dot", line_color="darkblue",
                               annotation_text=f"GWR median: {gwr_med:.3f}",
                               annotation_position="top left")
            fig_hist.update_layout(
                xaxis_title="Local crime coefficient",
                yaxis_title="Number of LSOAs",
                title="Distribution of GWR Crime Coefficients Across Bristol LSOAs",
                plot_bgcolor="white", height=380,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Minimum", f"{gwr_min:.4f}", "Strongest negative effect")
            col2.metric("Median",  f"{gwr_med:.4f}")
            col3.metric("Maximum", f"+{gwr_max:.4f}", "Weakest / positive effect")
        else:
            show_data_warning()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: KEY FINDINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Key Findings":
    st.markdown('<div class="main-header">Key Findings</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">What the analysis reveals about Bristol\'s housing market</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="finding-box">
    ✅ <b>Finding 1:</b> Crime and house prices are significantly negatively correlated
    across Bristol LSOAs (Pearson r = −0.23, p &lt; 0.05). High-crime neighbourhoods
    have statistically significantly lower median prices than low-crime areas.
    </div>

    <div class="finding-box">
    ✅ <b>Finding 2:</b> A global OLS model is insufficient. It explains only 10.8% of
    price variation, and Moran's I = 0.4775 (p = 0.001) confirms strong spatial
    autocorrelation in residuals — violating the OLS independence assumption.
    </div>

    <div class="finding-box">
    ✅ <b>Finding 3:</b> GWR substantially outperforms OLS (R² = 0.716 vs 0.108,
    AICc improvement of 115 points) using the same 5 predictors. The improvement
    comes entirely from allowing coefficients to vary spatially.
    </div>

    <div class="finding-box">
    ✅ <b>Finding 4:</b> The local crime coefficient ranges from −0.244 to +0.020
    across LSOAs — more than 3× the magnitude of the global OLS estimate of −0.078.
    79% of LSOAs show a meaningfully negative effect; ~21% show near-zero or
    positive effects concentrated near the city centre.
    </div>

    <div class="finding-box">
    ✅ <b>Finding 5:</b> High-crime, high-price hotspot LSOAs have nearly double
    the school density (1.20 vs 0.69 schools per LSOA) and slightly better bus
    access than other areas — explaining why crime does not suppress prices there.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Summary comparison table")
        summary_df = pd.DataFrame({
            "Metric":        ["R²", "Adjusted R²", "AICc",
                              "Crime coefficient", "Moran's I on residuals"],
            "OLS":           ["0.108", "0.083", "−84.17",
                              "−0.078 (global)", "0.4775 (p = 0.001)"],
            "GWR":           ["0.716", "0.636", "−199.38",
                              "−0.244 to +0.020 (local)", "—"],
            "GWR better?":   ["✓", "✓", "✓ (lower = better)", "✓ (richer)", "GWR justified"],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("### Hotspot analysis")
        hotspot_df = pd.DataFrame({
            "Indicator":              ["Distance to centre (km)",
                                       "Schools per LSOA",
                                       "Distance to bus (km)"],
            "Hotspot LSOAs":          ["3.151", "1.20", "0.162"],
            "Other LSOAs":            ["3.217", "0.69", "0.182"],
            "Difference":             ["−0.066", "+0.51 ✓", "−0.020"],
        })
        st.dataframe(hotspot_df, use_container_width=True, hide_index=True)
        st.caption("Hotspots = LSOAs in upper quartile for BOTH crime and price")

    with col2:
        st.markdown("### Policy implications")
        st.markdown("""
        **For policymakers and urban planners:**

        Crime reduction is likely to have the **largest positive effect on
        house prices in peripheral, amenity-poor neighbourhoods** — those
        showing the strongest negative GWR coefficients (dark red on the map).

        In **central, well-served areas** where the GWR coefficient is near
        zero, investment in schools, transport, and local services may be a
        more effective lever than crime reduction alone.

        A spatially differentiated policy approach — one that recognises
        local conditions rather than applying a single city-wide average —
        is both empirically justified and practically necessary.
        """)

        st.markdown("### Limitations")
        st.markdown("""
        - **Cross-sectional** analysis — cannot establish causality
        - Total crime count **aggregates all offence types** equally
        - **Omitted variables**: deprivation, property size, age, condition
        - GWR results **sensitive to bandwidth choice**
        - Local estimates **less stable** than global OLS (smaller effective n)
        """)

        st.markdown("### Future work")
        st.markdown("""
        - Add **IMD deprivation scores** as a covariate
        - Disaggregate by **individual crime types** (violent, burglary, ASB)
        - **Panel methods** to track annual crime-price changes
        - **Multi-city replication** to test generalisability
        """)

    st.markdown("---")
    st.markdown("### Resources")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.markdown("""
        **📂 GitHub Repository**
        Full code, notebooks, and data instructions.
        [layaung-linnlett/bristol-crime-houseprices-gwr](https://github.com/layaung-linnlett/bristol-crime-houseprices-gwr)
        """)
    with col_r2:
        st.markdown("""
        **📊 Data Sources**
        - [HM Land Registry](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads)
        - [Police.uk](https://data.police.uk/data/)
        - [ONS Geoportal](https://geoportal.statistics.gov.uk/)
        """)
    with col_r3:
        st.markdown("""
        **🛠️ Tools Used**
        - Python, GeoPandas, statsmodels
        - mgwr (GWR), esda (Moran's I)
        - Plotly, Streamlit
        """)
