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

    # Check if coordinates are in BNG (large numbers) or WGS84
    try:
        sample_coord = geojson["features"][0]["geometry"]["coordinates"][0][0]
        if isinstance(sample_coord[0], list):
            sample_coord = sample_coord[0]
        x, y = sample_coord[0], sample_coord[1]

        # BNG coordinates are large numbers (e.g. 360000, 170000)
        # WGS84 coordinates are small (e.g. -2.5, 51.4)
        if abs(x) > 1000:
            # Coordinates are in BNG — convert to WGS84
            import pyproj
            transformer = pyproj.Transformer.from_crs(
                "EPSG:27700", "EPSG:4326", always_xy=True)

            def convert_ring(ring):
                return [list(transformer.transform(c[0], c[1]))
                        for c in ring]

            for feature in geojson["features"]:
                geom = feature["geometry"]
                if geom["type"] == "Polygon":
                    geom["coordinates"] = [
                        convert_ring(ring)
                        for ring in geom["coordinates"]]
                elif geom["type"] == "MultiPolygon":
                    geom["coordinates"] = [
                        [convert_ring(ring) for ring in polygon]
                        for polygon in geom["coordinates"]]
    except Exception:
        pass

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
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=51.4545, lon=-2.5879),
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
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <div style="font-size:2.5rem;">🏠</div>
        <div style="font-size:1.1rem; font-weight:700; color:#1f4e79;
                    line-height:1.3; margin-top:0.3rem;">
            Bristol Crime<br>& House Prices
        </div>
        <div style="font-size:0.8rem; color:#888; margin-top:0.3rem;
                    font-style:italic;">
            Spatial Analysis Using GWR
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio("Navigate", options=[
        "🏠  Project Overview",
        "📊  Exploratory Analysis",
        "📈  OLS Baseline Model",
        "🗺️  GWR Results",
        "🔍  Key Findings",
    ], label_visibility="collapsed")

    st.markdown("---")

    # Dataset stats
    st.markdown("""
    <div style="font-size:0.85rem; color:#444;">
        <div style="font-weight:700; margin-bottom:0.5rem;
                    color:#1f4e79;">📁 Dataset</div>
    </div>
    """, unsafe_allow_html=True)

    n_lsoas   = stats.get("n_lsoas",          182)
    n_sales   = stats.get("n_house_sales",   34543)
    n_crimes  = stats.get("n_crime_records", 159666)

    st.markdown(f"""
    <div style="font-size:0.85rem; color:#444; line-height:2;">
        🏘️ &nbsp;<b>{n_lsoas:,}</b> LSOAs<br>
        🏠 &nbsp;<b>{n_sales:,}</b> house sales<br>
        🚨 &nbsp;<b>{n_crimes:,}</b> crime records<br>
        📅 &nbsp;<b>2021–2025</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model results summary
    gwr_r2 = stats.get("gwr_r2", 0.7393)
    ols_r2 = stats.get("ols_r2", 0.108)

    st.markdown(f"""
    <div style="font-size:0.85rem; color:#444;">
        <div style="font-weight:700; margin-bottom:0.5rem;
                    color:#1f4e79;">📊 Key Results</div>
        <div style="line-height:2.0;">
            OLS R²: &nbsp;<b>{ols_r2:.3f}</b><br>
            GWR R²: &nbsp;<b style="color:#2e7d32">{gwr_r2:.3f}</b>
            &nbsp;✓<br>
            Moran's I: &nbsp;<b>0.5408</b><br>
            Crime coef: &nbsp;<b>−0.256 to +0.025</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Author
    st.markdown("""
    <div style="font-size:0.82rem; color:#555; line-height:1.8;">
        <b>👤 Author</b><br>
        La Yaung Linn Lett<br>
        BSc Data Science & AI<br>
        UWE Bristol, 2026
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        "[📂 GitHub Repository](https://github.com/layaung-linnlett/"
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
            "5. Moran's I":           "I=0.5408 (p=0.001) → spatial autocorrelation confirmed",
            "6. GWR":                 "Adaptive bisquare kernel, AICc bandwidth selection",
            "7. Comparison":          "R² 0.108 → 0.739 using same 5 predictors",
        }.items():
            st.markdown(f"**{step}** — {detail}")

    with col_r:
        st.markdown("### Model comparison")
        st.dataframe(pd.DataFrame({
            "Metric":     ["R²", "Adjusted R²", "AICc"],
            "OLS":        ["0.108", "0.083", "−84.17"],
            "GWR":        ["0.739", "0.662", "−209.48"],
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
        '<div class="sub-header">Understanding Bristol\'s crime and '
        'housing landscape before modelling</div>',
        unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["💰 Price distribution", "🚨 Crime trends", "🗺️ Spatial maps",
         "⚠️ Data Coverage"])

    # ── Tab 1 — Price distribution ─────────────────────────────────────────────
    with tab1:
        st.markdown("### What do Bristol house prices look like?")
        st.markdown("""
        Before modelling, we need to understand the shape of the price data.
        The histogram below shows all **34,543 cleaned house transactions**
        across Bristol from 2021 to 2025.
        """)

        if house_df is not None:
            med = house_df["price"].median()
            avg = house_df["price"].mean()
            skew = house_df["price"].skew()

            # Key stats row
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Median price",  f"£{med:,.0f}")
            k2.metric("Mean price",    f"£{avg:,.0f}")
            k3.metric("Price skew",    f"{skew:.2f}")
            k4.metric("Transactions",  f"{len(house_df):,}")

            st.markdown("<br>", unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=house_df["price"], nbinsx=50,
                marker_color="#4c72b0", opacity=0.85,
                name="House sales",
                hovertemplate="Price: £%{x:,.0f}<br>Count: %{y}<extra></extra>",
            ))

            # Median line — more visible
            fig.add_shape(type="line",
                x0=med, x1=med, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="crimson", width=3, dash="dash"))
            fig.add_annotation(
                x=med, y=0.97, xref="x", yref="paper",
                text=f"<b>Median<br>£{med:,.0f}</b>",
                showarrow=True, arrowhead=2, arrowcolor="crimson",
                ax=-60, ay=0,
                font=dict(size=12, color="crimson"),
                bgcolor="white", bordercolor="crimson", borderwidth=1)

            # Mean line — more visible
            fig.add_shape(type="line",
                x0=avg, x1=avg, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="#e67e00", width=3, dash="dot"))
            fig.add_annotation(
                x=avg, y=0.85, xref="x", yref="paper",
                text=f"<b>Mean<br>£{avg:,.0f}</b>",
                showarrow=True, arrowhead=2, arrowcolor="#e67e00",
                ax=60, ay=0,
                font=dict(size=12, color="#e67e00"),
                bgcolor="white", bordercolor="#e67e00", borderwidth=1)

            fig.update_layout(
                xaxis_title="Sale Price (£)",
                yaxis_title="Number of Sales",
                title="<b>Distribution of House Sale Prices — Bristol (2021–2025)</b>",
                plot_bgcolor="white",
                height=450,
                showlegend=False,
            )
            fig.update_xaxes(tickformat="£,.0f")
            st.plotly_chart(fig, use_container_width=True)

            # Storytelling insight
            st.info(
                f"💡 **Why this matters for modelling:** The mean (£{avg:,.0f}) "
                f"sits £{avg-med:,.0f} above the median (£{med:,.0f}), "
                f"confirming a right-skewed distribution (skew = {skew:.2f}). "
                f"This means a small number of high-value properties pull the "
                f"average upward. **We apply a log transformation** to median "
                f"price before regression — this compresses the right tail, "
                f"stabilises variance, and produces a more linear relationship "
                f"with the predictors."
            )
        else:
            st.warning("⚠️ house_prices_bristol_clean.csv/zip not found in data/")

    # ── Tab 2 — Crime trends ───────────────────────────────────────────────────
    with tab2:
        st.markdown("### How has crime changed across Bristol (2021–2025)?")
        st.markdown("""
        We use Police.uk data covering **159,666 crime records** across
        all Bristol LSOAs. Understanding crime composition and stability
        over time justifies our use of a single aggregated crime count
        per LSOA in the regression models.
        """)

        if crime_df is not None:
            # Summary metrics
            total_crimes = len(crime_df)
            avg_annual   = crime_df.groupby("year").size().mean()
            top_type     = crime_df["Crime type"].value_counts().index[0]
            top_pct      = (crime_df["Crime type"].value_counts().iloc[0]
                           / total_crimes * 100)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total records",    f"{total_crimes:,}")
            m2.metric("Avg per year",     f"{avg_annual:,.0f}")
            m3.metric("Dominant type",    "Violence & sexual")
            m4.metric("% of all crimes",  f"{top_pct:.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)

            crime_pivot  = crime_df.pivot_table(
                index="year", columns="Crime type",
                aggfunc="size", fill_value=0)
            crime_totals = crime_df.groupby("year").size()
            top_types    = crime_df["Crime type"].value_counts().head(5).index
            colours      = ["#4c72b0", "#dd8452",
                            "#55a868", "#c44e52", "#8172b2"]

            fig = go.Figure()
            for ct, colour in zip(top_types, colours):
                if ct in crime_pivot.columns:
                    fig.add_trace(go.Scatter(
                        x=crime_pivot.index,
                        y=crime_pivot[ct],
                        mode="lines+markers",
                        name=ct,
                        line=dict(color=colour, width=2.5),
                        marker=dict(size=7),
                        hovertemplate=(
                            f"<b>{ct}</b><br>"
                            "Year: %{x}<br>"
                            "Count: %{y:,}<extra></extra>"),
                    ))

            fig.add_trace(go.Scatter(
                x=crime_totals.index,
                y=crime_totals.values,
                mode="lines+markers",
                name="Total (all types)",
                line=dict(color="black", width=3, dash="dash"),
                marker=dict(symbol="square", size=8),
                hovertemplate=(
                    "<b>Total crimes</b><br>"
                    "Year: %{x}<br>"
                    "Count: %{y:,}<extra></extra>"),
            ))

            # Annotation for dominant category
            last_year = crime_pivot.index[-1]
            if "Violence and sexual offences" in crime_pivot.columns:
                last_val = crime_pivot.loc[
                    last_year, "Violence and sexual offences"]
                fig.add_annotation(
                    x=last_year, y=last_val,
                    text=f"<b>Violence: {last_val:,}</b><br>~1/3 of all crime",
                    showarrow=True, arrowhead=2,
                    ax=-80, ay=-30,
                    font=dict(size=11, color="#4c72b0"),
                    bgcolor="white",
                    bordercolor="#4c72b0", borderwidth=1)

            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Recorded Crimes",
                title="<b>Crime Trends by Type — Bristol (2021–2025)</b>",
                plot_bgcolor="white",
                height=450,
                xaxis=dict(tickmode="linear", tick0=2021, dtick=1),
                legend=dict(
                    x=1.02, y=1,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#cccccc", borderwidth=1),
                yaxis=dict(tickformat=","),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "💡 **Why this matters for modelling:** Total crime is broadly "
                "stable across 2021–2025 (~31,000–33,000 incidents per year), "
                "with only a slight dip in 2023. This stability supports "
                "aggregating crime across the full study period into a single "
                "**total_crimes** count per LSOA, rather than modelling "
                "year-by-year changes. Violence and sexual offences consistently "
                "accounts for around one third of all crime — making it the "
                "dominant driver of neighbourhood crime levels across Bristol."
            )
        else:
            st.warning("⚠️ crime_bristol_clean.csv/zip not found in data/")

    # ── Tab 3 — Spatial maps ───────────────────────────────────────────────────
    with tab3:
        st.markdown("### Where are prices high — and where is crime high?")
        st.markdown("""
        The two maps below show the spatial distribution of **median house
        price** (left) and **total crime count** (right) across all 182
        Bristol LSOAs. The key question: do high-crime areas always have
        low prices — or is the relationship more complex?
        """)

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
                st.caption(
                    "🔵 **Highest prices** in the north and north-west "
                    "(Clifton, Redland, Cotham). Prices fall moving south "
                    "and east toward more peripheral neighbourhoods.")

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
                st.caption(
                    "🔴 **Highest crime** concentrated in the city centre "
                    "and inner-city LSOAs. Outer and northern residential "
                    "areas show markedly lower crime counts.")

            st.warning(
                "⚠️ **The spatial paradox:** Some city-centre LSOAs appear "
                "**dark on both maps** — high crime AND high prices. This "
                "directly contradicts a simple negative relationship and "
                "suggests that **location advantages** (centrality, schools, "
                "transport) can override the negative price effect of crime "
                "in specific neighbourhoods. A global OLS model cannot capture "
                "this — it forces a single average effect across all 182 LSOAs. "
                "**This is why we need GWR.**"
            )

            st.info(
                "🔍 **Why are some areas missing from the maps?** "
                "The maps show 182 LSOAs rather than all areas in Bristol. "
                "An investigation revealed that 86 LSOAs were dropped during "
                "the inner join merge because the ONS shapefile (268 LSOAs) "
                "covers a wider geographic boundary than the Police.uk crime "
                "recording area (189 LSOAs). All 86 missing LSOAs had house "
                "sales recorded but no crimes — they fall on Bristol's periphery "
                "bordering South Gloucestershire and North Somerset, outside the "
                "Avon & Somerset Police recording boundary. "
                "See the **⚠️ Data Coverage** tab for the full investigation."
            )

            # Quick stats
            st.markdown("#### At a glance")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Price range",
                      f"£{reg_df['median_price'].min():,.0f}–"
                      f"£{reg_df['median_price'].max():,.0f}")
            g2.metric("Crime range",
                      f"{reg_df['total_crimes'].min():.0f}–"
                      f"{reg_df['total_crimes'].max():,.0f}")
            g3.metric("Correlation (r)",  "−0.23")
            g4.metric("Significant?",     "Yes (p < 0.05)")
        else:
            st.warning(
                "⚠️ regression_dataset.geojson not found in data/. "
                "Upload it from Google Drive to enable the maps.")

    # ── Tab 4 — Data Coverage Investigation ───────────────────────────────────
    with tab4:
        st.markdown("### ⚠️ Why do the maps have missing areas?")

        # Pipeline counts
        st.markdown("#### 📊 LSOA counts at each pipeline stage")
        st.markdown("""
        By comparing LSOA counts at every stage of the data pipeline,
        we can pinpoint exactly where and why LSOAs go missing.
        """)

        pipeline_data = pd.DataFrame({
            "Pipeline Stage":   [
                "ONS Shapefile (all Bristol boundaries)",
                "House price data (after aggregation)",
                "Crime data (Police.uk)",
                "Final merged dataset",
            ],
            "LSOA Count": [268, 276, 189, 182],
            "Notes": [
                "Wider geographic boundary — includes peripheral areas",
                "All LSOAs with at least one house sale 2021–2025",
                "Only LSOAs within Avon & Somerset Police recording area",
                "After inner join: only LSOAs present in ALL datasets",
            ],
        })
        st.dataframe(pipeline_data, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual pipeline
        col_pipe, col_num = st.columns([1.5, 1])

        with col_pipe:
            st.markdown("#### 🔍 Where did the 86 LSOAs go?")
            st.markdown("""
            The investigation diagnosed **every single missing LSOA:**
            """)

            diag_data = pd.DataFrame({
                "Cause":        [
                    "🚔 In house data — NOT in crime data",
                    "🏠 In crime data — NOT in house data",
                    "❌ Missing from both datasets",
                    "🔗 Lost during inner join merge",
                ],
                "Count": [86, 0, 0, 0],
                "Verdict": [
                    "⬅️ ROOT CAUSE",
                    "Not applicable",
                    "Not applicable",
                    "Consequence of above",
                ],
            })
            st.dataframe(diag_data, use_container_width=True, hide_index=True)

            st.error(
                "🔴 **Root cause confirmed:** All 86 missing LSOAs had house "
                "sales recorded in Land Registry — but **zero crimes recorded** "
                "in Police.uk. This is not because these areas had no crime. "
                "It is because they fall **outside the Police.uk recording "
                "boundary** for Avon & Somerset Police."
            )

        with col_num:
            st.markdown("#### 📉 The numbers")
            st.metric("Shapefile LSOAs",     "268", "Full ONS boundary")
            st.metric("Crime data LSOAs",    "189", "Police recording area only",
                      delta_color="inverse")
            st.metric("Final dataset LSOAs", "182",
                      "After inner join + IQR cleaning")
            st.metric("Missing LSOAs",       "86",
                      "All due to boundary mismatch",
                      delta_color="inverse")

        st.markdown("---")

        # Explanation
        st.markdown("#### 🗺️ Why does this happen?")
        col_e1, col_e2 = st.columns(2)

        with col_e1:
            st.markdown("""
            **The ONS Shapefile boundary:**
            - Covers 268 LSOAs
            - Includes Bristol city proper
            - PLUS peripheral LSOAs that cross
              into South Gloucestershire and
              North Somerset

            These boundary LSOAs genuinely have
            residents and house sales — but they
            sit outside the Avon & Somerset Police
            operational recording area.
            """)

        with col_e2:
            st.markdown("""
            **The Police.uk Crime boundary:**
            - Covers only 189 LSOAs
            - Strictly within Avon & Somerset
              Police beat boundaries
            - Peripheral LSOAs recorded under
              neighbouring force areas instead

            This is a **data alignment limitation**
            — two government datasets using
            different geographic boundaries.
            """)

        st.markdown("---")

        # Where are they on the map?
        st.markdown("#### 📍 Where are the missing LSOAs?")
        st.markdown("""
        The missing LSOAs map (generated during investigation) showed
        they are **not in the city centre** — they cluster on Bristol's
        **outer edges**, confirming the boundary mismatch explanation:
        """)

        loc_col1, loc_col2 = st.columns(2)
        with loc_col1:
            st.markdown("""
            **Missing areas confirmed:**
            - 🔴 Large cluster — **north-west** (bordering S. Gloucestershire)
            - 🔴 Smaller cluster — **north-east** border
            - 🔴 Strip along **east** boundary
            - 🔴 Band along **south** boundary (bordering N. Somerset)
            - 🔴 Small **harbour** area cluster
            """)
        with loc_col2:
            st.info(
                "💡 If the missing areas were city-centre commercial zones "
                "(as initially assumed), they would appear in the centre "
                "of the map. The fact that they appear on the **outer edges** "
                "proves this is a boundary mismatch — not a residential "
                "population issue."
            )

        st.markdown("---")

        # What this means
        st.markdown("#### ✅ What this means for the analysis")

        imp1, imp2, imp3 = st.columns(3)

        with imp1:
            st.markdown("""
            **Is the analysis valid?**

            Yes — the 182 LSOAs in the final
            dataset represent the core of
            Bristol where both housing market
            data AND police crime data are
            reliably available.
            """)

        with imp2:
            st.markdown("""
            **Is it a limitation?**

            Yes — peripheral Bristol LSOAs
            are excluded. The analysis
            understates the full geographic
            scope of Bristol and cannot make
            claims about boundary areas.
            """)

        with imp3:
            st.markdown("""
            **How would you fix it?**

            Clip the shapefile to the exact
            Police.uk boundary **before** merging,
            or use a left join with explicit
            zero-filling for missing LSOAs
            rather than silently dropping them.
            """)

        st.success(
            "✅ **Conclusion:** The missing areas are an acknowledged "
            "boundary alignment limitation between ONS geographic boundaries "
            "and Police.uk recording boundaries. All 86 missing LSOAs were "
            "identified, diagnosed, and mapped. The core analysis of Bristol's "
            "182 reliably-covered LSOAs remains valid and interpretable."
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — OLS BASELINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  OLS Baseline Model":
    st.markdown(
        '<div class="main-header">Global OLS Baseline Model</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Step 1: Fit a single average relationship '
        'across all 182 LSOAs — then test whether it is adequate</div>',
        unsafe_allow_html=True)

    # ── Narrative intro ────────────────────────────────────────────────────────
    st.markdown("""
    The global OLS model asks: *on average across all of Bristol, how much
    does a 1-unit increase in log crime reduce log median house price?*
    It gives one answer for the entire city. We fit it first as a baseline —
    then test whether that single answer is actually appropriate.
    """)

    st.markdown("---")

    # ── Section 1: VIF ─────────────────────────────────────────────────────────
    st.markdown("### Step 1 — Check for multicollinearity (VIF)")
    st.markdown("""
    Before fitting OLS, we check whether any predictors are so highly
    correlated with each other that their individual effects cannot be
    reliably estimated. **Variance Inflation Factor (VIF) > 5 is a warning;
    VIF > 10 is serious.**
    """)

    col_vif, col_vif_explain = st.columns([1.2, 1])

    with col_vif:
        vif_data = {
            "Predictor":  ["log(Crime)", "Prop. Flats",
                           "⚠️ Prop. Leasehold",
                           "Dist. to Centre", "Schools", "Dist. to Bus"],
            "VIF":        [11.54, 29.64, 34.73, 7.80, 1.65, 3.86],
            "Status":     ["Keep", "Keep", "❌ Remove",
                           "Keep", "Keep", "Keep"],
        }
        vif_df = pd.DataFrame(vif_data)
        st.table(vif_df)

    with col_vif_explain:
        st.markdown("""
        **Why `prop_leasehold` was removed:**

        In England, flats are almost always sold as leasehold — meaning
        `prop_flats` and `prop_leasehold` capture nearly identical
        information. Their VIF values (29.64 and 34.73) confirm this.

        Including both would make individual coefficient estimates
        unreliable — the model cannot separate the effect of one
        from the other.

        **After removing `prop_leasehold`:**
        All remaining VIF values fall below 5 ✓
        No serious multicollinearity remains.
        """)
        st.success("✓ Final model: 5 predictors, all VIF < 5")

    st.markdown("---")

    # ── Section 2: Model results ───────────────────────────────────────────────
    st.markdown("### Step 2 — Fit OLS and interpret results")

    col_l, col_r = st.columns([1, 1.3])

    with col_l:
        st.markdown("#### Model fit")

        ols_r2     = stats.get("ols_r2",     0.1084)
        ols_adj_r2 = stats.get("ols_adj_r2", 0.0831)
        ols_aicc   = stats.get("ols_aicc",  -84.17)

        m1, m2, m3 = st.columns(3)
        m1.metric("R²",          f"{ols_r2:.3f}")
        m2.metric("Adjusted R²", f"{ols_adj_r2:.3f}")
        m3.metric("AICc",        f"{ols_aicc:.1f}")

        st.error(
            f"🔴 **R² = {ols_r2:.3f}** — OLS explains only "
            f"**{ols_r2*100:.1f}%** of the variation in log median "
            f"house prices across Bristol's 182 LSOAs. The remaining "
            f"**{(1-ols_r2)*100:.1f}%** is unexplained. This is a "
            f"deliberately parsimonious model — but the low R² also "
            f"signals that a *global* average is a poor fit for a "
            f"spatially heterogeneous city."
        )

        st.markdown("#### Crime coefficient interpretation")
        st.markdown("""
        The crime coefficient of **−0.0778** is the most important result.

        Using the formula `(exp(β) − 1) × 100`:
        """)
        crime_pct = (np.exp(-0.0778) - 1) * 100
        st.metric(
            "Price effect of crime",
            f"{crime_pct:.1f}% per log unit",
            delta="Statistically significant (p < 0.001)",
            delta_color="inverse"
        )
        st.markdown("""
        A one-unit increase in log(total crimes) is associated with a
        **~7.5% decrease** in median house price, holding all other
        predictors constant.

        But this is a *city-wide average* — GWR will show this effect
        ranges from **−22.6%** to **+2.6%** depending on location.
        """)

    with col_r:
        st.markdown("#### Which predictors matter? (95% confidence intervals)")
        st.markdown(
            "Blue dots = significant (p < 0.05). "
            "Grey = not significant. "
            "If the CI crosses zero, the effect is indistinguishable from chance.")

        coefs   = [-0.0778, -0.0079, -0.0336, -0.0253, -0.0842]
        pvals   = [ 0.001,   0.894,  0.018,  0.106,  0.534]
        labels  = ["log(Crime)", "Prop. Flats",
                   "Dist. to Centre (km)",
                   "Schools Count", "Dist. to Bus Stop (km)"]
        ci_half = [ 0.023,   0.095,  0.009,  0.016,  0.071]
        colours = ["#4c72b0" if p < 0.05 else "#bbbbbb" for p in pvals]

        fig = go.Figure()
        for i, (label, coef, p, ci, colour) in enumerate(
            zip(labels, coefs, pvals, ci_half, colours)
        ):
            fig.add_trace(go.Scatter(
                x=[coef], y=[label], mode="markers",
                marker=dict(size=13, color=colour,
                            line=dict(width=1.5, color="white")),
                error_x=dict(type="data", array=[ci], arrayminus=[ci],
                             visible=True, color=colour, thickness=2.5),
                showlegend=False,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"Coefficient: {coef:.4f}<br>"
                    f"95% CI: [{coef-ci:.4f}, {coef+ci:.4f}]<br>"
                    f"p-value: {p:.3f}<br>"
                    f"{'✓ Significant' if p < 0.05 else '✗ Not significant'}"
                    f"<extra></extra>"),
            ))

        fig.add_vline(x=0, line_dash="dash",
                      line_color="black", opacity=0.4)
        fig.add_vrect(x0=-0.02, x1=0.02,
                      fillcolor="grey", opacity=0.05,
                      line_width=0,
                      annotation_text="Near zero",
                      annotation_position="top")
        fig.update_layout(
            xaxis_title="Coefficient (effect on log median house price)",
            plot_bgcolor="white", height=320,
            margin=dict(l=180, r=20, t=30, b=50),
            xaxis=dict(zeroline=True, zerolinecolor="black",
                       zerolinewidth=1))
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "💡 Only **log(Crime)** (p < 0.001) and **Distance to Centre** "
            "(p = 0.018) are statistically significant. Proportion of flats, "
            "schools count, and bus distance show wide CIs crossing zero — "
            "their effects cannot be reliably estimated at the global level. "
            "GWR may reveal significant *local* effects hidden by the global average."
        )

    st.markdown("---")

    # ── Section 3: Moran's I ───────────────────────────────────────────────────
    st.markdown("### Step 3 — Test whether OLS residuals are spatially random")
    st.markdown("""
    If OLS is a good fit, its residuals (prediction errors) should be
    **spatially random** — no pattern in where the model over- or
    under-predicts. We test this formally with **Moran's I**.
    """)

    col_m, col_map = st.columns([1, 1.5])

    with col_m:
        mi   = stats.get("morans_I",      0.5408)
        mi_p = stats.get("morans_pvalue", 0.001)

        st.markdown("#### Moran's I result")
        a1, a2 = st.columns(2)
        a1.metric("Moran's I",   f"{mi:.4f}")
        a2.metric("p-value",     f"{mi_p:.3f}")

        st.markdown(f"""
        **Interpretation:**
        - Moran's I ranges from −1 (dispersed) to +1 (clustered)
        - Expected value under randomness: −0.006
        - Observed value: **{mi:.4f}** — far above expected
        - p = {mi_p:.3f} → **statistically significant**

        This means neighbouring LSOAs have **similar prediction errors**.
        Where OLS under-predicts in one area, it tends to
        under-predict in surrounding areas too.
        """)

        st.error(
            f"🔴 **OLS FAILS the spatial independence test.** "
            f"Moran's I = {mi:.4f} (p = {mi_p:.3f}) confirms strong "
            f"positive spatial autocorrelation in residuals. "
            f"The global model systematically misses spatial structure "
            f"in Bristol's housing market. **GWR is needed.**"
        )

    with col_map:
        if DATA_AVAILABLE and "ols_residual" in reg_df.columns:
            st.markdown("#### Residual map — can you see the clustering?")
            max_abs = reg_df["ols_residual"].abs().max()
            fig_r = make_choropleth(
                geojson=geojson,
                locations=reg_df["lsoa_code"],
                z=reg_df["ols_residual"],
                colorscale=[
                    [0,   "#2166ac"],
                    [0.5, "#f7f7f7"],
                    [1,   "#d6604d"]],
                title="OLS Residuals by LSOA — Bristol",
                colorbar_title="Residual<br>(log price)",
                zmin=-max_abs, zmax=max_abs,
                hover_template=(
                    "<b>%{location}</b><br>"
                    "Residual: %{z:.3f}<br>"
                    "Positive = under-predicted<br>"
                    "Negative = over-predicted"
                    "<extra></extra>"),
                height=420,
            )
            st.plotly_chart(fig_r, use_container_width=True)
            st.caption(
                "🔴 Red cluster (north/north-west) = OLS consistently "
                "under-predicts prices here. "
                "🔵 Blue cluster (south/south-east) = OLS consistently "
                "over-predicts. This spatial pattern is exactly what "
                "Moran's I = 0.5408 quantifies.")
        else:
            st.info("Upload regression_dataset.geojson to see the residual map.")

    st.markdown("---")
    st.success(
        "✅ **Conclusion from OLS:** The global model explains only 10.8% "
        "of price variation, and its residuals are strongly spatially "
        "clustered (Moran's I = 0.5408, p = 0.001). A spatially varying "
        "model — **Geographically Weighted Regression** — is both "
        "statistically justified and necessary. → See GWR Results"
    )
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — GWR RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️  GWR Results":
    st.markdown(
        '<div class="main-header">GWR Results</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Allowing the crime-price relationship '
        'to vary across Bristol\'s 182 neighbourhoods</div>',
        unsafe_allow_html=True)

    st.markdown("""
    Geographically Weighted Regression fits a **separate local regression
    at each LSOA**, weighted by distance so that nearby LSOAs have more
    influence than distant ones. The result: instead of one crime coefficient
    for all of Bristol, we get **182 local coefficients** — one per neighbourhood.
    """)

    st.markdown("---")

    # ── Key metrics ────────────────────────────────────────────────────────────
    st.markdown("### How much better is GWR than OLS?")

    gwr_r2  = stats.get("gwr_r2",             0.7393)
    ols_r2  = stats.get("ols_r2",             0.1084)
    gwr_adj = stats.get("gwr_adj_r2",         0.6622)
    ols_adj = stats.get("ols_adj_r2",         0.0831)
    gwr_aic = stats.get("gwr_aicc",         -209.48)
    ols_aic = stats.get("ols_aicc",          -84.17)
    gwr_min = stats.get("gwr_crime_coef_min", -0.2562)
    gwr_max = stats.get("gwr_crime_coef_max",  0.0253)
    gwr_med = stats.get("gwr_crime_coef_med", -0.1148)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color:#2e7d32">+{:.0f}%</div>
            <div class="metric-label">R² improvement<br>
            OLS: {:.3f} → GWR: {:.3f}<br>
            Same 5 predictors — spatial variation only</div>
        </div>
        """.format((gwr_r2 - ols_r2) * 100, ols_r2, gwr_r2),
        unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color:#2e7d32">{:.0f} pts</div>
            <div class="metric-label">AICc improvement<br>
            OLS: {:.1f} → GWR: {:.1f}<br>
            &gt;10 pts = strong evidence (Burnham &amp; Anderson, 2002)</div>
        </div>
        """.format(abs(gwr_aic - ols_aic), ols_aic, gwr_aic),
        unsafe_allow_html=True)

    with col_c:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color:#1f4e79">3×</div>
            <div class="metric-label">Range of local crime effect<br>
            OLS global: −0.078 (one number)<br>
            GWR local: {:.3f} to +{:.3f} (182 numbers)</div>
        </div>
        """.format(gwr_min, gwr_max),
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.success(
        f"✅ **GWR explains {gwr_r2*100:.1f}% of price variation** vs "
        f"{ols_r2*100:.1f}% for OLS — an improvement of "
        f"+{(gwr_r2-ols_r2)*100:.1f} percentage points using exactly "
        f"the same 5 predictors. The improvement comes **entirely** from "
        f"allowing coefficients to vary spatially across Bristol."
    )

    st.markdown("---")

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🗺️ Crime coefficient map",
        "📊 Coefficient distribution",
        "📋 Model comparison"
    ])

    # ── Tab 1: Map ─────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Where does crime hurt house prices most?")
        st.markdown("""
        Each LSOA is coloured by its **local GWR crime coefficient** —
        the estimated effect of crime on house prices *in that specific
        neighbourhood*, after controlling for housing type, centrality,
        schools, and bus access.
        """)

        c1, c2, c3 = st.columns(3)
        c1.metric("Strongest negative effect",
                  f"{gwr_min:.4f}",
                  f"≈ {(np.exp(gwr_min)-1)*100:.1f}% per log crime unit",
                  delta_color="inverse")
        c2.metric("Median effect",
                  f"{gwr_med:.4f}",
                  f"≈ {(np.exp(gwr_med)-1)*100:.1f}% per log crime unit",
                  delta_color="inverse")
        c3.metric("Weakest / positive effect",
                  f"+{gwr_max:.4f}",
                  f"≈ +{(np.exp(gwr_max)-1)*100:.1f}% per log crime unit")

        st.markdown("<br>", unsafe_allow_html=True)

        if DATA_AVAILABLE and "gwr_crime_coef" in reg_df.columns:
            max_abs = max(abs(reg_df["gwr_crime_coef"].min()),
                         abs(reg_df["gwr_crime_coef"].max()))

            fig_gwr = go.Figure(go.Choroplethmapbox(
                geojson=geojson,
                locations=reg_df["lsoa_code"],
                z=reg_df["gwr_crime_coef"],
                featureidkey="properties.lsoa_code",
                colorscale=[
                    [0.0, "#d6604d"],
                    [0.25, "#f4a582"],
                    [0.5,  "#f7f7f7"],
                    [0.75, "#92c5de"],
                    [1.0,  "#2166ac"]],
                zmin=-max_abs, zmax=max_abs,
                colorbar=dict(
                    title="Crime<br>coefficient",
                    thickness=14,
                    tickvals=[round(-max_abs, 2),
                              round(-max_abs/2, 2), 0,
                              round(max_abs/2, 2),
                              round(max_abs, 2)],
                    ticktext=[
                        f"{round(-max_abs,2)}<br>Strong −",
                        f"{round(-max_abs/2,2)}",
                        "0",
                        f"{round(max_abs/2,2)}",
                        f"+{round(max_abs,2)}<br>Positive"]),
                marker=dict(opacity=0.82, line_width=0.4),
                hovertemplate=(
                    "<b>%{location}</b><br>"
                    "Crime coefficient: %{z:.4f}<br>"
                    "Price effect: ~" +
                    "%{customdata:.1f}% per log crime unit"
                    "<extra></extra>"),
                customdata=[
                    (np.exp(c) - 1) * 100
                    for c in reg_df["gwr_crime_coef"]],
            ))

            fig_gwr.update_layout(
                mapbox=dict(
                    style="carto-positron",
                    center=dict(lat=51.4545, lon=-2.5879),
                    zoom=11.2),
                margin=dict(l=0, r=0, t=10, b=0),
                height=580,
            )
            st.plotly_chart(fig_gwr, use_container_width=True)

            pct_neg = (
                (reg_df["gwr_crime_coef"] < -0.05).sum()
                / len(reg_df) * 100
            )
            col_i1, col_i2 = st.columns(2)
            col_i1.error(
                f"🔴 **{pct_neg:.0f}% of LSOAs (dark red):** Crime "
                f"meaningfully lowers prices (coef < −0.05). Concentrated "
                f"in outer and south-eastern Bristol where no strong "
                f"amenity advantages exist to offset crime's negative signal."
            )
            col_i2.info(
                f"🔵 **{100-pct_neg:.0f}% of LSOAs (white/blue):** Crime "
                f"has little or no negative price effect. Concentrated near "
                f"the city centre where proximity, schools, and transport "
                f"sustain demand despite elevated crime levels."
            )
        else:
            st.warning(
                "GWR coefficient column not found. Re-run the save cell "
                "in your notebook to include `gwr_crime_coef`.")

    # ── Tab 2: Distribution ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("### How much does the crime effect vary?")
        st.markdown("""
        The histogram shows all 182 local GWR crime coefficients.
        The **global OLS estimate** (−0.078) is shown as a dashed red line —
        it represents the city-wide average that OLS forces on every LSOA.
        Notice how wide the actual distribution is.
        """)

        if DATA_AVAILABLE and "gwr_crime_coef" in reg_df.columns:
            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(
                x=reg_df["gwr_crime_coef"],
                nbinsx=25,
                marker_color="#4c72b0",
                opacity=0.85,
                name="GWR local coefficients",
                hovertemplate=(
                    "Coefficient: %{x:.3f}<br>"
                    "LSOAs: %{y}<extra></extra>"),
            ))

            # OLS line
            fig_h.add_vline(
                x=-0.0778, line_dash="dash",
                line_color="crimson", line_width=2.5,
                annotation_text="<b>OLS global: −0.078</b><br>"
                                "(what OLS forces on every LSOA)",
                annotation_position="top right",
                annotation_font_color="crimson",
                annotation_font_size=11)

            # GWR median line
            fig_h.add_vline(
                x=gwr_med, line_dash="dot",
                line_color="#1f4e79", line_width=2.5,
                annotation_text=f"<b>GWR median: {gwr_med:.3f}</b>",
                annotation_position="top left",
                annotation_font_color="#1f4e79",
                annotation_font_size=11)

            # Shade negative region
            fig_h.add_vrect(
                x0=reg_df["gwr_crime_coef"].min(), x1=-0.05,
                fillcolor="rgba(214,96,77,0.08)",
                line_width=0,
                annotation_text="Meaningful negative effect",
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color="#d6604d")

            fig_h.update_layout(
                xaxis_title="Local crime coefficient (GWR)",
                yaxis_title="Number of LSOAs",
                title="<b>Distribution of GWR Crime Coefficients "
                      "— 182 Bristol LSOAs</b>",
                plot_bgcolor="white",
                height=420,
                showlegend=False,
            )
            st.plotly_chart(fig_h, use_container_width=True)

            # Key insight
            range_val = gwr_max - gwr_min
            ols_val   = -0.0778
            st.warning(
                f"⚠️ **The global OLS estimate of −0.078 conceals a range "
                f"of {range_val:.3f} log points** ({gwr_min:.4f} to "
                f"+{gwr_max:.4f}). In the most crime-sensitive LSOAs, "
                f"a one-unit increase in log crime is associated with a "
                f"**{(np.exp(gwr_min)-1)*100:.1f}% price decrease** — "
                f"nearly three times the global OLS estimate of "
                f"{(np.exp(ols_val)-1)*100:.1f}%. Using OLS alone would "
                f"systematically mis-estimate the crime-price relationship "
                f"for the majority of Bristol neighbourhoods."
            )

            a, b, c = st.columns(3)
            a.metric("Minimum (strongest −)",
                     f"{gwr_min:.4f}",
                     f"{(np.exp(gwr_min)-1)*100:.1f}%",
                     delta_color="inverse")
            b.metric("Median",
                     f"{gwr_med:.4f}",
                     f"{(np.exp(gwr_med)-1)*100:.1f}%",
                     delta_color="inverse")
            c.metric("Maximum (weakest / +)",
                     f"+{gwr_max:.4f}",
                     f"+{(np.exp(gwr_max)-1)*100:.1f}%")

        else:
            st.warning("Upload regression_dataset.geojson to see this chart.")

    # ── Tab 3: Model comparison ─────────────────────────────────────────────────
    with tab3:
        st.markdown("### OLS vs GWR — a rigorous comparison")
        st.markdown("""
        Three standard metrics are used to compare the models.
        All three tell the same story: **GWR is substantially better.**
        """)

        col_t, col_e = st.columns([1, 1.2])

        with col_t:
            st.markdown("#### Performance metrics")
            st.table(pd.DataFrame({
                "Metric":      ["R²", "Adjusted R²", "AICc"],
                "OLS":         [f"{ols_r2:.4f}",
                                f"{ols_adj:.4f}",
                                f"{ols_aic:.2f}"],
                "GWR":         [f"{gwr_r2:.4f}",
                                f"{gwr_adj:.4f}",
                                f"{gwr_aic:.2f}"],
                "Improvement": [
                    f"+{gwr_r2-ols_r2:.4f}",
                    f"+{gwr_adj-ols_adj:.4f}",
                    f"{gwr_aic-ols_aic:.2f} (lower = better)"],
            }))

        with col_e:
            st.markdown("#### What each metric means")
            st.markdown(f"""
            **R² (+{gwr_r2-ols_r2:.3f}):**
            GWR explains {gwr_r2*100:.1f}% of price variation vs
            {ols_r2*100:.1f}% for OLS. The improvement comes
            entirely from spatial variation — same predictors, same data.

            **Adjusted R² (+{gwr_adj-ols_adj:.3f}):**
            Penalises model complexity. Even after accounting for GWR's
            additional effective parameters, it still outperforms OLS
            substantially. The extra parameters are earning their keep.

            **AICc ({gwr_aic-ols_aic:.1f}):**
            Balances fit vs complexity. A difference > 10 points is
            considered strong evidence in favour of the lower-AICc model
            (Burnham & Anderson, 2002). A difference of
            **{abs(gwr_aic-ols_aic):.0f} points** is overwhelming evidence.
            """)

        st.success(
            f"✅ **Verdict:** GWR outperforms OLS on all three metrics "
            f"simultaneously. The {abs(gwr_aic-ols_aic):.0f}-point AICc "
            f"difference provides overwhelming statistical evidence that "
            f"the crime-price relationship varies spatially across Bristol "
            f"— and that a spatially varying model is both warranted and "
            f"necessary. → See Key Findings for policy implications."
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — KEY FINDINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Key Findings":
    st.markdown(
        '<div class="main-header">Key Findings</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">What the analysis reveals about '
        'Bristol\'s housing market — and why it matters</div>',
        unsafe_allow_html=True)

    st.markdown("""
    This project set out to answer one question:
    **Does crime affect house prices in Bristol — and does this relationship
    vary spatially across neighbourhoods?**
    The answer is yes — but with important nuance that only GWR can reveal.
    """)

    st.markdown("---")

    # ── The story in numbers ───────────────────────────────────────────────────
    st.markdown("### The story in numbers")

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.markdown("""
    <div class="metric-card">
        <div class="metric-value">−0.23</div>
        <div class="metric-label">Pearson r<br>crime vs price<br>(p &lt; 0.05)</div>
    </div>""", unsafe_allow_html=True)
    s2.markdown("""
    <div class="metric-card">
        <div class="metric-value">10.8%</div>
        <div class="metric-label">Variance explained<br>by global OLS<br>(insufficient)</div>
    </div>""", unsafe_allow_html=True)
    s3.markdown("""
    <div class="metric-card">
        <div class="metric-value" style="color:#2e7d32">73.9%</div>
        <div class="metric-label">Variance explained<br>by GWR<br>(same predictors)</div>
    </div>""", unsafe_allow_html=True)
    s4.markdown("""
    <div class="metric-card">
        <div class="metric-value">3×</div>
        <div class="metric-label">Range of local effect<br>vs OLS global<br>estimate</div>
    </div>""", unsafe_allow_html=True)
    s5.markdown("""
    <div class="metric-card">
        <div class="metric-value">76%</div>
        <div class="metric-label">LSOAs with<br>meaningful negative<br>crime effect</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Five findings ──────────────────────────────────────────────────────────
    st.markdown("### Five key findings")

    # Finding 1
    with st.expander("✅ Finding 1 — Crime and prices ARE negatively correlated", expanded=True):
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.markdown("""
            Both Pearson and Spearman correlation tests return **Pearson r = −0.23,
            Spearman ρ = −0.28** (both p < 0.05) between log total crimes and
            log median house price across Bristol's 182 LSOAs.

            An independent samples t-test confirms that **high-crime LSOAs
            have statistically significantly lower median prices** than
            low-crime areas — approximately 7–10% cheaper on average.

            **What this means:** The negative relationship is real and
            statistically robust. It is not driven by outliers — the
            Spearman (rank-based) result matches Pearson exactly.
            """)
        with col2:
            st.metric("Pearson r",  "−0.23", "p < 0.05")
            st.metric("Spearman ρ", "−0.28", "p < 0.05")
            st.metric("Price gap",  "~7–10%",
                      "high vs low crime areas",
                      delta_color="inverse")

    # Finding 2
    with st.expander("✅ Finding 2 — A global model is insufficient"):
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.markdown("""
            The global OLS model explains only **10.8%** of price variation
            (R² = 0.108). More critically, Moran's I applied to the OLS
            residuals returns **I = 0.5408 (p = 0.001)** — confirming
            strong positive spatial autocorrelation.

            This means neighbouring LSOAs have **systematically similar
            prediction errors**. The global model consistently under-predicts
            in the north-west and over-predicts in the south — a clear
            spatial pattern that OLS cannot capture.

            **What this means:** The assumption that crime affects prices
            equally everywhere in Bristol is wrong. A spatially varying
            model is needed.
            """)
        with col2:
            st.metric("OLS R²",    "0.108",  "Only 10.8% explained")
            st.metric("Moran's I", "0.5408", "p = 0.001",
                      delta_color="inverse")
            st.metric("Verdict",   "❌ OLS fails",
                      "independence assumption violated",
                      delta_color="off")

    # Finding 3
    with st.expander("✅ Finding 3 — GWR substantially outperforms OLS"):
        col1, col2 = st.columns([1.5, 1])
        with col1:
            gwr_r2 = stats.get("gwr_r2", 0.7393)
            ols_r2 = stats.get("ols_r2", 0.1084)
            gwr_aic = stats.get("gwr_aicc", -209.48)
            ols_aic = stats.get("ols_aicc", -84.17)
            st.markdown(f"""
            Using the same 5 predictors, GWR achieves **R² = {gwr_r2:.3f}**
            compared to {ols_r2:.3f} for OLS — an improvement of
            **+{(gwr_r2-ols_r2):.3f}**.

            The AICc improves by **{abs(gwr_aic-ols_aic):.0f} points**
            (from {ols_aic:.1f} to {gwr_aic:.1f}). In model selection,
            a difference greater than 10 points is considered strong
            evidence in favour of the lower-AICc model — a difference
            of {abs(gwr_aic-ols_aic):.0f} points is overwhelming.

            **What this means:** The dramatic improvement comes entirely
            from allowing coefficients to vary spatially. Bristol's housing
            market cannot be adequately characterised by any single
            global model.
            """)
        with col2:
            st.metric("GWR R²",
                      f"{gwr_r2:.3f}",
                      f"+{gwr_r2-ols_r2:.3f} vs OLS")
            st.metric("AICc improvement",
                      f"{abs(gwr_aic-ols_aic):.0f} points",
                      "Overwhelming evidence")
            st.metric("Verdict", "✅ GWR wins",
                      "on all 3 metrics",
                      delta_color="off")

    # Finding 4
    with st.expander("✅ Finding 4 — The crime effect varies dramatically across space"):
        col1, col2 = st.columns([1.5, 1])
        with col1:
            gwr_min = stats.get("gwr_crime_coef_min", -0.2562)
            gwr_max = stats.get("gwr_crime_coef_max",  0.0253)
            st.markdown(f"""
            The local GWR crime coefficient ranges from **{gwr_min:.4f}
            to +{gwr_max:.4f}** across Bristol's 182 LSOAs.

            Converting to percentage terms:
            - **Strongest effect:** {(np.exp(gwr_min)-1)*100:.1f}% price
              decrease per log crime unit (outer/south-east Bristol)
            - **Global OLS:** {(np.exp(-0.0778)-1)*100:.1f}% (city-wide average)
            - **Weakest effect:** +{(np.exp(gwr_max)-1)*100:.1f}% (near-zero,
              city-centre LSOAs)

            The strongest local effect is nearly **3× larger** than the
            global OLS estimate — meaning OLS systematically under-estimates
            crime's impact in peripheral neighbourhoods and over-estimates
            it in central ones.

            **What this means:** Any policy using a single city-wide estimate
            will be wrong for most neighbourhoods.
            """)
        with col2:
            st.metric("Min coefficient",
                      f"{gwr_min:.4f}",
                      f"≈ {(np.exp(gwr_min)-1)*100:.1f}% effect",
                      delta_color="inverse")
            st.metric("OLS estimate",
                      "−0.0778",
                      "≈ −7.5% (global average)",
                      delta_color="inverse")
            st.metric("Max coefficient",
                      f"+{gwr_max:.4f}",
                      f"≈ +{(np.exp(gwr_max)-1)*100:.1f}% effect")

    # Finding 5
    with st.expander("✅ Finding 5 — Amenities explain the spatial paradox"):
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.markdown("""
            A small number of LSOAs show **high crime AND high prices**
            simultaneously — apparently defying the overall negative
            relationship. These hotspot LSOAs have:

            - **74% more schools** per LSOA (1.20 vs 0.69)
            - **Slightly better bus access** (0.162 vs 0.182 km)
            - **Marginally more central** location (3.151 vs 3.217 km)

            Strong educational provision in particular appears to sustain
            housing demand even in high-crime areas. Parents and families
            appear willing to accept higher local crime in exchange for
            access to more schools.

            **What this means:** Crime reduction is most effective at
            raising prices in peripheral, amenity-poor areas. In central
            areas, investing in schools and transport may be a more
            powerful lever than crime reduction alone.
            """)
        with col2:
            st.metric("Schools — hotspots",  "1.20 per LSOA")
            st.metric("Schools — others",    "0.69 per LSOA",
                      "−42% fewer",
                      delta_color="inverse")
            st.metric("School density gap",  "+74%",
                      "in high-crime, high-price areas")

    st.markdown("---")

    # ── Policy implications ────────────────────────────────────────────────────
    st.markdown("### Policy implications")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("""
        **🏘️ Peripheral neighbourhoods**

        Crime reduction will have the **largest positive effect on house
        prices** in outer and south-eastern Bristol — LSOAs with strong
        negative GWR coefficients and limited amenity provision.

        Targeted policing and community safety investment in these areas
        is likely to deliver measurable property value uplift.
        """)

    with p2:
        st.markdown("""
        **🏙️ City-centre neighbourhoods**

        In central, well-served areas where the GWR coefficient is near
        zero, **investment in schools and transport** may be a more
        effective lever for sustaining property values than crime
        reduction alone.

        Amenity provision sustains demand even where crime is elevated.
        """)

    with p3:
        st.markdown("""
        **📊 Evidence-based planning**

        A single city-wide crime statistic hides fundamentally different
        local dynamics. **Spatially differentiated policy** — using
        neighbourhood-level GWR coefficients to prioritise interventions
        — is both empirically justified and practically necessary.
        """)

    st.markdown("---")

    # ── Limitations and future work ────────────────────────────────────────────
    col_lim, col_fut = st.columns(2)

    with col_lim:
        st.markdown("### Limitations")
        for lim in [
            "Cross-sectional analysis — cannot establish causality",
            "Total crime aggregates all offence types equally",
            "Omitted variables: deprivation, property size, age, condition",
            "GWR sensitive to bandwidth choice",
            "Local estimates less stable than global OLS",
            "Boundary mismatch: ONS shapefile (268 LSOAs) vs Police.uk "
            "recording area (189 LSOAs) — 86 peripheral LSOAs excluded "
            "due to missing crime data (see Data Coverage tab for full investigation)",
        ]:
            st.markdown(f"- {lim}")

    with col_fut:
        st.markdown("### Future work")
        for fut in [
            "Add IMD deprivation scores to reduce omitted variable bias",
            "Disaggregate by individual crime types (violent, burglary, ASB)",
            "Panel methods to track annual crime-price changes 2021–2025",
            "Replicate in other UK cities to test generalisability",
            "Causal inference via instrumental variables or diff-in-diff",
        ]:
            st.markdown(f"- {fut}")

    st.markdown("---")

    # ── Resources ──────────────────────────────────────────────────────────────
    st.markdown("### Resources")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""
        **📂 Code & data**
        - [GitHub Repository](https://github.com/layaung-linnlett/bristol-crime-houseprices-gwr)
        - Full notebook with reproducible pipeline
        - src/ modules for reuse
        """)
    with r2:
        st.markdown("""
        **📊 Data sources**
        - [HM Land Registry](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads)
        - [Police.uk](https://data.police.uk/data/)
        - [ONS Geoportal](https://geoportal.statistics.gov.uk/)
        """)
    with r3:
        st.markdown("""
        **👤 Author**

        La Yaung Linn Lett

        BSc Data Science & AI

        UWE Bristol, 2026
        """)
