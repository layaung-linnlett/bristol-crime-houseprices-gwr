# Bristol Crime & House Prices — Spatial Analysis Using GWR

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)](https://colab.research.google.com/)

I investigate whether and how crime levels influence house prices across Bristol's 182 Lower Layer Super Output Areas (LSOAs), 2021–2025, using Geographically Weighted Regression (GWR) to see how that relationship changes across the city rather than assuming one number fits everywhere.

---

## Key findings

I compare a global OLS baseline against GWR to test whether a single average effect is hiding different local dynamics between neighbourhoods. It is.

| Metric | OLS | GWR |
|--------|-----|-----|
| R² | 0.108 | 0.739 |
| Adjusted R² | 0.083 | 0.662 |
| AICc | −84.17 | −209.48 |

- **Moran's I = 0.5408** (p = 0.001) on OLS residuals confirms strong spatial autocorrelation, which is the statistical justification for using GWR instead of a global model
- **GWR crime coefficients range from −0.2562 to +0.0253** across LSOAs — the global average of −0.078 hides very different local effects depending on where you look
- **76% of LSOAs** show a meaningfully negative crime effect (coefficient < −0.05)
- LSOAs where crime is high but prices stay high anyway have around 73% more schools per area on average than other LSOAs (1.20 vs 0.69 schools per LSOA) — a plausible reason crime doesn't suppress prices there
- Dataset: 182 LSOAs, 34,543 house transactions, 159,666 crime records over 2021–2025

---

## Screenshots

Not added yet — the project currently runs as a Jupyter notebook (Colab) plus a separate Streamlit dashboard (`app.py`). See [How to run](#how-to-run) below to run either one locally.

---

## Tech stack

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data handling |
| `geopandas`, `shapely` | Geospatial operations |
| `statsmodels` | OLS regression with full inference |
| `mgwr` | Geographically Weighted Regression |
| `esda`, `libpysal` | Moran's I spatial autocorrelation |
| `scikit-learn` | Nearest-neighbour distance calculations |
| `matplotlib`, `seaborn` | Static visualisation |
| `plotly` | Interactive visualisation (notebook + Streamlit app) |
| `streamlit` | Interactive dashboard (`app.py`) |
| `pyproj` | Coordinate reprojection (BNG ↔ WGS84) |

---

## Methodology

### Pipeline

```
Raw data (5 sources)
    → Data cleaning & filtering (Bristol only, IQR outlier removal)
    → LSOA-level aggregation (prices, crimes, housing composition)
    → Feature engineering (log transforms, distances, school density)
    → Regression dataset (182 LSOAs, GeoDataFrame)
    → Exploratory analysis (maps, heatmap, t-test)
    → VIF diagnostic (removed prop_leasehold, VIF > 29)
    → OLS baseline (statsmodels, full inference)
    → Moran's I (spatial autocorrelation test)
    → GWR (adaptive bisquare kernel, AICc bandwidth selection)
    → Model comparison + visualisation
    → Additional analysis (hotspots, property types)
```

### Model specification

```
log(median_price) = β₀ + β₁·log(crime) + β₂·prop_flats
                      + β₃·dist_centre_km + β₄·schools_count
                      + β₅·dist_nearest_bus_km + ε
```

**Why `prop_leasehold` was excluded:** the initial VIF check found severe collinearity between `prop_flats` (VIF = 29.64) and `prop_leasehold` (VIF = 34.73). In the UK, flats are almost always leasehold, so the two variables capture nearly identical information. I removed `prop_leasehold`; all remaining VIF values fell below 5.

### Data sources

All data is openly available. Download and place in the `data/raw/` directories as described in [`data/README.md`](data/README.md).

| Dataset | Source | Period |
|---------|--------|--------|
| House price (price-paid) | [HM Land Registry](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads) | 2021–2025 |
| Crime records | [Police.uk](https://data.police.uk/data/) | 2021–2025 |
| LSOA boundaries | [ONS Geoportal](https://geoportal.statistics.gov.uk/) | 2021 |
| Postcode-to-LSOA lookup | [ONS Geoportal](https://geoportal.statistics.gov.uk/) | 2025 |
| School locations | [West of England Combined Authority](https://opendata.westofengland-ca.gov.uk/explore/dataset/schools-lep/table/) | 2024 |
| Bus stops | [Bristol Open Data](https://opendata.bristol.gov.uk/datasets/bus-stops-2/explore) | Current |

### Visualisations produced

| Plot | Type | Description |
|------|------|-------------|
| Price distribution | Static (matplotlib) | Right-skewed histogram justifying log transform |
| Crime trends | Static (matplotlib) | Annual counts by top 5 crime types, 2021–2025 |
| Correlation heatmap | Static (seaborn) | Pairwise correlations — complements the VIF check |
| OLS coefficient plot | Static (matplotlib) | Forest plot with 95% CIs, coloured by significance |
| Crime vs price scatter | Interactive (Plotly) | Log-log scatter coloured by distance to centre |
| Price & crime maps | Interactive (Plotly) | Side-by-side choropleth — spatial EDA |
| OLS residual map | Interactive (Plotly) | Diverging scale — visualises the Moran's I result |
| GWR coefficient maps | Interactive (Plotly) | Local crime & distance-to-centre coefficients |

---

## Project structure

```
bristol-crime-houseprices-gwr/
│
├── app.py                                        # Streamlit dashboard (uses data/ directly, no geopandas)
├── requirements.txt                               # Dependencies for app.py
├── requirements-notebook.txt                      # Dependencies for the full notebook pipeline
│
├── notebooks/
│   ├── GWR_crime_price_project.ipynb             # Main analysis notebook
│   └── individual_project_final_version.ipynb    # Final submitted version (includes missing-LSOA investigation)
│
├── src/                                           # Analysis functions extracted from the notebook, for reuse
│   ├── data_loading.py
│   ├── cleaning.py
│   ├── aggregation.py
│   ├── features.py
│   ├── modelling.py
│   └── visualization.py
│
├── data/
│   ├── README.md                                 # Download instructions for raw data
│   ├── regression_dataset.geojson                # Processed data bundled for the Streamlit app
│   ├── full_boundaries.geojson
│   ├── summary_statistics.json
│   ├── house_prices_bristol_clean.zip
│   ├── crime_bristol_clean.zip
│   └── raw/                                      # Not tracked — populate locally, see data/README.md
│
├── outputs/                                       # Generated by the notebook (Colab), not tracked in git
│
├── .streamlit/config.toml                        # Streamlit theme config
├── LICENSE                                        # MIT licence
└── README.md
```

---

## How to run

### Option 1 — Google Colab (recommended for the notebook)

1. Upload the notebook to Google Colab
2. Mount your Google Drive
3. Place raw data in `My Drive/Bristol_Project/data/raw/`
4. Run all cells — the execute cell auto-detects saved outputs

```python
# First run: runs the full pipeline (~10–15 minutes for GWR)
# After restart: loads from saved GeoJSON (~30 seconds)
```

### Option 2 — Local Jupyter

```bash
git clone https://github.com/layaung-linnlett/bristol-crime-houseprices-gwr.git
cd bristol-crime-houseprices-gwr
pip install -r requirements-notebook.txt
jupyter notebook notebooks/individual_project_final_version.ipynb
```

Update `BASE_PATH` in the notebook to point to your local data directory.

### Option 3 — Streamlit dashboard

The dashboard reads the processed data already committed under `data/` (no raw data download needed):

```bash
git clone https://github.com/layaung-linnlett/bristol-crime-houseprices-gwr.git
cd bristol-crime-houseprices-gwr
pip install -r requirements.txt
streamlit run app.py
```

---

## Limitations & future work

**Limitations**

- Cross-sectional analysis — cannot establish causality
- Total crime count aggregates all offence types equally
- Key omitted variables: deprivation index, property size, age, condition
- GWR is sensitive to bandwidth choice; local estimates are less stable than global OLS

**Future work**

- Panel methods to track annual crime–price changes across 2021–2025, rather than aggregating the whole period
- Replicate the analysis in other UK cities to test how well it generalises
- Disaggregate crime by type (violent crime, burglary, anti-social behaviour may each have a different spatial price effect)
- Normalise crime counts by resident population or area instead of using raw counts

---

## Contact

**La Yaung Linn Lett**
BSc Data Science and AI, University of the West of England, Bristol — 2026
[GitHub repository](https://github.com/layaung-linnlett/bristol-crime-houseprices-gwr)

Supervisor: Eman Qaddoumi
Data: HM Land Registry, Police.uk, ONS, West of England Combined Authority, Bristol City Council

Licensed under the [MIT Licence](LICENSE).
