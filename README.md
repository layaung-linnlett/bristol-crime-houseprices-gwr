# Bristol Crime & House Prices — Spatial Analysis Using GWR

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)](https://colab.research.google.com/)

A spatial analysis of the relationship between neighbourhood crime rates and house prices across Bristol Lower Layer Super Output Areas (LSOAs), 2021–2025, using Geographically Weighted Regression (GWR).

---

## Project overview

This project investigates whether and how crime levels influence house prices across Bristol's 182 LSOAs, and crucially, **how this relationship varies spatially** across the city. A global OLS baseline is compared against GWR to demonstrate that a single average effect conceals fundamentally different local dynamics across Bristol's neighbourhoods.

### Key findings

| Metric | OLS | GWR |
|--------|-----|-----|
| R² | 0.108 | 0.739 |
| Adjusted R² | 0.083 | 0.662 |
| AICc | −84.17 | −209.48 |

- **Moran's I = 0.5408** (p = 0.001) on OLS residuals confirms strong spatial autocorrelation — GWR is statistically justified
- **GWR crime coefficients range from −0.2562 to +0.0253** across LSOAs — the global average of −0.078 masks fundamentally different local dynamics
- **76% of LSOAs** show a meaningfully negative crime effect (coef < −0.05)
- **High-crime, high-price hotspots** have nearly double the school density of other LSOAs, explaining why crime does not suppress prices in those areas

---

## Repository structure

```
bristol-crime-houseprices-gwr/
│
├── notebooks/
│   └── GWR_crime_price_project.ipynb   # Main analysis notebook
│
├── data/
│   ├── raw/                                      # Raw data 
│   │   ├── house_prices/                         # HM Land Registry pp-YYYY.csv files
│   │   ├── crime/                                # Police.uk monthly CSV files
│   │   ├── geo/                                  # LSOA shapefile + postcode directory
│   │   ├── schools-lep.csv                       # School locations
│   │   └── Bus_Stops.csv                         # Bristol bus stop locations
│   └── processed/                               
│       ├── house_prices_bristol_clean.csv
│       ├── crime_bristol_clean.csv
│       └── regression_dataset.geojson
│
├── outputs/                                      
│   ├── price_distribution.png
│   ├── crime_trends.png
│   ├── correlation_heatmap.png
│   ├── ols_coefficients.png
│   ├── scatter_crime_price.html
│   ├── map_price_crime.png
│   ├── map_ols_residuals.png
│   ├── map_gwr_coefficients.png
│   ├── model_comparison.csv
│   └── summary_statistics.json
│
├── requirements.txt                              # Python dependencies
├── LICENSE                                       # MIT licence
└── README.md                                     # This file
```

---

## Data sources

All data is openly available. Download and place in the `data/raw/` directories as shown above.

| Dataset | Source | Period |
|---------|--------|--------|
| House price (price-paid) | [HM Land Registry](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads) | 2021–2025 |
| Crime records | [Police.uk](https://data.police.uk/data/) | 2021–2025 |
| LSOA boundaries | [ONS Geoportal](https://geoportal.statistics.gov.uk/) | 2021 |
| Postcode-to-LSOA lookup | [ONS Geoportal](https://geoportal.statistics.gov.uk/) | 2025 |
| School locations | [West of England Combined Authority](https://opendata.westofengland-ca.gov.uk/explore/dataset/schools-lep/table/) | 2024 |
| Bus stops | [Bristol Open Data](https://opendata.bristol.gov.uk/datasets/bus-stops-2/explore) | Current |

---

## Methods

### Pipeline overview

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

**Why `prop_leasehold` was excluded:** Initial VIF check revealed severe collinearity between `prop_flats` (VIF = 29.64) and `prop_leasehold` (VIF = 34.73). In the UK, flats are almost always leasehold, so the two variables capture nearly identical information. `prop_leasehold` was removed; all remaining VIF values fell below 5.

---

## Setup and usage

### Option 1 — Google Colab (recommended)

1. Upload the notebook to Google Colab
2. Mount your Google Drive
3. Place raw data in `My Drive/Bristol_Project/data/raw/`
4. Run all cells — the smart execute cell auto-detects saved outputs

```python
# The execute cell handles first run vs restart automatically
# First run: runs full pipeline (~10–15 minutes for GWR)
# After restart: loads from saved GeoJSON (~30 seconds)
```

### Option 2 — Local Jupyter

```bash
git clone https://github.com/layaung-linnlett/bristol-crime-houseprices-gwr.git
cd bristol-crime-houseprices-gwr
pip install -r requirements.txt
jupyter notebook notebooks/individual_project_final_version.ipynb
```

Update `BASE_PATH` in the notebook to point to your local data directory.

---

## Visualisations produced

| Plot | Type | Description |
|------|------|-------------|
| Price distribution | Static (matplotlib) | Right-skewed histogram justifying log transform |
| Crime trends | Static (matplotlib) | Annual counts by top 5 crime types, 2021–2025 |
| Correlation heatmap | Static (seaborn) | Pairwise correlations — complements VIF check |
| OLS coefficient plot | Static (matplotlib) | Forest plot with 95% CIs, coloured by significance |
| Crime vs price scatter | Interactive (Plotly) | Log-log scatter coloured by distance to centre |
| Price & crime maps | Interactive (Plotly) | Side-by-side choropleth — spatial EDA |
| OLS residual map | Interactive (Plotly) | Diverging scale — visualises Moran's I result |
| GWR coefficient maps | Interactive (Plotly) | Local crime & distance-to-centre coefficients |

---

## Key libraries

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data handling |
| `geopandas`, `shapely` | Geospatial operations |
| `statsmodels` | OLS regression with full inference |
| `mgwr` | Geographically Weighted Regression |
| `esda`, `libpysal` | Moran's I spatial autocorrelation |
| `matplotlib`, `seaborn` | Static visualisation |
| `plotly` | Interactive visualisation |

---

## Results summary

```
Dataset
  LSOAs analysed     : 182
  House transactions : 34,543
  Crime records      : 159,666
  Study period       : 2021–2025

Spatial diagnostics
  Moran's I          : 0.5408  (p = 0.001) → GWR justified

OLS baseline
  R²                 : 0.108
  Crime coefficient  : −0.078  (≈ −7.5% per log crime unit)

GWR model
  R²                 : 0.739  
  Bandwidth          : 59 neighbours (adaptive)
  Crime coef range   : −0.256 to +0.025
  76% of LSOAs show meaningfully negative crime effect
```

---

## Limitations

- Cross-sectional analysis — cannot establish causality
- Total crime count aggregates all offence types equally
- Key omitted variables: deprivation index, property size, age, condition
- GWR sensitive to bandwidth choice; local estimates less stable than global OLS

---

## Author

**La Yaung Linn Lett**  
BSc Data Science and AI  
University of the West of England, Bristol  
2026

---

## Licence

This project is licensed under the MIT Licence — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

Supervisor: Eman Qaddoumi  
Data: HM Land Registry, Police.uk, ONS, West of England Combined Authority, Bristol City Council
