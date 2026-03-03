# Bristol crime–house price GWR project  
Spatial analysis of crime rates and house prices in Bristol using GWR

This repository contains the code for my final‑year project analysing how crime rates relate to house prices across Bristol LSOAs. Using HM Land Registry price‑paid data and Police.uk crime records (2021–2024), the project builds a clean LSOA‑level dataset and applies both global Ordinary Least Squares (OLS) regression and Geographically Weighted Regression (GWR) to quantify how the crime–price relationship varies by location.

---

## 1. Repository structure

- `notebooks/`  
  - `individual_project_final_version.ipynb` – main analysis notebook containing the full pipeline and commentary.
- `data/`  
  - `raw/` – raw CSVs and shapefiles (not tracked in version control).  
  - `processed/` – cleaned and aggregated datasets created by the notebook  
    (for example `house_prices_bristol_clean.csv`, `crime_bristol_clean.csv`, `regression_dataset.geojson`).
- `output/` – generated figures, maps, tables, and JSON summaries  
  (for example `pricedistribution.png`, `mapgwrcrimecoef.png`, `model_comparison.csv`, `summary_statistics.json`).
- `README.md` – project description and instructions.
- `.gitignore` – excludes raw data folders, temporary files, and notebook checkpoints from version control.

Raw data are not committed to the repository; they should be downloaded from the original public sources.

---

## 2. Data

The project uses only publicly available, anonymised datasets:

- **HM Land Registry Price Paid Data** – individual property transactions with price, date, postcode, property type, tenure, and other attributes.  
- **Police.uk crime data** – monthly street‑level crime records with crime type, date, location, and LSOA code.  
- **Office for National Statistics / boundary data** – LSOA 2021 boundary polygons and postcode‑to‑LSOA lookup for Bristol.

These sources are combined to produce an LSOA‑level dataset containing median house prices, crime counts, and information on housing composition for each neighbourhood.

---

## 3. Methods

The analysis follows these main steps:

- **Exploratory data analysis**
  - Distribution of house prices.
  - Crime trends over time by crime type.
  - Log–log scatter plot of crime vs median house price.
  - Choropleth maps of median price and total crime by LSOA.

- **Statistical tests**
  - Pearson and Spearman correlation between log total crimes and log median house prices.
  - Independent samples t‑test comparing log prices in high‑crime vs low‑crime LSOAs.

- **Regression models**
  - **Global OLS model**  
    - Specification: `log_median_price ~ log_crime + prop_flats + prop_leasehold`.  
    - Produces a single average crime effect for all Bristol LSOAs.
  - **Geographically Weighted Regression (GWR)**  
    - Same predictors as OLS.  
    - Uses LSOA centroids and an adaptive bisquare kernel.  
    - Estimates local coefficients so the crime effect can vary between neighbourhoods.

- **Spatial diagnostics and comparison**
  - Moran’s I on OLS residuals to test for spatial autocorrelation.  
  - Model comparison between OLS and GWR using R², adjusted R², and AICc.  
  - Mapping of the local GWR crime coefficient to show where crime has stronger or weaker associations with house prices.  
  - Optional analyses of property type composition (high‑flat vs low‑flat areas) and “hotspot” LSOAs with both high crime and high prices.

---

## 4. How to run the analysis

1. **Set up the environment**

   Create a Python environment and install the required libraries, for example:

   - `pandas`, `numpy`, `geopandas`, `scikit-learn`, `scipy`  
   - `mgwr`, `pysal` / `esda`, `libpysal`  
   - `matplotlib`, `seaborn`, `folium` (optional for interactive maps)

2. **Prepare data folders**

   From the project root, create:

   - `data/raw/house_prices/`  
   - `data/raw/crime/`  
   - `data/raw/geo/`  
   - `data/processed/`  
   - `output/`

   Download the raw HM Land Registry CSVs, Police.uk CSVs, and the Bristol LSOA shapefile and postcode directory into the appropriate `data/raw` subfolders.

3. **Run the notebook**

   - Open `notebooks/individual_project_final_version.ipynb` in Jupyter or Google Colab.  
   - Update the `BASE_PATH` variable if needed so it points to the project root.  
   - Run all cells; the main pipeline is executed by calling:

     ```python
     results = run_full_analysis()
     ```

   - The notebook will:
     - Save cleaned datasets into `data/processed/`.  
     - Generate figures and maps into `output/`.  
     - Write `model_comparison.csv` and `summary_statistics.json` to `output/`.

---

## 5. Results (high‑level)

The current implementation provides:

- Evidence on the overall relationship between crime and house prices in Bristol through correlation and hypothesis testing.  
- A global OLS estimate of how crime, housing composition, and tenure relate to log median prices across LSOAs.  
- Moran’s I statistics showing whether OLS residuals still exhibit spatial autocorrelation.  
- GWR results that demonstrate whether allowing coefficients to vary across space improves model fit and how the crime effect differs between neighbourhoods.  
- Maps highlighting areas where crime is more strongly or weakly associated with house prices, and LSOAs that combine high crime with high property values.

These outputs are interpreted in the accompanying project report, with implications for homebuyers, investors, and local policymakers.

---

## 6. Ethical and privacy considerations

- Only publicly available, anonymised datasets are used.  
- No personal or sensitive information about individuals is included.  
- All data sources are credited, and users are encouraged to follow the licences and terms of use specified by the original providers.
