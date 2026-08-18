# Bristol Crime & House Prices — Does Crime Affect House Prices Differently Across the City?

**Live dashboard:** https://bristol-crime-houseprices-gwr.streamlit.app

I wanted to find out whether the relationship between **crime and house prices is the same across Bristol**.

A standard regression gives one city-wide estimate: higher crime is associated with around **7.5% lower house prices**. But that number hides a lot of local variation.

Using **Geographically Weighted Regression (GWR)**, I estimated the relationship separately across Bristol's **182 neighbourhoods (LSOAs)**. The local crime coefficient ranges from about a **22.6% negative association to a slight positive association**, depending on the area.

The GWR model explains **73.9% of house price variation**, compared with **10.8% for the standard OLS model**, using the same five predictors.

I also tested the OLS residuals using **Moran's I**. The result was **0.5408 (p = 0.001)**, showing that neighbouring areas tend to have similar model errors. This gave me evidence that a single city-wide model was missing some of the spatial pattern in the data.

The project uses **34,543 house transactions** and **159,666 recorded crimes** across Bristol from **2021–2025**.

> **Important:** This project shows associations between crime and house prices. It does not prove that crime causes house prices to increase or decrease.

---

## Key Findings

### 1. The standard model hides a lot of local variation

The global OLS model explains only **10.8% of house price variation (R² = 0.108)**.

When I allow the relationships to change between neighbourhoods using GWR, the model explains **73.9% (R² = 0.739)**.

Both models use the same five predictors.

**Why this matters:** A single city-wide coefficient makes Bristol look much more uniform than it really is. The spatial model gives a clearer picture of how different neighbourhoods behave.

---

### 2. The estimated crime effect is very different across Bristol

The GWR crime coefficient ranges from approximately **−0.256 to +0.025** across the 182 LSOAs.

The global OLS model gives one average coefficient equivalent to roughly a **7.5% negative association**.

**Why this matters:** The average effect does not describe every neighbourhood well. In some areas, higher crime is much more strongly associated with lower house prices. In other areas, the relationship is weak or slightly positive.

A positive coefficient should not be interpreted as crime making houses more valuable. It is more likely that other local characteristics are stronger in those areas.

---

### 3. Around 76% of neighbourhoods have a negative local crime coefficient

Around **76% of Bristol's neighbourhoods** have a negative crime coefficient in the GWR results.

The remaining areas are more concentrated around the inner city and show weaker or positive associations.

**Why this matters:** Factors such as proximity to the city centre, transport and schools may be important enough in some areas to offset the negative relationship normally associated with crime.

This is something a single city-wide regression cannot show.

---

### 4. The OLS residuals are spatially clustered

The OLS residuals have a **Moran's I of 0.5408 (p = 0.001)**.

In simple terms, nearby neighbourhoods tend to have similar prediction errors.

**Why this matters:** If the global model captured the main spatial pattern in the data, we would expect the remaining errors to be more randomly distributed.

Instead, the errors are clustered. This was the main reason I tested a spatial model rather than using GWR just because it is a more advanced technique.

---

### 5. The analysis combines house prices, crime and location data

The project brings together:

* **34,543 house transactions**
* **159,666 recorded crimes**
* **182 Bristol LSOAs**
* data from **2021–2025**

I also included other local characteristics such as:

* property type
* distance to the city centre
* school density
* distance to the nearest bus stop

This means the model is not looking at crime on its own.

---

# Screenshots

### Local crime coefficient by LSOA

This is the main result from the GWR model.

The map shows how the estimated relationship between crime and house prices changes across Bristol. A single OLS coefficient reduces all of this variation to one number.

![GWR local coefficients by LSOA](outputs/figures/map_gwr_coefficients.png)

---

### OLS residuals

This map shows where the global OLS model over- or under-predicts house prices.

The clustering of similar errors matches the **Moran's I = 0.5408 (p = 0.001)** result.

![OLS residuals map](outputs/figures/map_ols_residuals.png)

---

### Crime and house prices

![Crime vs price scatter](outputs/figures/scatter_crime_price.png)

An interactive version of the scatter plot is available at:

```text
outputs/figures/scatter_crime_price.html
```

The Streamlit dashboard contains the rest of the analysis.

---

# Tech Stack

| Tool                             | Used for                                       |
| -------------------------------- | ---------------------------------------------- |
| **pandas / numpy**               | Data cleaning and calculations                 |
| **GeoPandas / Shapely / PyProj** | Spatial joins, geometry and coordinate systems |
| **statsmodels**                  | OLS regression                                 |
| **scikit-learn**                 | Regression diagnostics                         |
| **mgwr**                         | Geographically Weighted Regression             |
| **esda / libpysal**              | Moran's I and spatial analysis                 |
| **Matplotlib / Seaborn**         | Static visualisations                          |
| **Plotly / Kaleido**             | Interactive charts and image export            |
| **Streamlit**                    | Interactive dashboard                          |

---

# Methodology

The analysis followed these steps.

### 1. Clean the source data

I used house price data from **HM Land Registry** and crime records from **Police.uk**.

The data was filtered to Bristol, and house price outliers were removed using the **1.5 × IQR rule**.

### 2. Aggregate the data to LSOA level

The modelling was done at the neighbourhood level rather than at individual transaction level.

For each of Bristol's **182 LSOAs**, I calculated measures including:

* median house price
* number of sales
* property mix
* total recorded crime

### 3. Create additional features

I added several location-related variables:

* distance to Bristol city centre
* school density
* distance to the nearest bus stop

### 4. Check for multicollinearity

Before modelling, I checked whether the predictor variables were too closely related using **Variance Inflation Factor (VIF)**.

`prop_flats` and `prop_leasehold` were almost duplicates:

* `prop_flats` VIF = **12.7**
* `prop_leasehold` VIF = **12.9**
* correlation = **0.96**

I removed `prop_leasehold`.

All remaining VIF values were below 5.

### 5. Fit a standard OLS model

I first fitted a global **Ordinary Least Squares (OLS)** regression.

This gives one coefficient for each predictor across the whole of Bristol.

The model explained **10.8% of house price variation (R² = 0.108)**.

### 6. Test the OLS residuals

I used **Moran's I** to check whether the OLS errors were spatially clustered.

The result was:

**Moran's I = 0.5408, p = 0.001**

This indicates positive spatial autocorrelation: nearby neighbourhoods tend to have similar errors.

This was important because it suggested that the global model was missing some spatial structure.

### 7. Fit the GWR model

I then fitted **Geographically Weighted Regression**.

Unlike OLS, GWR allows the coefficients to change depending on location.

I used:

* an adaptive bisquare kernel
* AICc to select the bandwidth
* a final bandwidth of **57 neighbours**

This produced a separate crime coefficient for each of the 182 LSOAs.

### 8. Compare the models

I compared the models using:

* R²
* adjusted R²
* AICc
* residual patterns
* local crime coefficients

The main difference was that the GWR model captured much more of the variation in house prices and showed where the crime relationship changed across Bristol.

---

# Why I Used GWR

I did not start with GWR just because it is a more advanced model.

The reasoning was:

```text
Fit global OLS
       ↓
Check the residuals
       ↓
Moran's I = 0.5408, p = 0.001
       ↓
Residuals are spatially clustered
       ↓
A global model may be missing local patterns
       ↓
Fit GWR
       ↓
Estimate a separate relationship for each neighbourhood
```

This gave me a clear reason for using the more complex spatial model.

---

# Project Structure

```text
bristol-crime-houseprices-gwr/
├── data/
│   ├── raw/
│   │   ├── house_prices_bristol_clean.zip
│   │   └── crime_bristol_clean.zip
│   └── processed/
│       ├── regression_dataset.geojson
│       ├── summary_statistics.json
│       └── full_boundaries.geojson
├── notebooks/
│   └── GWR_crime_price_project.ipynb
├── outputs/
│   └── figures/
│       ├── 9 static PNGs
│       ├── scatter_crime_price.html
│       └── summary_statistics.json
├── src/
│   ├── data_loading.py
│   ├── cleaning.py
│   ├── aggregation.py
│   ├── features.py
│   ├── modelling.py
│   └── visualization.py
├── app.py
├── requirements.txt
├── requirements-notebook.txt
├── .gitignore
└── README.md
```

The processed dataset is already included in the repository, so the notebook can run without downloading the original data again.

---

# How To Run

The Streamlit dashboard requires **Python 3.11+**.

The analysis notebook requires **Python 3.12+**, because of the versions of NumPy and SciPy used in the project.

## Clone the repository

```bash
git clone https://github.com/layaung-linnlett/bristol-crime-houseprices-gwr.git
cd bristol-crime-houseprices-gwr
```

## Run the dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run the analysis notebook

```bash
pip install -r requirements-notebook.txt
jupyter notebook notebooks/GWR_crime_price_project.ipynb
```

The notebook uses the cleaned data already included in `data/`, so no additional download is required.

The analysis can also be rebuilt from the original government sources. See `data/README.md` for more information.

---

# Limitations

### This is an association, not a causal study

The analysis shows relationships between variables but does **not** prove that crime causes house prices to change.

There may be other factors affecting both crime and house prices that are not included in the model, such as deprivation, income or housing tenure.

### Crime is treated as one overall measure

All recorded crimes are combined into one count.

This means violent crime, burglary and anti-social behaviour are treated equally.

A future version could separate crime types and test whether they have different relationships with house prices.

### GWR local coefficients are less stable

The local coefficients are estimated from nearby observations rather than the whole city.

The adaptive bisquare kernel and **57-neighbour bandwidth** were selected using AICc, but individual local estimates should still be treated as **patterns in the data rather than precise predictions for each LSOA**.

### LSOA-level aggregation hides some local detail

Each LSOA contains many properties and households.

Aggregating the data to neighbourhood level makes the spatial analysis possible, but it also means that differences within an individual LSOA are not captured.

---

# Future Work

### 1. Look at different crime types

Separate:

* burglary
* violent crime
* anti-social behaviour
* theft
* other crime categories

This could show whether certain types of crime have a stronger relationship with house prices.

### 2. Add more neighbourhood variables

Potential additions include:

* deprivation
* income
* housing tenure
* green space
* school quality
* transport accessibility

These variables could help explain some of the remaining local differences.

### 3. Compare other spatial models

I would also compare GWR with approaches such as:

* Spatial Lag models
* Spatial Error models
* Multiscale Geographically Weighted Regression (MGWR)

This would help test whether the results are specific to the GWR approach.

---

# What I Learned

This project was my first deeper look at a problem where **location matters**.

I learned how to:

* combine datasets from different sources
* work with geospatial data in Python
* aggregate transaction-level data to neighbourhood level
* check multicollinearity before modelling
* use Moran's I to detect spatial autocorrelation
* understand the difference between global regression and GWR
* interpret local model coefficients
* build an interactive Streamlit dashboard
* explain statistical results without treating them as causal

The biggest lesson was that **one average number can hide important local differences**.

In this project, the global model suggested one Bristol-wide relationship between crime and house prices. The GWR analysis showed that the relationship changes substantially depending on where you are in the city.

---

# Contact

**La Yaung Linn Lett**
BSc Data Science and AI, University of the West of England, Bristol — 2026

[GitHub](https://github.com/layaung-linnlett) · [LinkedIn](https://www.linkedin.com/in/layaung-linnlett/)
