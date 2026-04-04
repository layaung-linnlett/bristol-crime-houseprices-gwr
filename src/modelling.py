"""
modelling.py
------------
Statistical and spatial modelling functions.

Functions
---------
check_vif               : Variance Inflation Factor diagnostic
perform_statistical_tests: Pearson, Spearman, t-test
fit_ols_model           : Global OLS regression (statsmodels)
fit_gwr_model           : Geographically Weighted Regression (mgwr)
calculate_morans_i      : Moran's I spatial autocorrelation test
compare_models          : OLS vs GWR comparison table
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr, ttest_ind


def check_vif(reg_gdf: gpd.GeoDataFrame,
              predictors: list) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for a set of predictors.

    VIF values above 5 indicate problematic collinearity. This is
    especially important for GWR, which fits local regressions with
    smaller effective sample sizes.

    Parameters
    ----------
    reg_gdf : gpd.GeoDataFrame
        Regression dataset containing all predictor columns.
    predictors : list of str
        Column names of predictors to check.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Predictor', 'VIF'], sorted by VIF descending.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X_vif = reg_gdf[predictors].dropna()

    vif_df = pd.DataFrame({
        'Predictor': predictors,
        'VIF': [variance_inflation_factor(X_vif.values, i)
                for i in range(len(predictors))]
    }).round(2).sort_values('VIF', ascending=False).reset_index(drop=True)

    print("Variance Inflation Factors")
    print("-" * 35)
    print(vif_df.to_string(index=False))
    print()

    if (vif_df['VIF'] > 5).any():
        high = vif_df[vif_df['VIF'] > 5]['Predictor'].tolist()
        print(f"  ⚠️  High VIF detected for: {high}")
    else:
        max_vif = vif_df['VIF'].max()
        print(f"  ✓  All VIF values below 5 (max = {max_vif:.2f})")
        print(f"     No serious multicollinearity — all predictors retained.")

    return vif_df


def perform_statistical_tests(reg_gdf: gpd.GeoDataFrame,
                               config: dict) -> dict:
    """
    Run Pearson/Spearman correlation and high- vs low-crime t-test.

    Parameters
    ----------
    reg_gdf : gpd.GeoDataFrame
        Regression dataset with log_crime, log_median_price, total_crimes.
    config : dict
        Must contain 'significance_level'.

    Returns
    -------
    dict
        Correlation coefficients, p-values, t-statistic, and group means.
    """
    print("Statistical testing")
    print()

    alpha = config['significance_level']
    sig   = lambda p: '✓ Significant' if p < alpha else '✗ Not significant'

    # Pearson and Spearman correlations
    r_p, p_p = pearsonr(reg_gdf['log_crime'],  reg_gdf['log_median_price'])
    r_s, p_s = spearmanr(reg_gdf['log_crime'], reg_gdf['log_median_price'])

    print(f"(1) Correlation: log(Crime) vs log(Price)")
    print(f"     Pearson  r = {r_p:>7.4f},  p = {p_p:.4f}  {sig(p_p)}")
    print(f"     Spearman ρ = {r_s:>7.4f},  p = {p_s:.4f}  {sig(p_s)}")
    print()

    # t-test: high- vs low-crime LSOAs
    median_crime = reg_gdf['total_crimes'].median()
    high_crime   = reg_gdf[reg_gdf['total_crimes'] >  median_crime]['log_median_price']
    low_crime    = reg_gdf[reg_gdf['total_crimes'] <= median_crime]['log_median_price']

    t_stat, p_value = ttest_ind(high_crime, low_crime)

    print(f"(2) Hypothesis test: high-crime vs low-crime areas")
    print(f"     H0: Mean log(price) equal in high- and low-crime LSOAs")
    print(f"     H1: Mean log(price) differs between groups")
    print()
    print(f"     Median crime threshold : {median_crime:.0f} incidents")
    print(f"     High-crime group       : {len(high_crime)} LSOAs")
    print(f"     Low-crime group        : {len(low_crime)} LSOAs")
    print(f"     t-statistic = {t_stat:.4f},  p-value = {p_value:.4f}")
    print()

    if p_value < alpha:
        mean_diff = high_crime.mean() - low_crime.mean()
        pct_diff  = (np.exp(mean_diff) - 1) * 100
        direction = 'higher' if pct_diff > 0 else 'lower'
        print(f"     Result: REJECT H0 (p < {alpha})")
        print(f"     High-crime areas are ~{abs(pct_diff):.1f}% {direction} priced")
    else:
        print(f"     Result: FAIL TO REJECT H0 (p ≥ {alpha})")

    return {
        'pearson_r': float(r_p), 'pearson_p': float(p_p),
        'spearman_r': float(r_s), 'spearman_p': float(p_s),
        't_statistic': float(t_stat), 'p_value': float(p_value),
        'significant': p_value < alpha,
        'high_crime_mean': float(high_crime.mean()),
        'low_crime_mean':  float(low_crime.mean()),
    }


def fit_ols_model(reg_gdf: gpd.GeoDataFrame) -> dict:
    """
    Fit global OLS regression model using statsmodels.

    Model: log(price) ~ log(crime) + prop_flats + dist_centre_km
                        + schools_count + dist_nearest_bus_km

    prop_leasehold excluded after VIF diagnostic (VIF = 34.73,
    collinear with prop_flats VIF = 29.64).

    Parameters
    ----------
    reg_gdf : gpd.GeoDataFrame
        Regression dataset with all model variables.

    Returns
    -------
    dict
        Model object, predictions, residuals, fit metrics, coefficients.
    """
    print()
    print("(1) OLS REGRESSION — GLOBAL MODEL")
    print()

    y = reg_gdf['log_median_price']
    X = reg_gdf[['log_crime', 'prop_flats',
                 'dist_centre_km', 'schools_count', 'dist_nearest_bus_km']]

    X_const   = sm.add_constant(X)
    ols_model = sm.OLS(y, X_const).fit()

    print("Model: log(price) = β0 + β1·log(crime) + β2·prop_flats")
    print("                       + β3·dist_centre + β4·schools + β5·bus_dist")
    print()
    print(ols_model.summary())

    y_pred    = ols_model.fittedvalues.values
    residuals = ols_model.resid.values
    r2        = ols_model.rsquared
    adj_r2    = ols_model.rsquared_adj
    aic       = ols_model.aic
    n, k      = len(y), 6
    aicc      = aic + (2 * k * (k + 1)) / (n - k - 1)
    rmse      = float(np.sqrt(np.mean(residuals ** 2)))

    print("Model fit:")
    print(f"   R²          : {r2:8.4f}")
    print(f"   Adjusted R² : {adj_r2:8.4f}")
    print(f"   AIC         : {aic:8.2f}")
    print(f"   AICc        : {aicc:8.2f}")
    print(f"   RMSE        : {rmse:8.4f}")
    print()

    crime_pct = (np.exp(ols_model.params['log_crime']) - 1) * 100
    direction = 'increase' if crime_pct > 0 else 'decrease'
    print(f"Crime interpretation: 1-unit increase in log(crime) → "
          f"{abs(crime_pct):.2f}% {direction} in median price")

    return {
        'model':     ols_model,
        'y_pred':    y_pred,
        'residuals': residuals,
        'r2':        r2,
        'adj_r2':    adj_r2,
        'aic':       aic,
        'aicc':      aicc,
        'rmse':      rmse,
        'coefficients': {
            'intercept':           float(ols_model.params['const']),
            'log_crime':           float(ols_model.params['log_crime']),
            'prop_flats':          float(ols_model.params['prop_flats']),
            'dist_centre_km':      float(ols_model.params['dist_centre_km']),
            'schools_count':       float(ols_model.params['schools_count']),
            'dist_nearest_bus_km': float(ols_model.params['dist_nearest_bus_km']),
        },
    }


def fit_gwr_model(reg_gdf: gpd.GeoDataFrame, config: dict):
    """
    Fit Geographically Weighted Regression (GWR) model.

    Model: log(price) ~ log(crime) + prop_flats + dist_centre_km
                        + schools_count + dist_nearest_bus_km

    Uses adaptive bisquare kernel with AICc-optimised bandwidth.

    Parameters
    ----------
    reg_gdf : gpd.GeoDataFrame
        Regression dataset with LSOA geometries and all model variables.
    config : dict
        Must contain 'gwr_kernel' and 'gwr_adaptive'.

    Returns
    -------
    dict or None
        GWR results including bandwidth, R², AICc, and local params.
        Returns None if the model fails.
    """
    from mgwr.gwr import GWR
    from mgwr.sel_bw import Sel_BW

    print()
    print("(2) GEOGRAPHICALLY WEIGHTED REGRESSION (GWR)")
    print()

    coords = np.column_stack([
        reg_gdf.geometry.centroid.x,
        reg_gdf.geometry.centroid.y
    ])

    y = reg_gdf['log_median_price'].values.reshape(-1, 1)
    X = reg_gdf[['log_crime', 'prop_flats',
                 'dist_centre_km', 'schools_count',
                 'dist_nearest_bus_km']].values

    print(f"     Input shape  : coords={coords.shape}, y={y.shape}, X={X.shape}")
    print(f"     Kernel       : {config['gwr_kernel']}, "
          f"adaptive={config['gwr_adaptive']}")
    print()

    if len(reg_gdf) < 30:
        print("     ⚠️  Too few LSOAs for stable GWR — returning None.")
        return None

    print("     Selecting bandwidth (this may take a moment)...")
    try:
        bw = Sel_BW(coords, y, X).search()
        print(f"     Optimal bandwidth: {bw:.1f} neighbours")
    except Exception as e:
        print(f"     ⚠️  Bandwidth selection failed ({e}) — using fallback bw=49")
        bw = 49

    print("     Fitting GWR model...")
    try:
        gwr_model = GWR(coords, y, X, bw=bw).fit()
        print(gwr_model.summary())

        crime_coefs = gwr_model.params[:, 1]
        print(f"     Local crime coefficient range: "
              f"{crime_coefs.min():.4f} to {crime_coefs.max():.4f}")

        return {
            'model':  gwr_model,
            'bw':     bw,
            'r2':     gwr_model.R2,
            'adj_r2': gwr_model.adj_R2,
            'aic':    gwr_model.aic,
            'aicc':   gwr_model.aicc,
            'params': gwr_model.params,
        }

    except Exception as e:
        print(f"     ⚠️  GWR failed: {e}")
        return None


def calculate_morans_i(reg_gdf: gpd.GeoDataFrame,
                       residuals,
                       spatial_stats_available: bool = True) -> dict:
    """
    Calculate Moran's I on model residuals.

    Uses Queen contiguity weights matrix (row-standardised). A significant
    positive result confirms spatial autocorrelation in OLS residuals,
    justifying GWR.

    Parameters
    ----------
    reg_gdf : gpd.GeoDataFrame
        Regression dataset with LSOA geometries.
    residuals : array-like
        Residuals from the fitted OLS model.
    spatial_stats_available : bool
        Whether esda/libpysal are installed.

    Returns
    -------
    dict or None
        Moran's I, expected I, p-value, and significance flag.
    """
    if not spatial_stats_available:
        print("     ⚠️  PySAL not available — Moran's I skipped.")
        return None

    from esda.moran import Moran
    from libpysal.weights import Queen

    print()
    print("Moran's I — spatial autocorrelation in OLS residuals")
    print()

    try:
        w           = Queen.from_dataframe(reg_gdf)
        w.transform = 'r'
        moran       = Moran(residuals, w)

        sig = '✓ Significant' if moran.p_sim < 0.05 else '✗ Not significant'
        print(f"     Moran's I  : {moran.I:.4f}")
        print(f"     Expected I : {moran.EI:.4f}")
        print(f"     p-value    : {moran.p_sim:.4f}  {sig}")
        print()

        if moran.p_sim < 0.05 and moran.I > 0:
            print("     OLS residuals are spatially clustered → GWR is justified.")

        return {
            'I': moran.I, 'EI': moran.EI,
            'p_value': moran.p_sim,
            'significant': moran.p_sim < 0.05,
        }

    except Exception as e:
        print(f"     ⚠️  Moran's I failed: {e}")
        return None


def compare_models(ols_results: dict,
                   gwr_results: dict,
                   reg_gdf: gpd.GeoDataFrame) -> dict:
    """
    Compare OLS and GWR models using R², adjusted R², and AICc.

    Parameters
    ----------
    ols_results : dict
        Output from fit_ols_model.
    gwr_results : dict or None
        Output from fit_gwr_model. If None, only OLS is reported.
    reg_gdf : gpd.GeoDataFrame
        Regression dataset (unused directly, included for extensions).

    Returns
    -------
    dict
        Comparison DataFrame and gwr_available flag.
    """
    print()
    print("MODEL COMPARISON: OLS vs GWR")
    print()

    print("(1) OLS — global model:")
    print(f"     R²          : {ols_results['r2']:.4f}")
    print(f"     Adjusted R² : {ols_results['adj_r2']:.4f}")
    print(f"     AICc        : {ols_results['aicc']:.2f}")
    print(f"     Crime coef  : {ols_results['coefficients']['log_crime']:.4f}")

    if gwr_results is None:
        print()
        print("(2) GWR — unavailable (model failed to fit)")
        return {'comparison_table': None, 'gwr_available': False}

    crime_coefs = gwr_results['params'][:, 1]
    print()
    print("(2) GWR — spatially varying model:")
    print(f"     R²          : {gwr_results['r2']:.4f}")
    print(f"     Adjusted R² : {gwr_results['adj_r2']:.4f}")
    print(f"     AICc        : {gwr_results['aicc']:.2f}")
    print(f"     Crime coef range: {crime_coefs.min():.4f} to {crime_coefs.max():.4f}")

    comparison = pd.DataFrame({
        'Metric': ['R²', 'Adj R²', 'AICc'],
        'OLS':    [ols_results['r2'],   ols_results['adj_r2'],  ols_results['aicc']],
        'GWR':    [gwr_results['r2'],   gwr_results['adj_r2'],  gwr_results['aicc']],
    })
    comparison['GWR Better'] = [
        'Yes' if gwr_results['r2']     > ols_results['r2']     else 'No',
        'Yes' if gwr_results['adj_r2'] > ols_results['adj_r2'] else 'No',
        'Yes' if gwr_results['aicc']   < ols_results['aicc']   else 'No',
    ]

    print()
    print(comparison.round(4).to_string(index=False))

    return {'comparison_table': comparison, 'gwr_available': True}
