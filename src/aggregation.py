"""
aggregation.py
--------------
Functions to aggregate transaction-level data to LSOA level.
"""

import numpy as np
import pandas as pd
import geopandas as gpd


def aggregate_house_prices_by_lsoa(house_clean: pd.DataFrame,
                                   postcode_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate cleaned house price data to LSOA level.

    Links each transaction to an LSOA via the postcode directory, then
    computes per LSOA: median price, mean price, number of sales,
    proportion of flats, and proportion of leasehold properties.

    Parameters
    ----------
    house_clean : pd.DataFrame
        Cleaned house price data.
    postcode_lookup : pd.DataFrame
        Postcode-to-LSOA mapping table.

    Returns
    -------
    pd.DataFrame
        LSOA-level house price statistics, one row per LSOA.
    """
    print("(1) Aggregating house prices to LSOA level...")

    house_clean     = house_clean.copy()
    postcode_lookup = postcode_lookup.copy()

    house_clean['postcode_clean']     = (
        house_clean['postcode'].astype(str).str.strip().str.upper()
    )
    postcode_lookup['postcode_clean'] = (
        postcode_lookup['postcode'].astype(str).str.strip().str.upper()
    )

    house_lsoa = house_clean.merge(
        postcode_lookup[['postcode_clean', 'lsoa_code']],
        on='postcode_clean', how='left'
    )

    matched    = house_lsoa['lsoa_code'].notna().sum()
    match_rate = matched / len(house_lsoa) * 100
    print(f"     Postcode match rate: {match_rate:.1f}% "
          f"({matched:,} of {len(house_lsoa):,} records)")

    if match_rate < 80:
        print("     ⚠️  Match rate below 80% — check postcode format")

    house_lsoa = house_lsoa.dropna(subset=['lsoa_code'])

    house_lsoa['is_flat']      = (house_lsoa['property_type'] == 'F').astype(int)
    house_lsoa['is_leasehold'] = (house_lsoa['tenure_type']   == 'L').astype(int)

    lsoa_agg = house_lsoa.groupby('lsoa_code').agg(
        median_price    =('price',        'median'),
        mean_price      =('price',        'mean'),
        n_sales         =('price',        'size'),
        flat_count      =('is_flat',      'sum'),
        leasehold_count =('is_leasehold', 'sum')
    ).reset_index()

    lsoa_agg['prop_flats']     = lsoa_agg['flat_count']      / lsoa_agg['n_sales']
    lsoa_agg['prop_leasehold'] = lsoa_agg['leasehold_count'] / lsoa_agg['n_sales']

    low_sales = (lsoa_agg['n_sales'] < 5).sum()
    if low_sales > 0:
        print(f"     ⚠️  {low_sales} LSOAs have fewer than 5 sales — "
              f"price medians may be unreliable")

    print(f"     Aggregated to {len(lsoa_agg):,} LSOAs")
    print(f"     Sales per LSOA — mean: {lsoa_agg['n_sales'].mean():.1f}, "
          f"min: {lsoa_agg['n_sales'].min()}, "
          f"max: {lsoa_agg['n_sales'].max()}")
    print()

    return lsoa_agg


def aggregate_crime_by_lsoa(crime_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate crime incident counts to LSOA level.

    Each row in the cleaned crime data represents one recorded incident.
    This function counts incidents per LSOA to produce a single
    total_crimes measure for each neighbourhood.

    Parameters
    ----------
    crime_clean : pd.DataFrame
        Cleaned crime data with an lsoa_code column.

    Returns
    -------
    pd.DataFrame
        LSOA-level crime counts, one row per LSOA.
    """
    print("(2) Aggregating crime counts to LSOA level...")

    crime_agg = (
        crime_clean
        .groupby('lsoa_code')
        .size()
        .reset_index(name='total_crimes')
    )

    print(f"     Aggregated to {len(crime_agg):,} LSOAs")
    print(f"     Crimes per LSOA — mean:   {crime_agg['total_crimes'].mean():.1f}")
    print(f"                      median: {crime_agg['total_crimes'].median():.1f}")
    print(f"                      min:    {crime_agg['total_crimes'].min():,}")
    print(f"                      max:    {crime_agg['total_crimes'].max():,}")
    print()

    return crime_agg


def create_regression_dataset(lsoa_house: pd.DataFrame,
                               lsoa_crime: pd.DataFrame,
                               lsoa_gdf: gpd.GeoDataFrame,
                               postcode_lookup: pd.DataFrame,
                               schools_path,
                               bus_path) -> gpd.GeoDataFrame:
    """
    Merge house price, crime, and spatial data into the final regression dataset.

    Parameters
    ----------
    lsoa_house : pd.DataFrame
        LSOA-level house price statistics.
    lsoa_crime : pd.DataFrame
        LSOA-level crime counts.
    lsoa_gdf : gpd.GeoDataFrame
        LSOA boundary polygons for Bristol.
    postcode_lookup : pd.DataFrame
        Postcode-to-LSOA mapping (for school density calculation).
    schools_path : Path
        Path to schools CSV file.
    bus_path : Path
        Path to bus stops CSV file.

    Returns
    -------
    gpd.GeoDataFrame
        Complete dataset for OLS and GWR, one row per LSOA.
    """
    from src.features import city_centre_distance, school_density, transport_accessibility

    print("(3) Building regression dataset...")

    reg_df = lsoa_house.merge(lsoa_crime, on='lsoa_code', how='inner')
    print(f"     After house + crime merge : {len(reg_df):,} LSOAs")

    lsoa_gdf = lsoa_gdf.copy()
    if 'lsoa_code' not in lsoa_gdf.columns:
        for col in ['LSOA21CD', 'lsoa21cd', 'LSOA Code']:
            if col in lsoa_gdf.columns:
                lsoa_gdf = lsoa_gdf.rename(columns={col: 'lsoa_code'})
                break

    reg_gdf = lsoa_gdf.merge(reg_df, on='lsoa_code', how='inner')
    print(f"     After spatial merge       : {len(reg_gdf):,} LSOAs")

    reg_gdf['log_median_price'] = np.log(reg_gdf['median_price'])
    reg_gdf['log_crime']        = np.log(reg_gdf['total_crimes'] + 1)

    core_vars   = ['log_median_price', 'log_crime', 'prop_flats', 'prop_leasehold']
    before_drop = len(reg_gdf)
    reg_gdf     = reg_gdf.dropna(subset=core_vars)
    if before_drop > len(reg_gdf):
        print(f"     Dropped {before_drop - len(reg_gdf)} LSOAs with missing core variables")

    print(f"     After cleaning            : {len(reg_gdf):,} LSOAs")
    print("     Adding neighbourhood covariates...")

    reg_gdf = city_centre_distance(reg_gdf)
    reg_gdf = school_density(reg_gdf, postcode_lookup, schools_path)
    reg_gdf = transport_accessibility(reg_gdf, bus_path)

    all_vars    = core_vars + ['dist_centre_km', 'schools_count', 'dist_nearest_bus_km']
    before_drop = len(reg_gdf)
    reg_gdf     = reg_gdf.dropna(subset=all_vars)
    if before_drop > len(reg_gdf):
        print(f"     Dropped {before_drop - len(reg_gdf)} LSOAs with missing covariate values")

    print(f"     Final dataset: {len(reg_gdf):,} LSOAs, {reg_gdf.shape[1]} columns")
    print(f"       - Median price range: £{reg_gdf['median_price'].min():,.0f} – "
          f"£{reg_gdf['median_price'].max():,.0f}")
    print(f"       - Crime range: {reg_gdf['total_crimes'].min():.0f} – "
          f"{reg_gdf['total_crimes'].max():.0f}")
    print()

    return reg_gdf
