"""
data_loading.py
---------------
Functions to load raw house price, crime, and geospatial data.

All functions follow the same pattern:
  - Accept a Path object pointing to the data directory
  - Return a DataFrame or GeoDataFrame
  - Print progress to stdout
  - Raise FileNotFoundError if expected files are missing
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path


def load_house_price_data(data_path: Path,
                          start_year: int = 2021,
                          end_year: int = 2025) -> pd.DataFrame:
    """
    Load and concatenate HM Land Registry house price data for Bristol.

    Parameters
    ----------
    data_path : Path
        Folder containing raw price-paid CSV files (e.g. pp-2021.csv).
    start_year : int
        First year to include.
    end_year : int
        Last year to include.

    Returns
    -------
    pd.DataFrame
        Combined house price dataset filtered to the chosen years.

    Raises
    ------
    FileNotFoundError
        If no pp-*.csv files are found in data_path.
    ValueError
        If all CSV files fail to load.
    """
    print(f"(1) Loading house price data from {data_path}...")

    column_names = [
        'transaction_id', 'price', 'date', 'postcode', 'property_type',
        'new_build_flag', 'tenure_type', 'paon', 'saon', 'street',
        'locality', 'town_city', 'district', 'county',
        'ppd_category', 'record_status'
    ]

    csv_files = list(data_path.glob("pp-*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No house price files found in {data_path}. "
            f"Expected format: pp-YYYY.csv"
        )

    print(f"     Found {len(csv_files)} CSV files")

    dataframes = []
    for file in sorted(csv_files):
        try:
            df = pd.read_csv(file, header=None, names=column_names,
                             low_memory=False)
            dataframes.append(df)
        except Exception as e:
            print(f"     Error loading {file.name}: {e}")
            continue

    if len(dataframes) == 0:
        raise ValueError("No valid house price data loaded")

    house_data = pd.concat(dataframes, ignore_index=True)

    house_data['date'] = pd.to_datetime(house_data['date'], errors='coerce')
    house_data['year'] = house_data['date'].dt.year

    house_data = house_data[
        (house_data['year'] >= start_year) &
        (house_data['year'] <= end_year)
    ].copy()

    print(f"     Loaded {len(house_data):,} transactions ({start_year}–{end_year})")
    print(f"     Date range: {house_data['date'].min().date()} "
          f"to {house_data['date'].max().date()}")
    print()

    return house_data


def load_crime_data(data_path: Path,
                    start_year: int = 2021,
                    end_year: int = 2025) -> pd.DataFrame:
    """
    Load and concatenate monthly Police.uk crime data for Bristol.

    Parameters
    ----------
    data_path : Path
        Folder containing raw monthly crime CSV files.
    start_year : int
        First year to include.
    end_year : int
        Last year to include.

    Returns
    -------
    pd.DataFrame
        Combined crime dataset filtered to the chosen years.

    Raises
    ------
    FileNotFoundError
        If no CSV files are found in data_path.
    ValueError
        If all CSV files fail to load.
    """
    print(f"(2) Loading crime data from {data_path}...")

    csv_files = list(data_path.rglob("*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No crime files found in {data_path}")

    print(f"     Found {len(csv_files)} CSV files")

    dataframes = []
    for file in sorted(csv_files):
        try:
            df = pd.read_csv(file, low_memory=False)
            dataframes.append(df)
        except Exception as e:
            print(f"     Error loading {file.name}: {e}")
            continue

    if len(dataframes) == 0:
        raise ValueError("No valid crime data loaded")

    crime_data = pd.concat(dataframes, ignore_index=True)

    crime_data['Month'] = pd.to_datetime(crime_data['Month'], errors='coerce')
    crime_data['year']  = crime_data['Month'].dt.year

    crime_data = crime_data[
        (crime_data['year'] >= start_year) &
        (crime_data['year'] <= end_year)
    ].copy()

    print(f"     Loaded {len(crime_data):,} crime records ({start_year}–{end_year})")
    print(f"     Date range: {crime_data['Month'].min().date()} "
          f"to {crime_data['Month'].max().date()}")
    print(f"     Unique crime types: {crime_data['Crime type'].nunique()}")
    print()

    return crime_data


def load_geospatial_data(geo_path: Path):
    """
    Load LSOA boundary polygons and postcode-to-LSOA lookup table.

    Parameters
    ----------
    geo_path : Path
        Folder containing the LSOA shapefile and postcode directory CSV.

    Returns
    -------
    tuple[gpd.GeoDataFrame, pd.DataFrame]
        LSOA boundary GeoDataFrame and postcode-to-LSOA lookup DataFrame,
        both with standardised column names.

    Raises
    ------
    FileNotFoundError
        If the LSOA shapefile or postcode directory cannot be found.
    """
    print(f"(3) Loading geospatial data from {geo_path}...")

    # LSOA boundaries
    lsoa_shp_files = list(
        geo_path.rglob("Lower_Layer_Super_Output_Areas_2021_(Precise).shp")
    )
    if len(lsoa_shp_files) == 0:
        lsoa_shp_files = list(geo_path.rglob("*.shp"))

    if len(lsoa_shp_files) == 0:
        raise FileNotFoundError(
            f"No LSOA shapefile found under {geo_path}. "
            f"Expected: Lower_Layer_Super_Output_Areas_2021_(Precise).shp"
        )

    lsoa_gdf = gpd.read_file(lsoa_shp_files[0])
    print(f"     Using LSOA file: {lsoa_shp_files[0].name}")

    lsoa_column_mappings = {
        'LSOA21CD': 'lsoa_code', 'LSOA Code': 'lsoa_code',
        'lsoa21cd': 'lsoa_code',
        'LSOA21NM': 'lsoa_name', 'LSOA Name': 'lsoa_name',
    }
    for old_col, new_col in lsoa_column_mappings.items():
        if old_col in lsoa_gdf.columns:
            lsoa_gdf = lsoa_gdf.rename(columns={old_col: new_col})

    print(f"     Loaded {len(lsoa_gdf):,} LSOA polygons")

    # Postcode-to-LSOA lookup
    postcode_files = list(geo_path.glob("postcode_directory_*.csv"))
    if len(postcode_files) == 0:
        raise FileNotFoundError(
            f"No postcode lookup found in {geo_path}. "
            f"Expected: postcode_directory_*.csv"
        )

    postcode_df = pd.read_csv(postcode_files[0], low_memory=False)
    print(f"     Using postcode file: {postcode_files[0].name}")

    postcode_column_mappings = {
        'pcds':      'postcode', 'pcd':       'postcode',
        'Postcode':  'postcode',
        'lsoa21cd':  'lsoa_code', 'LSOA Code': 'lsoa_code',
    }
    for old_col, new_col in postcode_column_mappings.items():
        if old_col in postcode_df.columns:
            postcode_df = postcode_df.rename(columns={old_col: new_col})

    print(f"     Loaded {len(postcode_df):,} postcode-to-LSOA mappings")
    print()

    return lsoa_gdf, postcode_df
