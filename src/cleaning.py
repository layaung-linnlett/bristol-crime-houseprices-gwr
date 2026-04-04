"""
cleaning.py
-----------
Functions to clean and filter raw datasets to Bristol.

Each function:
  - Accepts a raw DataFrame and a config dict
  - Returns a cleaned DataFrame
  - Prints a step-by-step cleaning log
  - Measures retention against Bristol records only
"""

import pandas as pd


def clean_house_prices(house_data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Clean and filter house price data for Bristol.

    Cleaning steps
    --------------
    1. Filter to the Bristol local authority district.
    2. Standardise price and postcode formats.
    3. Drop rows with missing values in critical fields.
    4. Remove clearly invalid prices (< £100).
    5. Remove extreme outliers using the 1.5×IQR rule.
       Lower bound clamped to £100 to avoid negative thresholds
       on right-skewed price data.

    Parameters
    ----------
    house_data : pd.DataFrame
        Raw house price data loaded from HM Land Registry.
    config : dict
        Must contain 'bristol_district' (str) and 'iqr_multiplier' (float).

    Returns
    -------
    pd.DataFrame
        Cleaned house price data for Bristol, ready for aggregation.
    """
    print("(1) Cleaning house price data...")

    df = house_data.copy()

    # 1) Filter for Bristol district
    df = df[df['district'].str.upper() == config['bristol_district']].copy()
    bristol_count = len(df)
    print(f"     Filtered to {config['bristol_district']}: {bristol_count:,} records")

    # 2) Standardise price and postcode formats
    df['price']    = pd.to_numeric(df['price'], errors='coerce')
    df['postcode'] = df['postcode'].astype(str).str.strip().str.upper()

    # 3) Remove rows with missing critical fields
    required_fields = ['price', 'date', 'postcode', 'property_type']
    before_missing  = len(df)
    df = df.dropna(subset=required_fields)
    print(f"     Removed {before_missing - len(df):,} rows with missing data")

    # 4) Remove clearly invalid prices (e.g. £1 family transfers)
    before_invalid = len(df)
    df = df[df['price'] >= 100]
    print(f"     Removed {before_invalid - len(df):,} records with price < £100")

    # 5) Remove outliers using the 1.5×IQR rule
    Q1  = df['price'].quantile(0.25)
    Q3  = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(Q1 - config['iqr_multiplier'] * IQR, 100)
    upper_bound = Q3 + config['iqr_multiplier'] * IQR

    before_outlier = len(df)
    df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

    print(f"     Removed {before_outlier - len(df):,} outliers "
          f"(IQR ×{config['iqr_multiplier']}: "
          f"£{lower_bound:,.0f} – £{upper_bound:,.0f})")
    print(f"       - Price range: £{df['price'].min():,.0f} – £{df['price'].max():,.0f}")
    print(f"       - Median price: £{df['price'].median():,.0f}")

    retention_rate = len(df) / bristol_count * 100
    print(f"     Final: {len(df):,} records "
          f"({retention_rate:.1f}% of Bristol records retained)")
    print()

    return df


def clean_crime_data(crime_data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Clean and filter Police.uk crime data for Bristol.

    Cleaning steps
    --------------
    1. Filter by LSOA code (Bristol codes start with E01014).
    2. Convert latitude/longitude to numeric.
    3. Apply a Bristol geographic bounding box.
    4. Drop records with missing key fields.
    5. Remove duplicate records (can occur across monthly downloads).
    6. Standardise the LSOA code column name.

    Parameters
    ----------
    crime_data : pd.DataFrame
        Raw crime data loaded from Police.uk.
    config : dict
        Must contain 'bristol_lsoa_prefix', 'lat_min', 'lat_max',
        'lon_min', 'lon_max'.

    Returns
    -------
    pd.DataFrame
        Cleaned crime data for Bristol LSOAs.
    """
    print("(2) Cleaning crime data...")

    df = crime_data.copy()

    # 1) Filter for Bristol LSOA codes
    df = df[
        df['LSOA code'].astype(str).str.startswith(
            config['bristol_lsoa_prefix'], na=False
        )
    ].copy()
    bristol_count = len(df)
    print(f"     Filtered to Bristol LSOA codes: {bristol_count:,} records")

    # 2) Convert coordinates to numeric
    df['Latitude']  = pd.to_numeric(df['Latitude'],  errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    # 3) Apply geographic bounding box
    before_geo = len(df)
    df = df[
        (df['Latitude']  >= config['lat_min']) &
        (df['Latitude']  <= config['lat_max']) &
        (df['Longitude'] >= config['lon_min']) &
        (df['Longitude'] <= config['lon_max'])
    ].copy()
    print(f"     Geographic bounding box: {before_geo - len(df):,} records removed")

    # 4) Remove records with missing key fields
    required_fields = ['Month', 'Crime type', 'LSOA code', 'Latitude', 'Longitude']
    before_missing  = len(df)
    df = df.dropna(subset=required_fields)
    print(f"       - Removed {before_missing - len(df):,} rows with missing data")

    # 5) Remove duplicates
    before_dup = len(df)
    df = df.drop_duplicates()
    print(f"       - Removed {before_dup - len(df):,} duplicate records")

    # 6) Standardise LSOA code column name
    df = df.rename(columns={'LSOA code': 'lsoa_code'})

    retention_rate = len(df) / bristol_count * 100
    print(f"     Final: {len(df):,} records "
          f"({retention_rate:.1f}% of Bristol records retained)")
    print(f"     Unique LSOAs      : {df['lsoa_code'].nunique()}")
    print(f"     Unique crime types: {df['Crime type'].nunique()}")
    print()

    return df
