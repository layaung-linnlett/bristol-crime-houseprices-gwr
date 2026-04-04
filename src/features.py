"""
features.py
-----------
Feature engineering functions for neighbourhood covariates.

Three features are computed and added to the regression GeoDataFrame:
  - dist_centre_km      : distance from LSOA centroid to Bristol city centre
  - schools_count       : number of schools per LSOA
  - dist_nearest_bus_km : distance from LSOA centroid to nearest bus stop
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path


def city_centre_distance(reg_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add distance-to-city-centre feature to the regression GeoDataFrame.

    Computes Euclidean distance (km) from each LSOA centroid to Bristol
    City Hall (−2.5879, 51.4545). The reference point is reprojected to
    match the LSOA CRS before distance calculation.

    Parameters
    ----------
    reg_gdf : gpd.GeoDataFrame
        Regression dataset with LSOA geometries.

    Returns
    -------
    gpd.GeoDataFrame
        Input GeoDataFrame with a new dist_centre_km column.
    """
    print("     Adding dist_centre_km...")

    reg_gdf = reg_gdf.copy()

    centre = gpd.GeoDataFrame(
        [{'geometry': Point(-2.5879, 51.4545)}], crs='EPSG:4326'
    )
    centre = centre.to_crs(reg_gdf.crs)

    reg_gdf['dist_centre_km'] = (
        reg_gdf.geometry.centroid.distance(centre.geometry.iloc[0]) / 1000
    )

    print(f"       - Range: {reg_gdf['dist_centre_km'].min():.2f} – "
          f"{reg_gdf['dist_centre_km'].max():.2f} km, "
          f"mean {reg_gdf['dist_centre_km'].mean():.2f} km")

    return reg_gdf


def school_density(reg_gdf: gpd.GeoDataFrame,
                   postcode_df: pd.DataFrame,
                   schools_path: Path) -> gpd.GeoDataFrame:
    """
    Add school density feature to the regression GeoDataFrame.

    Loads school data from schools_path, matches each school to its LSOA
    via the postcode lookup, and counts schools per LSOA. LSOAs with no
    matched schools receive a count of zero.

    Parameters
    ----------
    reg_gdf : gpd.GeoDataFrame
        Regression dataset with LSOA geometries.
    postcode_df : pd.DataFrame
        Postcode-to-LSOA lookup table.
    schools_path : Path
        Path to the schools CSV file (schools-lep.csv).

    Returns
    -------
    gpd.GeoDataFrame
        Input GeoDataFrame with a new schools_count column.
    """
    print("     Adding schools_count...")

    reg_gdf     = reg_gdf.copy()
    postcode_df = postcode_df.copy()

    schools_df = pd.read_csv(schools_path, low_memory=False)

    schools_df['postcode_clean'] = (
        schools_df['Postcode'].astype(str).str.strip().str.upper()
    )
    if 'postcode_clean' not in postcode_df.columns:
        postcode_df['postcode_clean'] = (
            postcode_df['postcode'].astype(str).str.strip().str.upper()
        )

    merged     = schools_df.merge(
        postcode_df[['postcode_clean', 'lsoa_code']],
        on='postcode_clean', how='left'
    )
    match_rate = merged['lsoa_code'].notna().sum() / len(merged) * 100
    print(f"       - School postcode match rate: {match_rate:.1f}%")

    if merged['lsoa_code'].notna().sum() == 0:
        reg_gdf['schools_count'] = 0
        print("       - ⚠️  No schools matched — schools_count set to 0")
        return reg_gdf

    schools_per_lsoa = (
        merged.dropna(subset=['lsoa_code'])
              .groupby('lsoa_code')
              .size()
              .reset_index(name='schools_count')
    )
    reg_gdf = reg_gdf.merge(schools_per_lsoa, on='lsoa_code', how='left')
    reg_gdf['schools_count'] = reg_gdf['schools_count'].fillna(0)

    print(f"       - Total schools matched: {int(reg_gdf['schools_count'].sum())}, "
          f"mean {reg_gdf['schools_count'].mean():.2f} per LSOA")

    return reg_gdf


def transport_accessibility(reg_gdf: gpd.GeoDataFrame,
                             bus_path: Path) -> gpd.GeoDataFrame:
    """
    Add nearest bus stop distance feature to the regression GeoDataFrame.

    Loads bus stop coordinates (EPSG:27700), reprojects to match the LSOA
    CRS, clips to the Bristol bounding box, then uses a vectorised
    nearest-neighbour search to compute minimum distance (km) from each
    LSOA centroid to the nearest bus stop.

    Parameters
    ----------
    reg_gdf : gpd.GeoDataFrame
        Regression dataset with LSOA geometries.
    bus_path : Path
        Path to the bus stops CSV. Must contain 'X' (easting) and
        'Y' (northing) columns in EPSG:27700.

    Returns
    -------
    gpd.GeoDataFrame
        Input GeoDataFrame with a new dist_nearest_bus_km column.
    """
    from sklearn.neighbors import NearestNeighbors

    print("     Adding dist_nearest_bus_km...")

    reg_gdf = reg_gdf.copy()

    if not bus_path.exists():
        reg_gdf['dist_nearest_bus_km'] = 0.5
        print("       - ⚠️  Bus stops file not found — dist_nearest_bus_km set to 0.5 km")
        return reg_gdf

    bus_df  = pd.read_csv(bus_path, low_memory=False)
    bus_gdf = gpd.GeoDataFrame(
        bus_df,
        geometry=gpd.points_from_xy(bus_df['X'], bus_df['Y']),
        crs='EPSG:27700'
    )
    bus_gdf = bus_gdf.to_crs(reg_gdf.crs)

    bounds          = reg_gdf.total_bounds
    bus_gdf_bristol = bus_gdf.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
    print(f"       - Bus stops within Bristol bounds: {len(bus_gdf_bristol):,}")

    lsoa_pts = np.array([[p.x, p.y] for p in reg_gdf.geometry.centroid])
    bus_pts  = np.array([[p.x, p.y] for p in bus_gdf_bristol.geometry])

    nbrs         = NearestNeighbors(n_neighbors=1).fit(bus_pts)
    distances, _ = nbrs.kneighbors(lsoa_pts)

    reg_gdf['dist_nearest_bus_km'] = distances.flatten() / 1000

    print(f"       - Range: {reg_gdf['dist_nearest_bus_km'].min():.3f} – "
          f"{reg_gdf['dist_nearest_bus_km'].max():.3f} km, "
          f"mean {reg_gdf['dist_nearest_bus_km'].mean():.3f} km")

    return reg_gdf
