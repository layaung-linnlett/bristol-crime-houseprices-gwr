"""Checks that the shipped data reproduces the headline numbers quoted in the README.

The tests refit the models from `data/processed/regression_dataset.geojson` using the
project's own functions in `src/modelling.py`, then compare against
`data/processed/summary_statistics.json` and the figures written in the README.

Run from the project root:  python -m pytest
(needs the notebook dependencies: `pip install -r requirements-notebook.txt pytest`)
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("statsmodels")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import modelling  # noqa: E402

STATS = json.loads((ROOT / "data" / "processed" / "summary_statistics.json").read_text())


@pytest.fixture(scope="module")
def reg_gdf():
    return gpd.read_file(ROOT / "data" / "processed" / "regression_dataset.geojson")


@pytest.fixture(scope="module")
def ols(reg_gdf):
    with contextlib.redirect_stdout(io.StringIO()):
        return modelling.fit_ols_model(reg_gdf)


@pytest.fixture(scope="module")
def gwr(reg_gdf):
    pytest.importorskip("mgwr")
    with contextlib.redirect_stdout(io.StringIO()):
        return modelling.fit_gwr_model(reg_gdf, {"gwr_kernel": "bisquare", "gwr_adaptive": True})


def test_dataset_has_182_lsoas(reg_gdf):
    assert len(reg_gdf) == STATS["n_lsoas"] == 182


def test_modelled_counts_are_the_in_model_totals_not_the_raw_totals(reg_gdf):
    """34,543 sales / 159,666 crimes are the cleaned raw totals; the model holds fewer."""
    assert int(reg_gdf["n_sales"].sum()) == STATS["n_house_sales_in_model"] == 24_318
    assert int(reg_gdf["total_crimes"].sum()) == STATS["n_crime_records_in_model"] == 147_689
    assert STATS["n_house_sales"] == 34_543
    assert STATS["n_crime_records"] == 159_666


def test_ols_reproduces_the_summary(ols):
    assert ols["r2"] == pytest.approx(STATS["ols_r2"], abs=1e-6)
    assert ols["r2"] == pytest.approx(0.108, abs=0.001)
    assert ols["coefficients"]["log_crime"] == pytest.approx(STATS["ols_crime_coef"], abs=1e-6)


def test_global_crime_effect_is_about_minus_7_5_percent_per_log_unit(ols):
    effect = (np.exp(ols["coefficients"]["log_crime"]) - 1) * 100
    assert effect == pytest.approx(-7.5, abs=0.05)


def test_per_doubling_table_in_the_readme(ols, reg_gdf):
    """The pound figures on a median-priced home (README, 'The cost of using one number')."""
    median_price = reg_gdf["median_price"].median()
    assert median_price == 344_000
    effect = lambda coef: median_price * (2 ** coef - 1)  # noqa: E731
    assert effect(ols["coefficients"]["log_crime"]) == pytest.approx(-18_064, abs=1)
    assert effect(STATS["gwr_crime_coef_med"]) == pytest.approx(-26_319, abs=1)
    assert effect(STATS["gwr_crime_coef_min"]) == pytest.approx(-55_966, abs=1)
    assert effect(STATS["gwr_crime_coef_max"]) == pytest.approx(6_083, abs=1)


def test_morans_i_on_ols_residuals(reg_gdf, ols):
    pytest.importorskip("esda")
    with contextlib.redirect_stdout(io.StringIO()):
        result = modelling.calculate_morans_i(reg_gdf, ols["residuals"])
    assert result["I"] == pytest.approx(0.5408, abs=1e-4)
    assert result["p_value"] <= 0.001


def test_gwr_reproduces_the_summary(gwr):
    assert gwr["bw"] == 57
    assert gwr["r2"] == pytest.approx(STATS["gwr_r2"], abs=1e-6)
    assert gwr["aicc"] == pytest.approx(STATS["gwr_aicc"], abs=1e-3)
    crime = gwr["params"][:, 1]
    assert crime.min() == pytest.approx(STATS["gwr_crime_coef_min"], abs=1e-6)
    assert crime.max() == pytest.approx(STATS["gwr_crime_coef_max"], abs=1e-6)


def test_share_of_lsoas_with_a_reliable_negative_effect(gwr):
    """The README says 67% (5% level) and 51% (multiple-test corrected); none positive."""
    t = gwr["model"].tvalues[:, 1]
    n = len(t)
    assert (t < -1.96).sum() == STATS["n_lsoas_sig_negative_5pct"] == 122
    assert (t < -gwr["model"].critical_tval()).sum() == STATS["n_lsoas_sig_negative_corrected"] == 92
    assert (t > 1.96).sum() == STATS["n_lsoas_sig_positive_5pct"] == 0
    assert round(100 * 122 / n, 1) == STATS["pct_lsoas_sig_negative_5pct"]
    assert round(100 * 92 / n, 1) == STATS["pct_lsoas_sig_negative_corrected"]


def test_76_percent_is_a_size_cutoff_not_a_significance_share(gwr):
    crime = gwr["params"][:, 1]
    assert round(100 * (crime < -0.05).mean(), 1) == STATS["pct_lsoas_coef_below_minus_0_05"] == 76.4
    assert "pct_lsoas_negative" not in STATS, "the ambiguous key was renamed"


def test_crime_price_hotspots_still_show_a_negative_effect(reg_gdf, gwr):
    """The five high-crime, high-price LSOAs are not exempt from the crime penalty."""
    q3_crime = reg_gdf["total_crimes"].quantile(0.75)
    q3_price = reg_gdf["median_price"].quantile(0.75)
    hot = (reg_gdf["total_crimes"] > q3_crime) & (reg_gdf["median_price"] > q3_price)
    assert hot.sum() == 5
    assert (gwr["params"][:, 1][hot.values] < -0.05).all()
