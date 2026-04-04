"""
visualisation.py
----------------
All plotting and mapping functions for the Bristol crime-house price analysis.

Static plots  : matplotlib / seaborn  → saved as PNG
Interactive   : Plotly                → displayed inline + saved as HTML/PNG

Plot inventory
--------------
1. plot_price_distribution      : House price histogram (static)
2. plot_crime_trends            : Crime by type over time (static)
3. plot_correlation_heatmap     : Pairwise correlation matrix (static)
4. plot_ols_coefficients        : OLS forest plot (static)
5. plot_scatter_crime_price     : Crime vs price scatter (interactive Plotly)
6. plot_price_crime_maps        : Side-by-side choropleth (interactive Plotly)
7. plot_ols_residual_map        : OLS residual choropleth (interactive Plotly)
8. plot_gwr_coefficient_map     : GWR local coefficients (interactive Plotly)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ── 1. Price distribution ──────────────────────────────────────────────────────

def plot_price_distribution(house_clean: pd.DataFrame,
                             output_path: Path) -> None:
    """Plot histogram of cleaned house sale prices with median and mean lines."""
    median_price = house_clean['price'].median()
    mean_price   = house_clean['price'].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(house_clean['price'], bins=50, color='#4c72b0',
            edgecolor='white', linewidth=0.4, alpha=0.85)
    ax.axvline(median_price, color='crimson',   linewidth=2,
               linestyle='--', label=f'Median: £{median_price:,.0f}')
    ax.axvline(mean_price,   color='darkorange', linewidth=2,
               linestyle=':',  label=f'Mean: £{mean_price:,.0f}')

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x/1000:.0f}k'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.set_xlabel('Sale Price (£)', fontsize=12)
    ax.set_ylabel('Number of Sales', fontsize=12)
    ax.set_title('Distribution of House Sale Prices in Bristol (2021–2025)',
                 fontsize=14, pad=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'price_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"     Saved: price_distribution.png")
    print(f"       - Median: £{median_price:,.0f} | Mean: £{mean_price:,.0f}")
    print(f"       - Skew: {house_clean['price'].skew():.2f} (log transform justified)")
    print()


# ── 2. Crime trends ────────────────────────────────────────────────────────────

def plot_crime_trends(crime_clean: pd.DataFrame, output_path: Path) -> None:
    """Plot annual crime counts by type with total reference line."""
    crime_pivot  = crime_clean.pivot_table(
        index='year', columns='Crime type', aggfunc='size', fill_value=0
    )
    crime_totals = crime_clean.groupby('year').size()
    top_types    = crime_clean['Crime type'].value_counts().head(5).index
    colours      = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b2']

    fig, ax = plt.subplots(figsize=(13, 6))

    for crime_type, colour in zip(top_types, colours):
        if crime_type in crime_pivot.columns:
            ax.plot(crime_pivot.index, crime_pivot[crime_type],
                    marker='o', linewidth=2, color=colour, label=crime_type)
            last_val = crime_pivot.loc[crime_pivot.index[-1], crime_type]
            ax.annotate(f'{last_val:,.0f}', xy=(crime_pivot.index[-1], last_val),
                        xytext=(6, 0), textcoords='offset points',
                        fontsize=8, color=colour, va='center')

    ax.plot(crime_totals.index, crime_totals.values,
            marker='s', linewidth=2.5, linestyle='--',
            color='black', label='Total (all types)', zorder=5)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Recorded Crimes', fontsize=12)
    ax.set_title('Crime Trends by Type in Bristol (2021–2025)', fontsize=14, pad=12)
    ax.set_xticks(crime_pivot.index)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=9, frameon=True, loc='upper left',
              bbox_to_anchor=(1.01, 1), borderaxespad=0)

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(output_path / 'crime_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("     Saved: crime_trends.png")
    print()


# ── 3. Correlation heatmap ─────────────────────────────────────────────────────

def plot_correlation_heatmap(reg_gdf: gpd.GeoDataFrame,
                              output_path: Path) -> None:
    """Plot pairwise Pearson correlation heatmap of all model variables."""
    variables = ['log_median_price', 'log_crime', 'prop_flats',
                 'dist_centre_km', 'schools_count', 'dist_nearest_bus_km']
    labels = {
        'log_median_price':    'log(Median Price)',
        'log_crime':           'log(Crime)',
        'prop_flats':          'Prop. Flats',
        'dist_centre_km':      'Dist. to Centre (km)',
        'schools_count':       'Schools Count',
        'dist_nearest_bus_km': 'Dist. to Bus (km)',
    }

    corr_df = reg_gdf[variables].rename(columns=labels).corr()

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                linecolor='white', annot_kws={'size': 10}, ax=ax,
                cbar_kws={'label': 'Pearson r', 'shrink': 0.8})
    ax.set_title('Pairwise correlations — model variables', fontsize=14, pad=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("     Saved: correlation_heatmap.png")
    print()


# ── 4. OLS coefficient forest plot ────────────────────────────────────────────

def plot_ols_coefficients(ols_results: dict, output_path: Path) -> None:
    """Forest plot of OLS coefficients with 95% confidence intervals."""
    from matplotlib.lines import Line2D

    model    = ols_results['model']
    params   = model.params.drop('const')
    conf_int = model.conf_int().drop('const')
    pvalues  = model.pvalues.drop('const')

    label_map = {
        'log_crime':           'log(Crime)',
        'prop_flats':          'Proportion Flats',
        'dist_centre_km':      'Distance to Centre (km)',
        'schools_count':       'Schools Count',
        'dist_nearest_bus_km': 'Distance to Bus Stop (km)',
    }
    labels  = [label_map.get(p, p) for p in params.index]
    coefs   = params.values
    lower   = coefs - conf_int[0].values
    upper   = conf_int[1].values - coefs
    sig     = pvalues.values < 0.05
    colours = ['#4c72b0' if s else '#aaaaaa' for s in sig]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, coef, lo, hi, colour) in enumerate(
        zip(labels, coefs, lower, upper, colours)
    ):
        ax.errorbar(x=coef, y=i, xerr=[[lo], [hi]], fmt='o',
                    color=colour, markersize=8, capsize=4, linewidth=1.5)

    ax.axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Coefficient (effect on log median house price)', fontsize=11)
    ax.set_title('OLS Regression Coefficients with 95% Confidence Intervals',
                 fontsize=13, pad=12)
    ax.grid(axis='x', alpha=0.3)

    for i, (coef, pval) in enumerate(zip(coefs, pvalues.values)):
        ax.annotate(f'p={pval:.3f}', xy=(coef, i), xytext=(8, 0),
                    textcoords='offset points', fontsize=8,
                    color='#444444', va='center')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4c72b0',
               markersize=9, label='Significant (p < 0.05)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#aaaaaa',
               markersize=9, label='Not significant (p ≥ 0.05)'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(output_path / 'ols_coefficients.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("     Saved: ols_coefficients.png")
    print()


# ── 5. Crime vs price scatter ──────────────────────────────────────────────────

def plot_scatter_crime_price(reg_gdf: gpd.GeoDataFrame,
                              ols_results: dict,
                              output_path: Path) -> None:
    """Interactive Plotly scatter of log crime vs log price, coloured by centrality."""
    import plotly.graph_objects as go

    df = reg_gdf[['lsoa_code', 'log_crime', 'log_median_price',
                  'total_crimes', 'median_price',
                  'dist_centre_km', 'prop_flats']].dropna().copy()

    x_range = np.linspace(df['log_crime'].min(), df['log_crime'].max(), 100)
    model   = ols_results['model']
    X_trend = pd.DataFrame({
        'const':               1,
        'log_crime':           x_range,
        'prop_flats':          df['prop_flats'].mean(),
        'dist_centre_km':      df['dist_centre_km'].mean(),
        'schools_count':       reg_gdf['schools_count'].mean(),
        'dist_nearest_bus_km': reg_gdf['dist_nearest_bus_km'].mean(),
    })
    y_trend = model.predict(X_trend)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['log_crime'], y=df['log_median_price'], mode='markers',
        marker=dict(size=8, color=df['dist_centre_km'], colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(title='Distance to<br>centre (km)', thickness=14),
                    opacity=0.75, line=dict(width=0.5, color='white')),
        customdata=np.column_stack([df['lsoa_code'], df['total_crimes'],
                                    df['median_price'], df['dist_centre_km'],
                                    df['prop_flats']]),
        hovertemplate=('<b>%{customdata[0]}</b><br>'
                       'Total crimes: %{customdata[1]:,.0f}<br>'
                       'Median price: £%{customdata[2]:,.0f}<br>'
                       'Dist. to centre: %{customdata[3]:.2f} km<br>'
                       '<extra></extra>'),
        name='Bristol LSOAs',
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=y_trend, mode='lines',
        line=dict(color='crimson', width=2.5, dash='dash'),
        name='OLS global trend', hoverinfo='skip',
    ))
    fig.update_layout(
        title='Crime vs house prices by LSOA — Bristol (log scale)',
        xaxis_title='log(Total Crimes)',
        yaxis_title='log(Median House Price)',
        plot_bgcolor='white', paper_bgcolor='white',
        height=550, width=850,
    )
    fig.show()

    html_path = output_path / 'scatter_crime_price.html'
    fig.write_html(str(html_path))
    print(f"     Saved: scatter_crime_price.html")

    try:
        fig.write_image(str(output_path / 'scatter_crime_price.png'), scale=2)
        print(f"     Saved: scatter_crime_price.png")
    except Exception:
        print(f"     ⚠️  PNG export skipped — install kaleido==0.2.1")
    print()


# ── 6. Price and crime maps ────────────────────────────────────────────────────

def plot_price_crime_maps(reg_gdf: gpd.GeoDataFrame,
                           output_path: Path) -> None:
    """Interactive side-by-side Plotly choropleth maps of price and crime."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import json

    reg_wgs      = reg_gdf.to_crs('EPSG:4326').copy()
    geojson_data = json.loads(reg_wgs[['lsoa_code', 'geometry']].to_json())

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Median House Price by LSOA', 'Total Crime Count by LSOA'),
        specs=[[{'type': 'choroplethmapbox'}, {'type': 'choroplethmapbox'}]]
    )
    fig.add_trace(go.Choroplethmapbox(
        geojson=geojson_data, locations=reg_wgs['lsoa_code'],
        z=reg_wgs['median_price'], featureidkey='properties.lsoa_code',
        colorscale='YlGnBu',
        colorbar=dict(title='Price (£)', x=0.46, thickness=12, len=0.8),
        marker=dict(opacity=0.75, line_width=0.3),
        hovertemplate='<b>%{location}</b><br>Median price: £%{z:,.0f}<extra></extra>',
        name='Median price',
    ), row=1, col=1)
    fig.add_trace(go.Choroplethmapbox(
        geojson=geojson_data, locations=reg_wgs['lsoa_code'],
        z=reg_wgs['total_crimes'], featureidkey='properties.lsoa_code',
        colorscale='Reds',
        colorbar=dict(title='Total crimes', x=1.0, thickness=12, len=0.8),
        marker=dict(opacity=0.75, line_width=0.3),
        hovertemplate='<b>%{location}</b><br>Total crimes: %{z:,.0f}<extra></extra>',
        name='Total crimes',
    ), row=1, col=2)

    fig.update_layout(
        mapbox1=dict(style='carto-positron',
                     center=dict(lat=51.4545, lon=-2.5879), zoom=10.5),
        mapbox2=dict(style='carto-positron',
                     center=dict(lat=51.4545, lon=-2.5879), zoom=10.5),
        height=520, margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text='Median house price and total crime by LSOA — Bristol',
                   font=dict(size=14), x=0.5),
    )
    fig.show()

    try:
        fig.write_image(str(output_path / 'map_price_crime.png'),
                        scale=2, width=1200, height=520)
        print("     Saved: map_price_crime.png")
    except Exception:
        print("     ⚠️  PNG export skipped — install kaleido==0.2.1")
    print()


# ── 7. OLS residual map ────────────────────────────────────────────────────────

def plot_ols_residual_map(reg_gdf: gpd.GeoDataFrame,
                           ols_results: dict,
                           output_path: Path) -> None:
    """Interactive Plotly choropleth of OLS residuals (diverging blue-white-red)."""
    import plotly.graph_objects as go
    import json

    reg_plot = reg_gdf.copy()
    reg_plot['ols_residual'] = ols_results['residuals']
    reg_wgs      = reg_plot.to_crs('EPSG:4326')
    geojson_data = json.loads(reg_wgs[['lsoa_code', 'geometry']].to_json())
    residuals    = reg_wgs['ols_residual']
    max_abs      = residuals.abs().max()

    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson_data, locations=reg_wgs['lsoa_code'],
        z=residuals, featureidkey='properties.lsoa_code',
        colorscale=[[0.0, '#2166ac'], [0.25, '#92c5de'],
                    [0.5, '#f7f7f7'],
                    [0.75, '#f4a582'], [1.0, '#d6604d']],
        zmin=-max_abs, zmax=max_abs,
        colorbar=dict(title='OLS residual<br>(log price)', thickness=14),
        marker=dict(opacity=0.8, line_width=0.3),
        hovertemplate='<b>%{location}</b><br>Residual: %{z:.3f}<extra></extra>',
    ))
    fig.update_layout(
        mapbox=dict(style='carto-positron',
                    center=dict(lat=51.4545, lon=-2.5879), zoom=10.5),
        height=540, margin=dict(l=0, r=0, t=40, b=0),
        title=dict(
            text='OLS residuals by LSOA — Bristol '
                 '(red = under-predicted, blue = over-predicted)',
            font=dict(size=13), x=0.5),
    )
    fig.show()

    try:
        fig.write_image(str(output_path / 'map_ols_residuals.png'),
                        scale=2, width=900, height=540)
        print("     Saved: map_ols_residuals.png")
    except Exception:
        print("     ⚠️  PNG export skipped — install kaleido==0.2.1")

    print(f"     Residual mean : {residuals.mean():.4f}  (should be ≈ 0)")
    print(f"     Residual std  : {residuals.std():.4f}")
    print()


# ── 8. GWR coefficient maps ────────────────────────────────────────────────────

def plot_gwr_coefficient_map(reg_gdf: gpd.GeoDataFrame,
                              gwr_results: dict,
                              output_path: Path) -> None:
    """Interactive Plotly maps of GWR local crime and distance coefficients."""
    import plotly.graph_objects as go
    import json

    if gwr_results is None:
        print("     ⚠️  GWR results not available — skipping.")
        return

    reg_plot = reg_gdf.copy()
    reg_plot['gwr_crime_coef']  = gwr_results['params'][:, 1]
    reg_plot['gwr_centre_coef'] = gwr_results['params'][:, 3]

    reg_wgs      = reg_plot.to_crs('EPSG:4326')
    geojson_data = json.loads(reg_wgs[['lsoa_code', 'geometry']].to_json())
    crime_coefs  = reg_wgs['gwr_crime_coef']
    max_abs      = max(abs(crime_coefs.min()), abs(crime_coefs.max()))

    # Map 1 — crime coefficient
    fig1 = go.Figure(go.Choroplethmapbox(
        geojson=geojson_data, locations=reg_wgs['lsoa_code'],
        z=crime_coefs, featureidkey='properties.lsoa_code',
        colorscale=[[0.0, '#d6604d'], [0.25, '#f4a582'],
                    [0.5, '#f7f7f7'],
                    [0.75, '#92c5de'], [1.0, '#2166ac']],
        zmin=-max_abs, zmax=max_abs,
        colorbar=dict(title='Crime coefficient', thickness=14),
        marker=dict(opacity=0.8, line_width=0.3),
        hovertemplate='<b>%{location}</b><br>Crime coef: %{z:.4f}<extra></extra>',
    ))
    fig1.update_layout(
        mapbox=dict(style='carto-positron',
                    center=dict(lat=51.4545, lon=-2.5879), zoom=11),
        height=550, margin=dict(l=0, r=0, t=40, b=0),
        title=dict(
            text='Local crime coefficient by LSOA — Bristol '
                 '(red = crime lowers prices, blue = weak/positive effect)',
            font=dict(size=13), x=0.5),
    )
    fig1.show()

    # Map 2 — distance to centre coefficient
    centre_coefs = reg_wgs['gwr_centre_coef']
    max_abs_c    = max(abs(centre_coefs.min()), abs(centre_coefs.max()))

    fig2 = go.Figure(go.Choroplethmapbox(
        geojson=geojson_data, locations=reg_wgs['lsoa_code'],
        z=centre_coefs, featureidkey='properties.lsoa_code',
        colorscale='RdYlBu_r', zmin=-max_abs_c, zmax=max_abs_c,
        colorbar=dict(title='Distance coefficient', thickness=14),
        marker=dict(opacity=0.8, line_width=0.3),
        hovertemplate='<b>%{location}</b><br>Distance coef: %{z:.4f}<extra></extra>',
    ))
    fig2.update_layout(
        mapbox=dict(style='carto-positron',
                    center=dict(lat=51.4545, lon=-2.5879), zoom=11),
        height=550, margin=dict(l=0, r=0, t=40, b=0),
        title=dict(
            text='Local distance-to-centre coefficient by LSOA — Bristol',
            font=dict(size=13), x=0.5),
    )
    fig2.show()

    # Static PNG via matplotlib
    fig_s, axes = plt.subplots(1, 2, figsize=(16, 8))
    reg_plot.plot(column='gwr_crime_coef',  cmap='RdBu', legend=True,
                  linewidth=0.2, edgecolor='grey', ax=axes[0],
                  legend_kwds={'label': 'Crime coefficient', 'shrink': 0.8})
    axes[0].set_title('Local crime effect (GWR)', fontsize=13)
    axes[0].axis('off')
    reg_plot.plot(column='gwr_centre_coef', cmap='RdYlBu_r', legend=True,
                  linewidth=0.2, edgecolor='grey', ax=axes[1],
                  legend_kwds={'label': 'Distance coefficient', 'shrink': 0.8})
    axes[1].set_title('Local distance-to-centre effect (GWR)', fontsize=13)
    axes[1].axis('off')
    plt.suptitle('GWR local coefficients — Bristol LSOAs', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path / 'map_gwr_coefficients.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    print("     Saved: map_gwr_coefficients.png")

    pct = (crime_coefs < -0.05).sum() / len(crime_coefs) * 100
    print(f"     Crime coef range: {crime_coefs.min():.4f} to {crime_coefs.max():.4f}")
    print(f"     {pct:.0f}% of LSOAs show meaningfully negative crime effect")
    print()
