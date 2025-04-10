# ------------------Import Libraries -------------#
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------CONSTANTS------------------#
DEFAULT_TRADING_PERIODS = 252  # Typical number of trading days in a year
DEFAULT_WINDOW = 30  # Default rolling window size for volatility calculations


# --------------------MAIN CODE-------------------#

def edge_table(options_json):
    symbols = options_json.keys()
    # Create Edge dataframe
    data = []
    for s in symbols:
        expiry_list = options_json['NIFTY']['CE'].keys()
        for expiry in expiry_list:
            try:
                data.append({
                    'Symbol': f"{s}",
                    'Maturity': expiry,
                    'Edge': options_json[s]['CE'][expiry]['implied_moments']['edge']
                })
            except:
                print(s)
                continue
    edge_df = pd.DataFrame(data)
    edge_df = edge_df.sort_values(by='Edge', ascending=False)

    return edge_df
def estimator_table(symbol_json, expiry):
    cones_list = ['Volatility_Cones_df', 'Skewness_Cones_df', 'Kurtosis_Cones_df']
    estimators = symbol_json['realized_volatility'].keys()
    estimators = [k for k in estimators if k not in cones_list]
    data = []
    volatility_json = symbol_json['realized_volatility']
    for estimator in estimators:
        data.append({
            'Estimator': estimator,
            'Point Forecast': volatility_json[estimator].mean(),
            '90% CI Lower Limit': volatility_json[estimator].quantile(0.05),
            '90% CI Upper Limit': volatility_json[estimator].quantile(0.95)
        })
    data.append({'Estimator': 'BSM Implied Volatility',
                 'Point Forecast': symbol_json['CE'][expiry]['implied_moments']['bsm_implied_vol'],
                 '90% CI Lower Limit': '-',
                 '90% CI Upper Limit': '-',
                 })
    data.append({'Estimator': 'Corrado-Su Implied Volatility',
                 'Point Forecast': symbol_json['CE'][expiry]['implied_moments']['cs_implied_vol'],
                 '90% CI Lower Limit': '-',
                 '90% CI Upper Limit': '-',
                 })
    data.append({'Estimator': 'Corrado-Su Implied Skewness',
                 'Point Forecast': symbol_json['CE'][expiry]['implied_moments']['cs_implied_skew'],
                 '90% CI Lower Limit': '-',
                 '90% CI Upper Limit': '-',
                 })
    data.append({'Estimator': 'Corrado-Su Implied Kurtosis',
                 'Point Forecast': symbol_json['CE'][expiry]['implied_moments']['cs_implied_kurt'],
                 '90% CI Lower Limit': '-',
                 '90% CI Upper Limit': '-',
                 })
    estimator_df = pd.DataFrame(data)

    return estimator_df


def plot_volatility_cone_plotly(cones_df, windows=[20, 40, 60, 120, 240], quantiles=[0.25, 0.75], moment='Volatility'):
    """
    Plots a volatility cone figure using Plotly.

    Parameters:
        cones_df : pd.DataFrame
            DataFrame containing volatility series for each rolling window.
            Expected columns: '20_day_vol', '40_day_vol', etc.
        windows : list of int
            Rolling window sizes (in days) to plot.
        quantiles : list of float
            Two-element list containing the lower and upper quantiles (e.g., [0.25, 0.75])
            used for the cone boundaries.

    Returns:
        fig : plotly.graph_objects.Figure
            The generated Plotly figure.
    """
    # Check that quantiles are correctly provided.
    if len(quantiles) != 2:
        raise ValueError("A two-element list for quantiles is required (e.g., [0.25, 0.75]).")
    if quantiles[0] >= quantiles[1]:
        raise ValueError("The first quantile must be less than the second.")

    # Prepare lists to store summary statistics and data for violin plots.
    max_vals = []
    min_vals = []
    top_quantiles = []
    medians = []
    bottom_quantiles = []
    violin_data = []  # For the violin plot

    for window in windows:
        col_name = f"{window}_day"
        if col_name not in cones_df.columns:
            raise ValueError(f"Expected column '{col_name}' not found in DataFrame.")

        # Drop NaNs from the series for accurate statistics.
        vol_series = cones_df[col_name].dropna()

        # Compute the summary statistics.
        max_vals.append(vol_series.max())
        min_vals.append(vol_series.min())
        top_quantiles.append(vol_series.quantile(quantiles[1]))
        medians.append(vol_series.median())
        bottom_quantiles.append(vol_series.quantile(quantiles[0]))

        violin_data.append(vol_series)

    # Create a subplot with 1 row and 2 columns.
    # Adjust column widths to mimic the original layout (e.g., wider left panel).
    fig = make_subplots(rows=1, cols=2,
                        column_widths=[0.75, 0.25],
                        subplot_titles=(f"{moment} Cone", "Distribution"))

    # Add the volatility cone line plots to the left subplot.
    fig.add_trace(go.Scatter(x=windows, y=max_vals, mode='lines+markers',
                             name='Max'), row=1, col=1)
    fig.add_trace(go.Scatter(x=windows, y=top_quantiles, mode='lines+markers',
                             name=f"{int(quantiles[1] * 100)} Prctl"), row=1, col=1)
    fig.add_trace(go.Scatter(x=windows, y=medians, mode='lines+markers',
                             name='Median'), row=1, col=1)
    fig.add_trace(go.Scatter(x=windows, y=bottom_quantiles, mode='lines+markers',
                             name=f"{int(quantiles[0] * 100)} Prctl"), row=1, col=1)
    fig.add_trace(go.Scatter(x=windows, y=min_vals, mode='lines+markers',
                             name='Min'), row=1, col=1)
    # Format the left subplot (volatility cone):
    # Format y-axis as percentages (e.g., 2% for 0.02).
    fig.update_yaxes(tickformat=',.0%', row=1, col=1)
    # Set custom x-axis ticks and limits.
    fig.update_xaxes(tickmode='array', tickvals=windows,
                     range=[windows[0] - 5, windows[-1] + 5], row=1, col=1)

    # Add violin plots to the right subplot for each rolling window.
    # Each trace corresponds to one window; using the window size as a categorical x value.
    for i, window in enumerate(windows):
        fig.add_trace(go.Violin(y=violin_data[i],
                                x=[str(window)] * len(violin_data[i]),
                                name=f"{window}-day",
                                box_visible=True,
                                meanline_visible=True,
                                showlegend=False),
                      row=1, col=2)

    # Format the right subplot's y-axis similarly.
    fig.update_yaxes(tickformat=',.0%', row=1, col=2)

    # Update the layout: adjust margins and position the legend at the bottom center.
    # Update the layout with a white background.
    fig.update_layout(
        template="plotly_white",  # Sets both paper and plot background to white.
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig
