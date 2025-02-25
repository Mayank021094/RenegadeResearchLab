# -----------------IMPORTING LIBRARIES-------------#
import pandas as pd


# -------------------MAIN CODE----------------------#
def equi_wt(scores, symbol_mapping):
    # Merge scores with symbol_mapping to get 'symbol_nse' for each 'symbol_yfinance'
    scores = scores.merge(symbol_mapping, on='symbol_yfinance', how='left')
    scores.drop_duplicates(inplace=True)

    # If scores is empty after merging, return an empty DataFrame
    if scores.empty:
        return pd.DataFrame(columns=['symbol_nse', 'wts'])

    # Assign equal weight
    wt = round(1 / len(scores), 3)
    symbols = scores['symbol_nse'].values.tolist()

    # Create DataFrame with symbols and equal weights
    wts_df = pd.DataFrame(symbols, columns=['symbol_nse'])
    wts_df['wts'] = wt
    wts_df['scores'] = scores['scores'].round(3).values.tolist()
    return wts_df


def cap_wt(scores, symbol_mapping):
    # Merge scores with symbol_mapping to get 'symbol_nse' and ensure 'mkt_cap' exists
    scores = scores.merge(symbol_mapping, on='symbol_yfinance', how='left')
    scores.drop_duplicates(inplace=True)

    # If scores is empty after merging, return an empty DataFrame
    if scores.empty:
        return pd.DataFrame(columns=['symbol_nse', 'wts'])

    # Check if required columns exist
    if 'symbol_nse' not in scores.columns or 'mkt_cap' not in scores.columns:
        raise ValueError("Required columns missing. Ensure 'symbol_nse' and 'mkt_cap' are present.")

    # Calculate market-cap weights
    total_mkt_cap = scores['mkt_cap'].sum()

    # If total market cap is zero (e.g., all mkt_cap values are zero), return empty weights
    if total_mkt_cap == 0:
        return pd.DataFrame(columns=['symbol_nse', 'wts'])

    wts = [round(w / total_mkt_cap, 3) for w in scores['mkt_cap']]

    # Create DataFrame with symbols and market-cap weights
    symbols = scores['symbol_nse'].values.tolist()
    wts_df = pd.DataFrame(symbols, columns=['symbol_nse'])
    wts_df['wts'] = wts
    wts_df['scores'] = scores['scores'].round(3).values.tolist()
    return wts_df
