# ------------------Import Libraries -------------#
import pandas as pd
import numpy as np


# -------------------- MAIN CODE -------------------#

def find_best_straddle_strike(option_chain_df, min_oi=500, max_spread_pct=0.1):
    """
    Find the optimal ATM strike for short straddle based on liquidity filters.

    Args:
        option_chain_df: DataFrame containing option chain data
        min_oi: Minimum open interest threshold (default 500)
        max_spread_pct: Maximum bid-ask spread percentage (default 10%)
        dte_range: Tuple of (min_dte, max_dte) to filter expiries

    Returns:
        Dictionary containing best strike details or None if no suitable strike found
    """
    # Make a copy to avoid modifying original dataframe
    df = option_chain_df.copy()

    if df.empty:
        return None

    df = df[
        (df['ce_oi'] >= min_oi) &
        (df['pe_oi'] >= min_oi) &
        (df['bid_ask_pct_ce'] <= max_spread_pct) &
        (df['bid_ask_pct_pe'] <= max_spread_pct)
        ].reset_index(drop=True)

    if df.empty:
        return None

    # Find current spot price (assuming same for all rows)
    spot_price = df['underlying_last'].iloc[0]

    # Calculate absolute distance from spot for each strike
    df['distance_from_spot'] = abs(df['strike'] - spot_price)

    if df.empty:
        return None

    # To find the strike closest to ATM with best liquidity
    # We'll score each strike based on:
    # 1. Distance from spot (lower is better)
    # 2. Combined OI (higher is better)
    # 3. Combined spread percentage (lower is better)

    df['combined_oi'] = df['ce_oi'] + df['pe_oi']
    df['combined_spread_pct'] = (df['bid_ask_pct_ce'] + df['bid_ask_pct_pe']) / 2

    # Normalize factors for scoring (0-1 scale)
    df['distance_score'] = 1 - (df['distance_from_spot'] / df['distance_from_spot'].max())
    df['oi_score'] = df['combined_oi'] / df['combined_oi'].max()
    df['spread_score'] = 1 - (df['combined_spread_pct'] / df['combined_spread_pct'].max())

    # Weighted score (adjust weights as needed)
    df['liquidity_score'] = (
            0.4 * df['distance_score'] +
            0.4 * df['oi_score'] +
            0.2 * df['spread_score']
    )

    # Sort by best score and get top strike
    best_strike_row = df.sort_values('liquidity_score', ascending=False).iloc[0]
    return {
        'underlying_last': spot_price,
        'strike': best_strike_row['strike'],
        'expiry': best_strike_row['expiry'],
        'dte': best_strike_row['dte'],
        'call_premium': best_strike_row['mid_ce'],
        'put_premium': best_strike_row['mid_pe'],
        'call_oi': best_strike_row['ce_oi'],
        'put_oi': best_strike_row['pe_oi'],
        'call_spread_pct': best_strike_row['bid_ask_pct_ce'],
        'put_spread_pct': best_strike_row['bid_ask_pct_pe'],
        'distance_from_spot': best_strike_row['distance_from_spot'],
        'liquidity_score': best_strike_row['liquidity_score']
    }


def select_strikes_for_corrado_su(option_chain_df, min_oi_atm=500, min_oi_otm=200, max_spread_pct=0.1):
    """
    Select strikes for Corrado-Su with liquidity-adjusted filters.

    Args:
        option_chain_df: DataFrame of option chain
        spot_price: Current underlying price
        min_oi_atm: OI threshold for ATM strikes
        min_oi_otm: OI threshold for OTM strikes
        max_spread_pct: Max bid-ask spread %

    Returns:
        DataFrame of filtered strikes
    """
    df = option_chain_df.copy()

    spot_price = df['underlying_last'].iloc[0]

    # Calculate distance from spot and % spread
    df['distance'] = abs(df['strike'] - spot_price)
    df['is_atm'] = df['distance'] <= (spot_price * 0.02)  # ±2% = ATM

    # Apply liquidity filters
    atm_ce_ok = df['is_atm'] & (df['ce_oi'] > min_oi_atm)
    otm_ce_ok = ~df['is_atm'] & (df['ce_oi'] > min_oi_otm)

    atm_pe_ok = df['is_atm'] & (df['pe_oi'] > min_oi_atm)
    otm_pe_ok = ~df['is_atm'] & (df['pe_oi'] > min_oi_otm)

    spread_ok = (df['bid_ask_pct_ce'] < max_spread_pct) & (df['bid_ask_pct_pe'] < max_spread_pct)

    liquidity_mask = (atm_ce_ok | otm_ce_ok) & (atm_pe_ok | otm_pe_ok) & spread_ok

    # Apply the filter
    df = df[liquidity_mask]

    # Sort by distance
    df = df.sort_values('distance')

    return df
