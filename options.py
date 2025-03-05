import numpy as np
from extract_options_chain import ExtractOptionsChain
from extract_options_data import ExtractOptionsData
from estimate_volatility import EstimateVolatility
from extract_rf import extract_risk_free_rate
import math
import pandas as pd


# -------------------- MAIN CODE -------------------#
def option_json():
    """
    Extract available option symbols, compute realized volatilities,
    implied moments and return a JSON with the results.
    """
    # Extract available option symbols using helper class
    options_data = ExtractOptionsData()
    symbols = options_data.extract_available_option_symbols()
    print(symbols)

    options_json = {}
    # Get risk-free rate data
    rf = extract_risk_free_rate()
    symbols = symbols[2:5]
    # Iterate over each symbol (unpacking index and row)
    for _, r in symbols.iterrows():
        symbols_json = {}
        symbols_json['underlying'] = r['underlying']
        symbols_json['type'] = r['type']

        # Estimate realized volatility using different methods
        vol_estimates = EstimateVolatility(ticker=r['symbol'], type=r['type'])
        realized_vol_json = {
            'Close-to-close': vol_estimates.close_to_close(),
            'Yang-Zhang': vol_estimates.yang_zhang(),
            'Parkinson': vol_estimates.parkinson(),
            'Garmann_Klass': vol_estimates.garman_klass(),
            'Rogers_Satchell': vol_estimates.rogers_satchell(),
            'High_Frequency': vol_estimates.high_frequency(period='30d', interval='15m'),
            'GARCH(1,1)': vol_estimates.garch()
        }

        # Compute the mean of each volatility measure
        means = [s.mean() for s in realized_vol_json.values()]
        forecasted_realized_volatility = sum(means) / len(means)
        symbols_json['realized_volatility'] = realized_vol_json

        #print(symbols_json)
        # Initialize implied moments dictionary
        implied_moments_json = {}

        # Extract options chain data for calls and puts
        op_chain = ExtractOptionsChain(ticker=r['symbol'], type_=r['type'])
        call_chain = op_chain.extract_call_data()
        put_chain = op_chain.extract_put_data()

        ce_json = {}
        pe_json = {}
        # Process each expiry date in the call chain
        for expiry in call_chain['expiryDate'].unique():
            expiry_json = {}
            # Use single bracket filtering (not double brackets)
            ce_chain_df = call_chain[call_chain['expiryDate'] == expiry]
            print(ce_chain_df)
            try:
                expiry_json['bsm_implied_vol'] = vol_estimates.bsm_implied_volatility(
                    mkt_price=ce_chain_df['mkt_price'].values[0],
                    S=ce_chain_df['ltp'].values[0],
                    K=ce_chain_df['strikePrice'].values[0],
                    rf=rf, maturity=expiry,
                    option_type='CE', q=0, current_date=None
                )
                cs_moments = vol_estimates.corrado_su_implied_moments(
                    mkt_price=np.array(ce_chain_df['mkt_price']),
                    S=ce_chain_df['ltp'].values[0],
                    K=np.array(ce_chain_df['strikePrice']),
                    rf=rf, maturity=expiry, option_type='CE', q=0,
                    current_date=None
                )
                expiry_json['cs_implied_vol'] = cs_moments['cs_implied_vol'].values[0]
                expiry_json['cs_implied_skew'] = cs_moments['cs_implied_skew'].values[0]
                expiry_json['cs_implied_kurt'] = cs_moments['cs_implied_kurt'].values[0]
                expiry_json['edge'] = np.abs(
                    expiry_json['bsm_implied_vol'] - forecasted_realized_volatility) / forecasted_realized_volatility
                print(expiry_json)
            except Exception as e:
                print(f"❌ Error processing call option for expiry {expiry}: {e}")
            ce_json[expiry] = expiry_json
            print(ce_json)

            # Process put options similarly
            expiry_json = {}
            pe_chain_df = put_chain[put_chain['expiryDate'] == expiry]
            try:
                expiry_json['bsm_implied_vol'] = vol_estimates.bsm_implied_volatility(
                    mkt_price=pe_chain_df['mkt_price'].values[0],
                    S=pe_chain_df['ltp'].values[0],
                    K=pe_chain_df['strikePrice'].values[0],
                    rf=rf, maturity=expiry,
                    option_type='PE', q=0, current_date=None
                )
                cs_moments = vol_estimates.corrado_su_implied_moments(
                    mkt_price=np.array(pe_chain_df['mkt_price']),
                    S=pe_chain_df['ltp'].values[0],
                    K=np.array(pe_chain_df['strikePrice']),
                    rf=rf, maturity=expiry, option_type='PE', q=0,
                    current_date=None
                )
                expiry_json['cs_implied_vol'] = cs_moments['cs_implied_vol'].values[0]
                expiry_json['cs_implied_skew'] = cs_moments['cs_implied_skew'].values[0]
                expiry_json['cs_implied_kurt'] = cs_moments['cs_implied_kurt'].values[0]
                expiry_json['edge'] = np.abs(
                    expiry_json['bsm_implied_vol'] - forecasted_realized_volatility) / forecasted_realized_volatility
                print(expiry_json)
            except Exception as e:
                print(f"❌ Error processing put option for expiry {expiry}: {e}")
            pe_json[expiry] = expiry_json
            print(pe_json)

        implied_moments_json['CE'] = ce_json
        implied_moments_json['PE'] = pe_json
        symbols_json['implied_moments'] = implied_moments_json
        print(symbols_json)

        options_json[r['symbol']] = symbols_json

    return options_json

#----------------Usage----------------------#
test_json = option_json()
print(test_json)