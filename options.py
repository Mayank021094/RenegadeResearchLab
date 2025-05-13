# ------------------Import Libraries -------------#
import numpy as np
from extract_options_chain import ExtractOptionsChain
from extract_options_data import ExtractOptionsData
from estimate_volatility import EstimateVolatility
from extract_rf import extract_risk_free_rate
import math
import pandas as pd
from greeks import Extract_Greeks
from options_strategies import Strategies
import pickle
from zerodha import Zerodha_Api
import datetime
from utilities import find_best_straddle_strike, select_strikes_for_corrado_su


# -------------------- MAIN CODE -------------------#
def option_json(api_key, secret_key, client_id):
    """
    Extract available option symbols, compute realized volatilities,
    implied moments and return a JSON with the results.
    """
    # Extract available option symbols using helper class
    init_zerodha_instance = Zerodha_Api(api_key=api_key, api_secret_key=secret_key, client_id=client_id)
    symbols = init_zerodha_instance.extract_available_symbols()
    # options_data = ExtractOptionsData()
    # symbols = options_data.extract_available_option_symbols()
    # if symbols.empty:
    #     raise ValueError("No available option symbols found.")

    options_json = {}
    # Get risk-free rate data
    rf = extract_risk_free_rate()
    rf = round(np.mean(rf['MIBOR Rate (%)']), 3)
    #indexes = np.random.randint(0, 220, size=50)
    #symbols = symbols.iloc[indexes]

    # Iterate over each symbol (unpacking index and row)
    for _, r in symbols.iterrows():
        symbols_json = {}
        symbols_json['underlying'] = r['symbol']
        symbols_json['category'] = r['category']

        # Estimate realized volatility using different methods
        vol_estimates = EstimateVolatility(ticker=r['symbol'], category=r['category'])

        symbols_json['expiry'] = {}

        # Extract options chain data for calls and puts
        op_chain = init_zerodha_instance.extract_options_chain(ticker=r['symbol'])
        # call_chain = op_chain.extract_call_data()
        if op_chain.empty:
            print("❌ Failed to extract Option Chain for symbol: {}".format(r['symbol']))
            continue
        # put_chain = op_chain.extract_put_data()

        q = ExtractOptionsData.extracting_dividend_yield(ticker=r['symbol'], category=r['category'])

        current_date = datetime.date.today()
        op_chain['dte'] = np.busday_count(
            current_date,
            op_chain['expiry'].values.astype('datetime64[D]'),
            weekmask='1111111')

        symbols_json['option_chain'] = op_chain

        # ce_json = {}
        # pe_json = {}
        strategies_json = {}

        # Process each expiry date in the call chain
        for expiry in op_chain['expiry'].unique():
            expiry_json = {}
            implied_moments_json = {}
            op_chain_copy = op_chain[op_chain['expiry'] == expiry]
            try:
                best_parameters = find_best_straddle_strike(op_chain_copy)
                if best_parameters == None:
                    continue
                bsm_imp_vol_ce = EstimateVolatility.bsm_implied_volatility(
                    mkt_price=best_parameters['call_premium'],
                    S=best_parameters['underlying_last'],
                    K=best_parameters['strike'],
                    rf=rf, maturity=best_parameters['expiry'],
                    option_category='CE', q=q, current_date=None
                )

                bsm_imp_vol_pe = EstimateVolatility.bsm_implied_volatility(
                    mkt_price=best_parameters['put_premium'],
                    S=best_parameters['underlying_last'],
                    K=best_parameters['strike'],
                    rf=rf, maturity=best_parameters['expiry'],
                    option_category='PE', q=q, current_date=None
                )

                bsm_iv_atm = (bsm_imp_vol_ce + bsm_imp_vol_pe) / 2

                implied_moments_json['bsm_implied_vol_ce'] = round(bsm_imp_vol_ce, 3)
                implied_moments_json['bsm_implied_vol_pe'] = round(bsm_imp_vol_pe, 3)
                implied_moments_json['bsm_atm_iv'] = round(bsm_iv_atm, 3)

            except Exception as e:
                print(f"❌ Error processing BSM implied volatility for {r['symbol']} expiry {expiry}: {e}")
                continue

                # binomial_iv_ce = EstimateVolatility.implied_vol_binomial(
                #     S=best_parameters['underlying_last'],
                #     K=best_parameters['strike'],
                #     T=best_parameters['dte'] / 365,
                #     option_type='CE',
                #     r=rf, q=q, market_price=best_parameters['call_premium'],
                #     N=100)
                #
                # binomial_iv_pe = EstimateVolatility.implied_vol_binomial(
                #     S=best_parameters['underlying_last'],
                #     K=best_parameters['strike'],
                #     T=best_parameters['dte'] / 365,
                #     option_type='PE',
                #     r=rf, q=q, market_price=best_parameters['put_premium'],
                #     N=100)
                #
                # binomial_iv_atm = (binomial_iv_ce + binomial_iv_pe) / 2
                #
                # implied_moments_json['binomial_implied_vol_ce'] = round(binomial_iv_ce, 3)
                # implied_moments_json['binomial_implied_vol_pe'] = round(binomial_iv_pe, 3)
                # implied_moments_json['binomial_atm_iv'] = round(binomial_iv_atm, 3)

            best_parameters_cs = select_strikes_for_corrado_su(op_chain_copy)
            try:
                if len(best_parameters_cs) >= 6:
                    # Calculate moments for calls and puts
                    cs_moments_ce = vol_estimates.corrado_su_implied_moments(
                        mkt_price=np.array(best_parameters_cs['mid_ce']),
                        S=best_parameters_cs['underlying_last'].values[0],
                        K=np.array(best_parameters_cs['strike']),
                        rf=rf,
                        maturity=expiry,
                        option_category='CE',
                        q=q,
                        current_date=None
                    )
                    cs_moments_pe = vol_estimates.corrado_su_implied_moments(
                        mkt_price=np.array(best_parameters_cs['mid_pe']),
                        S=best_parameters_cs['underlying_last'].values[0],
                        K=np.array(best_parameters_cs['strike']),
                        rf=rf,
                        maturity=expiry,
                        option_category='PE',
                        q=q,
                        current_date=None
                    )

                    # Check if optimization succeeded
                    if isinstance(cs_moments_ce, pd.DataFrame) and isinstance(cs_moments_pe, pd.DataFrame):
                        implied_moments_json['cs_implied_vol'] = round((
                                                                               cs_moments_ce[
                                                                                   'cs_implied_vol'].values[0] +
                                                                               cs_moments_pe[
                                                                                   'cs_implied_vol'].values[0]
                                                                       ) / 2, 3)
                        implied_moments_json['cs_implied_skew'] = round(cs_moments_ce['cs_implied_skew'].values[0],
                                                                        3)
                        implied_moments_json['cs_implied_kurt'] = round(cs_moments_ce['cs_implied_kurt'].values[0],
                                                                        3)
                    else:
                        print("❌ Corrado-Su optimization failed for symbol: {}, expiry: {}".format(r['symbol'],                                                                      expiry))
                        continue
                else:
                    print("❌ Insufficient strikes (<6) for symbol: {}, expiry: {}".format(r['symbol'], expiry))
                    continue
            except Exception as e:
                print("❌ Error in Corrado-Su calculation for {}: {}".format(r['symbol'], str(e)))
                continue

            expiry_json['implied_moments'] = implied_moments_json

            vol_forecast_json = {}
            vol_forecast_json['GARCH(1,1)'] = round(vol_estimates.garch(horizon=op_chain_copy['dte'].values[0])[0], 3)

            expiry_json['vol_forecast'] = vol_forecast_json

            forecasts_list = list(vol_forecast_json.values())
            if forecasts_list and len(forecasts_list) > 0:
                avg_vol_forecast = sum(forecasts_list) / len(forecasts_list)

                # Calculate edge: % difference between BSM IV and forecast, floored at 0
                expiry_json['edge'] = max((bsm_iv_atm - avg_vol_forecast) / avg_vol_forecast, 0)

                # Assign expiry data to symbols_json
                symbols_json['expiry'][expiry] = expiry_json
            else:
                print(f"❌ No volatility forecasts found for expiry: {expiry}")
                expiry_json['edge'] = 0  # Default to no edge if forecasts are missing

        if len(expiry_json) == 0:
            continue
        # else:
        realized_vol_json = {
            'Close-to-close': vol_estimates.close_to_close(),
            'Yang-Zhang': vol_estimates.yang_zhang(),
            'Parkinson': vol_estimates.parkinson(),
            'Garmann_Klass': vol_estimates.garman_klass(),
            'Rogers_Satchell': vol_estimates.rogers_satchell(),
            'High_Frequency': vol_estimates.high_frequency(period='30d', interval='15m')
        }
        cones_json = {
            'Volatility_Cones_df': vol_estimates.cones(moment='vol'),
            'Skewness_Cones_df': vol_estimates.cones(moment='skew'),
            'Kurtosis_Cones_df': vol_estimates.cones(moment='kurt')
        }

        symbols_json['realized_volatility'] = realized_vol_json
        symbols_json['cones'] = cones_json

        # Strategies processing (if needed) can be added here as well.
        # for expiry in call_chain['expiryDate'].unique():
        #     expiry_strategy_json = {}
        #     ce_chain_df = call_chain[call_chain['expiryDate'] == expiry]
        #     pe_chain_df = put_chain[put_chain['expiryDate'] == expiry]
        #     strategies = Strategies(expiry=expiry, ce_chain=ce_chain_df, pe_chain=pe_chain_df, rf=rf, q=q)
        #     expiry_strategy_json['Long Call'] = strategies.long_call()
        #     expiry_strategy_json['Long Put'] = strategies.long_put()
        #     expiry_strategy_json['Short Call'] = strategies.short_call()
        #     expiry_strategy_json['Short Put'] = strategies.short_put()
        #     expiry_strategy_json['Bull Call Spread'] = strategies.bull_call_spread()
        #     expiry_strategy_json['Bull Put Spread'] = strategies.bull_put_spread()
        #     expiry_strategy_json['Bear Call Spread'] = strategies.bear_call_spread()
        #     expiry_strategy_json['Bear Put Spread'] = strategies.bear_put_spread()
        #     expiry_strategy_json['Long Call Butterfly'] = strategies.long_call_butterfly()
        #     expiry_strategy_json['Long Put Butterfly'] = strategies.long_put_butterfly()
        #     expiry_strategy_json['Long Straddle'] = strategies.long_straddle()
        #     expiry_strategy_json['Short Straddle'] = strategies.short_straddle()
        #     expiry_strategy_json['Strip'] = strategies.strip()
        #     expiry_strategy_json['Strap'] = strategies.strap()
        #     expiry_strategy_json['Long Strangle'] = strategies.long_strangle()
        #     expiry_strategy_json['Short Strangle'] = strategies.short_strangle()
        #     strategies_json[expiry] = expiry_strategy_json
        #
        # symbols_json['CE'] = ce_json
        # symbols_json['PE'] = pe_json
        # symbols_json['strategies'] = strategies_json
        #
        # Add the symbol's data to the overall options JSON.
        options_json[r['symbol']] = symbols_json
    # Save the dictionary to a pickle file
    with open("options_dict.pkl", "wb") as file:
        pickle.dump(options_json, file)
    return options_json, init_zerodha_instance

# ----------------Usage----------------------#
# if __name__ == "__main__":
#     test_json = option_json()
#     # Save the dictionary to a pickle file
#     with open("options_json_for_ui_testing.pkl", "wb") as file:
#         pickle.dump(test_json, file)
