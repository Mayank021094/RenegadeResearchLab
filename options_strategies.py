# ------------------Import Libraries -------------#
from arch import arch_model
import pandas as pd
import numpy as np
import math
from extract_options_data import ExtractOptionsData  # Note: Duplicate imports removed
from scipy.optimize import minimize, brentq
import scipy.optimize as opt
from scipy.stats import norm
import scipy.stats as si
import warnings
from scipy.interpolate import CubicSpline
from extract_options_chain import ExtractOptionsChain
from extract_rf import extract_risk_free_rate
from greeks import Extract_Greeks

warnings.filterwarnings('ignore')
import datetime

# ---------------------CONSTANTS------------------#
DEFAULT_TRADING_PERIODS = 252  # Typical number of trading days in a year
DEFAULT_WINDOW = 30  # Default rolling window size for volatility calculations


# --------------------MAIN CODE-------------------#

class Strategies:
    def __init__(self, expiry, ce_chain, pe_chain, rf, maturity, q=0, current_date=None):
        """
                Initialize the option parameters and time to maturity.

                Parameters:

                    rf: DataFrame or dict containing risk-free rate data.
                        (Note: If rf values are in percentage, consider converting them by dividing by 100.)
                    maturity: Option expiration date (as a datetime.date or a compatible string).
                    q: Dividend yield (default is 0).
                    current_date: Valuation date (defaults to today's date if not provided).
        """
        self.ce_chain = ce_chain
        self.pe_chain = pe_chain
        self.rf = rf
        self.expiry = expiry

        if current_date is None:
            current_date = datetime.date.today()

        # Calculate time to maturity:
        # t1: trading days converted to years (used for volatility scaling).
        trading_days = np.busday_count(current_date, maturity)
        self.t1 = trading_days / DEFAULT_TRADING_PERIODS

        # t2: calendar days converted to years (used for discounting).
        calendar_days = np.busday_count(current_date, maturity, weekmask='1111111')
        self.t2 = calendar_days / 365

        self.q = q

    def long_call(self, imp_vol):
        """
        Compute the payoff and Greeks for a long call option strategy.

        This function performs the following steps:
          1. Filters the call option chain for in-the-money (ITM) call options.
          2. Extracts key parameters like the underlying price (S), strike (K), and market price.
          3. Sets up an underlying price range based on the implied volatility.
          4. Computes base option Greeks and Zakamouline delta band values at the current underlying price.
          5. Iterates over the price range to compute series data for payoff and Greeks.
          6. Returns a dictionary with the computed Greek values and a DataFrame containing the series data.

        Error handling is implemented via try-except blocks at critical steps.

        Parameters:
            imp_vol (float): The implied volatility.

        Returns:
            dict or None: A dictionary with keys "greeks" and "payoff" if successful,
                          or None if an error occurred.
        """

        # 1. Filter the option chain for in-the-money (ITM) call options (S > K)
        try:
            itm_calls = self.ce_chain[self.ce_chain['ltp'] - self.ce_chain['strikePrice'] > 0]
            if itm_calls.empty:
                raise ValueError("No in-the-money call options found in ce_chain.")
        except Exception as e:
            print(f"Error filtering ITM calls: {e}")
            return None

        # 2. Extract the underlying price, strike price, and market price
        try:
            S = self.ce_chain['ltp'].values[0]
            K = itm_calls['strikePrice'].values[0]
            mkt_price = itm_calls['ask_price'].values[0]
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        # 3. Compute the underlying price range for analysis using the implied volatility
        try:
            start = S - (imp_vol * S)
            stop = S + (imp_vol * S)
            S_array = np.linspace(start=start, stop=stop, num=100)
        except Exception as e:
            print(f"Error computing S_array: {e}")
            return None

        greeks_json = {}
        # 4. Instantiate the Extract_Greeks object and compute base Greeks
        try:
            object_greeks = Extract_Greeks(
                K=K,
                imp_vol=imp_vol,
                rf=self.rf,
                maturity=self.expiry,
                option_type='CE',
                q=self.q,
                current_date=None
            )
            greeks_json['delta'] = object_greeks.delta(S=S)
            greeks_json['gamma'] = object_greeks.gamma(S=S)
            greeks_json['vega'] = object_greeks.vega(S=S)
            greeks_json['rho'] = object_greeks.rho(S=S)
            greeks_json['theta'] = object_greeks.theta(S=S)
        except Exception as e:
            print(f"Error computing base Greeks: {e}")
            return None

        # 5. Compute the spread from the ITM call option data
        try:
            spread = (itm_calls['ask_price'].values[0] - itm_calls['bid_price'].values[0]) / itm_calls['mkt_price']
        except Exception as e:
            print(f"Error computing spread: {e}")
            return None

        # 6. Compute Zakamouline delta band for the current underlying price
        try:
            lower_band, upper_band = object_greeks.zakamouline_delta_band(S=S, spread=spread,
                                                                          option_gamma=greeks_json['gamma'])
            greeks_json['zak_lower_band'] = lower_band
            greeks_json['zak_upper_band'] = upper_band
        except Exception as e:
            print(f"Error computing Zakamouline delta band: {e}")
            return None

        # Initialize lists to store series data for the payoff and Greeks over the price range.
        payoff_series = []
        delta_series = []
        gamma_series = []
        vega_series = []
        zak_lower_band_series = []
        zak_upper_band_series = []

        # 7. Loop over the price range to compute payoff and Greek series
        for s in S_array:
            try:
                # Compute the payoff for a long call option.
                payoff_series.append(max(s - K - mkt_price, -mkt_price))
                # Compute delta, gamma, and vega for the given underlying price.
                delta_series.append(object_greeks.delta(S=s))
                gamma_temp = object_greeks.gamma(S=s)
                gamma_series.append(gamma_temp)
                vega_series.append(object_greeks.vega(S=s))
                # Compute the Zakamouline delta band for the current price.
                lb, ub = object_greeks.zakamouline_delta_band(S=s, spread=spread, option_gamma=gamma_temp)
                zak_lower_band_series.append(lb)
                zak_upper_band_series.append(ub)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue  # Skip this iteration and proceed with the next underlying price

        # 8. Create a DataFrame from the series data
        try:
            payoff_df = pd.DataFrame({
                'payoff': payoff_series,
                'delta': delta_series,
                'gamma': gamma_series,
                'vega': vega_series,
                'zak_lower_band': zak_lower_band_series,
                'zak_upper_band': zak_upper_band_series
            })
            payoff_json = {'payoffs': payoff_df}
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            return None

        # 9. Return the computed base Greeks and the payoff DataFrame in a structured dictionary.
        try:
            result = {"greeks": greeks_json, "payoff": payoff_json}
            return result
        except Exception as e:
            print(f"Error returning results: {e}")
            return None

    def short_call(self, imp_vol):
        """
                Compute the payoff and Greeks for a long call option strategy.

                This function performs the following steps:
                  1. Filters the call option chain for in-the-money (ITM) call options.
                  2. Extracts key parameters like the underlying price (S), strike (K), and market price.
                  3. Sets up an underlying price range based on the implied volatility.
                  4. Computes base option Greeks and Zakamouline delta band values at the current underlying price.
                  5. Iterates over the price range to compute series data for payoff and Greeks.
                  6. Returns a dictionary with the computed Greek values and a DataFrame containing the series data.

                Error handling is implemented via try-except blocks at critical steps.

                Parameters:
                    imp_vol (float): The implied volatility.

                Returns:
                    dict or None: A dictionary with keys "greeks" and "payoff" if successful,
                                  or None if an error occurred.
                """

        # 1. Filter the option chain for in-the-money (ITM) call options (S > K)
        try:
            itm_calls = self.ce_chain[self.ce_chain['ltp'] - self.ce_chain['strikePrice'] > 0]
            if itm_calls.empty:
                raise ValueError("No in-the-money call options found in ce_chain.")
        except Exception as e:
            print(f"Error filtering ITM calls: {e}")
            return None

        # 2. Extract the underlying price, strike price, and market price
        try:
            S = self.ce_chain['ltp'].values[0]
            K = itm_calls['strikePrice'].values[0]
            mkt_price = itm_calls['bid_price'].values[0]
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        # 3. Compute the underlying price range for analysis using the implied volatility
        try:
            start = S - (imp_vol * S)
            stop = S + (imp_vol * S)
            S_array = np.linspace(start=start, stop=stop, num=100)
        except Exception as e:
            print(f"Error computing S_array: {e}")
            return None

        greeks_json = {}
        # 4. Instantiate the Extract_Greeks object and compute base Greeks
        try:
            object_greeks = Extract_Greeks(
                K=K,
                imp_vol=imp_vol,
                rf=self.rf,
                maturity=self.expiry,
                option_type='CE',
                q=self.q,
                current_date=None
            )
            greeks_json['delta'] = -object_greeks.delta(S=S)
            greeks_json['gamma'] = -object_greeks.gamma(S=S)
            greeks_json['vega'] = -object_greeks.vega(S=S)
            greeks_json['rho'] = -object_greeks.rho(S=S)
            greeks_json['theta'] = -object_greeks.theta(S=S)
        except Exception as e:
            print(f"Error computing base Greeks: {e}")
            return None

        # 5. Compute the spread from the ITM call option data
        try:
            spread = (itm_calls['ask_price'].values[0] - itm_calls['bid_price'].values[0]) / itm_calls['mkt_price']
        except Exception as e:
            print(f"Error computing spread: {e}")
            return None

        # 6. Compute Zakamouline delta band for the current underlying price
        try:
            lower_band, upper_band = object_greeks.zakamouline_delta_band(S=S, spread=spread,
                                                                          option_gamma=greeks_json['gamma'])
            greeks_json['zak_lower_band'] = -upper_band
            greeks_json['zak_upper_band'] = -lower_band
        except Exception as e:
            print(f"Error computing Zakamouline delta band: {e}")
            return None

        # Initialize lists to store series data for the payoff and Greeks over the price range.
        payoff_series = []
        delta_series = []
        gamma_series = []
        vega_series = []
        zak_lower_band_series = []
        zak_upper_band_series = []

        # 7. Loop over the price range to compute payoff and Greek series
        for s in S_array:
            try:
                # Compute the payoff for a long call option.
                payoff_series.append(-max(s - K - mkt_price, -mkt_price))
                # Compute delta, gamma, and vega for the given underlying price.
                delta_series.append(-object_greeks.delta(S=s))
                gamma_temp = -object_greeks.gamma(S=s)
                gamma_series.append(gamma_temp)
                vega_series.append(-object_greeks.vega(S=s))
                # Compute the Zakamouline delta band for the current price.
                lb, ub = object_greeks.zakamouline_delta_band(S=s, spread=spread, option_gamma=gamma_temp)
                zak_lower_band_series.append(-ub)
                zak_upper_band_series.append(-lb)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue  # Skip this iteration and proceed with the next underlying price

        # 8. Create a DataFrame from the series data
        try:
            payoff_df = pd.DataFrame({
                'payoff': payoff_series,
                'delta': delta_series,
                'gamma': gamma_series,
                'vega': vega_series,
                'zak_lower_band': zak_lower_band_series,
                'zak_upper_band': zak_upper_band_series
            })
            payoff_json = {'payoffs': payoff_df}
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            return None

        # 9. Return the computed base Greeks and the payoff DataFrame in a structured dictionary.
        try:
            result = {"greeks": greeks_json, "payoff": payoff_json}
            return result
        except Exception as e:
            print(f"Error returning results: {e}")
            return None

    def long_put(self):

    def short_put(self):

    def bull_call_spread(self):

    def bull_put_spread(self):

    def bear_call_spread(self):

    def bear_put_spread(self):

    def long_call_butterfly(self):

    def long_put_butterfly(self):

    def long_straddle(self):

    def short_straddle(self):

    def strip(self):

    def strap(self):

    def long_strangle(self):

    def short_strangle(self):
