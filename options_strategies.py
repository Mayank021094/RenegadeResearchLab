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
from estimate_volatility import EstimateVolatility
warnings.filterwarnings('ignore')
import datetime

# ---------------------CONSTANTS------------------#
DEFAULT_TRADING_PERIODS = 252  # Typical number of trading days in a year
DEFAULT_WINDOW = 30  # Default rolling window size for volatility calculations


# --------------------MAIN CODE-------------------#

class Strategies:
    def __init__(self, expiry, ce_chain, pe_chain, rf, q=0, current_date=None):
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
        trading_days = np.busday_count(current_date, expiry)
        self.t1 = trading_days / DEFAULT_TRADING_PERIODS

        # t2: calendar days converted to years (used for discounting).
        calendar_days = np.busday_count(current_date, expiry, weekmask='1111111')
        self.t2 = calendar_days / 365

        self.q = q

    def long_call(self):
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
            imp_vol = EstimateVolatility.bsm_implied_volatility(mkt_price=itm_calls['mkt_price'].values[0],
                                                                S=S, K=K, rf=self.rf, maturity=self.expiry,
                                                                option_type='CE', q=self.q, current_date=None)
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

    def short_call(self):
        """
                Compute the payoff and Greeks for a short call option strategy.

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
            imp_vol = EstimateVolatility.bsm_implied_volatility(mkt_price=itm_calls['mkt_price'].values[0],
                                                                S=S, K=K, rf=self.rf, maturity=self.expiry,
                                                                option_type='CE', q=self.q, current_date=None)
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
        """
                        Compute the payoff and Greeks for a long put option strategy.

                        This function performs the following steps:
                          1. Filters the put option chain for in-the-money (ITM) call options.
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
            itm_puts = self.pe_chain[self.pe_chain['strikePrice'] - self.pe_chain['ltp'] > 0]
            if itm_puts.empty:
                raise ValueError("No in-the-money put options found in pe_chain.")
        except Exception as e:
            print(f"Error filtering ITM puts: {e}")
            return None

        # 2. Extract the underlying price, strike price, and market price
        try:
            S = self.pe_chain['ltp'].values[0]
            K = itm_puts['strikePrice'].values[0]
            mkt_price = itm_puts['bid_price'].values[0]
            imp_vol = EstimateVolatility.bsm_implied_volatility(mkt_price=itm_puts['mkt_price'].values[0],
                                                                S=S, K=K, rf=self.rf, maturity=self.expiry,
                                                                option_type='PE', q=self.q, current_date=None)
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
                option_type='PE',
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
            spread = (itm_puts['ask_price'].values[0] - itm_puts['bid_price'].values[0]) / itm_puts['mkt_price']
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
                payoff_series.append(max(K - S - mkt_price, -mkt_price))
                # Compute delta, gamma, and vega for the given underlying price.
                delta_series.append(object_greeks.delta(S=s))
                gamma_temp = object_greeks.gamma(S=s)
                gamma_series.append(gamma_temp)
                vega_series.append(object_greeks.vega(S=s))
                # Compute the Zakamouline delta band for the current price.
                lb, ub = object_greeks.zakamouline_delta_band(S=s, spread=spread, option_gamma=gamma_temp)
                zak_lower_band_series.append(ub)
                zak_upper_band_series.append(lb)
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

    def short_put(self):
        """
                                Compute the payoff and Greeks for a short put option strategy.

                                This function performs the following steps:
                                  1. Filters the put option chain for in-the-money (ITM) call options.
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
            itm_puts = self.pe_chain[self.pe_chain['strikePrice'] - self.pe_chain['ltp'] > 0]
            if itm_puts.empty:
                raise ValueError("No in-the-money put options found in pe_chain.")
        except Exception as e:
            print(f"Error filtering ITM puts: {e}")
            return None

        # 2. Extract the underlying price, strike price, and market price
        try:
            S = self.pe_chain['ltp'].values[0]
            K = itm_puts['strikePrice'].values[0]
            mkt_price = itm_puts['bid_price'].values[0]
            imp_vol = EstimateVolatility.bsm_implied_volatility(mkt_price=itm_puts['mkt_price'].values[0],
                                                                S=S, K=K, rf=self.rf, maturity=self.expiry,
                                                                option_type='PE', q=self.q, current_date=None)
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
                option_type='PE',
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
            spread = (itm_puts['ask_price'].values[0] - itm_puts['bid_price'].values[0]) / itm_puts['mkt_price']
        except Exception as e:
            print(f"Error computing spread: {e}")
            return None

        # 6. Compute Zakamouline delta band for the current underlying price
        try:
            lower_band, upper_band = object_greeks.zakamouline_delta_band(S=S, spread=spread,
                                                                          option_gamma=greeks_json['gamma'])
            greeks_json['zak_lower_band'] = - upper_band
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
                payoff_series.append(-max(K - S - mkt_price, -mkt_price))
                # Compute delta, gamma, and vega for the given underlying price.
                delta_series.append(-object_greeks.delta(S=s))
                gamma_temp = -object_greeks.gamma(S=s)
                gamma_series.append(gamma_temp)
                vega_series.append(-object_greeks.vega(S=s))
                # Compute the Zakamouline delta band for the current price.
                lb, ub = object_greeks.zakamouline_delta_band(S=s, spread=spread, option_gamma=gamma_temp)
                zak_lower_band_series.append(-lb)
                zak_upper_band_series.append(-ub)
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

    def bull_call_spread(self):
        """
        Calculate the Greeks and payoff profile for a bull call spread strategy.

        The bull call spread is constructed by buying an in-the-money (ITM) call (strike K1)
        and selling an out-of-the-money (OTM) call (strike K2). This function:
          1. Filters the call chain to identify ITM and OTM call options.
          2. Extracts the underlying price, strike prices, and market prices.
          3. Computes a range of underlying prices (S_array) based on the implied volatility.
          4. Instantiates separate Greeks calculators (Extract_Greeks) for each option leg.
          5. Calculates the portfolio Greeks by subtracting the short leg from the long leg.
          6. Computes Zakamouline's delta bands for each leg and combines them to obtain the portfolio band.
          7. Iterates over the underlying price range to compute the payoff and series for the Greeks.
          8. Assembles the results into a Pandas DataFrame and returns the overall portfolio data.

        Error handling via try-except blocks is used throughout to catch issues like missing data,
        indexing errors, or arithmetic issues.

        Returns:
            dict or None: A dictionary containing base Greeks under key "greeks" and payoff series in "payoff".
                          Returns None if an error is encountered.
        """

        # 1. Filter for ITM and OTM call options from the call option chain (ce_chain).
        try:
            itm_calls = self.ce_chain[self.ce_chain['ltp'] - self.ce_chain['strikePrice'] > 0]
            otm_calls = self.ce_chain[self.ce_chain['ltp'] - self.ce_chain['strikePrice'] < 0]
            if itm_calls.empty or otm_calls.empty:
                raise ValueError("No ITM or OTM call options found in ce_chain.")
        except Exception as e:
            print(f"Error filtering ITM and OTM calls: {e}")
            return None

        # 2. Extract underlying price, strike prices, and market prices.
        try:
            # Underlying price S is taken from the ce_chain (assumes first row is representative).
            S = self.ce_chain['ltp'].values[0]
            # For the bull spread, assume long call at K1 (from ITM calls) and short call at K2 (from OTM calls).
            K_1 = itm_calls['strikePrice'].values[0]
            # Here we use the second row (index 1) from otm_calls. Ensure your data has at least 2 rows.
            K_2 = otm_calls['strikePrice'].values[1]
            # Extract the ask price of the long call and bid price of the short call.
            long_call_price = itm_calls['ask_price'].values[0]
            short_call_price = otm_calls['bid_price'].values[1]
            # Compute the net cost of the strategy.
            price_of_strategy = np.abs(short_call_price - long_call_price)
            imp_vol_k1 = EstimateVolatility.bsm_implied_volatility(mkt_price=itm_calls['mkt_price'].values[0],
                                                                S=S, K=K_1, rf=self.rf, maturity=self.expiry,
                                                                option_type='CE', q=self.q, current_date=None)
            imp_vol_k2 = EstimateVolatility.bsm_implied_volatility(mkt_price=otm_calls['mkt_price'].values[0],
                                                                S=S, K=K_2, rf=self.rf, maturity=self.expiry,
                                                                option_type='CE', q=self.q, current_date=None)

        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        # 3. Compute the underlying price range using implied volatility.
        try:
            start = S - (imp_vol_k1 * S)
            stop = S + (imp_vol_k1 * S)
            S_array = np.linspace(start, stop, num=100)
        except Exception as e:
            print(f"Error computing S_array: {e}")
            return None

        # Dictionary to hold the base Greeks for the portfolio.
        greeks_json = {}

        # 4. Instantiate Extract_Greeks objects for the long and short call legs, and compute base Greeks.
        try:
            long_call_greeks = Extract_Greeks(
                K=K_1,
                imp_vol=imp_vol_k1,
                rf=self.rf,
                maturity=self.expiry,
                option_type='CE',
                q=self.q,
                current_date=None
            )
            short_call_greeks = Extract_Greeks(
                K=K_2,
                imp_vol=imp_vol_k2,
                rf=self.rf,
                maturity=self.expiry,
                option_type='CE',
                q=self.q,
                current_date=None
            )
            # Calculate portfolio Greeks as difference: long call (positive) minus short call (negative exposure).
            greeks_json['delta'] = long_call_greeks.delta(S=S) - short_call_greeks.delta(S=S)
            gamma_long_call = long_call_greeks.gamma(S=S)
            gamma_short_call = short_call_greeks.gamma(S=S)
            greeks_json['gamma'] = gamma_long_call - gamma_short_call
            greeks_json['vega'] = long_call_greeks.vega(S=S) - short_call_greeks.vega(S=S)
            greeks_json['rho'] = long_call_greeks.rho(S=S) - short_call_greeks.rho(S=S)
            greeks_json['theta'] = long_call_greeks.theta(S=S) - short_call_greeks.theta(S=S)
        except Exception as e:
            print(f"Error computing base Greeks: {e}")
            return None

        # 5. Compute the spread from the ITM call option data.
        try:
            # Use ITM call's ask and bid prices and its market price.
            spread = (itm_calls['ask_price'].values[0] - itm_calls['bid_price'].values[0]) / \
                     itm_calls['mkt_price'].values[0]
        except Exception as e:
            print(f"Error computing spread: {e}")
            return None

        # 6. Compute Zakamouline's delta band for both long and short calls, then combine for the portfolio.
        try:
            lower_band_long_call, upper_band_long_call = long_call_greeks.zakamouline_delta_band(S=S, spread=spread,
                                                                                                 option_gamma=gamma_long_call)
            lower_band_short_call, upper_band_short_call = short_call_greeks.zakamouline_delta_band(S=S, spread=spread,
                                                                                                    option_gamma=gamma_short_call)
            # For the portfolio, subtract the short call's band (note that its greeks contribute negatively).
            greeks_json['zak_lower_band'] = lower_band_long_call - upper_band_short_call
            greeks_json['zak_upper_band'] = upper_band_long_call - lower_band_short_call
        except Exception as e:
            print(f"Error computing Zakamouline delta band: {e}")
            return None

        # 7. Loop over the underlying price range (S_array) to compute series data for payoff and Greeks.
        payoff_series = []
        delta_series = []
        gamma_series = []
        vega_series = []
        zak_lower_band_series = []
        zak_upper_band_series = []

        for s in S_array:
            try:
                # Calculate the payoff of the bull call spread:
                # - If underlying price is above K2, the maximum payoff is achieved.
                # - If between K1 and K2, payoff increases linearly.
                # - Below K1, the loss is limited to the net premium paid.
                if s >= K_2:
                    payoff_series.append(K_2 - K_1 - price_of_strategy)
                elif K_1 <= s < K_2:
                    payoff_series.append(s - K_1 - price_of_strategy)
                else:  # s < K_1
                    payoff_series.append(-price_of_strategy)

                # Compute the portfolio delta at price s.
                delta_val = long_call_greeks.delta(S=s) - short_call_greeks.delta(S=s)
                delta_series.append(delta_val)

                # Compute portfolio gamma.
                gamma_val = long_call_greeks.gamma(S=s) - short_call_greeks.gamma(S=s)
                gamma_series.append(gamma_val)

                # Compute portfolio vega.
                vega_val = long_call_greeks.vega(S=s) - short_call_greeks.vega(S=s)
                vega_series.append(vega_val)

                # Compute Zakamouline bands for each leg at s.
                lb_lc, ub_lc = long_call_greeks.zakamouline_delta_band(S=s, spread=spread,
                                                                       option_gamma=long_call_greeks.gamma(S=s))
                lb_sc, ub_sc = short_call_greeks.zakamouline_delta_band(S=s, spread=spread,
                                                                        option_gamma=short_call_greeks.gamma(S=s))
                # Combine the bands for the portfolio.
                zak_lower_band_series.append(lb_lc - ub_sc)
                zak_upper_band_series.append(ub_lc - lb_sc)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue  # If an error occurs for a particular s, skip to the next

        # 8. Create a DataFrame from the series data for further analysis or visualization.
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

        # 9. Return a structured dictionary containing the base Greeks and the payoff DataFrame.
        try:
            result = {"greeks": greeks_json, "payoff": payoff_json}
            return result
        except Exception as e:
            print(f"Error returning results: {e}")
            return None

    def bull_put_spread(self):
        """
        Calculate the Greeks and payoff profile for a bull put spread strategy.

        The bull put spread is constructed by selling a put with a higher strike (K2, ITM put)
        and buying a put with a lower strike (K1, OTM put). This function:
          1. Filters the put chain to identify ITM and OTM put options.
          2. Extracts the underlying price, strike prices, and market prices.
          3. Computes a range of underlying prices (S_array) based on the implied volatility.
          4. Instantiates separate Greeks calculators (Extract_Greeks) for each option leg.
          5. Calculates the portfolio Greeks as the difference between the long (OTM) and short (ITM) legs.
          6. Computes Zakamouline's delta bands for each put leg and combines them to obtain the portfolio band.
          7. Iterates over the underlying price range to compute the payoff and Greeks series.
          8. Assembles the results into a Pandas DataFrame and returns the overall portfolio data.

        Error handling via try-except blocks is used throughout to catch issues like missing data,
        indexing errors, or arithmetic issues.

        Returns:
            dict or None: A dictionary containing base Greeks under key "greeks" and the payoff series in "payoff".
                          Returns None if an error is encountered.
        """

        # 1. Filter for ITM and OTM put options from the put option chain (pe_chain).
        try:
            # For put options, ITM when strike > underlying price and OTM when strike < underlying price.
            itm_puts = self.pe_chain[self.pe_chain['strikePrice'] - self.pe_chain['ltp'] > 0]
            otm_puts = self.pe_chain[self.pe_chain['strikePrice'] - self.pe_chain['ltp'] < 0]
            if itm_puts.empty or otm_puts.empty:
                raise ValueError("No ITM or OTM put options found in pe_chain.")
        except Exception as e:
            print(f"Error filtering ITM and OTM puts: {e}")
            return None

        # 2. Extract underlying price, strike prices, and market prices.
        try:
            # Underlying price S is taken from the pe_chain (assumes first row is representative).
            S = self.pe_chain['ltp'].values[0]
            # For the bull put spread, use the OTM put for the long leg and the ITM put for the short leg.
            # Thus, K1 (lower strike) is from otm_puts (long put) and K2 (higher strike) is from itm_puts (short put).
            K_1 = otm_puts['strikePrice'].values[0]
            K_2 = itm_puts['strikePrice'].values[0]
            # Extract the ask price of the long put and bid price of the short put.
            long_put_price = otm_puts['ask_price'].values[0]
            short_put_price = itm_puts['bid_price'].values[0]
            # Compute the net premium of the strategy (credit if positive, debit if negative).
            price_of_strategy = short_put_price - long_put_price

            # Compute implied volatilities for each leg using the corresponding market prices.
            imp_vol_k1 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=otm_puts['mkt_price'].values[0],
                S=S, K=K_1, rf=self.rf, maturity=self.expiry,
                option_type='PE', q=self.q, current_date=None
            )
            imp_vol_k2 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=itm_puts['mkt_price'].values[0],
                S=S, K=K_2, rf=self.rf, maturity=self.expiry,
                option_type='PE', q=self.q, current_date=None
            )

        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        # 3. Compute the underlying price range using implied volatility.
        try:
            start = S - (imp_vol_k1 * S)
            stop = S + (imp_vol_k1 * S)
            S_array = np.linspace(start, stop, num=100)
        except Exception as e:
            print(f"Error computing S_array: {e}")
            return None

        # Dictionary to hold the base Greeks for the portfolio.
        greeks_json = {}

        # 4. Instantiate Extract_Greeks objects for the long and short put legs, and compute base Greeks.
        try:
            long_put_greeks = Extract_Greeks(
                K=K_1,
                imp_vol=imp_vol_k1,
                rf=self.rf,
                maturity=self.expiry,
                option_type='PE',
                q=self.q,
                current_date=None
            )
            short_put_greeks = Extract_Greeks(
                K=K_2,
                imp_vol=imp_vol_k2,
                rf=self.rf,
                maturity=self.expiry,
                option_type='PE',
                q=self.q,
                current_date=None
            )
            # Calculate portfolio Greeks as the difference: long put (bought) minus short put (sold).
            greeks_json['delta'] = long_put_greeks.delta(S=S) - short_put_greeks.delta(S=S)
            gamma_long_put = long_put_greeks.gamma(S=S)
            gamma_short_put = short_put_greeks.gamma(S=S)
            greeks_json['gamma'] = gamma_long_put - gamma_short_put
            greeks_json['vega'] = long_put_greeks.vega(S=S) - short_put_greeks.vega(S=S)
            greeks_json['rho'] = long_put_greeks.rho(S=S) - short_put_greeks.rho(S=S)
            greeks_json['theta'] = long_put_greeks.theta(S=S) - short_put_greeks.theta(S=S)
        except Exception as e:
            print(f"Error computing base Greeks: {e}")
            return None

        # 5. Compute the spread from the ITM put option data.
        try:
            # Use ITM put's ask and bid prices and its market price to calculate a relative spread.
            spread = (itm_puts['ask_price'].values[0] - itm_puts['bid_price'].values[0]) / itm_puts['mkt_price'].values[
                0]
        except Exception as e:
            print(f"Error computing spread: {e}")
            return None

        # 6. Compute Zakamouline's delta band for both long and short puts, then combine for the portfolio.
        try:
            lower_band_long_put, upper_band_long_put = long_put_greeks.zakamouline_delta_band(
                S=S, spread=spread, option_gamma=gamma_long_put
            )
            lower_band_short_put, upper_band_short_put = short_put_greeks.zakamouline_delta_band(
                S=S, spread=spread, option_gamma=gamma_short_put
            )
            # For the portfolio, subtract the short put's band from the long put's band.
            greeks_json['zak_lower_band'] = lower_band_long_put - upper_band_short_put
            greeks_json['zak_upper_band'] = upper_band_long_put - lower_band_short_put
        except Exception as e:
            print(f"Error computing Zakamouline delta band: {e}")
            return None

        # 7. Loop over the underlying price range (S_array) to compute series data for payoff and Greeks.
        payoff_series = []
        delta_series = []
        gamma_series = []
        vega_series = []
        zak_lower_band_series = []
        zak_upper_band_series = []

        for s in S_array:
            try:
                # Calculate the payoff of the bull put spread:
                # - If underlying price is above K₂, both puts expire worthless, so payoff = net premium.
                # - If between K₁ and K₂, the payoff decreases linearly.
                # - If below K₁, both options are exercised, and the maximum loss is realized.
                if s <= K_1:
                    payoff_series.append(K_1 - K_2 + price_of_strategy)
                elif K_1 < s <= K_2:
                    payoff_series.append(s - K_2 + price_of_strategy)
                else:  # s > K_2
                    payoff_series.append(price_of_strategy)

                # Compute the portfolio delta at price s.
                delta_val = long_put_greeks.delta(S=s) - short_put_greeks.delta(S=s)
                delta_series.append(delta_val)

                # Compute portfolio gamma.
                gamma_val = long_put_greeks.gamma(S=s) - short_put_greeks.gamma(S=s)
                gamma_series.append(gamma_val)

                # Compute portfolio vega.
                vega_val = long_put_greeks.vega(S=s) - short_put_greeks.vega(S=s)
                vega_series.append(vega_val)

                # Compute Zakamouline bands for each leg at s.
                lb_lp, ub_lp = long_put_greeks.zakamouline_delta_band(
                    S=s, spread=spread, option_gamma=long_put_greeks.gamma(S=s)
                )
                lb_sp, ub_sp = short_put_greeks.zakamouline_delta_band(
                    S=s, spread=spread, option_gamma=short_put_greeks.gamma(S=s)
                )
                # Combine the bands for the portfolio.
                zak_lower_band_series.append(lb_lp - ub_sp)
                zak_upper_band_series.append(ub_lp - lb_sp)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue  # Skip this s value if an error occurs

        # 8. Create a DataFrame from the series data for further analysis or visualization.
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

        # 9. Return a structured dictionary containing the base Greeks and the payoff DataFrame.
        try:
            result = {"greeks": greeks_json, "payoff": payoff_json}
            return result
        except Exception as e:
            print(f"Error returning results: {e}")
            return None

    def bear_call_spread(self):
        """
        Calculate the Greeks and payoff profile for a bear call spread strategy.

        In a bear call spread you sell an in-the-money (ITM) call (strike K₁) and buy an out-of-the-money (OTM) call (strike K₂).
        This function:
          1. Filters the call chain (ce_chain) to identify ITM and OTM call options.
          2. Extracts the underlying price, strike prices, and market prices.
          3. Computes a range of underlying prices (S_array) based on the implied volatility.
          4. Instantiates separate Greeks calculators (Extract_Greeks) for each option leg.
          5. Calculates the portfolio Greeks as the difference between the long and short call legs.
          6. Computes Zakamouline's delta bands for each leg and combines them for the portfolio.
          7. Iterates over the underlying price range to compute the payoff and Greeks series.
          8. Assembles the results into a Pandas DataFrame and returns the overall portfolio data.

        Error handling via try-except blocks is used to catch missing data, indexing errors, or arithmetic issues.

        Returns:
            dict or None: A dictionary containing base Greeks under "greeks" and the payoff series under "payoff".
                          Returns None if an error is encountered.
        """

        # 1. Filter for ITM and OTM call options from the call option chain (ce_chain).
        try:
            # For call options, ITM when underlying (ltp) > strike; OTM when ltp < strike.
            itm_calls = self.ce_chain[self.ce_chain['ltp'] - self.ce_chain['strikePrice'] > 0]
            otm_calls = self.ce_chain[self.ce_chain['ltp'] - self.ce_chain['strikePrice'] < 0]
            if itm_calls.empty or otm_calls.empty:
                raise ValueError("No ITM or OTM call options found in ce_chain.")
        except Exception as e:
            print(f"Error filtering ITM and OTM calls: {e}")
            return None

        # 2. Extract underlying price, strike prices, and market prices.
        try:
            # Underlying price S is taken from ce_chain (assumes the first row is representative).
            S = self.ce_chain['ltp'].values[0]

            # For a bear call spread:
            #   - Sell the ITM call (strike K₁) and buy the OTM call (strike K₂).
            # The ITM and OTM chains are already sorted, so take the first row of each.
            K_1 = itm_calls['strikePrice'].values[0]
            K_2 = otm_calls['strikePrice'].values[0]

            # Extract the bid price of the ITM call (to sell) and the ask price of the OTM call (to buy).
            short_call_price = itm_calls['bid_price'].values[0]
            long_call_price = otm_calls['ask_price'].values[0]

            # Compute the net premium (credit) of the strategy.
            price_of_strategy = short_call_price - long_call_price

            # Compute implied volatilities for each leg.
            imp_vol_k1 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=itm_calls['mkt_price'].values[0],
                S=S, K=K_1, rf=self.rf, maturity=self.expiry,
                option_type='CE', q=self.q, current_date=None
            )
            imp_vol_k2 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=otm_calls['mkt_price'].values[0],
                S=S, K=K_2, rf=self.rf, maturity=self.expiry,
                option_type='CE', q=self.q, current_date=None
            )
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        # 3. Compute the underlying price range using implied volatility.
        try:
            start = S - (imp_vol_k1 * S)
            stop = S + (imp_vol_k1 * S)
            S_array = np.linspace(start, stop, num=100)
        except Exception as e:
            print(f"Error computing S_array: {e}")
            return None

        # Dictionary to hold the base Greeks for the portfolio.
        greeks_json = {}

        # 4. Instantiate Extract_Greeks objects for the long and short call legs, and compute base Greeks.
        try:
            # For the bear call spread:
            #   - Short call: ITM call at strike K₁.
            #   - Long call: OTM call at strike K₂.
            short_call_greeks = Extract_Greeks(
                K=K_1,
                imp_vol=imp_vol_k1,
                rf=self.rf,
                maturity=self.expiry,
                option_type='CE',
                q=self.q,
                current_date=None
            )
            long_call_greeks = Extract_Greeks(
                K=K_2,
                imp_vol=imp_vol_k2,
                rf=self.rf,
                maturity=self.expiry,
                option_type='CE',
                q=self.q,
                current_date=None
            )
            # Calculate portfolio Greeks as: long call (positive) minus short call (negative exposure).
            greeks_json['delta'] = long_call_greeks.delta(S=S) - short_call_greeks.delta(S=S)
            gamma_short_call = short_call_greeks.gamma(S=S)
            gamma_long_call = long_call_greeks.gamma(S=S)
            greeks_json['gamma'] = gamma_long_call - gamma_short_call
            greeks_json['vega'] = long_call_greeks.vega(S=S) - short_call_greeks.vega(S=S)
            greeks_json['rho'] = long_call_greeks.rho(S=S) - short_call_greeks.rho(S=S)
            greeks_json['theta'] = long_call_greeks.theta(S=S) - short_call_greeks.theta(S=S)
        except Exception as e:
            print(f"Error computing base Greeks: {e}")
            return None

        # 5. Compute the relative spread from the ITM call data.
        try:
            # Calculate a relative spread using the ITM call's bid-ask spread.
            spread = (itm_calls['ask_price'].values[0] - itm_calls['bid_price'].values[0]) / \
                     itm_calls['mkt_price'].values[0]
        except Exception as e:
            print(f"Error computing spread: {e}")
            return None

        # 6. Compute Zakamouline's delta band for both long and short calls, then combine for the portfolio.
        try:
            lower_band_long_call, upper_band_long_call = long_call_greeks.zakamouline_delta_band(
                S=S, spread=spread, option_gamma=gamma_long_call
            )
            lower_band_short_call, upper_band_short_call = short_call_greeks.zakamouline_delta_band(
                S=S, spread=spread, option_gamma=gamma_short_call
            )
            # For the portfolio, subtract the short call's band.
            greeks_json['zak_lower_band'] = lower_band_long_call - upper_band_short_call
            greeks_json['zak_upper_band'] = upper_band_long_call - lower_band_short_call
        except Exception as e:
            print(f"Error computing Zakamouline delta band: {e}")
            return None

        # 7. Loop over the underlying price range (S_array) to compute series data for payoff and Greeks.
        payoff_series = []
        delta_series = []
        gamma_series = []
        vega_series = []
        zak_lower_band_series = []
        zak_upper_band_series = []

        for s in S_array:
            try:
                # Calculate the payoff for the bear call spread:
                # - If s <= K₁: Both calls expire worthless; payoff equals the net premium.
                # - If K₁ < s < K₂: Loss increases linearly; payoff = net premium - (s - K₁).
                # - If s >= K₂: Maximum loss is realized; payoff = net premium - (K₂ - K₁).
                if s >= K_2:
                    payoff_series.append(K_1 - K_2 + price_of_strategy)
                elif K_1 <= s < K_2:
                    payoff_series.append(K_1 - s + price_of_strategy)
                else:  # s < K₁
                    payoff_series.append(price_of_strategy)

                # Compute portfolio delta at price s.
                delta_val = long_call_greeks.delta(S=s) - short_call_greeks.delta(S=s)
                delta_series.append(delta_val)

                # Compute portfolio gamma.
                gamma_val = long_call_greeks.gamma(S=s) - short_call_greeks.gamma(S=s)
                gamma_series.append(gamma_val)

                # Compute portfolio vega.
                vega_val = long_call_greeks.vega(S=s) - short_call_greeks.vega(S=s)
                vega_series.append(vega_val)

                # Compute Zakamouline bands for each leg at s.
                lb_lc, ub_lc = long_call_greeks.zakamouline_delta_band(
                    S=s, spread=spread, option_gamma=long_call_greeks.gamma(S=s)
                )
                lb_sc, ub_sc = short_call_greeks.zakamouline_delta_band(
                    S=s, spread=spread, option_gamma=short_call_greeks.gamma(S=s)
                )
                # Combine the bands for the portfolio.
                zak_lower_band_series.append(lb_lc - ub_sc)
                zak_upper_band_series.append(ub_lc - lb_sc)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue  # Skip this s value if an error occurs

        # 8. Create a DataFrame from the series data for further analysis or visualization.
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

        # 9. Return a structured dictionary containing the base Greeks and the payoff DataFrame.
        try:
            result = {"greeks": greeks_json, "payoff": payoff_json}
            return result
        except Exception as e:
            print(f"Error returning results: {e}")
            return None

    def bear_put_spread(self):
        """
        Calculate the Greeks and payoff profile for a bear put spread strategy.

        In a bear put spread you buy an in-the-money (ITM) put (strike K₂) and sell an out-of-the-money (OTM) put (strike K₁).
        This function:
          1. Filters the put chain (pe_chain) to identify ITM and OTM put options.
          2. Extracts the underlying price, strike prices, and market prices.
          3. Computes a range of underlying prices (S_array) based on the implied volatility.
          4. Instantiates separate Greeks calculators (Extract_Greeks) for each put leg.
          5. Calculates the portfolio Greeks as the difference between the long (ITM) and short (OTM) legs.
          6. Computes Zakamouline's delta bands for each put leg and combines them to obtain the portfolio band.
          7. Iterates over the underlying price range to compute the payoff and Greeks series.
          8. Assembles the results into a Pandas DataFrame and returns the overall portfolio data.

        Error handling via try-except blocks is used to catch missing data, indexing errors, or arithmetic issues.

        Returns:
            dict or None: A dictionary containing base Greeks under key "greeks" and the payoff series in "payoff".
                          Returns None if an error is encountered.
        """

        # 1. Filter for ITM and OTM put options from the put option chain (pe_chain).
        try:
            # For put options, ITM when strike > underlying price and OTM when strike < underlying price.
            itm_puts = self.pe_chain[self.pe_chain['strikePrice'] - self.pe_chain['ltp'] > 0]
            otm_puts = self.pe_chain[self.pe_chain['strikePrice'] - self.pe_chain['ltp'] < 0]
            if itm_puts.empty or otm_puts.empty:
                raise ValueError("No ITM or OTM put options found in pe_chain.")
        except Exception as e:
            print(f"Error filtering ITM and OTM puts: {e}")
            return None

        # 2. Extract underlying price, strike prices, and market prices.
        try:
            # Underlying price S is taken from pe_chain (assumes first row is representative).
            S = self.pe_chain['ltp'].values[0]

            # For a bear put spread:
            #   - Buy the ITM put (strike K₂) and sell the OTM put (strike K₁).
            # Thus, K₁ (lower strike) is from otm_puts and K₂ (higher strike) is from itm_puts.
            K_1 = otm_puts['strikePrice'].values[0]
            K_2 = itm_puts['strikePrice'].values[0]

            # Extract the ask price of the long ITM put and the bid price of the short OTM put.
            long_put_price = itm_puts['ask_price'].values[0]
            short_put_price = otm_puts['bid_price'].values[0]

            # Compute the net premium (debit) of the strategy: the cost to buy the ITM put minus the premium received.
            price_of_strategy = long_put_price - short_put_price

            # Compute implied volatilities for each leg using the corresponding market prices.
            imp_vol_k1 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=otm_puts['mkt_price'].values[0],
                S=S, K=K_1, rf=self.rf, maturity=self.expiry,
                option_type='PE', q=self.q, current_date=None
            )
            imp_vol_k2 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=itm_puts['mkt_price'].values[0],
                S=S, K=K_2, rf=self.rf, maturity=self.expiry,
                option_type='PE', q=self.q, current_date=None
            )

        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        # 3. Compute the underlying price range using implied volatility.
        try:
            start = S - (imp_vol_k1 * S)
            stop = S + (imp_vol_k1 * S)
            S_array = np.linspace(start, stop, num=100)
        except Exception as e:
            print(f"Error computing S_array: {e}")
            return None

        # Dictionary to hold the base Greeks for the portfolio.
        greeks_json = {}

        # 4. Instantiate Extract_Greeks objects for the long and short put legs, and compute base Greeks.
        try:
            # For the bear put spread:
            #   - Long put: ITM put at strike K₂.
            #   - Short put: OTM put at strike K₁.
            long_put_greeks = Extract_Greeks(
                K=K_2,
                imp_vol=imp_vol_k2,
                rf=self.rf,
                maturity=self.expiry,
                option_type='PE',
                q=self.q,
                current_date=None
            )
            short_put_greeks = Extract_Greeks(
                K=K_1,
                imp_vol=imp_vol_k1,
                rf=self.rf,
                maturity=self.expiry,
                option_type='PE',
                q=self.q,
                current_date=None
            )
            # Calculate portfolio Greeks as the difference: long (ITM) minus short (OTM).
            greeks_json['delta'] = long_put_greeks.delta(S=S) - short_put_greeks.delta(S=S)
            gamma_long_put = long_put_greeks.gamma(S=S)
            gamma_short_put = short_put_greeks.gamma(S=S)
            greeks_json['gamma'] = gamma_long_put - gamma_short_put
            greeks_json['vega'] = long_put_greeks.vega(S=S) - short_put_greeks.vega(S=S)
            greeks_json['rho'] = long_put_greeks.rho(S=S) - short_put_greeks.rho(S=S)
            greeks_json['theta'] = long_put_greeks.theta(S=S) - short_put_greeks.theta(S=S)
        except Exception as e:
            print(f"Error computing base Greeks: {e}")
            return None

        # 5. Compute the relative spread from the ITM put data.
        try:
            # Calculate a relative spread using the ITM put's bid-ask spread.
            spread = (itm_puts['ask_price'].values[0] - itm_puts['bid_price'].values[0]) / itm_puts['mkt_price'].values[
                0]
        except Exception as e:
            print(f"Error computing spread: {e}")
            return None

        # 6. Compute Zakamouline's delta band for both long and short puts, then combine for the portfolio.
        try:
            lower_band_long_put, upper_band_long_put = long_put_greeks.zakamouline_delta_band(
                S=S, spread=spread, option_gamma=gamma_long_put
            )
            lower_band_short_put, upper_band_short_put = short_put_greeks.zakamouline_delta_band(
                S=S, spread=spread, option_gamma=gamma_short_put
            )
            # For the portfolio, subtract the short put's band from the long put's band.
            greeks_json['zak_lower_band'] = lower_band_long_put - lower_band_short_put
            greeks_json['zak_upper_band'] = upper_band_long_put - upper_band_short_put
        except Exception as e:
            print(f"Error computing Zakamouline delta band: {e}")
            return None

        # 7. Loop over the underlying price range (S_array) to compute series data for payoff and Greeks.
        payoff_series = []
        delta_series = []
        gamma_series = []
        vega_series = []
        zak_lower_band_series = []
        zak_upper_band_series = []

        for s in S_array:
            try:
                # Calculate the payoff of the bear put spread:
                # For a bear put spread, the net payoff at expiration is:
                #    max(K₂ - S, 0) - max(K₁ - S, 0) - net_debit
                # Thus:
                # - If s >= K₂: both puts expire worthless → payoff = - price_of_strategy.
                # - If K₁ < s < K₂: payoff = (K₂ - s) - price_of_strategy.
                # - If s <= K₁: both puts are in the money → payoff = (K₂ - K₁) - price_of_strategy.
                if s >= K_2:
                    payoff_series.append(-price_of_strategy)
                elif K_1 < s < K_2:
                    payoff_series.append((K_2 - s) - price_of_strategy)
                else:  # s <= K_1
                    payoff_series.append((K_2 - K_1) - price_of_strategy)

                # Compute the portfolio delta at price s.
                delta_val = long_put_greeks.delta(S=s) - short_put_greeks.delta(S=s)
                delta_series.append(delta_val)

                # Compute portfolio gamma.
                gamma_val = long_put_greeks.gamma(S=s) - short_put_greeks.gamma(S=s)
                gamma_series.append(gamma_val)

                # Compute portfolio vega.
                vega_val = long_put_greeks.vega(S=s) - short_put_greeks.vega(S=s)
                vega_series.append(vega_val)

                # Compute Zakamouline bands for each leg at s.
                lb_lp, ub_lp = long_put_greeks.zakamouline_delta_band(
                    S=s, spread=spread, option_gamma=long_put_greeks.gamma(S=s)
                )
                lb_sp, ub_sp = short_put_greeks.zakamouline_delta_band(
                    S=s, spread=spread, option_gamma=short_put_greeks.gamma(S=s)
                )
                # Combine the bands for the portfolio.
                zak_lower_band_series.append(lb_lp - lb_sp)
                zak_upper_band_series.append(ub_lp - ub_sp)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue  # Skip this s value if an error occurs

        # 8. Create a DataFrame from the series data for further analysis or visualization.
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

        # 9. Return a structured dictionary containing the base Greeks and the payoff DataFrame.
        try:
            result = {"greeks": greeks_json, "payoff": payoff_json}
            return result
        except Exception as e:
            print(f"Error returning results: {e}")
            return None

    def long_call_butterfly(self):


    def long_put_butterfly(self):

    def long_straddle(self):

    def short_straddle(self):

    def strip(self):

    def strap(self):

    def long_strangle(self):

    def short_strangle(self):
