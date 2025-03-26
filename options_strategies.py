# ------------------Import Libraries -------------#
import datetime
import math
import warnings
import numpy as np
import pandas as pd
from arch import arch_model
from extract_options_data import ExtractOptionsData
from extract_options_chain import ExtractOptionsChain
from extract_rf import extract_risk_free_rate
from greeks import Extract_Greeks
from estimate_volatility import EstimateVolatility
from scipy.optimize import minimize, brentq
import scipy.optimize as opt
from scipy.stats import norm
import scipy.stats as si
from scipy.interpolate import CubicSpline

warnings.filterwarnings('ignore')

# ---------------------CONSTANTS------------------#
DEFAULT_TRADING_PERIODS = 252  # Typical number of trading days in a year
DEFAULT_WINDOW = 30  # Default rolling window size for volatility calculations

# --------------------MAIN CODE-------------------#
class Strategies:
    def __init__(self, expiry, ce_chain, pe_chain, rf, q=0, current_date=None):
        self.ce_chain = ce_chain
        self.pe_chain = pe_chain
        self.rf = rf
        self.expiry = expiry
        self.q = q

        if current_date is None:
            current_date = datetime.date.today()

        # Calculate time to maturity:
        trading_days = np.busday_count(current_date, expiry)
        self.t1 = trading_days / DEFAULT_TRADING_PERIODS
        calendar_days = np.busday_count(current_date, expiry, weekmask='1111111')
        self.t2 = calendar_days / 365

    # === Helper Methods (common functionality) ===

    def _compute_S_array(self, S, imp_vol, num_points=100):
        """Compute an array of underlying prices based on volatility."""
        try:
            start = S - (imp_vol * S)
            stop = S + (imp_vol * S)
            return np.linspace(start, stop, num_points)
        except Exception as e:
            print(f"Error computing S_array: {e}")
            return None
    def _build_payoff_dataframe(self, payoff_series, delta_series, gamma_series, vega_series,
                                  zak_lower_series, zak_upper_series):
        """Build a DataFrame from the computed series."""
        try:
            payoff_df = pd.DataFrame({
                'payoff': payoff_series,
                'delta': delta_series,
                'gamma': gamma_series,
                'vega': vega_series,
                'zak_lower_band': zak_lower_series,
                'zak_upper_band': zak_upper_series
            })
            return {'payoffs': payoff_df}
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            return None
    def _compute_base_greeks(self, K, imp_vol, option_category, S):
        """Instantiate an Extract_Greeks object and compute the base Greeks."""
        try:
            eg = Extract_Greeks(K=K, imp_vol=imp_vol, rf=self.rf, maturity=self.expiry,
                                option_category=option_category, q=self.q, current_date=None)
            greeks = {
                'delta': eg.delta(S=S),
                'gamma': eg.gamma(S=S),
                'vega': eg.vega(S=S),
                'rho': eg.rho(S=S),
                'theta': eg.theta(S=S)
            }
            return eg, greeks
        except Exception as e:
            print(f"Error computing base Greeks: {e}")
            return None, None
    def _compute_spread(self, chain):
        """Compute the relative spread from an option chain."""
        try:
            return (chain['ask_price'].values[0] - chain['bid_price'].values[0]) / chain['mkt_price'].values[0]
        except Exception as e:
            print(f"Error computing spread: {e}")
            return None
    def _compute_zakamouline_band(self, eg, S, spread, option_gamma):
        """Compute the Zakamouline delta band given an Extract_Greeks object."""
        try:
            return eg.zakamouline_delta_band(S=S, spread=spread, option_gamma=option_gamma)
        except Exception as e:
            print(f"Error computing Zakamouline delta band: {e}")
            return None, None

    # === Strategy Methods ===

    def long_call(self):
        """
        Compute the payoff and Greeks for a long call option strategy.
        """
        # 1. Filter ITM calls.
        try:
            itm_calls = self.ce_chain[self.ce_chain['ltp'] - self.ce_chain['strikePrice'] > 0]
            if itm_calls.empty:
                raise ValueError("No in-the-money call options found in ce_chain.")
        except Exception as e:
            print(f"Error filtering ITM calls: {e}")
            return None

        # 2. Extract parameters and compute implied volatility.
        try:
            S = self.ce_chain['ltp'].values[0]
            K = itm_calls['strikePrice'].values[0]
            mkt_price = itm_calls['ask_price'].values[0]
            imp_vol = EstimateVolatility.bsm_implied_volatility(
                mkt_price=itm_calls['mkt_price'].values[0],
                S=S, K=K, rf=self.rf, maturity=self.expiry,
                option_category='CE', q=self.q, current_date=None)
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        S_array = self._compute_S_array(S, imp_vol)
        if S_array is None:
            return None

        eg, base_greeks = self._compute_base_greeks(K, imp_vol, 'CE', S)
        if eg is None or base_greeks is None:
            return None

        spread = self._compute_spread(itm_calls)
        if spread is None:
            return None

        lb, ub = self._compute_zakamouline_band(eg, S, spread, base_greeks['gamma'])
        if lb is None or ub is None:
            return None
        base_greeks['zak_lower_band'] = lb
        base_greeks['zak_upper_band'] = ub

        # 6. Compute series data.
        payoff_series, delta_series, gamma_series, vega_series = [], [], [], []
        zak_lower_series, zak_upper_series = [], []
        for s in S_array:
            try:
                payoff_series.append(max(s - K - mkt_price, -mkt_price))
                delta_series.append(eg.delta(S=s))
                gamma_val = eg.gamma(S=s)
                gamma_series.append(gamma_val)
                vega_series.append(eg.vega(S=s))
                lb_s, ub_s = self._compute_zakamouline_band(eg, s, spread, gamma_val)
                zak_lower_series.append(lb_s)
                zak_upper_series.append(ub_s)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue

        payoff_json = self._build_payoff_dataframe(payoff_series, delta_series, gamma_series,
                                                   vega_series, zak_lower_series, zak_upper_series)
        if payoff_json is None:
            return None
        return {"greeks": base_greeks, "payoff": payoff_json}

    def short_call(self):
        """
        Compute the payoff and Greeks for a short call option strategy.
        """
        try:
            itm_calls = self.ce_chain[self.ce_chain['ltp'] - self.ce_chain['strikePrice'] > 0]
            if itm_calls.empty:
                raise ValueError("No in-the-money call options found in ce_chain.")
        except Exception as e:
            print(f"Error filtering ITM calls: {e}")
            return None

        try:
            S = self.ce_chain['ltp'].values[0]
            K = itm_calls['strikePrice'].values[0]
            mkt_price = itm_calls['bid_price'].values[0]
            imp_vol = EstimateVolatility.bsm_implied_volatility(
                mkt_price=itm_calls['mkt_price'].values[0],
                S=S, K=K, rf=self.rf, maturity=self.expiry,
                option_category='CE', q=self.q, current_date=None)
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        S_array = self._compute_S_array(S, imp_vol)
        if S_array is None:
            return None

        eg, base_greeks = self._compute_base_greeks(K, imp_vol, 'CE', S)
        if eg is None or base_greeks is None:
            return None
        for key in base_greeks:
            base_greeks[key] = -base_greeks[key]

        spread = self._compute_spread(itm_calls)
        if spread is None:
            return None

        lb, ub = self._compute_zakamouline_band(eg, S, spread, -base_greeks['gamma'])
        if lb is None or ub is None:
            return None
        base_greeks['zak_lower_band'] = -ub
        base_greeks['zak_upper_band'] = -lb

        payoff_series, delta_series, gamma_series, vega_series = [], [], [], []
        zak_lower_series, zak_upper_series = [], []
        for s in S_array:
            try:
                payoff_series.append(-max(s - K - mkt_price, -mkt_price))
                delta_series.append(-eg.delta(S=s))
                gamma_val = -eg.gamma(S=s)
                gamma_series.append(gamma_val)
                vega_series.append(-eg.vega(S=s))
                lb_s, ub_s = self._compute_zakamouline_band(eg, s, spread, gamma_val)
                zak_lower_series.append(-ub_s)
                zak_upper_series.append(-lb_s)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue

        payoff_json = self._build_payoff_dataframe(payoff_series, delta_series, gamma_series,
                                                   vega_series, zak_lower_series, zak_upper_series)
        if payoff_json is None:
            return None
        return {"greeks": base_greeks, "payoff": payoff_json}

    def long_put(self):
        """
        Compute the payoff and Greeks for a long put option strategy.
        """
        try:
            itm_puts = self.pe_chain[self.pe_chain['strikePrice'] - self.pe_chain['ltp'] > 0]
            if itm_puts.empty:
                raise ValueError("No in-the-money put options found in pe_chain.")
        except Exception as e:
            print(f"Error filtering ITM puts: {e}")
            return None

        try:
            S = self.pe_chain['ltp'].values[0]
            K = itm_puts['strikePrice'].values[0]
            mkt_price = itm_puts['bid_price'].values[0]
            imp_vol = EstimateVolatility.bsm_implied_volatility(
                mkt_price=itm_puts['mkt_price'].values[0],
                S=S, K=K, rf=self.rf, maturity=self.expiry,
                option_category='PE', q=self.q, current_date=None)
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        S_array = self._compute_S_array(S, imp_vol)
        if S_array is None:
            return None

        eg, base_greeks = self._compute_base_greeks(K, imp_vol, 'PE', S)
        if eg is None or base_greeks is None:
            return None

        spread = self._compute_spread(itm_puts)
        if spread is None:
            return None

        lb, ub = self._compute_zakamouline_band(eg, S, spread, base_greeks['gamma'])
        if lb is None or ub is None:
            return None
        base_greeks['zak_lower_band'] = lb
        base_greeks['zak_upper_band'] = ub

        payoff_series, delta_series, gamma_series, vega_series = [], [], [], []
        zak_lower_series, zak_upper_series = [], []
        for s in S_array:
            try:
                payoff_series.append(max(K - S - mkt_price, -mkt_price))
                delta_series.append(eg.delta(S=s))
                gamma_val = eg.gamma(S=s)
                gamma_series.append(gamma_val)
                vega_series.append(eg.vega(S=s))
                lb_s, ub_s = self._compute_zakamouline_band(eg, s, spread, gamma_val)
                # For a long put, the Zakamouline bands are switched.
                zak_lower_series.append(ub_s)
                zak_upper_series.append(lb_s)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue

        payoff_json = self._build_payoff_dataframe(payoff_series, delta_series, gamma_series,
                                                   vega_series, zak_lower_series, zak_upper_series)
        if payoff_json is None:
            return None
        return {"greeks": base_greeks, "payoff": payoff_json}

    def short_put(self):
        """
        Compute the payoff and Greeks for a short put option strategy.
        """
        try:
            itm_puts = self.pe_chain[self.pe_chain['strikePrice'] - self.pe_chain['ltp'] > 0]
            if itm_puts.empty:
                raise ValueError("No in-the-money put options found in pe_chain.")
        except Exception as e:
            print(f"Error filtering ITM puts: {e}")
            return None

        try:
            S = self.pe_chain['ltp'].values[0]
            K = itm_puts['strikePrice'].values[0]
            mkt_price = itm_puts['bid_price'].values[0]
            imp_vol = EstimateVolatility.bsm_implied_volatility(
                mkt_price=itm_puts['mkt_price'].values[0],
                S=S, K=K, rf=self.rf, maturity=self.expiry,
                option_category='PE', q=self.q, current_date=None)
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        S_array = self._compute_S_array(S, imp_vol)
        if S_array is None:
            return None

        eg, base_greeks = self._compute_base_greeks(K, imp_vol, 'PE', S)
        if eg is None or base_greeks is None:
            return None
        for key in base_greeks:
            base_greeks[key] = -base_greeks[key]

        spread = self._compute_spread(itm_puts)
        if spread is None:
            return None

        lb, ub = self._compute_zakamouline_band(eg, S, spread, -base_greeks['gamma'])
        if lb is None or ub is None:
            return None
        base_greeks['zak_lower_band'] = -ub
        base_greeks['zak_upper_band'] = -lb

        payoff_series, delta_series, gamma_series, vega_series = [], [], [], []
        zak_lower_series, zak_upper_series = [], []
        for s in S_array:
            try:
                payoff_series.append(-max(K - S - mkt_price, -mkt_price))
                delta_series.append(-eg.delta(S=s))
                gamma_val = -eg.gamma(S=s)
                gamma_series.append(gamma_val)
                vega_series.append(-eg.vega(S=s))
                lb_s, ub_s = self._compute_zakamouline_band(eg, s, spread, gamma_val)
                zak_lower_series.append(-lb_s)
                zak_upper_series.append(-ub_s)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue

        payoff_json = self._build_payoff_dataframe(payoff_series, delta_series, gamma_series,
                                                   vega_series, zak_lower_series, zak_upper_series)
        if payoff_json is None:
            return None
        return {"greeks": base_greeks, "payoff": payoff_json}

    def bull_call_spread(self):
        """
        Calculate the Greeks and payoff profile for a bull call spread strategy.
        """
        try:
            itm_calls = self.ce_chain[self.ce_chain['ltp'] - self.ce_chain['strikePrice'] > 0]
            otm_calls = self.ce_chain[self.ce_chain['ltp'] - self.ce_chain['strikePrice'] < 0]
            if itm_calls.empty or otm_calls.empty:
                raise ValueError("No ITM or OTM call options found in ce_chain.")
        except Exception as e:
            print(f"Error filtering ITM and OTM calls: {e}")
            return None

        try:
            S = self.ce_chain['ltp'].values[0]
            K_1 = itm_calls['strikePrice'].values[0]
            K_2 = otm_calls['strikePrice'].values[1]
            long_call_price = itm_calls['ask_price'].values[0]
            short_call_price = otm_calls['bid_price'].values[1]
            price_of_strategy = np.abs(short_call_price - long_call_price)
            # Compute implied volatilities for each leg.
            imp_vol_k1 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=itm_calls['mkt_price'].values[0],
                S=S, K=K_1, rf=self.rf, maturity=self.expiry,
                option_category='CE', q=self.q, current_date=None)
            imp_vol_k2 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=otm_calls['mkt_price'].values[0],
                S=S, K=K_2, rf=self.rf, maturity=self.expiry,
                option_category='CE', q=self.q, current_date=None)
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        S_array = self._compute_S_array(S, imp_vol_k1)
        if S_array is None:
            return None

        try:
            eg_long, _ = self._compute_base_greeks(K_1, imp_vol_k1, 'CE', S)
            eg_short, _ = self._compute_base_greeks(K_2, imp_vol_k2, 'CE', S)
            base_greeks = {}
            base_greeks['delta'] = eg_long.delta(S=S) - eg_short.delta(S=S)
            gamma_long = eg_long.gamma(S=S)
            gamma_short = eg_short.gamma(S=S)
            base_greeks['gamma'] = gamma_long - gamma_short
            base_greeks['vega'] = eg_long.vega(S=S) - eg_short.vega(S=S)
            base_greeks['rho'] = eg_long.rho(S=S) - eg_short.rho(S=S)
            base_greeks['theta'] = eg_long.theta(S=S) - eg_short.theta(S=S)
        except Exception as e:
            print(f"Error computing base Greeks: {e}")
            return None

        spread = self._compute_spread(itm_calls)
        if spread is None:
            return None

        try:
            lb_long, ub_long = self._compute_zakamouline_band(eg_long, S, spread, gamma_long)
            lb_short, ub_short = self._compute_zakamouline_band(eg_short, S, spread, gamma_short)
            base_greeks['zak_lower_band'] = lb_long - ub_short
            base_greeks['zak_upper_band'] = ub_long - lb_short
        except Exception as e:
            print(f"Error computing Zakamouline delta band: {e}")
            return None

        payoff_series, delta_series, gamma_series, vega_series = [], [], [], []
        zak_lower_series, zak_upper_series = [], []
        for s in S_array:
            try:
                if s >= K_2:
                    payoff = K_2 - K_1 - price_of_strategy
                elif K_1 <= s < K_2:
                    payoff = s - K_1 - price_of_strategy
                else:
                    payoff = -price_of_strategy
                payoff_series.append(payoff)
                delta_series.append(eg_long.delta(S=s) - eg_short.delta(S=s))
                gamma_series.append(eg_long.gamma(S=s) - eg_short.gamma(S=s))
                vega_series.append(eg_long.vega(S=s) - eg_short.vega(S=s))
                lb_l, ub_l = self._compute_zakamouline_band(eg_long, s, spread, eg_long.gamma(S=s))
                lb_s, ub_s = self._compute_zakamouline_band(eg_short, s, spread, eg_short.gamma(S=s))
                zak_lower_series.append(lb_l - ub_s)
                zak_upper_series.append(ub_l - lb_s)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue

        payoff_json = self._build_payoff_dataframe(payoff_series, delta_series, gamma_series,
                                                    vega_series, zak_lower_series, zak_upper_series)
        if payoff_json is None:
            return None
        return {"greeks": base_greeks, "payoff": payoff_json}

    def bull_put_spread(self):
        """
        Calculate the Greeks and payoff profile for a bull put spread strategy.
        """
        try:
            # For puts: ITM when strike > underlying, OTM when strike < underlying.
            itm_puts = self.pe_chain[self.pe_chain['strikePrice'] - self.pe_chain['ltp'] > 0]
            otm_puts = self.pe_chain[self.pe_chain['strikePrice'] - self.pe_chain['ltp'] < 0]
            if itm_puts.empty or otm_puts.empty:
                raise ValueError("No ITM or OTM put options found in pe_chain.")
        except Exception as e:
            print(f"Error filtering ITM and OTM puts: {e}")
            return None

        try:
            S = self.pe_chain['ltp'].values[0]
            # For bull put spread: long (buy) the OTM put (lower strike) and short (sell) the ITM put (higher strike).
            K_1 = otm_puts['strikePrice'].values[0]
            K_2 = itm_puts['strikePrice'].values[0]
            long_put_price = otm_puts['ask_price'].values[0]
            short_put_price = itm_puts['bid_price'].values[0]
            price_of_strategy = short_put_price - long_put_price
            imp_vol_k1 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=otm_puts['mkt_price'].values[0],
                S=S, K=K_1, rf=self.rf, maturity=self.expiry,
                option_category='PE', q=self.q, current_date=None)
            imp_vol_k2 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=itm_puts['mkt_price'].values[0],
                S=S, K=K_2, rf=self.rf, maturity=self.expiry,
                option_category='PE', q=self.q, current_date=None)
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        S_array = self._compute_S_array(S, imp_vol_k1)
        if S_array is None:
            return None

        try:
            eg_long, _ = self._compute_base_greeks(K_1, imp_vol_k1, 'PE', S)
            eg_short, _ = self._compute_base_greeks(K_2, imp_vol_k2, 'PE', S)
            base_greeks = {}
            base_greeks['delta'] = eg_long.delta(S=S) - eg_short.delta(S=S)
            gamma_long = eg_long.gamma(S=S)
            gamma_short = eg_short.gamma(S=S)
            base_greeks['gamma'] = gamma_long - gamma_short
            base_greeks['vega'] = eg_long.vega(S=S) - eg_short.vega(S=S)
            base_greeks['rho'] = eg_long.rho(S=S) - eg_short.rho(S=S)
            base_greeks['theta'] = eg_long.theta(S=S) - eg_short.theta(S=S)
        except Exception as e:
            print(f"Error computing base Greeks: {e}")
            return None

        spread = self._compute_spread(itm_puts)
        if spread is None:
            return None

        try:
            lb_long, ub_long = self._compute_zakamouline_band(eg_long, S, spread, gamma_long)
            lb_short, ub_short = self._compute_zakamouline_band(eg_short, S, spread, gamma_short)
            base_greeks['zak_lower_band'] = lb_long - ub_short
            base_greeks['zak_upper_band'] = ub_long - lb_short
        except Exception as e:
            print(f"Error computing Zakamouline delta band: {e}")
            return None

        payoff_series, delta_series, gamma_series, vega_series = [], [], [], []
        zak_lower_series, zak_upper_series = [], []
        for s in S_array:
            try:
                if s <= K_1:
                    payoff = K_1 - K_2 + price_of_strategy
                elif K_1 < s <= K_2:
                    payoff = s - K_2 + price_of_strategy
                else:
                    payoff = price_of_strategy
                payoff_series.append(payoff)
                delta_series.append(eg_long.delta(S=s) - eg_short.delta(S=s))
                gamma_series.append(eg_long.gamma(S=s) - eg_short.gamma(S=s))
                vega_series.append(eg_long.vega(S=s) - eg_short.vega(S=s))
                lb_l, ub_l = self._compute_zakamouline_band(eg_long, s, spread, eg_long.gamma(S=s))
                lb_s, ub_s = self._compute_zakamouline_band(eg_short, s, spread, eg_short.gamma(S=s))
                zak_lower_series.append(lb_l - ub_s)
                zak_upper_series.append(ub_l - lb_s)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue

        payoff_json = self._build_payoff_dataframe(payoff_series, delta_series, gamma_series,
                                                    vega_series, zak_lower_series, zak_upper_series)
        if payoff_json is None:
            return None
        return {"greeks": base_greeks, "payoff": payoff_json}

    def bear_call_spread(self):
        """
        Calculate the Greeks and payoff profile for a bear call spread strategy.
        """
        try:
            itm_calls = self.ce_chain[self.ce_chain['ltp'] - self.ce_chain['strikePrice'] > 0]
            otm_calls = self.ce_chain[self.ce_chain['ltp'] - self.ce_chain['strikePrice'] < 0]
            if itm_calls.empty or otm_calls.empty:
                raise ValueError("No ITM or OTM call options found in ce_chain.")
        except Exception as e:
            print(f"Error filtering ITM and OTM calls: {e}")
            return None

        try:
            S = self.ce_chain['ltp'].values[0]
            # For bear call spread: sell ITM call and buy OTM call.
            K_1 = itm_calls['strikePrice'].values[0]
            K_2 = otm_calls['strikePrice'].values[0]
            short_call_price = itm_calls['bid_price'].values[0]
            long_call_price = otm_calls['ask_price'].values[0]
            price_of_strategy = short_call_price - long_call_price
            imp_vol_k1 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=itm_calls['mkt_price'].values[0],
                S=S, K=K_1, rf=self.rf, maturity=self.expiry,
                option_category='CE', q=self.q, current_date=None)
            imp_vol_k2 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=otm_calls['mkt_price'].values[0],
                S=S, K=K_2, rf=self.rf, maturity=self.expiry,
                option_category='CE', q=self.q, current_date=None)
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        S_array = self._compute_S_array(S, imp_vol_k1)
        if S_array is None:
            return None

        try:
            eg_short, _ = self._compute_base_greeks(K_1, imp_vol_k1, 'CE', S)  # short call leg
            eg_long, _ = self._compute_base_greeks(K_2, imp_vol_k2, 'CE', S)   # long call leg
            base_greeks = {}
            # For bear call spread, portfolio Greeks = long call (OTM) minus short call (ITM)
            base_greeks['delta'] = eg_long.delta(S=S) - eg_short.delta(S=S)
            gamma_long = eg_long.gamma(S=S)
            gamma_short = eg_short.gamma(S=S)
            base_greeks['gamma'] = gamma_long - gamma_short
            base_greeks['vega'] = eg_long.vega(S=S) - eg_short.vega(S=S)
            base_greeks['rho'] = eg_long.rho(S=S) - eg_short.rho(S=S)
            base_greeks['theta'] = eg_long.theta(S=S) - eg_short.theta(S=S)
        except Exception as e:
            print(f"Error computing base Greeks: {e}")
            return None

        spread = self._compute_spread(itm_calls)
        if spread is None:
            return None

        try:
            lb_long, ub_long = self._compute_zakamouline_band(eg_long, S, spread, gamma_long)
            lb_short, ub_short = self._compute_zakamouline_band(eg_short, S, spread, gamma_short)
            base_greeks['zak_lower_band'] = lb_long - ub_short
            base_greeks['zak_upper_band'] = ub_long - lb_short
        except Exception as e:
            print(f"Error computing Zakamouline delta band: {e}")
            return None

        payoff_series, delta_series, gamma_series, vega_series = [], [], [], []
        zak_lower_series, zak_upper_series = [], []
        for s in S_array:
            try:
                if s >= K_2:
                    payoff = K_1 - K_2 + price_of_strategy
                elif K_1 <= s < K_2:
                    payoff = K_1 - s + price_of_strategy
                else:
                    payoff = price_of_strategy
                payoff_series.append(payoff)
                delta_series.append(eg_long.delta(S=s) - eg_short.delta(S=s))
                gamma_series.append(eg_long.gamma(S=s) - eg_short.gamma(S=s))
                vega_series.append(eg_long.vega(S=s) - eg_short.vega(S=s))
                lb_l, ub_l = self._compute_zakamouline_band(eg_long, s, spread, eg_long.gamma(S=s))
                lb_s, ub_s = self._compute_zakamouline_band(eg_short, s, spread, eg_short.gamma(S=s))
                zak_lower_series.append(lb_l - ub_s)
                zak_upper_series.append(ub_l - lb_s)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue

        payoff_json = self._build_payoff_dataframe(payoff_series, delta_series, gamma_series,
                                                    vega_series, zak_lower_series, zak_upper_series)
        if payoff_json is None:
            return None
        return {"greeks": base_greeks, "payoff": payoff_json}

    def bear_put_spread(self):
        """
        Calculate the Greeks and payoff profile for a bear put spread strategy.
        """
        try:
            # For puts: ITM when strike > underlying; OTM when strike < underlying.
            itm_puts = self.pe_chain[self.pe_chain['strikePrice'] - self.pe_chain['ltp'] > 0]
            otm_puts = self.pe_chain[self.pe_chain['strikePrice'] - self.pe_chain['ltp'] < 0]
            if itm_puts.empty or otm_puts.empty:
                raise ValueError("No ITM or OTM put options found in pe_chain.")
        except Exception as e:
            print(f"Error filtering ITM and OTM puts: {e}")
            return None

        try:
            S = self.pe_chain['ltp'].values[0]
            # For bear put spread: long (buy) the ITM put (higher strike) and short (sell) the OTM put (lower strike).
            K_1 = otm_puts['strikePrice'].values[0]
            K_2 = itm_puts['strikePrice'].values[0]
            long_put_price = itm_puts['ask_price'].values[0]
            short_put_price = otm_puts['bid_price'].values[0]
            price_of_strategy = long_put_price - short_put_price
            imp_vol_k1 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=otm_puts['mkt_price'].values[0],
                S=S, K=K_1, rf=self.rf, maturity=self.expiry,
                option_category='PE', q=self.q, current_date=None)
            imp_vol_k2 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=itm_puts['mkt_price'].values[0],
                S=S, K=K_2, rf=self.rf, maturity=self.expiry,
                option_category='PE', q=self.q, current_date=None)
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        S_array = self._compute_S_array(S, imp_vol_k1)
        if S_array is None:
            return None

        try:
            eg_long, _ = self._compute_base_greeks(K_2, imp_vol_k2, 'PE', S)  # long put (ITM)
            eg_short, _ = self._compute_base_greeks(K_1, imp_vol_k1, 'PE', S)  # short put (OTM)
            base_greeks = {}
            base_greeks['delta'] = eg_long.delta(S=S) - eg_short.delta(S=S)
            gamma_long = eg_long.gamma(S=S)
            gamma_short = eg_short.gamma(S=S)
            base_greeks['gamma'] = gamma_long - gamma_short
            base_greeks['vega'] = eg_long.vega(S=S) - eg_short.vega(S=S)
            base_greeks['rho'] = eg_long.rho(S=S) - eg_short.rho(S=S)
            base_greeks['theta'] = eg_long.theta(S=S) - eg_short.theta(S=S)
        except Exception as e:
            print(f"Error computing base Greeks: {e}")
            return None

        spread = self._compute_spread(itm_puts)
        if spread is None:
            return None

        try:
            lb_long, ub_long = self._compute_zakamouline_band(eg_long, S, spread, gamma_long)
            lb_short, ub_short = self._compute_zakamouline_band(eg_short, S, spread, gamma_short)
            base_greeks['zak_lower_band'] = lb_long - lb_short
            base_greeks['zak_upper_band'] = ub_long - ub_short
        except Exception as e:
            print(f"Error computing Zakamouline delta band: {e}")
            return None

        payoff_series, delta_series, gamma_series, vega_series = [], [], [], []
        zak_lower_series, zak_upper_series = [], []
        for s in S_array:
            try:
                if s >= K_2:
                    payoff = -price_of_strategy
                elif K_1 < s < K_2:
                    payoff = (K_2 - s) - price_of_strategy
                else:
                    payoff = (K_2 - K_1) - price_of_strategy
                payoff_series.append(payoff)
                delta_series.append(eg_long.delta(S=s) - eg_short.delta(S=s))
                gamma_series.append(eg_long.gamma(S=s) - eg_short.gamma(S=s))
                vega_series.append(eg_long.vega(S=s) - eg_short.vega(S=s))
                lb_l, ub_l = self._compute_zakamouline_band(eg_long, s, spread, eg_long.gamma(S=s))
                lb_s, ub_s = self._compute_zakamouline_band(eg_short, s, spread, eg_short.gamma(S=s))
                zak_lower_series.append(lb_l - lb_s)
                zak_upper_series.append(ub_l - ub_s)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue

        payoff_json = self._build_payoff_dataframe(payoff_series, delta_series, gamma_series,
                                                    vega_series, zak_lower_series, zak_upper_series)
        if payoff_json is None:
            return None
        return {"greeks": base_greeks, "payoff": payoff_json}

    def long_call_butterfly(self):
        """
        Calculate the Greeks and payoff profile for a long call butterfly strategy.
        """
        try:
            itm_calls = self.ce_chain[self.ce_chain['ltp'] - self.ce_chain['strikePrice'] > 0]
            otm_calls = self.ce_chain[self.ce_chain['ltp'] - self.ce_chain['strikePrice'] < 0]
            if itm_calls.empty or otm_calls.empty:
                raise ValueError("No ITM or OTM call options found in ce_chain.")
        except Exception as e:
            print(f"Error filtering ITM and OTM calls: {e}")
            return None

        try:
            S = self.ce_chain['ltp'].values[0]
            # For butterfly: use second available ITM strike for K1, first row for K2, second available OTM strike for K3.
            K_1 = itm_calls['strikePrice'].values[1]
            K_2 = self.ce_chain['strikePrice'].values[0]
            K_3 = otm_calls['strikePrice'].values[1]
            long_call_price = itm_calls['ask_price'].values[1] + otm_calls['ask_price'].values[1]
            short_call_price = 2 * self.ce_chain['ask_price'].values[0]
            price_of_strategy = short_call_price - long_call_price
            imp_vol_k1 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=itm_calls['mkt_price'].values[1],
                S=S, K=K_1, rf=self.rf, maturity=self.expiry,
                option_category='CE', q=self.q, current_date=None)
            imp_vol_k2 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=self.ce_chain['mkt_price'].values[0],
                S=S, K=K_2, rf=self.rf, maturity=self.expiry,
                option_category='CE', q=self.q, current_date=None)
            imp_vol_k3 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=otm_calls['mkt_price'].values[1],
                S=S, K=K_3, rf=self.rf, maturity=self.expiry,
                option_category='CE', q=self.q, current_date=None)
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        S_array = self._compute_S_array(S, imp_vol_k1)
        if S_array is None:
            return None

        try:
            eg_k1, _ = self._compute_base_greeks(K_1, imp_vol_k1, 'CE', S)
            eg_short, _ = self._compute_base_greeks(K_2, imp_vol_k2, 'CE', S)
            eg_k3, _ = self._compute_base_greeks(K_3, imp_vol_k3, 'CE', S)
            base_greeks = {}
            base_greeks['delta'] = eg_k1.delta(S=S) + eg_k3.delta(S=S) - 2 * eg_short.delta(S=S)
            gamma_k1 = eg_k1.gamma(S=S)
            gamma_k3 = eg_k3.gamma(S=S)
            gamma_short = eg_short.gamma(S=S)
            base_greeks['gamma'] = gamma_k1 + gamma_k3 - 2 * gamma_short
            base_greeks['vega'] = eg_k1.vega(S=S) + eg_k3.vega(S=S) - 2 * eg_short.vega(S=S)
            base_greeks['rho'] = eg_k1.rho(S=S) + eg_k3.rho(S=S) - 2 * eg_short.rho(S=S)
            base_greeks['theta'] = eg_k1.theta(S=S) + eg_k3.theta(S=S) - 2 * eg_short.theta(S=S)
        except Exception as e:
            print(f"Error computing base Greeks: {e}")
            return None

        spread = self._compute_spread(itm_calls)
        if spread is None:
            return None

        try:
            lb_k1, ub_k1 = self._compute_zakamouline_band(eg_k1, S, spread, eg_k1.gamma(S=S))
            lb_k3, ub_k3 = self._compute_zakamouline_band(eg_k3, S, spread, eg_k3.gamma(S=S))
            lb_short, ub_short = self._compute_zakamouline_band(eg_short, S, spread, eg_short.gamma(S=S))
            base_greeks['zak_lower_band'] = (lb_k1 + lb_k3) / 2 - ub_short
            base_greeks['zak_upper_band'] = (ub_k1 + ub_k3) / 2 - lb_short
        except Exception as e:
            print(f"Error computing Zakamouline delta band: {e}")
            return None

        payoff_series, delta_series, gamma_series, vega_series = [], [], [], []
        zak_lower_series, zak_upper_series = [], []
        for s in S_array:
            try:
                if s < K_1:
                    payoff = price_of_strategy
                elif K_1 <= s <= K_2:
                    payoff = s - K_1 + price_of_strategy
                elif K_2 < s <= K_3:
                    payoff = 2 * K_2 - s - K_1 + price_of_strategy
                else:
                    payoff = 2 * K_2 - K_1 - K_3 + price_of_strategy
                payoff_series.append(payoff)
                delta_val = eg_k1.delta(S=s) + eg_k3.delta(S=s) - 2 * eg_short.delta(S=s)
                delta_series.append(delta_val)
                gamma_val = eg_k1.gamma(S=s) + eg_k3.gamma(S=s) - 2 * eg_short.gamma(S=s)
                gamma_series.append(gamma_val)
                vega_val = eg_k1.vega(S=s) + eg_k3.vega(S=s) - 2 * eg_short.vega(S=s)
                vega_series.append(vega_val)
                lb1, ub1 = self._compute_zakamouline_band(eg_k1, s, spread, eg_k1.gamma(S=s))
                lb3, ub3 = self._compute_zakamouline_band(eg_k3, s, spread, eg_k3.gamma(S=s))
                lb_short_s, ub_short_s = self._compute_zakamouline_band(eg_short, s, spread, eg_short.gamma(S=s))
                zak_lower_series.append((lb1 + lb3) / 2 - ub_short_s)
                zak_upper_series.append((ub1 + ub3) / 2 - lb_short_s)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue

        payoff_json = self._build_payoff_dataframe(payoff_series, delta_series, gamma_series,
                                                    vega_series, zak_lower_series, zak_upper_series)
        if payoff_json is None:
            return None
        return {"greeks": base_greeks, "payoff": payoff_json}

    def long_put_butterfly(self):
        """
        Calculate the Greeks and payoff profile for a long put butterfly strategy.
        """
        try:
            itm_puts = self.pe_chain[self.pe_chain['strikePrice'] - self.pe_chain['ltp'] > 0]
            otm_puts = self.pe_chain[self.pe_chain['strikePrice'] - self.pe_chain['ltp'] < 0]
            if itm_puts.empty or otm_puts.empty:
                raise ValueError("No ITM or OTM put options found in pe_chain.")
        except Exception as e:
            print(f"Error filtering ITM and OTM puts: {e}")
            return None

        try:
            S = self.pe_chain['ltp'].values[0]
            # For butterfly: choose K1 from otm_puts, K2 from full chain, K3 from itm_puts.
            K_1 = otm_puts['strikePrice'].values[1]
            K_2 = self.pe_chain['strikePrice'].values[0]
            K_3 = itm_puts['strikePrice'].values[1]
            long_put_price = otm_puts['ask_price'].values[1] + itm_puts['ask_price'].values[1]
            short_put_price = 2 * self.pe_chain['bid_price'].values[0]
            price_of_strategy = short_put_price - long_put_price
            imp_vol_k1 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=otm_puts['mkt_price'].values[1],
                S=S, K=K_1, rf=self.rf, maturity=self.expiry,
                option_category='PE', q=self.q, current_date=None)
            imp_vol_k2 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=self.pe_chain['mkt_price'].values[0],
                S=S, K=K_2, rf=self.rf, maturity=self.expiry,
                option_category='PE', q=self.q, current_date=None)
            imp_vol_k3 = EstimateVolatility.bsm_implied_volatility(
                mkt_price=itm_puts['mkt_price'].values[1],
                S=S, K=K_3, rf=self.rf, maturity=self.expiry,
                option_category='PE', q=self.q, current_date=None)
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        S_array = self._compute_S_array(S, imp_vol_k1)
        if S_array is None:
            return None

        try:
            eg_k1, _ = self._compute_base_greeks(K_1, imp_vol_k1, 'PE', S)
            eg_short, _ = self._compute_base_greeks(K_2, imp_vol_k2, 'PE', S)
            eg_k3, _ = self._compute_base_greeks(K_3, imp_vol_k3, 'PE', S)
            base_greeks = {}
            base_greeks['delta'] = eg_k1.delta(S=S) + eg_k3.delta(S=S) - 2 * eg_short.delta(S=S)
            gamma_k1 = eg_k1.gamma(S=S)
            gamma_k3 = eg_k3.gamma(S=S)
            gamma_short = eg_short.gamma(S=S)
            base_greeks['gamma'] = gamma_k1 + gamma_k3 - 2 * gamma_short
            base_greeks['vega'] = eg_k1.vega(S=S) + eg_k3.vega(S=S) - 2 * eg_short.vega(S=S)
            base_greeks['rho'] = eg_k1.rho(S=S) + eg_k3.rho(S=S) - 2 * eg_short.rho(S=S)
            base_greeks['theta'] = eg_k1.theta(S=S) + eg_k3.theta(S=S) - 2 * eg_short.theta(S=S)
        except Exception as e:
            print(f"Error computing base Greeks: {e}")
            return None

        spread = self._compute_spread(itm_puts)
        if spread is None:
            return None

        try:
            lb_k1, ub_k1 = self._compute_zakamouline_band(eg_k1, S, spread, eg_k1.gamma(S=S))
            lb_k3, ub_k3 = self._compute_zakamouline_band(eg_k3, S, spread, eg_k3.gamma(S=S))
            lb_short, ub_short = self._compute_zakamouline_band(eg_short, S, spread, eg_short.gamma(S=S))
            base_greeks['zak_lower_band'] = lb_k1 - lb_short
            base_greeks['zak_upper_band'] = ub_k1 - ub_short
        except Exception as e:
            print(f"Error computing Zakamouline delta band: {e}")
            return None

        payoff_series, delta_series, gamma_series, vega_series = [], [], [], []
        zak_lower_series, zak_upper_series = [], []
        for s in S_array:
            try:
                if s <= K_1:
                    payoff = K_1 - K_2 + price_of_strategy
                elif K_1 < s <= K_2:
                    payoff = s - 2 * K_2 + K_3 + price_of_strategy
                elif K_2 < s <= K_3:
                    payoff = K_3 - s + price_of_strategy
                else:
                    payoff = price_of_strategy
                payoff_series.append(payoff)
                delta_val = eg_k1.delta(S=s) + eg_k3.delta(S=s) - 2 * eg_short.delta(S=s)
                delta_series.append(delta_val)
                gamma_val = eg_k1.gamma(S=s) + eg_k3.gamma(S=s) - 2 * eg_short.gamma(S=s)
                gamma_series.append(gamma_val)
                vega_val = eg_k1.vega(S=s) + eg_k3.vega(S=s) - 2 * eg_short.vega(S=s)
                vega_series.append(vega_val)
                lb1, ub1 = self._compute_zakamouline_band(eg_k1, s, spread, eg_k1.gamma(S=s))
                lb3, ub3 = self._compute_zakamouline_band(eg_k3, s, spread, eg_k3.gamma(S=s))
                lb_short_s, ub_short_s = self._compute_zakamouline_band(eg_short, s, spread, eg_short.gamma(S=s))
                zak_lower_series.append((lb1 + lb3) / 2 - ub_short_s)
                zak_upper_series.append((ub1 + ub3) / 2 - lb_short_s)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue

        payoff_json = self._build_payoff_dataframe(payoff_series, delta_series, gamma_series,
                                                    vega_series, zak_lower_series, zak_upper_series)
        if payoff_json is None:
            return None
        return {"greeks": base_greeks, "payoff": payoff_json}

    def long_straddle(self):
        """
        Calculate the Greeks and payoff profile for a long straddle strategy.
        """
        try:
            S = self.ce_chain['ltp'].values[0]
            K = self.ce_chain['strikePrice'].values[0]
            long_call_price = self.ce_chain['ask_price'].values[0]
            long_put_price = self.pe_chain['ask_price'].values[0]
            price_of_strategy = long_call_price + long_put_price
            imp_vol_call = EstimateVolatility.bsm_implied_volatility(
                mkt_price=self.ce_chain['mkt_price'].values[0],
                S=S, K=K, rf=self.rf, maturity=self.expiry,
                option_category='CE', q=self.q, current_date=None)
            imp_vol_put = EstimateVolatility.bsm_implied_volatility(
                mkt_price=self.pe_chain['mkt_price'].values[0],
                S=S, K=K, rf=self.rf, maturity=self.expiry,
                option_category='PE', q=self.q, current_date=None)
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        S_array = self._compute_S_array(S, imp_vol_call)
        if S_array is None:
            return None

        try:
            eg_call, _ = self._compute_base_greeks(K, imp_vol_call, 'CE', S)
            eg_put, _ = self._compute_base_greeks(K, imp_vol_put, 'PE', S)
            base_greeks = {}
            base_greeks['delta'] = eg_call.delta(S=S) + eg_put.delta(S=S)
            base_greeks['gamma'] = eg_call.gamma(S=S) + eg_put.gamma(S=S)
            base_greeks['vega'] = eg_call.vega(S=S) + eg_put.vega(S=S)
            base_greeks['rho'] = eg_call.rho(S=S) + eg_put.rho(S=S)
            base_greeks['theta'] = eg_call.theta(S=S) + eg_put.theta(S=S)
        except Exception as e:
            print(f"Error computing base Greeks: {e}")
            return None

        spread = self._compute_spread(self.ce_chain)
        if spread is None:
            return None

        try:
            lb_call, ub_call = self._compute_zakamouline_band(eg_call, S, spread, eg_call.gamma(S=S))
            lb_put, ub_put = self._compute_zakamouline_band(eg_put, S, spread, eg_put.gamma(S=S))
            base_greeks['zak_lower_band'] = lb_call + lb_put
            base_greeks['zak_upper_band'] = ub_call + ub_put
        except Exception as e:
            print(f"Error computing Zakamouline delta band: {e}")
            return None

        payoff_series, delta_series, gamma_series, vega_series = [], [], [], []
        zak_lower_series, zak_upper_series = [], []
        for s in S_array:
            try:
                if s <= K:
                    payoff = (K - s) - price_of_strategy
                else:
                    payoff = (s - K) - price_of_strategy
                payoff_series.append(payoff)
                delta_series.append(eg_call.delta(S=s) + eg_put.delta(S=s))
                gamma_series.append(eg_call.gamma(S=s) + eg_put.gamma(S=s))
                vega_series.append(eg_call.vega(S=s) + eg_put.vega(S=s))
                lb_c, ub_c = self._compute_zakamouline_band(eg_call, s, spread, eg_call.gamma(S=s))
                lb_p, ub_p = self._compute_zakamouline_band(eg_put, s, spread, eg_put.gamma(S=s))
                zak_lower_series.append(lb_c + lb_p)
                zak_upper_series.append(ub_c + ub_p)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue

        payoff_json = self._build_payoff_dataframe(payoff_series, delta_series, gamma_series,
                                                    vega_series, zak_lower_series, zak_upper_series)
        if payoff_json is None:
            return None
        return {"greeks": base_greeks, "payoff": payoff_json}

    def short_straddle(self):
        """
        Calculate the Greeks and payoff profile for a short straddle strategy.
        """
        try:
            S = self.ce_chain['ltp'].values[0]
            K = self.ce_chain['strikePrice'].values[0]
            short_call_price = self.ce_chain['bid_price'].values[0]
            short_put_price = self.pe_chain['bid_price'].values[0]
            price_of_strategy = short_call_price + short_put_price
            imp_vol_call = EstimateVolatility.bsm_implied_volatility(
                mkt_price=self.ce_chain['mkt_price'].values[0],
                S=S, K=K, rf=self.rf, maturity=self.expiry,
                option_category='CE', q=self.q, current_date=None)
            imp_vol_put = EstimateVolatility.bsm_implied_volatility(
                mkt_price=self.pe_chain['mkt_price'].values[0],
                S=S, K=K, rf=self.rf, maturity=self.expiry,
                option_category='PE', q=self.q, current_date=None)
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        S_array = self._compute_S_array(S, imp_vol_call)
        if S_array is None:
            return None

        try:
            eg_call, _ = self._compute_base_greeks(K, imp_vol_call, 'CE', S)
            eg_put, _ = self._compute_base_greeks(K, imp_vol_put, 'PE', S)
            base_greeks = {}
            base_greeks['delta'] = - (eg_call.delta(S=S) + eg_put.delta(S=S))
            base_greeks['gamma'] = - (eg_call.gamma(S=S) + eg_put.gamma(S=S))
            base_greeks['vega'] = - (eg_call.vega(S=S) + eg_put.vega(S=S))
            base_greeks['rho'] = - (eg_call.rho(S=S) + eg_put.rho(S=S))
            base_greeks['theta'] = - (eg_call.theta(S=S) + eg_put.theta(S=S))
        except Exception as e:
            print(f"Error computing base Greeks: {e}")
            return None

        spread = self._compute_spread(self.ce_chain)
        if spread is None:
            return None

        try:
            lb_call, ub_call = self._compute_zakamouline_band(eg_call, S, spread, eg_call.gamma(S=S))
            lb_put, ub_put = self._compute_zakamouline_band(eg_put, S, spread, eg_put.gamma(S=S))
            base_greeks['zak_lower_band'] = - (lb_call + lb_put)
            base_greeks['zak_upper_band'] = - (ub_call + ub_put)
        except Exception as e:
            print(f"Error computing Zakamouline delta band: {e}")
            return None

        payoff_series, delta_series, gamma_series, vega_series = [], [], [], []
        zak_lower_series, zak_upper_series = [], []
        for s in S_array:
            try:
                payoff_series.append(price_of_strategy - abs(s - K))
                delta_series.append(- (eg_call.delta(S=s) + eg_put.delta(S=s)))
                gamma_series.append(- (eg_call.gamma(S=s) + eg_put.gamma(S=s)))
                vega_series.append(- (eg_call.vega(S=s) + eg_put.vega(S=s)))
                lb_c, ub_c = self._compute_zakamouline_band(eg_call, s, spread, eg_call.gamma(S=s))
                lb_p, ub_p = self._compute_zakamouline_band(eg_put, s, spread, eg_put.gamma(S=s))
                zak_lower_series.append(- (lb_c + lb_p))
                zak_upper_series.append(- (ub_c + ub_p))
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue

        payoff_json = self._build_payoff_dataframe(payoff_series, delta_series, gamma_series,
                                                    vega_series, zak_lower_series, zak_upper_series)
        if payoff_json is None:
            return None
        return {"greeks": base_greeks, "payoff": payoff_json}

    def strip(self):
        """
        Calculate the Greeks and payoff profile for a strip strategy.
        (Buy one call and two puts)
        """
        try:
            S = self.ce_chain['ltp'].values[0]
            K = self.ce_chain['strikePrice'].values[0]
            long_call_price = self.ce_chain['ask_price'].values[0]
            long_put_price = 2 * self.pe_chain['ask_price'].values[0]
            price_of_strategy = long_call_price + long_put_price
            imp_vol_call = EstimateVolatility.bsm_implied_volatility(
                mkt_price=self.ce_chain['mkt_price'].values[0],
                S=S, K=K, rf=self.rf, maturity=self.expiry,
                option_category='CE', q=self.q, current_date=None)
            imp_vol_put = EstimateVolatility.bsm_implied_volatility(
                mkt_price=self.pe_chain['mkt_price'].values[0],
                S=S, K=K, rf=self.rf, maturity=self.expiry,
                option_category='PE', q=self.q, current_date=None)
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        S_array = self._compute_S_array(S, imp_vol_call)
        if S_array is None:
            return None

        try:
            eg_call, _ = self._compute_base_greeks(K, imp_vol_call, 'CE', S)
            eg_put, _ = self._compute_base_greeks(K, imp_vol_put, 'PE', S)
            base_greeks = {}
            base_greeks['delta'] = eg_call.delta(S=S) + 2 * eg_put.delta(S=S)
            base_greeks['gamma'] = eg_call.gamma(S=S) + 2 * eg_put.gamma(S=S)
            base_greeks['vega'] = eg_call.vega(S=S) + 2 * eg_put.vega(S=S)
            base_greeks['rho'] = eg_call.rho(S=S) + 2 * eg_put.rho(S=S)
            base_greeks['theta'] = eg_call.theta(S=S) + 2 * eg_put.theta(S=S)
        except Exception as e:
            print(f"Error computing base Greeks: {e}")
            return None

        spread = self._compute_spread(self.ce_chain)
        if spread is None:
            return None

        try:
            lb_call, ub_call = self._compute_zakamouline_band(eg_call, S, spread, eg_call.gamma(S=S))
            lb_put, ub_put = self._compute_zakamouline_band(eg_put, S, spread, eg_put.gamma(S=S))
            base_greeks['zak_lower_band'] = lb_call + 2 * lb_put
            base_greeks['zak_upper_band'] = ub_call + 2 * ub_put
        except Exception as e:
            print(f"Error computing Zakamouline delta band: {e}")
            return None

        payoff_series, delta_series, gamma_series, vega_series = [], [], [], []
        zak_lower_series, zak_upper_series = [], []
        for s in S_array:
            try:
                if s <= K:
                    payoff = 2 * (K - s) - price_of_strategy
                else:
                    payoff = (s - K) - price_of_strategy
                payoff_series.append(payoff)
                delta_series.append(eg_call.delta(S=s) + 2 * eg_put.delta(S=s))
                gamma_series.append(eg_call.gamma(S=s) + 2 * eg_put.gamma(S=s))
                vega_series.append(eg_call.vega(S=s) + 2 * eg_put.vega(S=s))
                lb_c, ub_c = self._compute_zakamouline_band(eg_call, s, spread, eg_call.gamma(S=s))
                lb_p, ub_p = self._compute_zakamouline_band(eg_put, s, spread, eg_put.gamma(S=s))
                zak_lower_series.append(lb_c + 2 * lb_p)
                zak_upper_series.append(ub_c + 2 * ub_p)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue

        payoff_json = self._build_payoff_dataframe(payoff_series, delta_series, gamma_series,
                                                    vega_series, zak_lower_series, zak_upper_series)
        if payoff_json is None:
            return None
        return {"greeks": base_greeks, "payoff": payoff_json}

    def strap(self):
        """
        Calculate the Greeks and payoff profile for a strap strategy.
        (Buy two calls and one put)
        """
        try:
            S = self.ce_chain['ltp'].values[0]
            K = self.ce_chain['strikePrice'].values[0]
            long_call_price = 2 * self.ce_chain['ask_price'].values[0]
            long_put_price = self.pe_chain['ask_price'].values[0]
            price_of_strategy = long_call_price + long_put_price
            imp_vol_call = EstimateVolatility.bsm_implied_volatility(
                mkt_price=self.ce_chain['mkt_price'].values[0],
                S=S, K=K, rf=self.rf, maturity=self.expiry,
                option_category='CE', q=self.q, current_date=None)
            imp_vol_put = EstimateVolatility.bsm_implied_volatility(
                mkt_price=self.pe_chain['mkt_price'].values[0],
                S=S, K=K, rf=self.rf, maturity=self.expiry,
                option_category='PE', q=self.q, current_date=None)
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        S_array = self._compute_S_array(S, imp_vol_call)
        if S_array is None:
            return None

        try:
            eg_call, _ = self._compute_base_greeks(K, imp_vol_call, 'CE', S)
            eg_put, _ = self._compute_base_greeks(K, imp_vol_put, 'PE', S)
            base_greeks = {}
            base_greeks['delta'] = 2 * eg_call.delta(S=S) + eg_put.delta(S=S)
            base_greeks['gamma'] = 2 * eg_call.gamma(S=S) + eg_put.gamma(S=S)
            base_greeks['vega'] = 2 * eg_call.vega(S=S) + eg_put.vega(S=S)
            base_greeks['rho'] = 2 * eg_call.rho(S=S) + eg_put.rho(S=S)
            base_greeks['theta'] = 2 * eg_call.theta(S=S) + eg_put.theta(S=S)
        except Exception as e:
            print(f"Error computing base Greeks: {e}")
            return None

        spread = self._compute_spread(self.ce_chain)
        if spread is None:
            return None

        try:
            lb_call, ub_call = self._compute_zakamouline_band(eg_call, S, spread, eg_call.gamma(S=S))
            lb_put, ub_put = self._compute_zakamouline_band(eg_put, S, spread, eg_put.gamma(S=S))
            base_greeks['zak_lower_band'] = 2 * lb_call + lb_put
            base_greeks['zak_upper_band'] = 2 * ub_call + ub_put
        except Exception as e:
            print(f"Error computing Zakamouline delta band: {e}")
            return None

        payoff_series, delta_series, gamma_series, vega_series = [], [], [], []
        zak_lower_series, zak_upper_series = [], []
        for s in S_array:
            try:
                if s <= K:
                    payoff = (K - s) - price_of_strategy
                else:
                    payoff = 2 * (s - K) - price_of_strategy
                payoff_series.append(payoff)
                delta_series.append(2 * eg_call.delta(S=s) + eg_put.delta(S=s))
                gamma_series.append(2 * eg_call.gamma(S=s) + eg_put.gamma(S=s))
                vega_series.append(2 * eg_call.vega(S=s) + eg_put.vega(S=s))
                lb_c, ub_c = self._compute_zakamouline_band(eg_call, s, spread, eg_call.gamma(S=s))
                lb_p, ub_p = self._compute_zakamouline_band(eg_put, s, spread, eg_put.gamma(S=s))
                zak_lower_series.append(2 * lb_c + lb_p)
                zak_upper_series.append(2 * ub_c + ub_p)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue

        payoff_json = self._build_payoff_dataframe(payoff_series, delta_series, gamma_series,
                                                    vega_series, zak_lower_series, zak_upper_series)
        if payoff_json is None:
            return None
        return {"greeks": base_greeks, "payoff": payoff_json}

    def long_strangle(self):
        """
        Calculate the Greeks and payoff profile for a long strangle strategy.
        (Buy an OTM put and an OTM call)
        """
        try:
            otm_puts = self.pe_chain[self.pe_chain['strikePrice'] - self.pe_chain['ltp'] < 0]
            otm_calls = self.ce_chain[self.ce_chain['ltp'] - self.ce_chain['strikePrice'] < 0]
            if otm_puts.empty or otm_calls.empty:
                raise ValueError("No OTM options found in chains.")
        except Exception as e:
            print(f"Error filtering OTM options: {e}")
            return None

        try:
            S = self.ce_chain['ltp'].values[0]
            K_1 = otm_puts['strikePrice'].values[1]  # lower strike (put)
            K_2 = otm_calls['strikePrice'].values[1]  # higher strike (call)
            long_put_price = otm_puts['ask_price'].values[1]
            long_call_price = otm_calls['ask_price'].values[1]
            price_of_strategy = long_call_price + long_put_price
            imp_vol_call = EstimateVolatility.bsm_implied_volatility(
                mkt_price=otm_calls['mkt_price'].values[1],
                S=S, K=K_2, rf=self.rf, maturity=self.expiry,
                option_category='CE', q=self.q, current_date=None)
            imp_vol_put = EstimateVolatility.bsm_implied_volatility(
                mkt_price=otm_puts['mkt_price'].values[1],
                S=S, K=K_1, rf=self.rf, maturity=self.expiry,
                option_category='PE', q=self.q, current_date=None)
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        S_array = self._compute_S_array(S, imp_vol_call)
        if S_array is None:
            return None

        try:
            eg_call, _ = self._compute_base_greeks(K_2, imp_vol_call, 'CE', S)
            eg_put, _ = self._compute_base_greeks(K_1, imp_vol_put, 'PE', S)
            base_greeks = {}
            base_greeks['delta'] = eg_call.delta(S=S) + eg_put.delta(S=S)
            base_greeks['gamma'] = eg_call.gamma(S=S) + eg_put.gamma(S=S)
            base_greeks['vega'] = eg_call.vega(S=S) + eg_put.vega(S=S)
            base_greeks['rho'] = eg_call.rho(S=S) + eg_put.rho(S=S)
            base_greeks['theta'] = eg_call.theta(S=S) + eg_put.theta(S=S)
        except Exception as e:
            print(f"Error computing base Greeks: {e}")
            return None

        spread = self._compute_spread(otm_calls)
        if spread is None:
            return None

        try:
            lb_call, ub_call = self._compute_zakamouline_band(eg_call, S, spread, eg_call.gamma(S=S))
            lb_put, ub_put = self._compute_zakamouline_band(eg_put, S, spread, eg_put.gamma(S=S))
            base_greeks['zak_lower_band'] = lb_call + lb_put
            base_greeks['zak_upper_band'] = ub_call + ub_put
        except Exception as e:
            print(f"Error computing Zakamouline delta band: {e}")
            return None

        payoff_series, delta_series, gamma_series, vega_series = [], [], [], []
        zak_lower_series, zak_upper_series = [], []
        for s in S_array:
            try:
                if s <= K_1:
                    payoff = (K_1 - s) - price_of_strategy
                elif s > K_2:
                    payoff = (s - K_2) - price_of_strategy
                else:
                    payoff = -price_of_strategy
                payoff_series.append(payoff)
                delta_series.append(eg_call.delta(S=s) + eg_put.delta(S=s))
                gamma_series.append(eg_call.gamma(S=s) + eg_put.gamma(S=s))
                vega_series.append(eg_call.vega(S=s) + eg_put.vega(S=s))
                lb_c, ub_c = self._compute_zakamouline_band(eg_call, s, spread, eg_call.gamma(S=s))
                lb_p, ub_p = self._compute_zakamouline_band(eg_put, s, spread, eg_put.gamma(S=s))
                zak_lower_series.append(lb_c + lb_p)
                zak_upper_series.append(ub_c + ub_p)
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue

        payoff_json = self._build_payoff_dataframe(payoff_series, delta_series, gamma_series,
                                                    vega_series, zak_lower_series, zak_upper_series)
        if payoff_json is None:
            return None
        return {"greeks": base_greeks, "payoff": payoff_json}

    def short_strangle(self):
        """
        Calculate the Greeks and payoff profile for a short strangle strategy.
        (Sell an OTM put and an OTM call)
        """
        try:
            otm_puts = self.pe_chain[self.pe_chain['strikePrice'] - self.pe_chain['ltp'] < 0]
            otm_calls = self.ce_chain[self.ce_chain['ltp'] - self.ce_chain['strikePrice'] < 0]
            if otm_puts.empty or otm_calls.empty:
                raise ValueError("No OTM options found in chains.")
        except Exception as e:
            print(f"Error filtering OTM options: {e}")
            return None

        try:
            S = self.ce_chain['ltp'].values[0]
            K_1 = otm_puts['strikePrice'].values[1]
            K_2 = otm_calls['strikePrice'].values[1]
            short_put_price = otm_puts['bid_price'].values[1]
            short_call_price = otm_calls['bid_price'].values[1]
            price_of_strategy = short_call_price + short_put_price
            imp_vol_call = EstimateVolatility.bsm_implied_volatility(
                mkt_price=otm_calls['mkt_price'].values[1],
                S=S, K=K_2, rf=self.rf, maturity=self.expiry,
                option_category='CE', q=self.q, current_date=None)
            imp_vol_put = EstimateVolatility.bsm_implied_volatility(
                mkt_price=otm_puts['mkt_price'].values[1],
                S=S, K=K_1, rf=self.rf, maturity=self.expiry,
                option_category='PE', q=self.q, current_date=None)
        except Exception as e:
            print(f"Error extracting price data: {e}")
            return None

        S_array = self._compute_S_array(S, imp_vol_call)
        if S_array is None:
            return None

        try:
            eg_call, _ = self._compute_base_greeks(K_2, imp_vol_call, 'CE', S)
            eg_put, _ = self._compute_base_greeks(K_1, imp_vol_put, 'PE', S)
            base_greeks = {}
            base_greeks['delta'] = - (eg_call.delta(S=S) + eg_put.delta(S=S))
            base_greeks['gamma'] = - (eg_call.gamma(S=S) + eg_put.gamma(S=S))
            base_greeks['vega'] = - (eg_call.vega(S=S) + eg_put.vega(S=S))
            base_greeks['rho'] = - (eg_call.rho(S=S) + eg_put.rho(S=S))
            base_greeks['theta'] = - (eg_call.theta(S=S) + eg_put.theta(S=S))
        except Exception as e:
            print(f"Error computing base Greeks: {e}")
            return None

        spread = self._compute_spread(self.ce_chain)
        if spread is None:
            return None

        try:
            lb_call, ub_call = self._compute_zakamouline_band(eg_call, S, spread, eg_call.gamma(S=S))
            lb_put, ub_put = self._compute_zakamouline_band(eg_put, S, spread, eg_put.gamma(S=S))
            base_greeks['zak_lower_band'] = - (lb_call + lb_put)
            base_greeks['zak_upper_band'] = - (ub_call + ub_put)
        except Exception as e:
            print(f"Error computing Zakamouline delta band: {e}")
            return None

        payoff_series, delta_series, gamma_series, vega_series = [], [], [], []
        zak_lower_series, zak_upper_series = [], []
        for s in S_array:
            try:
                if s <= K_1:
                    payoff = price_of_strategy - (K_1 - s)
                elif s >= K_2:
                    payoff = price_of_strategy - (s - K_2)
                else:
                    payoff = price_of_strategy
                payoff_series.append(payoff)
                delta_series.append(- (eg_call.delta(S=s) + eg_put.delta(S=s)))
                gamma_series.append(- (eg_call.gamma(S=s) + eg_put.gamma(S=s)))
                vega_series.append(- (eg_call.vega(S=s) + eg_put.vega(S=s)))
                lb_c, ub_c = self._compute_zakamouline_band(eg_call, s, spread, eg_call.gamma(S=s))
                lb_p, ub_p = self._compute_zakamouline_band(eg_put, s, spread, eg_put.gamma(S=s))
                zak_lower_series.append(- (lb_c + lb_p))
                zak_upper_series.append(- (ub_c + ub_p))
            except Exception as e:
                print(f"Error computing series values at underlying price {s}: {e}")
                continue

        payoff_json = self._build_payoff_dataframe(payoff_series, delta_series, gamma_series,
                                                    vega_series, zak_lower_series, zak_upper_series)
        if payoff_json is None:
            return None
        return {"greeks": base_greeks, "payoff": payoff_json}

