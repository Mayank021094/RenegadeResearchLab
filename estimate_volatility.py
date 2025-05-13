# ------------------Import Libraries -------------#
from arch import arch_model
import pandas as pd
import numpy as np
import math
from extract_options_data import ExtractOptionsData
from scipy.optimize import minimize, brentq, newton
import scipy.optimize as opt
from scipy.stats import norm
import scipy.stats as si
import warnings
from scipy.interpolate import CubicSpline
from extract_options_chain import ExtractOptionsChain
from extract_options_data import ExtractOptionsData
from extract_rf import extract_risk_free_rate
from typing import Literal

warnings.filterwarnings('ignore')
import datetime

# ---------------------CONSTANTS------------------#
DEFAULT_TRADING_PERIODS = 252  # Typical number of trading days in a year
DEFAULT_WINDOW = 30  # Default rolling window size for volatility calculations


# --------------------MAIN CODE-------------------#

class EstimateVolatility:
    """
    A class to estimate volatility using various methods.
    """

    def __init__(self, ticker, category):
        """
        Initialize the class with the ticker symbol and data category.

        :param ticker: The ticker symbol of the asset.
        :param category: The category of data (e.g., 'stock', 'option').
        """
        self.ticker = ticker
        self.category = category
        extract_data_object = ExtractOptionsData()
        self.daily_data = extract_data_object.extracting_ohlc(ticker=self.ticker, category=self.category, period='1y',
                                                              interval='1d')
        self.cones_data = extract_data_object.extracting_ohlc(ticker=self.ticker, category=self.category, period='5y',
                                                              interval='1d')
        if self.daily_data.empty:
            raise ValueError(f"No data found for ticker: {self.ticker} and category: {self.category}")

    # --------------------------------------Historical Volatility---------------------------------------------

    def close_to_close(self, window=DEFAULT_WINDOW, trading_periods=DEFAULT_TRADING_PERIODS, clean=True):
        """
        Calculate close-to-close volatility.

        :param window: Rolling window size for volatility calculation.
        :param trading_periods: Number of trading periods in a year for annualization.
        :param clean: If True, drop NaN values from the result.
        :return: Annualized close-to-close volatility.
        """
        if self.daily_data['Close'].isnull().any():
            raise ValueError("Close prices contain NaN values.")

        log_return = (self.daily_data['Close'] / self.daily_data['Close'].shift(1)).apply(np.log)
        result = log_return.rolling(window=window, center=False).std() * math.sqrt(trading_periods)

        return result.dropna() if clean else result

    def yang_zhang(self, window=DEFAULT_WINDOW, trading_periods=DEFAULT_TRADING_PERIODS, clean=True):
        """
        Calculate volatility using the Yang-Zhang method.

        :param window: Rolling window size for volatility calculation.
        :param trading_periods: Number of trading periods in a year for annualization.
        :param clean: If True, drop NaN values from the result.
        :return: Annualized volatility using the Yang-Zhang method.
        """
        price_data = self.daily_data

        # Calculate log returns
        log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
        log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
        log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
        log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
        log_oc_sq = log_oc ** 2
        log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
        log_cc_sq = log_cc ** 2

        # Calculate components
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        close_vol = log_cc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
        open_vol = log_oc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
        window_rs = rs.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))

        # Combine components
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * math.sqrt(trading_periods)

        return result.dropna() if clean else result

    def parkinson(self, window=DEFAULT_WINDOW, trading_periods=DEFAULT_TRADING_PERIODS, clean=True):
        """
        Calculate volatility using the Parkinson method.

        :param window: Rolling window size for volatility calculation.
        :param trading_periods: Number of trading periods in a year for annualization.
        :param clean: If True, drop NaN values from the result.
        :return: Annualized volatility using the Parkinson method.
        """
        price_data = self.daily_data
        rs = (1.0 / (4.0 * math.log(2.0))) * ((price_data['High'] / price_data['Low']).apply(np.log)) ** 2.0

        def f(v):
            return (trading_periods * v.mean()) ** 0.5

        result = rs.rolling(window=window, center=False).apply(func=f)
        return result.dropna() if clean else result

    def garman_klass(self, window=DEFAULT_WINDOW, trading_periods=DEFAULT_TRADING_PERIODS, clean=True):
        """
        Calculate volatility using the Garman-Klass method.

        :param window: Rolling window size for volatility calculation.
        :param trading_periods: Number of trading periods in a year for annualization.
        :param clean: If True, drop NaN values from the result.
        :return: Annualized volatility using the Garman-Klass method.
        """
        price_data = self.daily_data

        log_hl = (price_data['High'] / price_data['Low']).apply(np.log)
        log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
        rs = 0.5 * log_hl ** 2 - (2 * math.log(2) - 1) * log_co ** 2

        def f(v):
            return (trading_periods * v.mean()) ** 0.5

        result = rs.rolling(window=window, center=False).apply(func=f)
        return result.dropna() if clean else result

    def rogers_satchell(self, window=DEFAULT_WINDOW, trading_periods=DEFAULT_TRADING_PERIODS, clean=True):
        """
        Calculate volatility using the Rogers-Satchell method.

        :param window: Rolling window size for volatility calculation.
        :param trading_periods: Number of trading periods in a year for annualization.
        :param clean: If True, drop NaN values from the result.
        :return: Annualized volatility using the Rogers-Satchell method.
        """
        price_data = self.daily_data

        log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
        log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
        log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

        def f(v):
            return (trading_periods * v.mean()) ** 0.5

        result = rs.rolling(window=window, center=False).apply(func=f)
        return result.dropna() if clean else result

    def high_frequency(self, clean=True, trading_period=DEFAULT_TRADING_PERIODS, **kwargs):
        """
        Calculate volatility using high-frequency data.

        :param clean: If True, drop NaN values from the result.
        :param trading_period: Number of trading periods in a year for annualization.
        :param kwargs: Additional arguments for data extraction.
        :return: Annualized high-frequency volatility.
        """
        price_series = ExtractOptionsData().extracting_ohlc(ticker=self.ticker, category=self.category, **kwargs)
        try:
            price_series['Date'] = price_series.index.date
        except:
            price_series['Date'] = price_series.index
        price_series['prev_close'] = price_series['Close'].shift(1)

        # Identify the first entry of each day
        first_of_day = price_series.groupby('Date').head(1).index

        # Compute log returns
        price_series['return'] = np.log(price_series['Close'] / price_series['prev_close'])
        price_series.loc[first_of_day, 'return'] = np.log(
            price_series.loc[first_of_day, 'Open'] / price_series.loc[first_of_day, 'prev_close'])

        return_series = price_series[['Date', 'return']].dropna()
        return_series['return'] = return_series['return'] ** 2

        # Compute daily variances and annualize
        annualized_daily_variance = return_series.groupby('Date')['return'].sum() * trading_period
        annualized_daily_volatility = annualized_daily_variance ** 0.5

        return annualized_daily_volatility.dropna() if clean else annualized_daily_volatility

    # -----------------------------------------------------------------------------------------------------
    # --------------------------------------Implied Volatility---------------------------------------------
    # -----------------------------------------------------------------------------------------------------
    @staticmethod
    def binomial_option_price(S: float, K: float, T: float, r: float, sigma: float,
                              q: float = 0.0, N: int = 100, option_type: Literal['CE', 'PE'] = 'CE',
                              american: bool = False, model: Literal['CRR', 'JR'] = 'CRR') -> float:
        """
        Price a European or American option via a recombining binomial tree.

        Parameters
        ----------
        S : float
            Current stock price.
        K : float
            Strike price.
        T : float
            Time to maturity in years.
        r : float
            Risk-free interest rate (annual).
        sigma : float
            Volatility (annual).
        q : float, default 0.0
            Continuous dividend yield.
        N : int, default 100
            Number of time steps.
        option_type : {'CE', 'PE'}, default ''CE
            Option type.
        american : bool, default False
            If True, price an American option.
        model : {'CRR', 'JR'}, default 'CRR'
            Binomial model: Cox–Ross–Rubinstein or Jarrow–Rudd.

        Returns
        -------
        float
            The option price.
        """
        dt = T / N
        if model == 'CRR':
            up_factor = np.exp(sigma * np.sqrt(dt))
        else:  # Jarrow–Rudd
            up_factor = np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt))
        down_factor = 1 / up_factor
        disc = np.exp(-r * dt)

        # Risk-neutral probability
        p = (np.exp((r - q) * dt) - down_factor) / (up_factor - down_factor)

        # Stock prices at maturity, vectorized: S * up_factor**j * down_factor**(N-j)
        j = np.arange(N + 1)
        ST = S * (up_factor ** j) * (down_factor ** (N - j))

        # Option values at maturity
        if option_type == 'CE':
            values = np.maximum(ST - K, 0.0)
        else:
            values = np.maximum(K - ST, 0.0)

        # Step backwards
        for i in range(N - 1, -1, -1):
            # Discounted expectation
            values = disc * (p * values[1:] + (1 - p) * values[:-1])

            if american:
                # Early-exercise payoff at node i
                S_t = S * (up_factor ** np.arange(i + 1)) * (down_factor ** (i - np.arange(i + 1)))
                if option_type == 'CE':
                    values = np.maximum(values, S_t - K)
                else:
                    values = np.maximum(values, K - S_t)

        return float(values[0])


    @classmethod
    def implied_vol_binomial(cls, S: float, K: float, T: float, r: float, q: float, market_price: float,
                             N: int = 100, option_type: Literal['CE', 'PE'] = 'CE',
                             american: bool = False, model: Literal['CRR', 'JR'] = 'CRR',
                             tol: float = 1e-6, vol_bounds: tuple = (1e-5, 5.0)) -> float:
        """
        Solve implied volatility under a binomial-tree pricer.

        Raises
        ------
        ValueError
            If `market_price` lies outside the no‐arbitrage bounds.
        """
        # No‐arbitrage bounds for European call: [max(0, S e^{-qT}-K e^{-rT}), S e^{-qT}]
        lower_bound = max(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))
        upper_bound = S * np.exp(-q * T)
        if not (lower_bound <= market_price <= upper_bound):
            msg = (
                f"Market price {market_price:.4f} outside arbitrage bounds "
                f"[{lower_bound:.4f}, {upper_bound:.4f}]."
            )
            raise ValueError

        def objective(vol: float) -> float:
            return (cls.binomial_option_price(
                S, K, T, r, vol, q, N, option_type, american, model
            ) - market_price)

        iv = brentq(objective, *vol_bounds, xtol=tol)
        return iv

    @staticmethod
    def bsm_implied_volatility(mkt_price, S, K, rf, maturity, option_category, q=0, current_date=None):
        """
        Calculate the implied volatility using the Black-Scholes-Merton model.
        If the computed volatility is unrealistically low, raises Value Error.
        """

        if current_date is None:
            current_date = datetime.date.today()

        # Calculate time to maturity in trading days and calendar days
        trading_days = np.busday_count(current_date, maturity)
        calendar_days = np.busday_count(current_date, maturity, weekmask='1111111')

        r = rf

        t1 = trading_days / 252  # Trading days to years
        t2 = calendar_days / 365  # Calendar days to years

        if t1 <= 0 or t2 <= 0:
            return np.nan

        # --- Intrinsic value check ---
        if option_category == 'CE':
            intrinsic_value = max(S * np.exp(-q * t2) - K * np.exp(-r * t2), 0)
        elif option_category == 'PE':
            intrinsic_value = max(K * np.exp(-r * t2) - S * np.exp(-q * t2), 0)
        else:
            raise ValueError("Invalid option category. Use 'CE' for Call or 'PE' for Put.")

        if mkt_price < intrinsic_value:
            print("Market price is below intrinsic value.")
            raise ValueError

        # --- End intrinsic value check ---

        def bsm_price(sigma):
            """Calculate the Black-Scholes-Merton option price."""
            d1 = (np.log(S / K) + (r - q) * t2 + 0.5 * sigma ** 2 * t1) / (sigma * np.sqrt(t1))
            d2 = d1 - sigma * np.sqrt(t1)
            if option_category == 'CE':
                return S * np.exp(-q * t2) * si.norm.cdf(d1) - K * np.exp(-r * t2) * si.norm.cdf(d2)
            elif option_category == 'PE':
                return K * np.exp(-r * t2) * si.norm.cdf(-d2) - S * np.exp(-q * t2) * si.norm.cdf(-d1)

        def objective_function(sigma):
            """Objective function for root-finding."""
            return bsm_price(sigma) - mkt_price

        return newton(objective_function, 0.01)
    @staticmethod
    def corrado_su_implied_moments(mkt_price, S, K, rf, maturity, option_category, q=0, current_date=None):
        """
        Compute the implied moments (volatility, skew, and kurtosis) for an option using the Corrado-Su adjustment.

        Parameters:
            mkt_price : array-like
                Market prices of the option.
            S : float
                Underlying asset price.
            K : array-like
                Strike prices.
            rf : DataFrame
                Risk-free rate data containing 'MIBOR Rate (%)'.
            maturity : datetime.date
                Maturity date of the option.
            option_category : str
                Option category: 'CE' for call or 'PE' for put.
            q : float, optional
                Dividend yield (default is 0).
            current_date : datetime.date, optional
                Current date (default is today's date).

        Returns:
            pd.DataFrame or np.nan:
                DataFrame with the implied volatility, skew, and kurtosis if successful; otherwise, NaN.
        """

        # Compute risk-free rate as the average of MIBOR rates.
        r = rf

        # Use today's date if current_date is not provided.
        if current_date is None:
            current_date = datetime.date.today()

        # Calculate time to maturity:
        # trading_days: business days between current_date and maturity.
        trading_days = np.busday_count(current_date, maturity)
        # calendar_days: calendar days between current_date and maturity (using all days of the week).
        calendar_days = np.busday_count(current_date, maturity, weekmask='1111111')

        # Convert trading days and calendar days to year fractions.
        t1 = trading_days / 252  # Trading days to years.
        t2 = calendar_days / 365  # Calendar days to years.

        # If the computed time to maturity is zero or negative, return NaN.
        if t1 <= 0 or t2 <= 0:
            return np.nan

        def bsm_price(k, sigma):
            """
            Calculate the Black-Scholes-Merton option price for a given strike (k) and volatility (sigma).

            Parameters:
                k : float
                    Strike price.
                sigma : float
                    Volatility.

            Returns:
                float:
                    Option price computed using the BSM formula.
            """
            d1 = (np.log(S / k) + (r - q) * t2 + 0.5 * sigma ** 2 * t1) / (sigma * np.sqrt(t1))
            d2 = d1 - sigma * np.sqrt(t1)

            # Return price based on option category.
            if option_category == 'CE':
                # European Call Option.
                return S * np.exp(-q * t2) * si.norm.cdf(d1) - k * np.exp(-r * t2) * si.norm.cdf(d2)
            elif option_category == 'PE':
                # European Put Option.
                return k * np.exp(-r * t2) * si.norm.cdf(-d2) - S * np.exp(-q * t2) * si.norm.cdf(-d1)
            else:
                raise ValueError("Invalid option category. Use 'CE' for Call or 'PE' for Put.")

        def gram_charlier_adjustment(k, sigma, skew, kurt):
            """
            Compute the Gram-Charlier adjustment terms for skewness and kurtosis.

            Parameters:
                k : float
                    Strike price.
                sigma : float
                    Volatility.
                skew : float
                    Skewness parameter.
                kurt : float
                    Kurtosis parameter.

            Returns:
                tuple:
                    Q3 and Q4 adjustment terms.
            """
            d1 = (np.log(S / k) + (r - q) * t2 + 0.5 * sigma ** 2 * t1) / (sigma * np.sqrt(t1))
            d2 = d1 - sigma * np.sqrt(t1)

            # Compute Q3 and Q4 using the standard normal density function.
            term_1 = S * sigma * np.sqrt(t1)
            term_2 = (2 * sigma * np.sqrt(t1) - d1) * si.norm.pdf(d1)
            term_3 = si.norm.cdf(d1) * (sigma ** 2)

            Q3 = (1 / 6) * term_1 * (term_2 + term_3)

            term_4 = (d1 ** 2) - 1 - (3 * sigma * np.sqrt(t1) * d2) * si.norm.pdf(d1)
            term_5 = (sigma ** 3) * np.sqrt(t1 ** 3) * si.norm.cdf(d1)
            Q4 = (1 / 24) * term_1 * (term_4 + term_5)
            return Q3, Q4

        def corrado_su_price(k, sigma, skew, kurt):
            """
            Calculate the option price using the Corrado-Su model, which adjusts the BSM price with skew and kurtosis corrections.

            Parameters:
                k : float
                    Strike price.
                sigma : float
                    Volatility.
                skew : float
                    Skewness parameter.
                kurt : float
                    Kurtosis parameter.

            Returns:
                float:
                    Adjusted option price.
            """
            # Base price from the Black-Scholes-Merton model.
            BSM_PRICE = bsm_price(k, sigma)
            # Compute adjustment terms.
            Q3, Q4 = gram_charlier_adjustment(k, sigma, skew, kurt)
            # Adjust the BSM price with skew and kurtosis corrections.
            cs_price = BSM_PRICE + skew * Q3 + (kurt - 3) * Q4
            return cs_price

        def objective(params, K):
            """
            Objective function for optimization: minimize the sum of squared differences between
            the model prices (using Corrado-Su adjustments) and the market prices.

            Parameters:
                params : list or array-like
                    Parameters to optimize: [sigma, skew, kurt].
                K : array-like
                    Strike prices.

            Returns:
                float:
                    Sum of squared errors.
            """
            sigma, skew, kurt = params
            # Compute the model price for each strike.
            model_prices = np.array([corrado_su_price(k, sigma, skew, kurt) for k in K])
            return np.sum((model_prices - mkt_price) ** 2)

        # Initial guess for the parameters: volatility, skew, kurtosis.
        initial_guess = [0.1, 0, 3]

        try:
            # Optimize the parameters to best fit the market prices.
            result = opt.minimize(objective, initial_guess, args=(K,), bounds=[(0.01, 1), (-2, 2), (1, 10)])
        except Exception as e:
            # If an exception occurs during optimization, print the error and return NaN.
            print(f"Optimization failed: {e}")
            return np.nan

        # If optimization was unsuccessful, notify and return NaN.
        if not result.success:
            print("Optimization did not converge.")
            return np.nan

        # Extract optimal parameters: implied volatility, skew, and kurtosis.
        implied_vol, implied_skew, implied_kurt = result.x

        # Create and return a DataFrame with the implied moments.
        df = pd.DataFrame({
            'cs_implied_vol': [implied_vol],
            'cs_implied_skew': [implied_skew],
            'cs_implied_kurt': [implied_kurt]
        })
        return df

    # --------------------------------------Forecasted Volatility---------------------------------------------
    def garch(self, horizon=30, trading_periods=DEFAULT_TRADING_PERIODS):
        """
        Calculate volatility using a GARCH(1,1) model.

        :param horizon: Forecast horizon for GARCH model.
        :param trading_periods: Number of trading periods in a year for annualization.
        :return: Annualized GARCH volatility forecast.
        """
        return_series = (self.daily_data['Close'] / self.daily_data['Close'].shift(1)).apply(np.log).dropna()
        return_series = return_series * 100  # Scale for numerical stability

        model = arch_model(return_series, vol='GARCH', p=1, q=1, dist='t').fit(disp='off')
        forecast = model.forecast(horizon=horizon)
        # Extract the volatility for the last day of the horizon
        # This is the volatility over the entire horizon (e.g., 30 days)
        horizon_volatility = np.sqrt(forecast.variance.iloc[:, -1]) / 100  # Undo scaling

        # Convert horizon volatility to daily volatility
        daily_volatility = horizon_volatility  # / np.sqrt(horizon)

        # Annualize the daily volatility
        annualized_volatility = daily_volatility * np.sqrt(trading_periods)

        return annualized_volatility

    # -----------------------------------------Cones-----------------------------------------------------------
    def cones(self, moment='vol'):
        """
        Calculates volatility cones using logarithmic returns and multiple rolling window sizes.

        Returns:
            pd.DataFrame: DataFrame containing volatility cone values for various window sizes.
        Raises:
            AttributeError: If the 'close' column is missing in self.cones_data.
            ValueError: If there's insufficient data for a specific rolling window or invalid parameters.
            NameError: If DEFAULT_TRADING_PERIODS is not defined.
        """
        try:
            # Ensure the expected 'close' column exists in the data.
            if 'Close' not in self.cones_data.columns:
                raise AttributeError("The data does not contain a 'close' column.")

            # Initialize an empty DataFrame to store our computed values.
            cones_df = pd.DataFrame()

            # Define a helper function to calculate the scaling factor (m)
            def func_m(window, size_return_series):
                # Ensure that there is enough data for the given window size.
                if size_return_series < window:
                    raise ValueError(f"Not enough data points ({size_return_series}) for the window size {window}.")
                n = size_return_series - window + 1
                if n <= 0:
                    raise ValueError(f"Computed number of periods ({n}) is invalid. Check window size and data length.")
                t1 = window / n
                t2 = (window ** 2) - 1
                t3 = 3 * (n ** 2)
                denominator = 1 - t1 + (t2 / t3)
                if denominator <= 0:
                    raise ValueError("Denominator in scaling factor calculation is non-positive. Check inputs.")
                m = (1 / denominator) ** 0.5
                return m

            # Calculate logarithmic returns and store them in the DataFrame.
            cones_df['ln_returns'] = (self.cones_data['Close'] / self.cones_data['Close'].shift(1)).apply(np.log)
            cones_df.dropna(inplace=True)  # Corrected: remove assignment here.

            # Define the rolling windows (in days) for volatility calculations.
            windows = [20, 40, 60, 120, 240]
            for window in windows:
                if moment == 'vol':
                    rolling_std = cones_df['ln_returns'].rolling(window=window).std()
                    try:
                        scaling = math.sqrt(DEFAULT_TRADING_PERIODS)
                    except NameError:
                        raise NameError("DEFAULT_TRADING_PERIODS is not defined.")

                    try:
                        m_value = func_m(window, len(cones_df['ln_returns']))
                    except ValueError as ve:
                        raise ValueError(f"Error computing scaling factor for window {window}: {ve}")

                    cones_df[f'{window}_day'] = rolling_std * scaling * m_value

                elif moment == 'skew':
                    rolling_skew = cones_df['ln_returns'].rolling(window=window).skew()
                    try:
                        trading_days = DEFAULT_TRADING_PERIODS  # Typically 252
                    except NameError:
                        trading_days = 252
                    scaling = 1 / math.sqrt(trading_days)
                    cones_df[f'{window}_day'] = rolling_skew * scaling

                elif moment == 'kurt':
                    rolling_excess_kurt = cones_df['ln_returns'].rolling(window=window).kurt()
                    try:
                        trading_days = DEFAULT_TRADING_PERIODS  # Typically 252
                    except NameError:
                        trading_days = 252
                    scaling = 1 / trading_days
                    cones_df[f'{window}_day'] = rolling_excess_kurt * scaling

                else:
                    raise ValueError(f"Error computing cones for {moment}. Please enter vol/skew/kurt")

            cones_df.dropna(inplace=True)
            return cones_df

        except Exception as e:
            print(f"An error occurred in volatility_cones: {e}")
            return pd.DataFrame()

# -------------------- USAGE -------------------#
# if __name__ == "__main__":
#     # Example usage
#     ticker = "NIFTY"
#     category = 'index'
#     volatility_estimator = EstimateVolatility(ticker=ticker, category=category)
#     annualized_vol = volatility_estimator.high_frequency(period='30d', interval='15m')
#     print(f"Annualized High Frequency Volatility: {annualized_vol}")

# Implied Volatility Usage
# if __name__ == "__main__":
#     ticker = "NIFTY"
#     category = 'index'
#     volatility_estimator = EstimateVolatility(ticker=ticker, category=category)
#     op_chain = ExtractOptionsChain(ticker=ticker, category_=category)
#     call_chain = op_chain.extract_call_data()
#     rf = extract_risk_free_rate()
#     vol_estimates = EstimateVolatility(ticker=ticker, category=category)
#     for expiry in call_chain['expiryDate'].unique():
#         ce_chain_df = call_chain[call_chain['expiryDate'] == expiry]
#         imp_vol = vol_estimates.bsm_implied_volatility(
#             mkt_price=ce_chain_df['mkt_price'].values[0],
#             S=ce_chain_df['ltp'].values[0],
#             K=ce_chain_df['strikePrice'].values[0],
#             rf=rf, maturity=expiry,
#             option_category='CE', q=0, current_date=None
#         )
#         cor_su_vol = vol_estimates.corrado_su_implied_moments(
#                     mkt_price=np.array(ce_chain_df['mkt_price']),
#                     S=ce_chain_df['ltp'].values[0],
#                     K=np.array(ce_chain_df['strikePrice']),
#                     rf=rf, maturity=expiry, option_category='CE', q=0,
#                     current_date=None
#                 )
#     print(f"BSM Implied volatility of NIFTY call option:{imp_vol}")
#     print(f"Corrado Su Implied moments of NIFTY call option:{cor_su_vol}")
