# ------------------Import Libraries -------------#
from arch import arch_model
import pandas as pd
import numpy as np
import math
from extract_options_data import ExtractOptionsData
from scipy.optimize import minimize
from scipy.special import gamma
import warnings

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

    def __init__(self, ticker, type):
        """
        Initialize the class with the ticker symbol and data type.

        :param ticker: The ticker symbol of the asset.
        :param type: The type of data (e.g., 'stock', 'option').
        """
        self.ticker = ticker
        self.type = type
        extract_data_object = ExtractOptionsData()
        self.daily_data = extract_data_object.extracting_ohlc(ticker=self.ticker, type=self.type, period='1y',
                                                              interval='1d')
        if self.daily_data.empty:
            raise ValueError(f"No data found for ticker: {self.ticker} and type: {self.type}")

    def close_to_close(self, window=DEFAULT_WINDOW, trading_periods=DEFAULT_TRADING_PERIODS, clean=True):
        """
        Calculate close-to-close volatility.

        :param window: Rolling window size for volatility calculation.
        :param trading_periods: Number of trading periods in a year for annualization.
        :param clean: If True, drop NaN values from the result.
        :return: Annualized close-to-close volatility.
        """
        if self.daily_data['close'].isnull().any():
            raise ValueError("Close prices contain NaN values.")

        log_return = (self.daily_data['close'] / self.daily_data['close'].shift(1)).apply(np.log)
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
        log_ho = (price_data['high'] / price_data['open']).apply(np.log)
        log_lo = (price_data['low'] / price_data['open']).apply(np.log)
        log_co = (price_data['close'] / price_data['open']).apply(np.log)
        log_oc = (price_data['open'] / price_data['close'].shift(1)).apply(np.log)
        log_oc_sq = log_oc ** 2
        log_cc = (price_data['close'] / price_data['close'].shift(1)).apply(np.log)
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
        rs = (1.0 / (4.0 * math.log(2.0))) * ((price_data['high'] / price_data['low']).apply(np.log)) ** 2.0

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

        log_hl = (price_data['high'] / price_data['low']).apply(np.log)
        log_co = (price_data['close'] / price_data['open']).apply(np.log)
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

        log_ho = (price_data['high'] / price_data['open']).apply(np.log)
        log_lo = (price_data['low'] / price_data['open']).apply(np.log)
        log_co = (price_data['close'] / price_data['open']).apply(np.log)
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

        def f(v):
            return (trading_periods * v.mean()) ** 0.5

        result = rs.rolling(window=window, center=False).apply(func=f)
        return result.dropna() if clean else result

    def garch(self, horizon=30, trading_periods=DEFAULT_TRADING_PERIODS):
        """
        Calculate volatility using a GARCH(1,1) model.

        :param horizon: Forecast horizon for GARCH model.
        :param trading_periods: Number of trading periods in a year for annualization.
        :return: Annualized GARCH volatility forecast.
        """
        return_series = (self.daily_data['close'] / self.daily_data['close'].shift(1)).apply(np.log).dropna()
        return_series = return_series * 100  # Scale for numerical stability

        model = arch_model(return_series, vol='GARCH', p=1, q=1, dist='t').fit(disp='off')
        forecast = model.forecast(horizon=horizon)
        # Extract the volatility for the last day of the horizon
        # This is the volatility over the entire horizon (e.g., 30 days)
        horizon_volatility = np.sqrt(forecast.variance.iloc[:, -1]) / 100  # Undo scaling

        # Convert horizon volatility to daily volatility
        daily_volatility = horizon_volatility #/ np.sqrt(horizon)

        # Annualize the daily volatility
        annualized_volatility = daily_volatility * np.sqrt(trading_periods)

        return annualized_volatility

    def high_frequency(self, clean=True, trading_period=DEFAULT_TRADING_PERIODS, **kwargs):
        """
        Calculate volatility using high-frequency data.

        :param clean: If True, drop NaN values from the result.
        :param trading_period: Number of trading periods in a year for annualization.
        :param kwargs: Additional arguments for data extraction.
        :return: Annualized high-frequency volatility.
        """
        price_series = ExtractOptionsData().extracting_ohlc(ticker=self.ticker, type=self.type, **kwargs)
        price_series['Date'] = price_series.index.date
        price_series['prev_close'] = price_series['close'].shift(1)

        # Identify the first entry of each day
        first_of_day = price_series.groupby('Date').head(1).index

        # Compute log returns
        price_series['return'] = np.log(price_series['close'] / price_series['prev_close'])
        price_series.loc[first_of_day, 'return'] = np.log(
            price_series.loc[first_of_day, 'open'] / price_series.loc[first_of_day, 'prev_close'])

        return_series = price_series[['Date', 'return']].dropna()
        return_series['return'] = return_series['return'] ** 2

        # Compute daily variances and annualize
        annualized_daily_variance = return_series.groupby('Date')['return'].sum() * trading_period
        annualized_daily_volatility = annualized_daily_variance ** 0.5

        return annualized_daily_volatility.dropna() if clean else annualized_daily_volatility

    def bsm_implied_volatility(self, option_chain):



# -------------------- USAGE -------------------#
if __name__ == "__main__":
    # Example usage
    ticker = "NIFTY"
    type = 'index'
    volatility_estimator = EstimateVolatility(ticker=ticker, type=type)
    annualized_vol = volatility_estimator.high_frequency(period='30d', interval='15m')
    print(f"Annualized High Frequency Volatility: {annualized_vol}")