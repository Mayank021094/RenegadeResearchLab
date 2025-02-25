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


# ---------------------CONSTANTS------------------#


# --------------------MAIN CODE-------------------#

class EstimateVolatility:
    def __init__(self, ticker, type):
        self.ticker = ticker
        self.type = type
        extract_data_object = ExtractOptionsData()
        self.daily_data = extract_data_object.extracting_ohlc(ticker=self.ticker, type=self.type, period='1y',
                                                              interval='1d')

    def close_to_close(self, window=30, trading_periods=252, clean=True):
        log_return = (self.daily_data['close'] / self.daily_data['close'].shift(1)).apply(np.log)
        result = log_return.rolling(
            window=window,
            center=False
        ).std() * math.sqrt(trading_periods)
        if clean:
            return result.dropna()
        else:
            return result

    def yang_zhang(self, window=30, trading_periods=252, clean=True):

        price_data = self.daily_data
        log_ho = (price_data['high'] / price_data['open']).apply(np.log)
        log_lo = (price_data['low'] / price_data['open']).apply(np.log)
        log_co = (price_data['close'] / price_data['open']).apply(np.log)

        log_oc = (price_data['open'] / price_data['close'].shift(1)).apply(np.log)
        log_oc_sq = log_oc ** 2

        log_cc = (price_data['close'] / price_data['close'].shift(1)).apply(np.log)
        log_cc_sq = log_cc ** 2

        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

        close_vol = log_cc_sq.rolling(
            window=window,
            center=False
        ).sum() * (1.0 / (window - 1.0))
        open_vol = log_oc_sq.rolling(
            window=window,
            center=False
        ).sum() * (1.0 / (window - 1.0))
        window_rs = rs.rolling(
            window=window,
            center=False
        ).sum() * (1.0 / (window - 1.0))

        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * math.sqrt(trading_periods)

        if clean:
            return result.dropna()
        else:
            return result

    def parkinson(self, window=30, trading_periods=252, clean=True):

        price_data = self.daily_data
        rs = (1.0 / (4.0 * math.log(2.0))) * ((price_data['high'] / price_data['low']).apply(np.log)) ** 2.0

        def f(v):
            return (trading_periods * v.mean()) ** 0.5

        result = rs.rolling(
            window=window,
            center=False
        ).apply(func=f)
        if clean:
            return result.dropna()
        else:
            return result

    def garman_klass(self, window=30, trading_periods=252, clean=True):

        price_data = self.daily_data

        log_hl = (price_data['high'] / price_data['low']).apply(np.log)
        log_co = (price_data['close'] / price_data['open']).apply(np.log)

        rs = 0.5 * log_hl ** 2 - (2 * math.log(2) - 1) * log_co ** 2

        def f(v):
            return (trading_periods * v.mean()) ** 0.5

        result = rs.rolling(window=window, center=False).apply(func=f)

        if clean:
            return result.dropna()
        else:
            return result

    def rogers_satchell(self, window=30, trading_periods=252, clean=True):
        price_data = self.daily_data

        log_ho = (price_data['high'] / price_data['open']).apply(np.log)
        log_lo = (price_data['low'] / price_data['open']).apply(np.log)
        log_co = (price_data['close'] / price_data['open']).apply(np.log)

        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

        def f(v):
            return (trading_periods * v.mean()) ** 0.5

        result = rs.rolling(
            window=window,
            center=False
        ).apply(func=f)

        if clean:
            return result.dropna()
        else:
            return result

    def garch(self, horizon=30, trading_periods=252):
        # Compute log returns, ensuring no NaNs
        return_series = (self.daily_data['close'] / self.daily_data['close'].shift(1)).apply(np.log).dropna()
        return_series = return_series*100
        # Fit GARCH(1,1) model to the full return series
        model = arch_model(return_series, vol='GARCH', p=1, q=1, dist='t').fit(disp='off')
        forecast = model.forecast(horizon=horizon)
        future_volatility = np.sqrt(forecast.variance.iloc[:, -1]) * math.sqrt(trading_periods)/100

        return future_volatility

    def high_frequency(self, frequency='15min', trading_periods='30d'):


