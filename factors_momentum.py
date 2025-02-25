# ------------------Import Libraries ------------#
import pandas as pd
import sqlite3
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import seaborn as sns
from pykalman import KalmanFilter
import datetime as dt
from ta.momentum import RSIIndicator
from weighting_strategy import equi_wt, cap_wt
from sklearn.linear_model import LinearRegression
from extract_data import Extract_Data

# ---------------------CONSTANTS------------------#
def get_db_connection():
    return sqlite3.connect("./raw_datasets/head_database.db")

# --------------------MAIN CODE-------------------#
class Momentum:
    def __init__(self, strat, univ, wt_strat):

        # Use the function to create a new connection
        self.DB = get_db_connection()

        # Load the columns from the database into DataFrames
        symbol_yfinance_df = pd.read_sql_query("SELECT symbol_yfinance FROM table_symbol_yfinance", self.DB)
        symbol_nse_df = pd.read_sql_query("SELECT symbol_nse FROM table_symbol_nse", self.DB)
        # Combine both columns into a single DataFrame assuming index positions are the mapping
        symbol_mapping = pd.concat([symbol_yfinance_df, symbol_nse_df], axis=1)

        # Store Universe
        query = "SELECT * FROM table_symbol_yfinance"
        symbols = pd.read_sql_query(query, self.DB)
        if univ == 'All':
            self.tickers = symbols['symbol_yfinance'].tolist()
        elif univ == 'Nifty 50':
            symbol_nse_df = pd.read_sql_query("SELECT * FROM table_symbol_nse", self.DB)
            flag = symbol_nse_df['nifty_50'].tolist()
            self.tickers = symbols['symbol_yfinance'][pd.Series(flag).astype(bool)].tolist()
        elif univ == 'Nifty 500':
            self.tickers = symbols['symbol_yfinance'].head(500).tolist()
        elif univ == 'Nifty Next 50':
            pass
        # Calculate Scores based on strategy
        if strat == 'RSI':
            scores = self.rsi()
        elif strat == 'Price Momentum (12-1)':
            scores = self.price_momentum_1_12()
        elif strat == 'Price Momentum (12-3)':
            scores = self.price_momentum_3_12()
        elif strat == 'Price Acceleration':
            scores = self.price_acceleration()

        scores = scores.sort_values(by='scores', ascending=False)
        num_rows = int(len(scores) * 0.2)
        scores = scores.iloc[:num_rows]
        # Calculate Weights based on weighting strategy
        if wt_strat == 'equal_wt':
            self.wts = equi_wt(scores, symbol_mapping)
        elif wt_strat == 'mkt_cap':
            symbol_list = scores['symbol_yfinance'].tolist()
            mkt_cap = [Extract_Data(ticker).extract_market_cap() for ticker in symbol_list]
            scores['mkt_cap'] = mkt_cap
            self.wts = cap_wt(scores, symbol_mapping)

    def __del__(self):
        # Close the connection when the object is deleted
        self.DB.close()

    # Method to return the weights DataFrame
    def get_wts(self):
        return self.wts

    # Function to extract market capitalization with error handling
    # def extract_market_cap(self, ticker):
    #     try:
    #         stock = yf.Ticker(ticker)
    #         market_cap = stock.info.get('marketCap')
    #         return market_cap if market_cap else 'N/A'
    #     except Exception as e:
    #         print(f"Error fetching market capitalization for {ticker}: {e}")
    #         return 'N/A'

    # def extract_stock_data(self, ticker, **kwargs):
    #     try:
    #         stock = yf.Ticker(ticker)
    #         stock_data = stock.history(**kwargs)
    #         if stock_data.empty:
    #             raise ValueError(f"No data found for {ticker}")
    #         return stock_data
    #     except Exception as e:
    #         print(f"Error fetching stock data for {ticker}: {e}")
    #         return pd.DataFrame()

    def calculate_monthly_returns_for_lags(self, monthly_prices, lags):
        data = pd.DataFrame(index=monthly_prices.index)
        for lag in lags:
            data[f'return_{lag}m'] = (
                monthly_prices
                .pct_change(lag)
                .add(1)
                .pow(1 / lag)
                .sub(1)
            )
        return data
#---------------------------STRATEGIES--------------------------#
    def rsi(self):
        rsi_dict = {}

        for ticker in self.tickers:
            # Fetch 1 year of daily adjusted close prices
            data = Extract_Data(ticker)
            temp = data.extract_stock_data(period='1mo', interval='1d')
            prices = temp['Close'].dropna()

            if len(prices) < 14:  # Ensure we have sufficient data for RSI calculation
                continue

            # Calculate RSI with a 14-day window
            rsi_indicator = RSIIndicator(close=prices, window=14)
            rsi_values = rsi_indicator.rsi()

            # Get the most recent RSI value as the score
            latest_rsi = rsi_values.iloc[-1]

            # Store RSI value as a score (invert or transform as needed)
            rsi_dict[ticker] = latest_rsi

        # Convert the dictionary to a DataFrame for sorting and ranking
        rsi_df = pd.DataFrame(list(rsi_dict.items()), columns=['symbol_yfinance', 'scores'])
        rsi_df = rsi_df[rsi_df['scores'] < 30]
        # Sort stocks based on the scores in descending order
        rsi_df = rsi_df.sort_values(by='scores', ascending=True)
        return rsi_df

    def price_momentum_1_12(self):
        mom_dict = {}
        for ticker in self.tickers:
            data = Extract_Data(ticker)
            temp = data.extract_stock_data(period='2y', interval='1d')
            prices = temp['Close']
            monthly_prices = prices.resample('ME').last()
            lags = [1, 12]

            data = self.calculate_monthly_returns_for_lags(monthly_prices, lags)
            data['momentum_1_12'] = data['return_12m'].sub(data['return_1m'])
            mom_dict[ticker] = data['momentum_1_12'].tail(1).values[0]

        mom_df = pd.DataFrame(list(mom_dict.items()), columns=['symbol_yfinance', 'scores'])
        return mom_df

    def price_momentum_3_12(self):
        mom_dict = {}
        for ticker in self.tickers:
            data = Extract_Data(ticker)
            temp = data.extract_stock_data(period='2y', interval='1d')
            prices = temp['Close']
            monthly_prices = prices.resample('ME').last()
            lags = [3, 12]

            data = self.calculate_monthly_returns_for_lags(monthly_prices, lags)
            data['momentum_3_12'] = data['return_12m'].sub(data['return_3m'])
            mom_dict[ticker] = data['momentum_3_12'].tail(1).values[0]

        mom_df = pd.DataFrame(list(mom_dict.items()), columns=['symbol_yfinance', 'scores'])
        return mom_df

    def price_acceleration(self):
        acc_dict = {}

        for ticker in self.tickers:
            # Fetch 1 year of daily adjusted close prices
            data = Extract_Data(ticker)
            temp = data.extract_stock_data(period='2y', interval='1d')
            prices = temp['Close'].dropna()

            if len(prices) < 252:  # Ensure we have sufficient data for a 1-year period
                continue

            # Calculate slopes for the last 1 year (252 trading days) and 3 months (63 trading days)
            long_period = 252
            short_period = 63

            # Create arrays for linear regression (X as time index, y as prices)
            X_long = np.arange(long_period).reshape(-1, 1)
            y_long = prices.tail(long_period).values.reshape(-1, 1)

            X_short = np.arange(short_period).reshape(-1, 1)
            y_short = prices.tail(short_period).values.reshape(-1, 1)


            # Apply linear regression to get the slope for each period
            model_long = LinearRegression().fit(X_long, y_long)
            slope_long = model_long.coef_[0][0]

            model_short = LinearRegression().fit(X_short, y_short)
            slope_short = model_short.coef_[0][0]
            # Calculate price acceleration as the change in slopes, adjusted for volatility
            volatility = prices[-long_period:].std()
            price_acceleration_adjusted = (slope_short - slope_long) / volatility

            # Store the result in the dictionary
            acc_dict[ticker] = price_acceleration_adjusted
            # Create a DataFrame from the dictionary
        acc_df = pd.DataFrame(list(acc_dict.items()), columns=['symbol_yfinance', 'scores'])
        return acc_df
