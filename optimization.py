# ------------------Import Libraries ------------#
import pandas as pd
import sqlite3
import numpy as np
from numpy.random import random, uniform, dirichlet, choice
from numpy.linalg import inv
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
from sympy import symbols, solve, log, diff
from scipy.optimize import minimize_scalar, newton, minimize
from scipy.integrate import quad
from scipy.stats import norm
from expected_volatility import Expected_Volatility
from expected_mean import Expected_Mean


# ---------------------CONSTANTS------------------#
def get_db_connection():
    return sqlite3.connect("./raw_datasets/head_database.db")


# --------------------MAIN CODE-------------------#
class Optimization:
    def __init__(self, strat, univ):

        self.periods_per_year = None
        self.rf = None
        self.wts = None
        self.price_df = None
        self.start = None
        self.end = None
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

        self.price_df = self.get_weekly_prices_df()
        self.price_df = self.price_df.iloc[:, :1]
        print(np.log(self.price_df).diff().dropna().mean().mul(100))
        self.ret_df = Expected_Mean(self.price_df).bayesian_dist()

        # Calculate Scores based on strategy
        if strat == 'Max Sharpe Ratio':
            self.max_sharpe()
        elif strat == 'Min Volatility':
            self.min_vol()
        elif strat == 'Risk Parity':
            self.risk_parity()
        elif strat == 'Inverse Volatility':
            self.inverse_volatility()
        elif strat == 'Kelly':
            self.kelly()

    def __del__(self):
        # Close the connection when the object is deleted
        self.DB.close()

    # Method to return the weights DataFrame
    def get_wts(self):
        return self.wts

    # -----------------------STRATEGIES----------------------#

    def max_sharpe(self):

        # self.ret_df = self.get_weekly_returns_df()

        # Compute mean returns, covariance, risk-free rate and precision matrix
        mean_returns = self.ret_df.mean()
        cov_matrix = self.ret_df.cov()
        # precision_matrix = pd.DataFrame(inv(cov_matrix), index=self.ret_df.columns, columns=self.ret_df.columns)
        self.rf = self.extract_weekly_rf()

        # Define the constraint that weights sum to 1
        weight_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        # Initial guess for weights (equal weights)
        x0 = np.array([1 / len(self.ret_df.columns)] * len(self.ret_df.columns))
        # Optimize the portfolio to maximize the Sharpe Ratio
        wts = minimize(
            fun=self.neg_sharpe_ratio,
            x0=x0,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=[(0, 1) for _ in range(len(self.ret_df.columns))],
            constraints=weight_constraint,
            options={'maxiter': 1e4}
        )

        # Store and display the optimal weights
        self.wts = pd.DataFrame({'Symbol_NSE': self.ret_df.columns, 'Weights': wts.x})

    def min_vol(self):
        # self.ret_df, self.start, self.end = self.get_weekly_returns_df()

        # Compute mean returns, covariance, risk-free rate and precision matrix
        mean_returns = self.ret_df.mean()
        cov_matrix = self.ret_df.cov()
        # precision_matrix = pd.DataFrame(inv(cov_matrix), index=self.ret_df.columns, columns=self.ret_df.columns)
        self.rf = self.extract_weekly_rf()

        # Define the constraint that weights sum to 1
        weight_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        # Initial guess for weights (equal weights)
        x0 = np.array([1 / len(self.ret_df.columns)] * len(self.ret_df.columns))
        # Optimize the portfolio to maximize the Sharpe Ratio
        wts = minimize(
            fun=self.portfolio_std,
            x0=x0,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=[(0, 1) for _ in range(len(self.ret_df.columns))],
            constraints=weight_constraint,
            options={'tol': 1e-10, 'maxiter': 1e4}
        )
        # Store and display the optimal weights
        self.wts = pd.DataFrame({'Symbol_NSE': self.ret_df.columns, 'Weights': wts.x})

    def risk_parity(self):
        # Get the weekly returns DataFrame and the date range
        # self.ret_df, _, _ = self.get_weekly_returns_df()

        # Calculate the covariance matrix
        cov_matrix = self.ret_df.cov()

        # Define the objective function for risk parity
        def risk_parity_obj(weights):
            # Calculate the portfolio variance and standard deviation
            portfolio_var = weights.T @ cov_matrix @ weights
            portfolio_std = np.sqrt(portfolio_var)

            # Calculate the marginal risk contribution for each asset
            marginal_risk_contrib = cov_matrix @ weights

            # Calculate the risk contribution for each asset
            risk_contrib = weights * marginal_risk_contrib / portfolio_std

            # The target risk contribution for each asset is the total risk divided by the number of assets
            target_risk_contrib = portfolio_std / len(weights)

            # Minimize the squared difference between each asset's risk contribution and the target
            return np.sum((risk_contrib - target_risk_contrib) ** 2)

        # Constraints: weights must sum to 1
        weight_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        # Initial guess for weights (equal weighting)
        x0 = np.array([1 / len(self.ret_df.columns)] * len(self.ret_df.columns))

        # Bounds for each weight (between 0 and 1)
        bounds = [(0, 1) for _ in range(len(self.ret_df.columns))]

        # Run the optimization to minimize the risk parity objective
        result = minimize(
            fun=risk_parity_obj,
            x0=x0,
            bounds=bounds,
            constraints=weight_constraint,
            method='SLSQP'
        )

        # Store and print the optimal weights
        self.wts = pd.DataFrame({'Symbol_NSE': self.ret_df.columns, 'Weights': result.x})

    def inverse_volatility(self):
        # Get the weekly returns DataFrame
        # self.ret_df, _, _ = self.get_weekly_returns_df()
        # Calculate the standard deviation (volatility) of each asset's returns
        volatilities = self.ret_df.std()
        # Calculate the inverse of volatilities
        inv_volatilities = 1 / volatilities

        # Normalize so weights sum to 1
        weights = inv_volatilities / inv_volatilities.sum()

        # Store and display the weights
        self.wts = pd.DataFrame({'Weights': weights})
        self.wts = self.wts.reset_index()

    def kelly(self):
        # Get the weekly returns DataFrame and the date range
        # self.ret_df, self.start, self.end = self.get_weekly_returns_df()
        # Calculate the covariance matrix
        mean_returns = self.ret_df.mean()
        cov_matrix = self.ret_df.cov()
        rf = self.extract_weekly_rf()
        # Calculate the precision matrix (inverse of covariance matrix)
        precision_matrix = np.linalg.inv(cov_matrix)
        # Calculate raw Kelly weights
        raw_weights = precision_matrix.dot(mean_returns) / rf
        # Shift weights to make them non-negative
        shifted_weights = raw_weights - np.min(raw_weights)
        # Normalize weights so that they sum to 1
        normalized_weights = shifted_weights / np.sum(shifted_weights)
        # Normalize weights so that they sum to 1
        self.wts = pd.DataFrame({'Symbol_NSE': self.ret_df.columns, 'Weights': normalized_weights})
        print(self.wts)

    # ------------------------STRATEGIES BACKEND---------------------#
    # Portfolio metrics
    def portfolio_std(self, weights, cov=None):
        return np.sqrt(weights @ cov @ weights * self.periods_per_year)

    def portfolio_returns(self, weights, rt=None):
        return (weights @ rt + 1) ** self.periods_per_year - 1

    def portfolio_performance(self, weights, rt, cov):
        r = self.portfolio_returns(weights, rt=rt)
        sd = self.portfolio_std(weights, cov=cov)
        return r, sd

    def neg_sharpe_ratio(self, weights, mean_ret, cov):
        """Calculate the negative Sharpe ratio given portfolio weights."""
        r, sd = self.portfolio_performance(weights, mean_ret, cov)
        return -(r - self.rf) / sd

    def get_weekly_prices_df(self):
        price_dict = {}
        for ticker in self.tickers:
            data = Extract_Data(ticker)
            temp = data.extract_stock_data(period='10y', interval='1d')
            prices = temp['Close']
            weekly_prices = prices.resample('W').last()
            price_dict[ticker] = weekly_prices

        # Determine the maximum length and align date range
        max_length = max(len(v) for v in price_dict.values())
        self.end = max(series.index[-1] for series in price_dict.values())
        self.start = self.end - pd.DateOffset(weeks=max_length - 1)
        aligned_dict = {}
        for ticker, weekly_prices in price_dict.items():
            aligned_series = weekly_prices[self.start:self.end]
            aligned_series = aligned_series.reindex(pd.date_range(start=self.start, end=self.end, freq='W'),
                                                    method='ffill')
            aligned_dict[ticker] = aligned_series.tolist()

        # Create DataFrame from aligned returns
        price_df = pd.DataFrame(aligned_dict)
        price_df.index = pd.date_range(start=self.start, end=self.end, freq='W')

        # Calculate average number of weekly periods per year
        price_df['Year'] = price_df.index.year
        periods_per_year = price_df.groupby('Year').size().mean()
        self.periods_per_year = round(periods_per_year)
        price_df.drop(['Year'], axis=1, inplace=True)

        return price_df

    def extract_weekly_rf(self):
        # Fetch risk-free rate (using '^TNX' as proxy for 10-year rate)

        rf_data = yf.Ticker("^TNX")
        rf = rf_data.history(start=self.start, end=self.end)['Close'].resample('W').last()
        rf = rf.div(self.periods_per_year).div(100).mean()
        return rf

