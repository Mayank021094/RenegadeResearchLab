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
import datetime as dt
from weighting_strategy import equi_wt, cap_wt
import statistics
from extract_data import Extract_Data


# ---------------------CONSTANTS------------------#
def get_db_connection():
    return sqlite3.connect("./raw_datasets/head_database.db")


# --------------------MAIN CODE-------------------#
class Quality:
    def __init__(self, strat, univ, wt_strat):
        # Use the function to create a new connection
        self.DB = get_db_connection()

        # Load the columns from the database into DataFrames
        symbol_yfinance_df = pd.read_sql_query("SELECT symbol_yfinance FROM table_symbol_yfinance", self.DB)
        symbol_nse_df = pd.read_sql_query("SELECT * FROM table_symbol_nse", self.DB)
        # Combine both columns into a single DataFrame assuming index positions are the mapping
        symbol_mapping = pd.concat([symbol_yfinance_df, symbol_nse_df], axis=1)

        # Store Universe
        query = "SELECT * FROM table_symbol_yfinance"
        symbols = pd.read_sql_query(query, self.DB)
        if univ == 'All':
            self.tickers = symbols['symbol_yfinance'].tolist()
        elif univ == 'Nifty 50':
            flag = symbol_nse_df['nifty_50'].tolist()
            self.tickers = symbols['symbol_yfinance'][pd.Series(flag).astype(bool)].tolist()
        elif univ == 'Nifty 500':
            self.tickers = symbols['symbol_yfinance'].head(500).tolist()
        elif univ == 'Nifty Next 50':
            pass

        # Calculate Scores based on strategy
        if strat == 'Asset Turnover Ratio':
            scores = self.asset_turnover()
        elif strat == 'Current Ratio':
            scores = self.current_ratio()
        elif strat == 'Interest Coverage Ratio':
            scores = self.interest_coverage()
        elif strat == 'Leverage':
            scores = self.leverage()
            scores = scores.sort_values(by='scores', ascending=True)
        elif strat == 'Payout Ratio':
            scores = self.payout_ratio()
        elif strat == 'ROE':
            scores = self.roe()

        if strat == 'Leverage':
            pass
        else:
            scores = scores.sort_values(by='scores', ascending=False)
        num_rows = int(len(scores) * 0.2)
        scores = scores.iloc[:num_rows]

        # Calculate Weights based on weighting strategy
        if wt_strat == 'equal_wt':
            self.wts = equi_wt(scores, symbol_mapping)
        elif wt_strat == 'mkt_cap':
            symbol_list = scores['symbol_yfinance'].tolist()
            mkt_cap = [Extract_Data(ticker).extract_line_item('marketCap') for ticker in symbol_list]
            scores['mkt_cap'] = mkt_cap
            self.wts = cap_wt(scores, symbol_mapping)

    def __del__(self):
        # Close the connection when the object is deleted
        if hasattr(self, 'DB') and self.DB:
            self.DB.close()

    def get_wts(self):
        return self.wts

    # def extract_line_item(self, ticker, key):
    #     try:
    #         stock = yf.Ticker(ticker)
    #         line_item = stock.info.get(key)
    #         return line_item if line_item is not None else 'N/A'
    #     except Exception as e:
    #         print(f"Error fetching {key} for {ticker}: {e}")
    #         return 'N/A'
    #
    # def extract_balance_sheet_item(self, ticker, key):
    #     try:
    #         stock = yf.Ticker(ticker)
    #         bs = stock.balance_sheet
    #         if key in bs.index and not bs.empty:
    #             line_item = bs.loc[key].iloc[0]
    #             return line_item if line_item is not None else 'N/A'
    #         else:
    #             return 'N/A'
    #     except Exception as e:
    #         print(f"Error fetching {key} for {ticker}: {e}")
    #         return 'N/A'

    # def extract_pnl_item(self, ticker, key):
    #     try:
    #         stock = yf.Ticker(ticker)
    #         income_statement = stock.financials
    #         if key in income_statement.index and not income_statement.empty:
    #             line_item = income_statement.loc[key].iloc[0]
    #             return line_item if line_item is not None else 'N/A'
    #         else:
    #             return 'N/A'
    #     except Exception as e:
    #         print(f"Error fetching {key} for {ticker}: {e}")
    #         return 'N/A'
    #
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
#----------------------STRATEGY--------------------#
    def asset_turnover(self):
        print("Inside Asset Turnover")
        dict_asset_turnover = {}
        for ticker in self.tickers:
            data = Extract_Data(ticker)
            rev = data.extract_pnl_item('Total Revenue')
            asset = data.extract_balance_sheet_item('Total Assets')
            if rev != 'N/A' and asset != 'N/A' and asset != 0:
                dict_asset_turnover[ticker] = rev / asset
            else:
                dict_asset_turnover[ticker] = 0
        df_asset_turnover = pd.DataFrame(list(dict_asset_turnover.items()), columns=['symbol_yfinance', 'scores'])
        return df_asset_turnover

    def current_ratio(self):
        dict_cr = {}
        for ticker in self.tickers:
            data = Extract_Data(ticker)
            curr_assets = data.extract_balance_sheet_item('Current Assets')
            curr_debt = data.extract_balance_sheet_item('Current Liabilities')
            if curr_assets != 'N/A' and curr_debt != 'N/A' and curr_debt != 0:
                dict_cr[ticker] = curr_assets / curr_debt
            else:
                dict_cr[ticker] = 0
        df_cr = pd.DataFrame(list(dict_cr.items()), columns=['symbol_yfinance', 'scores'])
        return df_cr

    def interest_coverage(self):
        dict_ic = {}
        for ticker in self.tickers:
            data = Extract_Data(ticker)
            ebit = data.extract_pnl_item('EBIT')
            interest_expense = data.extract_pnl_item('Interest Expense')
            if ebit != 'N/A' and interest_expense != 'N/A' and interest_expense != 0:
                dict_ic[ticker] = ebit / interest_expense
            else:
                dict_ic[ticker] = 0
        df_ic = pd.DataFrame(list(dict_ic.items()), columns=['symbol_yfinance', 'scores'])
        return df_ic

    def leverage(self):
        dict_leverage = {}
        for ticker in self.tickers:
            data = Extract_Data(ticker)
            leverage = data.extract_line_item('debtToEquity')
            dict_leverage[ticker] = leverage if leverage != 'N/A' else 2000
        df_leverage = pd.DataFrame(list(dict_leverage.items()), columns=['symbol_yfinance', 'scores'])
        print(df_leverage)
        return df_leverage

    def payout_ratio(self):
        dict_payout = {}
        for ticker in self.tickers:
            data = Extract_Data(ticker)
            payout = data.extract_line_item('payoutRatio')
            dict_payout[ticker] = payout if payout != 'N/A' else 0
        df_payout = pd.DataFrame(list(dict_payout.items()), columns=['symbol_yfinance', 'scores'])
        return df_payout

    def roe(self):
        dict_roe = {}
        for ticker in self.tickers:
            data = Extract_Data(ticker)
            net_income = data.extract_pnl_item('Net Income')
            equity = data.extract_balance_sheet_item('Total Stockholder Equity')  # Ensure field names match
            if net_income != 'N/A' and equity != 'N/A' and equity != 0:
                dict_roe[ticker] = net_income / equity
            else:
                dict_roe[ticker] = 0
        df_roe = pd.DataFrame(list(dict_roe.items()), columns=['symbol_yfinance', 'scores'])
        return df_roe
