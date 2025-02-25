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
class Value:
    def __init__(self, strat, univ, wt_strat):
        # Use the function to create a new connection
        self.DB = get_db_connection()

        # Load the columns from the database into DataFrames
        symbol_yfinance_df = pd.read_sql_query("SELECT symbol_yfinance FROM table_symbol_yfinance", self.DB)
        symbol_nse_df = pd.read_sql_query("SELECT *  FROM table_symbol_nse", self.DB)
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
        if strat == 'Cash Flow Yield':
            scores = self.casf_flow_yield()
        elif strat == 'Book-to-Market':
            scores = self.book_to_mkt()
        elif strat == 'Dividend Yield':
            scores = self.div_yield()
        elif strat == 'Earnings Yield':
            scores = self.earnings_yield()
        elif strat == 'Sales Yield':
            scores = self.sales_yield()
        elif strat == 'PEG Ratio':
            scores = self.peg()
        elif strat == 'Composite Score':
            scores = self.composite_score()

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
        self.DB.close()

    def get_wts(self):
        return self.wts

    # def extract_line_item(self, ticker, key):
    #     try:
    #         stock = yf.Ticker(ticker)
    #         line_item = stock.info.get(key)
    #         if line_item is not None:
    #             return line_item
    #         else:
    #             return 'N/A'
    #     except Exception as e:
    #         print(f"Error fetching {key} for {ticker}: {e}")
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

#------------------------STRATEGIES-----------------------------#
    def casf_flow_yield(self):
        cfy_dict = {}
        for ticker in self.tickers:
            data = Extract_Data(ticker)
            temp = data.extract_stock_data(period='1d')
            if not temp.empty:
                price = temp['Close'].tail(1).values[0]
                data = Extract_Data(ticker)
                cash_flow = data.extract_line_item('totalCashPerShare')
                if cash_flow != 'N/A' and price > 0:
                    cash_flow_yield = cash_flow / price
                    cfy_dict[ticker] = cash_flow_yield
                else:
                    cfy_dict[ticker] = 0
        cfy_df = pd.DataFrame(list(cfy_dict.items()), columns=['symbol_yfinance', 'scores'])
        return cfy_df

    def book_to_mkt(self):
        bm_dict = {}
        for ticker in self.tickers:
            data = Extract_Data(ticker)
            bm_yield = data.extract_line_item('priceToBook')
            if bm_yield != 'N/A' and bm_yield != 0:
                bm_dict[ticker] = 1 / bm_yield
            else:
                bm_dict[ticker] = 0
        bm_df = pd.DataFrame(list(bm_dict.items()), columns=['symbol_yfinance', 'scores'])
        return bm_df

    def div_yield(self):
        dy_dict = {}
        for ticker in self.tickers:
            data = Extract_Data(ticker)
            dy = data.extract_line_item('dividendYield')
            dy_dict[ticker] = dy if dy != 'N/A' else 0
        dy_df = pd.DataFrame(list(dy_dict.items()), columns=['symbol_yfinance', 'scores'])
        return dy_df

    def earnings_yield(self):
        earnings_yield_dict = {}
        for ticker in self.tickers:
            data = Extract_Data(ticker)
            eps = data.extract_line_item('trailingEps')
            temp = data.extract_stock_data(period='1d')
            if eps != 'N/A' and not temp.empty:
                price = temp['Close'].tail(1).values[0]
                if price > 0:
                    earnings_yield_dict[ticker] = eps / price
                else:
                    earnings_yield_dict[ticker] = 0
            else:
                earnings_yield_dict[ticker] = 0
        earnings_yield_df = pd.DataFrame(list(earnings_yield_dict.items()), columns=['symbol_yfinance', 'scores'])
        return earnings_yield_df

    def sales_yield(self):
        sales_yield_dict = {}
        for ticker in self.tickers:
            data = Extract_Data(ticker)
            sales_per_share = data.extract_line_item('revenuePerShare')
            temp = data.extract_stock_data(period='1d')
            if sales_per_share != 'N/A' and not temp.empty:
                price = temp['Close'].tail(1).values[0]
                if price > 0:
                    sales_yield_dict[ticker] = sales_per_share / price
                else:
                    sales_yield_dict[ticker] = 0
            else:
                sales_yield_dict[ticker] = 0
        sales_yield_df = pd.DataFrame(list(sales_yield_dict.items()), columns=['symbol_yfinance', 'scores'])
        return sales_yield_df

    def peg(self):
        peg_dict = {}
        for ticker in self.tickers:
            data = Extract_Data(ticker)
            peg = data.extract_line_item('pegRatio')
            peg_dict[ticker] = peg if peg != 'N/A' else 0
        peg_df = pd.DataFrame(list(peg_dict.items()), columns=['symbol_yfinance', 'scores'])
        return peg_df

    def composite_score(self):
        composite_score_dict = {}

        for ticker in self.tickers:
            data = Extract_Data(ticker)
            temp = data.extract_stock_data(period='1d')

            if not temp.empty:
                # Ensure price is a scalar value
                price = float(temp['Close'].tail(1).values[0])
                # Extract line items and convert to floats where applicable
                cf = data.extract_line_item('totalCashPerShare')
                bm = data.extract_line_item('priceToBook')
                dy = data.extract_line_item('dividendYield')
                eps = data.extract_line_item('trailingEps')
                sps = data.extract_line_item('revenuePerShare')

                # Convert to floats if they aren't 'N/A' or other invalid entries
                try:
                    cf = float(cf) if cf != 'N/A' else 'N/A'
                    bm = float(bm) if bm != 'N/A' else 'N/A'
                    dy = float(dy) if dy != 'N/A' else 'N/A'
                    eps = float(eps) if eps != 'N/A' else 'N/A'
                    sps = float(sps) if sps != 'N/A' else 'N/A'
                except ValueError:
                    composite_score_dict[ticker] = 0
                    continue  # Skip to the next ticker if conversion fails

                # Calculate composite score if all values are valid
                if price > 0 and all(x != 'N/A' for x in [cf, bm, dy, eps, sps]):
                    composite = statistics.mean([cf / price, 1 / bm, dy, eps / price, sps / price])
                    composite_score_dict[ticker] = composite
                else:
                    composite_score_dict[ticker] = 0

        composite_score_df = pd.DataFrame(list(composite_score_dict.items()), columns=['symbol_yfinance', 'scores'])
        return composite_score_df
