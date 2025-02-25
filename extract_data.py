# ------------------Import Libraries ------------#
import yfinance as yf
import pandas as pd


# --------------------MAIN CODE-------------------#
class Extract_Data:
    def __init__(self, ticker):
        self.ticker = ticker
        
    def extract_market_cap(self):
        try:
            stock = yf.Ticker(self.ticker)
            market_cap = stock.info.get('marketCap')
            return market_cap if market_cap else 'N/A'
        except Exception as e:
            print(f"Error fetching market capitalization for {self.ticker}: {e}")
            return 'N/A'
    
    def extract_line_item(self, key):
        try:
            stock = yf.Ticker(self.ticker)
            line_item = stock.info.get(key)
            return line_item if line_item is not None else 'N/A'
        except Exception as e:
            print(f"Error fetching {key} for {self.ticker}: {e}")
            return 'N/A'
        
    def extract_balance_sheet_item(self, key):
        try:
            stock = yf.Ticker(self.ticker)
            bs = stock.balance_sheet
            if key in bs.index and not bs.empty:
                line_item = bs.loc[key].iloc[0]
                return line_item if line_item is not None else 'N/A'
            else:
                return 'N/A'
        except Exception as e:
            print(f"Error fetching {key} for {self.ticker}: {e}")
            return 'N/A'
    
    def extract_pnl_item(self, key):
        try:
            stock = yf.Ticker(self.ticker)
            income_statement = stock.financials
            if key in income_statement.index and not income_statement.empty:
                line_item = income_statement.loc[key].iloc[0]
                return line_item if line_item is not None else 'N/A'
            else:
                return 'N/A'
        except Exception as e:
            print(f"Error fetching {key} for {self.ticker}: {e}")
            return 'N/A'
    
    def extract_stock_data(self, **kwargs):
        try:
            stock = yf.Ticker(self.ticker)
            stock_data = stock.history(**kwargs)
            if stock_data.empty:
                raise ValueError(f"No data found for {self.ticker}")
            return stock_data
        except Exception as e:
            print(f"Error fetching stock data for {self.ticker}: {e}")
            return pd.DataFrame()
    