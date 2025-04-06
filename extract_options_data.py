# ------------------Import Libraries ------------#
import yfinance as yf
import pandas as pd
import requests
import re
import random
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import datetime as dt
import datetime
from requests.exceptions import ConnectionError, Timeout, RequestException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
from yahooquery import Ticker


# ---------------------CONSTANTS------------------#
# (No constants defined here, but this section can be used for future variables.)

# --------------------MAIN CODE-------------------#

class ExtractOptionsData:
    # Class to extract options data and related information.

    def __init__(self):
        # Initialize with URL for NSE underlying information.
        self.url_underlying = "https://www.nseindia.com/api/underlying-information"

    def extract_available_option_symbols(self, max_retries=5, delay=5):
        # Extracts the list of available option symbols from NSE, including indices and equities.
        # max_retries: Maximum number of attempts to fetch data.
        # delay: Delay (in seconds) between retries.

        url = self.url_underlying
        retries = 0
        data = None

        # Loop to attempt data fetching up to max_retries times
        while retries <= max_retries:
            try:
                # For the first 3 retries, use Chrome; afterwards, use Edge
                if retries < 3:
                    chrome_options = webdriver.ChromeOptions()
                    chrome_options.add_experimental_option('detach', True)
                    driver = webdriver.Chrome(options=chrome_options)
                else:
                    edge_options = webdriver.EdgeOptions()
                    edge_options.use_chromium = True
                    edge_options.add_experimental_option('detach', True)
                    driver = webdriver.Edge(options=edge_options)

                print(f"Attempt {retries + 1}: Fetching data from NSE...")
                driver.get(url)

                # Wait until the body tag is present to ensure page is loaded
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )

                # Extract JSON content from the body of the page
                json_text = driver.find_element(By.TAG_NAME, "body").text
                data = json.loads(json_text)
                print("✅ Data fetched successfully!")
                driver.quit()
                break  # Exit the loop if data fetched successfully
            except Exception as e:
                # If an error occurs, close the driver, print the error, increment retries, and wait
                driver.quit()
                print(f"⚠️ Error fetching data: {e}")
                retries += 1
                time.sleep(delay)

        # If data remains None after all retries, return an empty DataFrame
        if not data:
            print("⚠️ Error fetching available symbols.")
            return pd.DataFrame()

        # Extract index and equity records from the JSON response
        index_records = data.get("data", {}).get("IndexList", [])
        equity_records = data.get("data", {}).get("UnderlyingList", [])

        # Build tuples of (symbol, underlying, category)
        index_symbols = [(d['symbol'], d['underlying'], 'index') for d in index_records if 'symbol' in d]
        equity_symbols = [(d['symbol'], d['underlying'], 'equity') for d in equity_records if 'symbol' in d]

        # Combine index and equity symbols
        combined_symbols = index_symbols + equity_symbols

        # Create a DataFrame with columns: symbol, underlying, and category
        df_symbols = pd.DataFrame(combined_symbols, columns=['symbol', 'underlying', 'category'])

        return df_symbols

    def extracting_ohlc(self, ticker, category, **kwargs):
        # Extracts OHLC data for the given ticker (equity or index) at a specified interval.

        # Map the ticker to the appropriate Yahoo Finance symbol
        if category == 'equity':
            self.ticker = ticker + '.NS'
        elif category == 'index':
            if ticker == 'NIFTY':
                self.ticker = '^NSEI'
            elif ticker == 'BANKNIFTY':
                self.ticker = '^NSEBANK'
            elif ticker == 'FINNIFTY':
                self.ticker = 'NIFTY_FIN_SERVICE.NS'
            elif ticker == 'MIDCPNIFTY':
                self.ticker = 'NIFTY_MIDCAP_100.NS'
            elif ticker == 'NIFTYNXT50':
                self.ticker = '^NSMIDCP'
        else:
            print("⚠️ Error fetching available symbols.")

        # Download OHLC data (here using yahooquery's Ticker instead of yfinance directly)
        # The code is set to handle adjustments and attempt multiple retries if needed.
        stock = Ticker(self.ticker, asynchronous=True, retry=20, status_forcelist=[404, 429, 500, 502, 503, 504])
        data = stock.history(adj_ohlc=True, **kwargs)

        # The returned MultiIndex often has the symbol in the first level; drop it for clarity
        data.index = data.index.droplevel('symbol')
        data.index = data.index.map(lambda x: x.date() if hasattr(x, 'date') else x)
        return data
    @staticmethod
    def extracting_dividend_yield(ticker, category, **kwargs):
        """
        Extracts the dividend yield for a given ticker based on its asset category.

        Parameters:
            ticker (str): The base ticker symbol.
            category (str): The category of asset ('equity' or 'index').
            kwargs: Additional keyword arguments (currently not used).

        Returns:
            float or None: The dividend yield as a decimal (e.g., 0.03 for 3%) if available,
                           otherwise None if data is missing or an error occurs.
        """
        # Determine the correct ticker format based on asset category.
        if category == 'equity':
            # Append the '.NS' suffix for equities traded on the NSE.
            ticker = ticker + '.NS'
        elif category == 'index':
            # Mapping for known index tickers.
            index_mapping = {
                'NIFTY': '^NSEI',
                'BANKNIFTY': '^NSEBANK',
                'FINNIFTY': 'NIFTY_FIN_SERVICE.NS',
                'MIDCPNIFTY': 'NIFTY_MIDCAP_100.NS',
                'NIFTYNXT50': '^NSMIDCP'
            }
            if ticker in index_mapping:
                ticker = index_mapping[ticker]
            else:
                # If the provided index ticker is not recognized, print an error and exit.
                print("⚠️ Error: Unknown index ticker provided.")
                return 0
        else:
            # For an unknown asset category, print an error and exit.
            print("⚠️ Error: Unknown asset category provided.")
            return 0

        # Create a yfinance Ticker object.
        stock = yf.Ticker(ticker)

        try:
            # Try to retrieve the dividend yield from the stock info.
            dividend_yield = stock.info.get('dividendYield')
            if dividend_yield is not None:
                # dividend yield is given as a percentage, therefore converting it to a decimal.
                return dividend_yield / 100.0
            else:
                # Inform the user if dividend yield data is not available.
                print("⚠️ Dividend yield information is not available for this ticker.")
                return 0
        except Exception as e:
            # Catch any errors that occur during data retrieval.
            print(f"⚠️ An error occurred while fetching dividend yield: {e}")
            return 0




# -------------------- USAGE -------------------#
# if __name__ == "__main__":
#     ticker = "NIFTY"
#     category = 'index'
#     option_data_object = ExtractOptionsData()
#     nifty_ohlc_data = option_data_object.extracting_ohlc(ticker=ticker, category=category, period='60d', interval='5m')
#     print(nifty_ohlc_data.head(10))
# -----------Check risk-free rate code----------
# if __name__ == "__main__":
#     ticker = "NIFTY"
#     option_chain = ExtractOptionsChain(ticker, 'index')
#     rf = option_chain.extract_risk_free_rate()
#     print(rf)
