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

# --------------------MAIN CODE-------------------#

class ExtractOptionsData:

    def __init__(self):
        self.url_underlying = "https://www.nseindia.com/api/underlying-information"

    def extract_available_option_symbols(self, max_retries=5, delay=5):
        url = self.url_underlying
        retries = 0
        data = None
        while retries < max_retries:
            try:
                if retries <= 3:
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
                # Wait for page to load completely
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                # Extract JSON content
                json_text = driver.find_element(By.TAG_NAME, "body").text
                data = json.loads(json_text)
                print("✅ Data fetched successfully!")
                driver.quit()
                break  # Exit loop if successful
            except Exception as e:
                driver.quit()
                print(f"⚠️ Error fetching data: {e}")
                retries += 1
                time.sleep(delay)
        if not data:
            print("⚠️ Error fetching available symbols.")
            return pd.DataFrame()
        index_records = data.get("data", {}).get("IndexList", [])
        equity_records = data.get("data", {}).get("UnderlyingList", [])

        index_symbols = [(d['symbol'], d['underlying'], 'index') for d in index_records if 'symbol' in d]
        equity_symbols = [(d['symbol'], d['underlying'], 'equity') for d in equity_records if 'symbol' in d]

        # Combine both lists into a single list
        combined_symbols = index_symbols + equity_symbols

        # Create a DataFrame with three columns: 'symbol', 'underlying', and 'type'
        df_symbols = pd.DataFrame(combined_symbols, columns=['symbol', 'underlying', 'type'])

        return df_symbols

    def extracting_ohlc(self, ticker, type, **kwargs):

        if type == 'equity':
            self.ticker = ticker + '.NS'
        elif type == 'index':
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

        # Download OHLC data at 5-minute intervals
        # try:
        #     stock = yf.Ticker(self.ticker)
        #     data = stock.history(**kwargs)
        # except:
        stock = Ticker(self.ticker, asynchronous=True, retry=20, status_forcelist=[404, 429, 500, 502, 503, 504])
        data = stock.history(adj_ohlc=True, **kwargs)
        data.index = data.index.droplevel('symbol')

        return data


# -------------------- USAGE -------------------#
# if __name__ == "__main__":
#     ticker = "NIFTY"
#     type = 'index'
#     option_data_object = ExtractOptionsData()
#     nifty_ohlc_data = option_data_object.extracting_ohlc(ticker=ticker, type=type, period='60d', interval='5m')
#     print(nifty_ohlc_data.head(10))
