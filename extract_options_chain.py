# ------------------Import Libraries ------------#
import numpy as np
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
from requests.exceptions import ConnectionError, Timeout, RequestException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import datetime

# ---------------------CONSTANTS------------------#
MIN_OPEN_INTEREST = 500  # Minimum open interest for liquidity
MAX_SPREAD_PCT = 0.10  # Maximum allowable bid-ask spread as a fraction of mid-price (10%)


# --------------------MAIN CODE-------------------#
class ExtractOptionsChain:
    """
    A class to extract option chain data from NSE (India) using Selenium/HTTP.

    Workflow:
      1. __init__() calls extract_option_chain() to fetch data.
      2. Once the data is fetched, an internal function prepares and splits the data:
         - It merges raw call and put data (from the JSON) on ('strikePrice', 'expiryDate'),
         - It removes rows where either the call or put open interest is below MIN_OPEN_INTEREST,
         - It then splits the merged data into call and put DataFrames.
      3. These are stored in self.call_final and self.put_final.
      4. extract_call_data() and extract_put_data() then return these DataFrames.

    Importantly, the final output DataFrames use the same column names as in the original code.
    """

    def __init__(self, ticker, category):
        """
        Initialize the ExtractOptionsChain object.
        :param ticker: Symbol of the security (e.g., 'BANKNIFTY' or 'RELIANCE')
        :param category: 'index' or 'equity'
        """
        # Remove any suffix after a dot (e.g., ".NS")
        self.ticker = re.sub(r"\..*", "", ticker) if "." in ticker else ticker
        self.category = category
        self.data = None
        # self.call_final = pd.DataFrame()
        # self.put_final = pd.DataFrame()

        self.url_index = "https://www.nseindia.com/api/option-chain-indices?symbol="
        self.url_equity = "https://www.nseindia.com/api/option-chain-equities?symbol="

        # Automatically fetch the option chain data and process it.
        try:
            self.extract_option_chain()
        except Exception as e:
            print(f"Error during initialization: {e}")






#----------------------------------------Using Web Scrapping, and API calls.--------------------------
"""
Use this method when we don't have subscription to any of the broker APIs
"""

    # def extract_option_chain(self, max_retries=5, delay=5):
    #     """
    #     Fetch the option chain data via HTTP requests first; if that fails, use Selenium.
    #     After successful fetch, the data is processed (merging and filtering) to populate
    #     self.call_final and self.put_final.
    #     """
    #     url = self.url_index + self.ticker if self.category == 'index' else self.url_equity + self.ticker
    #     baseurl = 'https://www.nseindia.com/'
    #
    #     # -------------------- Method 1: HTTP Requests --------------------
    #     user_agent_list = [
    #         'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    #         'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
    #         'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
    #         'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
    #         'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
    #         'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
    #     ]
    #     headers = {
    #         "user-agent": random.choice(user_agent_list),
    #         "accept-language": "en,gu;q=0.9,hi;q=0.8",
    #         "accept-encoding": "gzip, deflate, br",
    #         "referer": "https://www.nseindia.com",
    #         "connection": "keep-alive",
    #         "DNT": "1",
    #         "sec-fetch-mode": "navigate"
    #     }
    #     session = requests.Session()
    #
    #     for attempt in range(1, max_retries + 1):
    #         try:
    #             initial_request = session.get(baseurl, headers=headers, timeout=5)
    #             cookies = dict(initial_request.cookies)
    #             response = session.get(url, headers=headers, timeout=10, cookies=cookies)
    #             if response.status_code != 200:
    #                 print(f"HTTP attempt {attempt}/{max_retries} failed (Status: {response.status_code}). Retrying...")
    #                 time.sleep(delay)
    #                 continue
    #             if "<html>" in response.text.lower():
    #                 print(f"HTTP attempt {attempt}/{max_retries}: HTML detected (possible bot detection). Retrying...")
    #                 time.sleep(delay)
    #                 continue
    #             try:
    #                 self.data = response.json()
    #                 print("✅ Data fetched via HTTP requests.")
    #                 break
    #             except ValueError as json_err:
    #                 print(f"HTTP attempt {attempt}/{max_retries}: JSON decoding error: {json_err}. Retrying...")
    #                 time.sleep(delay)
    #         except requests.exceptions.RequestException as req_err:
    #             print(f"HTTP attempt {attempt}/{max_retries}: {req_err}. Retrying in {delay} seconds...")
    #             time.sleep(delay)
    #     else:
    #         # -------------------- Method 2: Selenium-based Extraction --------------------
    #         print("❌ HTTP method failed. Switching to Selenium...")
    #         retries = 0
    #         while retries < max_retries:
    #             try:
    #                 if retries <= 3:
    #                     chrome_options = webdriver.ChromeOptions()
    #                     chrome_options.add_experimental_option('detach', True)
    #                     driver = webdriver.Chrome(options=chrome_options)
    #                 else:
    #                     edge_options = webdriver.EdgeOptions()
    #                     edge_options.use_chromium = True
    #                     edge_options.add_experimental_option('detach', True)
    #                     driver = webdriver.Edge(options=edge_options)
    #                 print(f"Selenium attempt {retries + 1}/{max_retries}: Fetching data...")
    #                 driver.get(url)
    #                 WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    #                 json_text = driver.find_element(By.TAG_NAME, "body").text
    #                 self.data = json.loads(json_text)
    #                 print("✅ Data fetched via Selenium.")
    #                 driver.quit()
    #                 break
    #             except Exception as e:
    #                 try:
    #                     driver.quit()
    #                 except Exception:
    #                     pass
    #                 print(f"Selenium attempt {retries + 1}/{max_retries} failed: {e}")
    #                 retries += 1
    #                 time.sleep(delay)
    #         if self.data is None:
    #             print("❌ Failed to fetch options data using both methods.")
    #
    #     # Automatically process and split the data if it has been fetched.
    #     if self.data:
    #         self.prepare_and_split_data()
    #
    # def prepare_and_split_data(self):
    #     """
    #     Process self.data to:
    #       1. Extract raw call and put data.
    #       2. Merge them on ('strikePrice', 'expiryDate') using suffixes.
    #       3. Remove rows where either side's open interest is below MIN_OPEN_INTEREST.
    #       4. Save the resulting call and put DataFrames (with original column names) to
    #          self.call_final and self.put_final.
    #     """
    #     records = self.data.get("records", {}).get("data", [])
    #     call_data = []
    #     put_data = []
    #
    #     # Loop through each record and extract raw call (CE) and put (PE) data.
    #     for d in records:
    #         if "CE" in d:
    #             ce = d["CE"]
    #             call_data.append((
    #                 ce.get('underlying', 'N/A'),
    #                 ce.get('openInterest', 0),
    #                 ce.get('strikePrice', 0),
    #                 pd.to_datetime(ce.get('expiryDate', '1970-01-01'), errors='coerce').date(),
    #                 ce.get('bidQty', 0),
    #                 ce.get('bidprice', 0),
    #                 ce.get('askQty', 0),
    #                 ce.get('askPrice', 0),
    #                 ce.get('underlyingValue', 0)
    #             ))
    #         if "PE" in d:
    #             pe = d["PE"]
    #             put_data.append((
    #                 pe.get('underlying', 'N/A'),
    #                 pe.get('openInterest', 0),
    #                 pe.get('strikePrice', 0),
    #                 pd.to_datetime(pe.get('expiryDate', '1970-01-01'), errors='coerce').date(),
    #                 pe.get('bidQty', 0),
    #                 pe.get('bidprice', 0),
    #                 pe.get('askQty', 0),
    #                 pe.get('askPrice', 0),
    #                 pe.get('underlyingValue', 0)
    #             ))
    #
    #     # Create raw DataFrames for calls and puts using the original column names.
    #     call_df_raw = pd.DataFrame(call_data, columns=[
    #         'symbol', 'openInterest', 'strikePrice', 'expiryDate',
    #         'bid_qty', 'bid_price', 'ask_qty', 'ask_price', 'ltp'
    #     ])
    #     put_df_raw = pd.DataFrame(put_data, columns=[
    #         'symbol', 'openInterest', 'strikePrice', 'expiryDate',
    #         'bid_qty', 'bid_price', 'ask_qty', 'ask_price', 'ltp'
    #     ])
    #
    #     # Merge the DataFrames on strikePrice and expiryDate.
    #     # Suffix _call and _put help us distinguish the two sides temporarily.
    #     merged_df = pd.merge(call_df_raw, put_df_raw, on=['strikePrice', 'expiryDate'], suffixes=('_call', '_put'))
    #
    #     # Filter out rows where either side's open interest is below MIN_OPEN_INTEREST.
    #     merged_df = merged_df[
    #         (merged_df['openInterest_call'] >= MIN_OPEN_INTEREST) &
    #         (merged_df['openInterest_put'] >= MIN_OPEN_INTEREST)
    #         ]
    #
    #     # Prepare the final call and put DataFrames with original column names.
    #     self.call_final = self._prepare_call_df(merged_df)
    #     self.put_final = self._prepare_put_df(merged_df)
    #
    # def _prepare_call_df(self, df):
    #     """
    #     From the merged DataFrame, extract and compute the call side fields,
    #     then rename them to match the original column names.
    #     """
    #     # Select call side columns (from raw data, with suffix _call) and the common fields.
    #     call_df = df[['symbol_call', 'openInterest_call', 'strikePrice', 'expiryDate',
    #                   'bid_qty_call', 'bid_price_call', 'ask_qty_call', 'ask_price_call', 'ltp_call']].copy()
    #     # Rename columns to the original names.
    #     call_df = call_df.rename(columns={
    #         'symbol_call': 'symbol',
    #         'openInterest_call': 'openInterest',
    #         'bid_qty_call': 'bid_qty',
    #         'bid_price_call': 'bid_price',
    #         'ask_qty_call': 'ask_qty',
    #         'ask_price_call': 'ask_price',
    #         'ltp_call': 'ltp'
    #     })
    #
    #     # Compute additional fields.
    #     call_df['mkt_price'] = (call_df['bid_price'] + call_df['ask_price']) / 2
    #     call_df['S-K'] = np.abs(call_df['ltp'] - call_df['strikePrice'])
    #     current_date = np.datetime64(datetime.date.today(), 'D')
    #     expiry_np = call_df['expiryDate'].values.astype('datetime64[D]')
    #     call_df['calendar_days_to_maturity'] = np.busday_count(current_date, expiry_np, weekmask='1111111')
    #     return call_df
    #
    # def _prepare_put_df(self, df):
    #     """
    #     From the merged DataFrame, extract and compute the put side fields,
    #     then rename them to match the original column names.
    #     """
    #     # Select put side columns (from raw data, with suffix _put) and the common fields.
    #     put_df = df[['symbol_put', 'openInterest_put', 'strikePrice', 'expiryDate',
    #                  'bid_qty_put', 'bid_price_put', 'ask_qty_put', 'ask_price_put', 'ltp_put']].copy()
    #     # Rename columns to the original names.
    #     put_df = put_df.rename(columns={
    #         'symbol_put': 'symbol',
    #         'openInterest_put': 'openInterest',
    #         'bid_qty_put': 'bid_qty',
    #         'bid_price_put': 'bid_price',
    #         'ask_qty_put': 'ask_qty',
    #         'ask_price_put': 'ask_price',
    #         'ltp_put': 'ltp'
    #     })
    #
    #     # Compute additional fields.
    #     put_df['mkt_price'] = (put_df['bid_price'] + put_df['ask_price']) / 2
    #     put_df['K-S'] = np.abs(put_df['strikePrice'] - put_df['ltp'])
    #     current_date = np.datetime64(datetime.date.today(), 'D')
    #     expiry_np = put_df['expiryDate'].values.astype('datetime64[D]')
    #     put_df['calendar_days_to_maturity'] = np.busday_count(current_date, expiry_np, weekmask='1111111')
    #     return put_df
    #
    # def extract_call_data(self):
    #     """
    #     Return the preprocessed call DataFrame with original column names.
    #     """
    #     if self.call_final.empty:
    #         print("Call data is empty. Check if data was fetched correctly.")
    #     return self.call_final
    #
    # def extract_put_data(self):
    #     """
    #     Return the preprocessed put DataFrame with original column names.
    #     """
    #     if self.put_final.empty:
    #         print("Put data is empty. Check if data was fetched correctly.")
    #     return self.put_final


# ------------------ Example Usage ------------------#
# if __name__ == "__main__":
#     # Instantiate the extractor (e.g., using 'BANKNIFTY' for an index)
#     extractor = ExtractOptionsChain(ticker='BANKNIFTY', category='index')
#
#     # Extract the call and put data (these functions simply return the processed data).
#     call_options = extractor.extract_call_data()
#     put_options = extractor.extract_put_data()
#
#     print("Filtered & Matched Call Options:")
#     print(call_options.head())
#
#     print("\nFiltered & Matched Put Options:")
#     print(put_options.head())
