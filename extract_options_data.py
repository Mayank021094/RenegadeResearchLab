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
import yfinance as yf


# ---------------------CONSTANTS------------------#
# (No constants defined here, but this section can be used for future variables.)

# --------------------MAIN CODE-------------------#

class ExtractOptionsData:
    # Class to extract options data and related information.

    def __init__(self):
        # Initialize with URL for NSE underlying information.
        self.url_underlying = "https://www.nseindia.com/api/underlying-information"

    def extract_available_option_symbols(self, max_retries=4, delay=5):
        """
        Extract the list of available option symbols from NSE, including both indices and equities.
        This function first tries to fetch data using direct HTTP requests. If that fails,
        it falls back to Selenium-based extraction.

        :param max_retries: Maximum number of attempts to fetch data via each method.
        :param delay: Time (in seconds) to wait between retries.
        :return: A pandas DataFrame with columns: symbol, underlying, and category.
        """
        url = self.url_underlying
        data = None

        # ===================== Method 1: Direct HTTP Requests (Preferred) =====================
        # This block uses direct HTTP requests to fetch underlying information.
        # It mimics a regular browser by using randomized user-agent headers,
        # sets up a session to acquire cookies, and handles network errors, non-200 responses,
        # HTML responses (indicative of bot detection), and JSON decode issues.
        baseurl = 'https://www.nseindia.com/'
        user_agent_list = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
        ]
        headers = {
            "user-agent": random.choice(user_agent_list),
            "accept-language": "en,gu;q=0.9,hi;q=0.8",
            "accept-encoding": "gzip, deflate, br",
            "referer": "https://www.nseindia.com",
            "connection": "keep-alive",
            "DNT": "1",
            "sec-fetch-mode": "navigate"
        }

        session = requests.Session()

        for attempt in range(1, max_retries + 1):
            try:
                # Step 1: Hit the base URL to establish necessary cookies.
                initial_request = session.get(baseurl, headers=headers, timeout=5)
                cookies = dict(initial_request.cookies)

                # Step 2: Use the acquired cookies to request the underlying information.
                response = session.get(url, headers=headers, timeout=10, cookies=cookies)

                # Check for a non-successful HTTP status code.
                if response.status_code != 200:
                    print(f"HTTP Request - Attempt {attempt}/{max_retries} failed. Status Code: {response.status_code}")
                    time.sleep(delay)
                    continue

                # If response contains HTML, NSE might be flagging bot activity.
                if "<html>" in response.text.lower():
                    print(
                        f"HTTP Request - Attempt {attempt}/{max_retries}: Received HTML content (bot detection). Retrying...")
                    time.sleep(delay)
                    continue

                # Try parsing the JSON response.
                try:
                    data = response.json()
                    print("✅ Data fetched successfully via HTTP requests!")
                    break  # Exit the retry loop if successful
                except ValueError as json_err:
                    print(f"HTTP Request - Attempt {attempt}/{max_retries}: JSON decoding error: {json_err}")
                    print("Response snippet (first 500 chars):", response.text[:500])
                    time.sleep(delay)
            except requests.exceptions.RequestException as req_err:
                print(
                    f"HTTP Request - Attempt {attempt}/{max_retries}: Network error: {req_err}. Retrying in {delay} seconds...")
                time.sleep(delay)

        # ===================== Method 2: Selenium-Based Extraction (Fallback) =====================
        # If the HTTP method fails after all retries, fall back to Selenium.
        if not data:
            print("❌ HTTP Requests method failed. Switching to Selenium-based extraction...")
            retries = 0
            while retries <= max_retries:
                try:
                    # Use Chrome for the first few retries; if those fail, switch to Edge.
                    if retries < 3:
                        chrome_options = webdriver.ChromeOptions()
                        chrome_options.add_experimental_option('detach', True)
                        driver = webdriver.Chrome(options=chrome_options)
                    else:
                        edge_options = webdriver.EdgeOptions()
                        edge_options.use_chromium = True
                        edge_options.add_experimental_option('detach', True)
                        driver = webdriver.Edge(options=edge_options)

                    print(f"Selenium - Attempt {retries + 1}/{max_retries}: Fetching data from NSE...")
                    driver.get(url)

                    # Wait until the page's body tag is present to ensure that the page has loaded.
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )

                    # Extract JSON content from the body.
                    json_text = driver.find_element(By.TAG_NAME, "body").text
                    data = json.loads(json_text)
                    print("✅ Data fetched successfully via Selenium!")
                    driver.quit()
                    break  # Exit the loop if data is fetched successfully
                except Exception as e:
                    # Ensure the driver is closed if an error occurs.
                    try:
                        driver.quit()
                    except Exception:
                        pass
                    print(f"Selenium - Attempt {retries + 1}/{max_retries}: Error fetching data: {e}")
                    retries += 1
                    time.sleep(delay)

        # If both methods fail, return an empty DataFrame.
        if not data:
            print("⚠️ Error fetching available symbols after using both HTTP requests and Selenium.")
            return pd.DataFrame()

        # ===================== Processing the Fetched Data =====================
        # Extract index and equity records from the JSON response.
        index_records = data.get("data", {}).get("IndexList", [])
        equity_records = data.get("data", {}).get("UnderlyingList", [])

        # Build tuples of (symbol, underlying, category) filtering for valid records.
        index_symbols = [(d['symbol'], d['underlying'], 'index') for d in index_records if 'symbol' in d]
        equity_symbols = [(d['symbol'], d['underlying'], 'equity') for d in equity_records if 'symbol' in d]

        # Combine the index and equity symbols.
        combined_symbols = index_symbols + equity_symbols

        # Create a DataFrame with the extracted data.
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
        # stock = Ticker(self.ticker, asynchronous=True, retry=20, status_forcelist=[404, 429, 500, 502, 503, 504])
        # data = stock.history(adj_ohlc=True, **kwargs)
        #
        # # The returned MultiIndex often has the symbol in the first level; drop it for clarity
        # data.index = data.index.droplevel('symbol')
        stock = yf.Ticker(self.ticker)
        data = stock.history(auto_adjust=True, **kwargs)

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
                print(f"⚠️ Dividend yield information is not available for {ticker}.")
                return 0
        except Exception as e:
            # Catch any errors that occur during data retrieval.
            print(f"⚠️ An error occurred while fetching dividend yield for {ticker} : {e}")
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
