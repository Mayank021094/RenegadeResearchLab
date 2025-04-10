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
# (No explicit constants defined here, but this section can be used for them if needed.)

# --------------------MAIN CODE-------------------#
class ExtractOptionsChain:
    """
    A class to extract option chain data from NSE (India) using Selenium.
    It fetches JSON data for either an index or equity symbol and then
    provides methods to parse call and put data into DataFrames.
    """

    def __init__(self, ticker, category):
        """
        Initialize the ExtractOptionsChain object with a given ticker and category.

        :param ticker: Symbol of the security (e.g., 'BANKNIFTY' or 'RELIANCE')
        :param category: 'index' or 'equity', indicating the NSE URL to be used
        """
        # Remove any suffix after a dot (e.g., ".NS") from the ticker if present
        self.ticker = re.sub(r"\..*", "", ticker) if "." in ticker else ticker
        self.category = category
        self.data = None

        # Two different NSE endpoints for indices vs. equities
        self.url_index = "https://www.nseindia.com/api/option-chain-indices?symbol="
        self.url_equity = "https://www.nseindia.com/api/option-chain-equities?symbol="

        # Automatically fetch the option chain data upon instantiation.
        try:
            self.extract_option_chain()
        except Exception as e:
            print(f"Error during initialization when fetching data: {e}")

    def extract_option_chain(self, max_retries=5, delay=5):
        """
        Fetch the option chain data from NSE by first trying HTTP requests,
        and falling back to Selenium if necessary.

        This function uses two methods:
        1. Direct HTTP Requests (preferred method)
        2. Selenium-based extraction (fallback method)

        If both methods fail after multiple attempts, self.data is set to None.
        """
        # Construct the URL based on whether the category is 'index' or 'equity'
        url = self.url_index + self.ticker if self.category == 'index' else self.url_equity + self.ticker
        baseurl = 'https://www.nseindia.com/'

        # ===================== Method 1: Direct HTTP Requests (Preferred) =====================
        # Prepare a list of user agent strings to bypass basic bot protection
        user_agent_list = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
        ]

        # Randomize headers to simulate different browser requests
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

        # Attempt multiple retries using HTTP requests
        for attempt in range(1, max_retries + 1):
            try:
                # Access the base URL to set cookies
                initial_request = session.get(baseurl, headers=headers, timeout=5)
                cookies = dict(initial_request.cookies)

                # Fetch the option chain data using the cookies
                response = session.get(url, headers=headers, timeout=10, cookies=cookies)

                # Check for successful HTTP response
                if response.status_code != 200:
                    print(f"HTTP Request - Attempt {attempt}/{max_retries} failed. Status Code: {response.status_code}")
                    time.sleep(delay)
                    continue

                # If the response contains HTML, it may indicate bot detection
                if "<html>" in response.text.lower():
                    print(
                        f"HTTP Request - Attempt {attempt}/{max_retries}: Received HTML content, likely due to bot detection. Retrying...")
                    time.sleep(delay)
                    continue

                # Try to parse the response as JSON
                try:
                    self.data = response.json()
                    print("✅ Data fetched successfully via HTTP requests!")
                    return  # Exit the function if successful
                except ValueError as json_err:
                    print(f"HTTP Request - Attempt {attempt}/{max_retries}: JSON decoding error: {json_err}")
                    print("Response snippet (first 500 characters):", response.text[:500])
                    time.sleep(delay)

            except requests.exceptions.RequestException as req_err:
                print(
                    f"HTTP Request - Attempt {attempt}/{max_retries}: Network error: {req_err}. Retrying in {delay} seconds...")
                time.sleep(delay)

        print("❌ HTTP Requests method failed. Switching to Selenium-based extraction...")

        # ===================== Method 2: Selenium-Based Extraction (Fallback) =====================
        # Initialize retries and the data variable
        retries = 0
        data = None

        # Attempt multiple retries using Selenium
        while retries < max_retries:
            try:
                # For the first few retries, use Chrome; if those fail, switch to Edge
                if retries <= 3:
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

                # Wait for the body element to appear, indicating the page has loaded
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )

                # Retrieve the page's text, which is expected to be JSON data
                json_text = driver.find_element(By.TAG_NAME, "body").text
                data = json.loads(json_text)
                print("✅ Data fetched successfully via Selenium!")
                driver.quit()
                self.data = data
                return  # Exit function if successful

            except Exception as e:
                # Ensure that the driver is closed before retrying
                try:
                    driver.quit()
                except Exception:
                    pass
                print(f"Selenium - Attempt {retries + 1}/{max_retries}: Error fetching data: {e}")
                retries += 1
                time.sleep(delay)

        # If both methods fail, set self.data to None and log the failure
        print("❌ Failed to fetch options data after multiple attempts using both HTTP requests and Selenium.")
        self.data = None

    def extract_call_data(self):
        """
        Extract call option data from the fetched JSON (self.data).

        :return: A DataFrame containing call option details
        """
        # Check if self.data has been set by extract_option_chain()
        if not self.data:
            print("⚠️ No data available. Run extract_option_chain() first.")
            return pd.DataFrame()

        # Locate records under "records" -> "data" in the JSON
        records = self.data.get("records", {}).get("data", [])

        # Filter out only those entries that have a 'CE' key (Call Option data)
        calls = [d["CE"] for d in records if "CE" in d]

        # Create a list of tuples containing relevant data for each call option
        call_data = [
            (
                d.get('underlying', 'N/A'),  # Symbol
                d.get('strikePrice', 0),  # Strike Price
                pd.to_datetime(d.get('expiryDate', '1970-01-01'), errors='coerce').date(),  # Expiry Date
                d.get('bidQty', 0),  # Bid Quantity
                d.get('bidprice', 0),  # Bid Price
                d.get('askQty', 0),  # Ask Quantity
                d.get('askPrice', 0),  # Ask Price
                d.get('underlyingValue', 0),  # underlying's current price
            )
            for d in calls
        ]

        # Convert the list of tuples into a DataFrame
        call_df = pd.DataFrame(call_data, columns=[
            'symbol', 'strikePrice', 'expiryDate', 'bid_qty', 'bid_price',
            'ask_qty', 'ask_price', 'ltp'
        ])
        # Create a column 'S-K' as the absolute difference between 'ltp' and 'strike'
        call_df['S-K'] = np.abs(call_df['ltp'] - call_df['strikePrice'])
        # Get the current date and convert it to numpy.datetime64 with daily resolution
        current_date = np.datetime64(datetime.date.today(), 'D')
        # Convert the 'expiry_date' column to a numpy array of datetime64[D]
        expiry_dates_np = call_df['expiryDate'].values.astype('datetime64[D]')
        # Compute the number of business days (with all days considered as business days using weekmask='1111111')
        call_df['calendar_days_to_maturity'] = np.busday_count(current_date, expiry_dates_np, weekmask='1111111')
        call_df = call_df[(call_df['calendar_days_to_maturity'] >= 30) & (call_df['calendar_days_to_maturity'] <= 60)]
        call_df['mkt_price'] = (call_df['bid_price'] + call_df['ask_price'])/2
        call_df = call_df.sort_values(by='S-K')
        return call_df

    def extract_put_data(self):
        """
        Extract put option data from the fetched JSON (self.data).

        :return: A DataFrame containing put option details
        """
        # Check if self.data has been set by extract_option_chain()
        if not self.data:
            print("⚠️ No data available. Run extract_option_chain() first.")
            return pd.DataFrame()

        # Locate records under "records" -> "data" in the JSON
        records = self.data.get("records", {}).get("data", [])

        # Filter out only those entries that have a 'PE' key (Put Option data)
        puts = [d["PE"] for d in records if "PE" in d]

        # Create a list of tuples containing relevant data for each put option
        put_data = [
            (
                d.get('underlying', 'N/A'),  # Symbol
                d.get('strikePrice', 0),  # Strike Price
                pd.to_datetime(d.get('expiryDate', '1970-01-01'), errors='coerce').date(),  # Expiry Date
                d.get('bidQty', 0),  # Bid Quantity
                d.get('bidprice', 0),  # Bid Price
                d.get('askQty', 0),  # Ask Quantity
                d.get('askPrice', 0),  # Ask Price
                d.get('underlyingValue', 0),  # Possibly the underlying's current price, not the option's LTP
            )
            for d in puts
        ]

        # Convert the list of tuples into a DataFrame
        put_df = pd.DataFrame(put_data, columns=[
            'symbol', 'strikePrice', 'expiryDate', 'bid_qty', 'bid_price',
            'ask_qty', 'ask_price', 'ltp'
        ])

        # Create a column 'K-S' as the absolute difference between 'strike' and 'ltp'
        put_df['K-S'] = np.abs(put_df['strikePrice'] - put_df['ltp'])

        # Get the current date and convert it to numpy.datetime64 with daily resolution
        current_date = np.datetime64(datetime.date.today(), 'D')
        # Convert the 'expiry_date' column to a numpy array of datetime64[D]
        expiry_dates_np = put_df['expiryDate'].values.astype('datetime64[D]')
        # Compute the number of business days (with all days considered as business days using weekmask='1111111')
        put_df['calendar_days_to_maturity'] = np.busday_count(current_date, expiry_dates_np, weekmask='1111111')
        put_df = put_df[(put_df['calendar_days_to_maturity'] >= 30) & (put_df['calendar_days_to_maturity'] <= 60)]
        put_df['mkt_price'] = (put_df['bid_price'] + put_df['ask_price']) / 2
        put_df = put_df.sort_values(by='K-S')
        return put_df

# -------------------- USAGE -------------------#
# if __name__ == "__main__":
#     ticker = "NIFTY"
#     option_chain = ExtractOptionsChain(ticker, 'index')
#
#     print("Fetching options data...")
#     option_chain.extract_option_chain()
#
#     print("\nCALL OPTIONS:")
#     call_df = option_chain.extract_call_data()
#     print(call_df.head())
#
#     print("\nPUT OPTIONS:")
#     put_df = option_chain.extract_put_data()
#     print(put_df.head())
