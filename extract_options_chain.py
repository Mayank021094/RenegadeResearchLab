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
from requests.exceptions import ConnectionError, Timeout, RequestException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json


# ---------------------CONSTANTS------------------#

# --------------------MAIN CODE-------------------#
class ExtractOptionsChain:
    def __init__(self, ticker, type_):
        # Strip suffix if ticker contains "."
        self.ticker = re.sub(r"\..*", "", ticker) if "." in ticker else ticker
        self.type = type_
        self.data = None
        self.url_index = "https://www.nseindia.com/api/option-chain-indices?symbol="
        self.url_equity = "https://www.nseindia.com/api/option-chain-equities?symbol="
    def extract_option_chain(self, max_retries=5, delay=5):
        """Fetch option chain data from NSE using Selenium"""
        url = self.url_index + self.ticker if self.type == 'index' else self.url_equity + self.ticker
        retries = 0
        data = None

        while retries < max_retries:
            try:
                chrome_options = webdriver.ChromeOptions()
                chrome_options.add_experimental_option('detach', True)
                driver = webdriver.Chrome(options=chrome_options)
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
            print("❌ Failed to fetch data after multiple retries.")
        self.data = data

        # baseurl = 'https://www.nseindia.com/'
        #
        # user_agent_list = [
        #     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        #     'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
        #     'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
        #     'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
        #     'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
        #     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
        # ]
        #
        # headers = {
        #     # "user-agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        #     "user-agent": random.choice(user_agent_list),
        #     "accept-language": "en,gu;q=0.9,hi;q=0.8",
        #     "accept-encoding": "gzip, deflate, br",
        #     "referer": "https://www.nseindia.com",
        #     "connection": "keep-alive",
        #     "DNT": "1",
        #     "sec-fetch-mode": "navigate"
        # }
        #
        # session = requests.Session()
        #
        # for attempt in range(1, max_retries + 1):
        #     try:
        #         request = session.get(baseurl, headers=headers, timeout=5)
        #         cookies = dict(request.cookies)
        #         response = session.get(url, headers=headers, timeout=10, cookies=cookies)
        #
        #         if response.status_code != 200:
        #             print(f"Attempt {attempt}/{max_retries} failed. Status Code: {response.status_code}")
        #             time.sleep(delay)
        #             continue
        #
        #         # Check if response is HTML instead of JSON (bot protection)
        #         if "<html>" in response.text.lower():
        #             print("⚠️ NSE detected bot activity. Retrying...")
        #             time.sleep(delay)
        #             continue
        #
        #         # Try parsing JSON
        #         self.data = response.json()
        #         return  # Exit if successful
        #
        #     except requests.exceptions.RequestException as e:
        #         print(f"Network error: {e}. Retrying in {delay} seconds...")
        #         time.sleep(delay)
        #
        #     except ValueError:  # JSON decode error
        #         print(f"Attempt {attempt}/{max_retries} failed. Response is not JSON.")
        #         print("Response Content (First 500 chars):", response.text[:500])  # Debugging
        #         time.sleep(delay)
        #
        # print("❌ Failed to fetch options data after multiple retries.")
        # self.data = None

    def extract_call_data(self):
        """Extract call option data from NSE"""
        if not self.data:
            print("⚠️ No data available. Run extract_option_chain() first.")
            return pd.DataFrame()  # Return empty DataFrame

        records = self.data.get("records", {}).get("data", [])
        calls = [d["CE"] for d in records if "CE" in d]

        call_data = [
            (
                d.get('underlying', 'N/A'),
                d.get('strikePrice', 0),
                pd.to_datetime(d.get('expiryDate', '1970-01-01'), errors='coerce'),  # Convert expiry date
                d.get('bidQty', 0),
                d.get('bidprice', 0),
                d.get('askQty', 0),
                d.get('askPrice', 0),
                d.get('underlyingValue', 0),
            )
            for d in calls
        ]

        call_df = pd.DataFrame(call_data, columns=['symbol', 'strike', 'expiry_date', 'bid_qty', 'bid_price', 'ask_qty',
                                                   'ask_price', 'ltp'])
        call_df['S-K'] = call_df['ltp'] - call_df['strike']
        return call_df

    def extract_put_data(self):
        """Extract put option data from NSE"""
        if not self.data:
            print("⚠️ No data available. Run extract_option_chain() first.")
            return pd.DataFrame()  # Return empty DataFrame

        records = self.data.get("records", {}).get("data", [])
        puts = [d["PE"] for d in records if "PE" in d]

        put_data = [
            (
                d.get('underlying', 'N/A'),
                d.get('strikePrice', 0),
                pd.to_datetime(d.get('expiryDate', '1970-01-01'), errors='coerce'),  # Convert expiry date
                d.get('bidQty', 0),
                d.get('bidprice', 0),
                d.get('askQty', 0),
                d.get('askPrice', 0),
                d.get('underlyingValue', 0),
            )
            for d in puts
        ]

        put_df = pd.DataFrame(put_data, columns=['symbol', 'strike', 'expiry_date', 'bid_qty', 'bid_price', 'ask_qty',
                                                 'ask_price', 'ltp'])
        put_df['K-S'] = put_df['strike'] - put_df['ltp']
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
