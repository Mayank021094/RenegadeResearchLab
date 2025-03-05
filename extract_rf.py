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


# -----------------------------Main Code----------------------------
def extract_risk_free_rate(max_retries=5, delay=5):
    # Extracts the risk-free rate (MIBOR) data from the FBIL website.
    # max_retries: Maximum number of attempts to fetch data.
    # delay: Delay (in seconds) between retries.

    url = "https://www.fbil.org.in/#/home"
    retries = 0
    data = []

    # Loop to attempt data fetching up to max_retries times
    while retries < max_retries:
        try:
            # For the first 3 retries, use Chrome; afterwards, use Edge
            if retries <= 3:
                chrome_options = webdriver.ChromeOptions()
                chrome_options.add_experimental_option('detach', True)
                driver = webdriver.Chrome(options=chrome_options)
            else:
                edge_options = webdriver.EdgeOptions()
                edge_options.use_chromium = True
                edge_options.add_experimental_option('detach', True)
                driver = webdriver.Edge(options=edge_options)

            print(f"Attempt {retries + 1}: Fetching data from FBIL...")
            driver.get(url)
            time.sleep(3)

            # Click on the "MONEY MARKET/INTEREST RATES" link
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.LINK_TEXT, "MONEY MARKET/INTEREST RATES"))
            ).click()

            # Click on the "Term MIBOR" link
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.LINK_TEXT, "Term MIBOR"))
            ).click()

            # Wait for the table with ID "termMibor" to appear
            table = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "termMibor"))
            )
            print("✅ Data fetched successfully!")

            # Find all rows in the table
            rows = table.find_elements(By.TAG_NAME, "tr")

            # Define a mapping to convert tenor text to numeric days
            tenor_mapping = {
                "14 DAYS": 14,
                "1 MONTH": 30,
                "3 MONTHS": 90
            }

            # Extract data from each row (skipping the header row)
            for row in rows[1:]:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) < 4:
                    continue  # Skip rows that don't have enough columns

                date = cols[0].text.strip()
                tenor_text = cols[1].text.strip()
                rate = cols[3].text.strip()

                # Convert tenor to numeric days if available in our mapping
                if tenor_text in tenor_mapping:
                    data.append([date, tenor_mapping[tenor_text], float(rate)])

            # Convert the extracted data to a DataFrame
            df = pd.DataFrame(data, columns=["Date", "Tenor", "MIBOR Rate (%)"])

            # Convert "Date" to a Python datetime.date object
            df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y").dt.date
            df['MIBOR Rate (%)'] = df['MIBOR Rate (%)']/100

            # Filter the DataFrame to keep only the latest date's rates (optional)
           # df_filtered = df[df["Date"] == df["Date"].max()]

            # Return the full DataFrame (df) or the filtered one as needed
            return df
        except Exception as e:
            # If an error occurs, close the driver, print the error, increment retries, and wait
            driver.quit()
            print(f"⚠️ Error: {e}")
            retries += 1
            time.sleep(delay)
        finally:
            # Ensure the driver is quit in every case (success or exception)
            if driver:
                driver.quit()

    # If all retries fail, return an empty DataFrame
    print("❌ Failed to fetch data after multiple retries.")
    return pd.DataFrame()
