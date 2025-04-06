# ------------------Import Libraries -------------#
import re
import logging
import pandas as pd
from kiteconnect import KiteConnect, KiteTicker
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from requests.exceptions import ConnectionError, Timeout, RequestException
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------CONSTANTS------------------#
API_KEY = ''  # Your Kite API key
CLIENT_ID = 'CCC807'  # Your client ID


# --------------------MAIN CODE-------------------#
class Zerodha_Api:
    def __init__(self, api_secret_key, wait_timeout=300):
        """
        Initializes the Zerodha_Api instance with the API credentials,
        and generates a session with a configurable timeout.
        """
        self.api_key = API_KEY
        self.client_id = CLIENT_ID
        self.wait_timeout = wait_timeout  # Timeout in seconds for waiting the token in the URL
        self.kite = None
        self.request_token = None
        self.generate_session(api_secret_key)
    def generate_session(self, api_secret_key):
        """
        Opens the Kite login URL in a Chrome browser, waits for the request token in the URL,
        extracts it, and generates a session using the provided API secret key.
        """
        try:
            self.kite = KiteConnect(api_key=self.api_key)
        except Exception as e:
            logging.error("Failed to create KiteConnect instance: %s", e)
            raise

        # Set up Chrome options (no detach option so the browser will close automatically)
        chrome_options = Options()

        try:
            driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            logging.error("Failed to initialize Chrome WebDriver: %s", e)
            raise

        try:
            driver.get(self.kite.login_url())
        except Exception as e:
            logging.error("Failed to navigate to login URL: %s", e)
            driver.quit()
            raise

        try:
            # Wait until the URL contains the 'request_token' parameter
            wait = WebDriverWait(driver, self.wait_timeout)
            wait.until(lambda d: "request_token=" in d.current_url)
        except Exception as e:
            logging.error("Timeout or error waiting for request token in URL: %s", e)
            driver.quit()
            raise

        try:
            url = driver.current_url
            pattern = r"request_token=([^&]+)"
            match = re.search(pattern, url)
            if match:
                self.request_token = match.group(1)
                logging.info("Request token extracted: %s", self.request_token)
            else:
                logging.error("Request token not found in URL.")
                driver.quit()
                raise ValueError("Request token not found in URL")
        except Exception as e:
            logging.error("Error processing URL: %s", e)
            driver.quit()
            raise

        driver.quit()

        try:
            user = self.kite.generate_session(self.request_token, api_secret_key)
            self.kite.set_access_token(user["access_token"])
            logging.info("Session generated and access token set.")
        except Exception as e:
            logging.error("Error generating session with KiteConnect: %s", e)
            raise
    def extract_instrument_token(self, name, instrument_type, expiry, strike):
        """
        Extracts the instrument token from the instruments CSV by filtering based on
        the provided criteria: name, instrument type, expiry date, and strike value.

        Parameters:
            name (str): The name of the instrument.
            instrument_type (str): The type of the instrument.
            expiry (datetime.date): The expiry date of the instrument.
            strike (float): The strike price of the instrument.
        """
        # Load instrument data from the CSV URL.
        # The parse_dates parameter ensures that date columns are parsed automatically.
        instrument_df = pd.read_csv("https://api.kite.trade/instruments", parse_dates=True)

        # Convert the 'expiry' column to datetime.date objects.
        instrument_df['expiry'] = pd.to_datetime(instrument_df['expiry']).dt.date

        # Filter the DataFrame to select the row that matches the provided criteria.
        filtered_df = instrument_df[
            (instrument_df['name'] == name) &
            (instrument_df['instrument_type'] == instrument_type) &
            (instrument_df['expiry'] == expiry) &
            (instrument_df['strike'] == strike)
            ]

        # Check if any matching instrument is found.
        if filtered_df.empty:
            raise ValueError("No instrument found matching the given criteria.")

        # Extract the instrument token from the first matching row.
        # Using .iloc[0] ensures that we get the first match safely.
        self.instrument_token = filtered_df.iloc[0]['instrument_token']
        self.trading_symbol = filtered_df.iloc[0]['tradingsymbol']
    def place_orders(self):
        """
        Uses the active session (self.kite) to place orders.
        Uncomment and modify the sample code below as needed.
        """
        if not self.kite:
            logging.error("Session is not initialized. Cannot place orders.")
            raise ValueError("Session is not initialized")

        # Sample order placement code (customize parameters as needed)
        # try:
        #     order_id = self.kite.place_order(
        #         variety=self.kite.VARIETY_REGULAR,
        #         exchange="NSE",
        #         tradingsymbol="RELIANCE",
        #         transaction_type=self.kite.TRANSACTION_TYPE_BUY,
        #         quantity=1,
        #         product=self.kite.PRODUCT_CNC,
        #         order_type=self.kite.ORDER_TYPE_MARKET
        #     )
        #     logging.info("Order placed successfully. Order ID: %s", order_id)
        # except Exception as e:
        #     logging.error("Error placing order: %s", e)
        #     raise
        pass


if __name__ == "__main__":
    try:
        # Replace 'your_api_secret' with your actual API secret key.
        api = Zerodha_Api(api_secret_key="your_api_secret")
        # api.place_orders()  # Uncomment this line to place orders after session creation.
    except Exception as e:
        logging.error("Error in Zerodha_Api: %s", e)
