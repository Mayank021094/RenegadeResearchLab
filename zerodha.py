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
import json
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------------CONSTANTS------------------#
# API_KEY = ''  # Your Kite API key - ensure this is set
# CLIENT_ID = 'CCC807'  # Your client ID


# --------------------MAIN CODE-------------------#
class Zerodha_Api:
    def __init__(self, api_key, api_secret_key, client_id, wait_timeout=300):
        """
        Initializes the Zerodha_Api instance with the API credentials,
        and generates a session with a configurable timeout.
        """
        # Check that the API key is provided
        self.api_key = api_key
        if not self.api_key:
            logging.error("API_KEY is not set. Please set your Kite API key.")
            raise ValueError("API_KEY is not set.")

        # Check that the client ID is provided
        self.client_id = client_id
        if not self.client_id:
            logging.error("CLIENT_ID is not set. Please set your client ID.")
            raise ValueError("CLIENT_ID is not set.")

        # Check that the API secret key is provided
        if not api_secret_key:
            logging.error("API_SECRET_KEY is not set. Please set your Kite API secret key.")
            raise ValueError("API_SECRET_KEY is not set.")

        self.wait_timeout = wait_timeout  # Timeout in seconds for waiting for the token in the URL
        self.kite = None
        self.request_token = None

        # Attempt to generate the session and catch any errors
        try:
            self.generate_session(api_secret_key)
        except Exception as e:
            logging.exception("Failed to generate session during initialization.")
            raise

    def generate_session(self, api_secret_key):
        """
        Opens the Kite login URL in a Chrome browser, waits for the request token in the URL,
        extracts it, and generates a session using the provided API secret key.
        """
        try:
            self.kite = KiteConnect(api_key=self.api_key)
        except Exception as e:
            logging.exception("Failed to create KiteConnect instance")
            raise

        # Set up Chrome options
        chrome_options = Options()
        # Uncomment below to run Chrome in headless mode if needed:
        # chrome_options.add_argument('--headless')
        try:
            driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            logging.exception("Failed to initialize Chrome WebDriver")
            raise

        try:
            login_url = self.kite.login_url()
            driver.get(login_url)
        except Exception as e:
            logging.exception("Failed to navigate to login URL")
            driver.quit()
            raise

        try:
            # Wait until the URL contains the 'request_token' parameter
            wait = WebDriverWait(driver, self.wait_timeout)
            wait.until(lambda d: "request_token=" in d.current_url)
        except Exception as e:
            logging.exception("Timeout or error waiting for request token in URL")
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
            logging.exception("Error processing URL")
            driver.quit()
            raise

        driver.quit()

        try:
            user = self.kite.generate_session(self.request_token, api_secret_key)
            self.kite.set_access_token(user["access_token"])
            logging.info("Session generated and access token set.")
        except Exception as e:
            logging.exception("Error generating session with KiteConnect")
            raise

    def extract_available_symbols(self, current_date=None):

        # 1. Fetch instruments
        try:
            fno_inst = self.kite.instruments(exchange="NFO")
        except Exception as e:
            logging.error(f"[extract_available_symbols] Kite API error: {e}")
            return pd.DataFrame()  # or raise

        # 2. Build DataFrame
        try:
            fno_df = pd.DataFrame(fno_inst)
        except Exception as e:
            logging.error(f"[extract_available_symbols] Failed to create DataFrame: {e}")
            return pd.DataFrame()

        # 3. Filter for options CE/PE
        try:
            options_df = fno_df[fno_df['instrument_type'].isin(['CE', 'PE'])]
        except KeyError as e:
            logging.error(f"[extract_available_symbols] Missing column: {e}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"[extract_available_symbols] Error filtering instrument_type: {e}")
            return pd.DataFrame()

        # 4. Filter by expiry window
        try:

            if current_date == None:
                today = pd.Timestamp.now().normalize()
            else:
                today = current_date
            # parse expiry safely (invalid parse → NaT)
            expiries = pd.to_datetime(options_df['expiry'], errors='coerce')
            mask = (
                    (expiries > today + pd.Timedelta(days=3)) &
                    (expiries < today + pd.Timedelta(days=60))
            )
            options_df = options_df[mask]
            self.options_df = options_df
        except KeyError as e:
            logging.error(f"[extract_available_symbols] Missing expiry column: {e}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"[extract_available_symbols] Error filtering by expiry: {e}")
            return pd.DataFrame()

        # 5. Extract symbols & categorize
        try:
            symbols = options_df['name'].dropna().unique()
            index_symbols = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'NIFTYNXT50']
            categories = [
                'index' if s in index_symbols else 'equity'
                for s in symbols
            ]
        except KeyError as e:
            logging.error(f"[extract_available_symbols] Missing name column: {e}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"[extract_available_symbols] Error extracting symbols: {e}")
            return pd.DataFrame()

        # 6. Build result DataFrame
        try:
            symbols_df = pd.DataFrame({
                'symbol': symbols,
                'underlying': categories
            })
            return symbols_df
        except Exception as e:
            logging.error(f"[extract_available_symbols] Failed to build symbols_df: {e}")
            return pd.DataFrame()

    def extract_options_chain(self, ticker, category):

        def f_cmp(ticker):
            if ticker == 'NIFTY':
                symbol = '^NSEI'
            elif ticker == 'BANKNIFTY':
                symbol = '^NSEBANK'
            elif ticker == 'FINNIFTY':
                symbol = 'NIFTY_FIN_SERVICE.NS'
            elif ticker == 'MIDCPNIFTY':
                symbol = 'NIFTY_MIDCAP_100.NS'
            elif ticker == 'NIFTYNXT50':
                symbol = '^NSMIDCP'
            else:
                symbol = ticker + '.NS'
            yf_obj = yf.Ticker(symbol)
            cmp = yf_obj.history(period='1d')['Close'][0]

            return cmp

        pass

    @staticmethod
    def extract_instrument_token(name, instrument_type, expiry=None, strike=None):
        """
        Extracts the instrument token from the instruments CSV by filtering based on
        the provided criteria: name, instrument type, and optionally expiry date and strike value.

        Parameters:
            name (str): The name of the instrument.
            instrument_type (str): The type of the instrument.
            expiry (datetime.date, optional): The expiry date of the instrument. Defaults to None.
            strike (float, optional): The strike price of the instrument. Defaults to None.

        Returns:
            tradingsymbol (str): The trading symbol corresponding to the instrument.
        """
        try:
            instrument_df = pd.read_csv("https://api.kite.trade/instruments", parse_dates=True)
        except Exception as e:
            logging.exception("Failed to load instruments CSV")
            raise

        # Ensure the CSV contains the necessary columns
        required_columns = {'name', 'instrument_type', 'expiry', 'strike', 'tradingsymbol'}
        if not required_columns.issubset(instrument_df.columns):
            msg = "CSV is missing one or more required columns: " + ", ".join(required_columns)
            logging.error(msg)
            raise ValueError(msg)

        try:
            # Convert the 'expiry' column to datetime.date objects.
            instrument_df['expiry'] = pd.to_datetime(instrument_df['expiry']).dt.date
        except Exception as e:
            logging.exception("Error converting expiry column to datetime.date")
            raise

        # Build the mask for filtering
        mask = (instrument_df['name'] == name) & (instrument_df['instrument_type'] == instrument_type)
        if expiry is not None:
            mask &= (instrument_df['expiry'] == expiry)
        if strike is not None:
            mask &= (instrument_df['strike'] == strike)

        filtered_df = instrument_df[mask]

        if filtered_df.empty:
            msg = "No instrument found matching the given criteria."
            logging.error(msg)
            raise ValueError(msg)

        return filtered_df.iloc[0]['tradingsymbol']

    def place_orders(self, trading_symbol='SETFNIFBK', exchange='NSE', transaction_type='BUY', quantity=1,
                     variety='regular', order_type='MARKET',
                     product='CNC', validity='DAY'):
        """
        Uses the active session (self.kite) to place orders.
        Validates the input hyperparameters before order placement.

        Hyperparameter allowed values:
        - exchange: {'NSE', 'BSE', 'NFO', 'CDS', 'BCD', 'MCX'}
        - quantity: must be numeric and greater than 0
        - variety: {'regular', 'amo', 'co', 'iceberg', 'auction'}
        - order_type: {'MARKET', 'LIMIT', 'SL', 'SL-M'}
        - product: {'CNC', 'NRML', 'MIS', 'MTF'}
        - validity: {'DAY', 'IOC', 'TTL'}
        """
        if not self.kite:
            logging.error("Session is not initialized. Cannot place orders.")
            raise ValueError("Session is not initialized")

        # Validate 'exchange'
        allowed_exchanges = {"NSE", "BSE", "NFO", "CDS", "BCD", "MCX"}
        if exchange not in allowed_exchanges:
            msg = f"Invalid exchange: '{exchange}'. Accepted values are: {allowed_exchanges}"
            logging.error(msg)
            raise ValueError(msg)

        # Validate 'quantity'
        if not isinstance(quantity, (int, float)) or quantity <= 0:
            msg = f"Invalid quantity: '{quantity}'. Quantity must be a positive number."
            logging.error(msg)
            raise ValueError(msg)

        allowed_varieties = {"regular", "amo", "co", "iceberg", "auction"}
        allowed_order_types = {"MARKET", "LIMIT", "SL", "SL-M"}
        allowed_products = {"CNC", "NRML", "MIS", "MTF"}
        allowed_validity = {"DAY", "IOC", "TTL"}

        if variety not in allowed_varieties:
            msg = f"Invalid variety: '{variety}'. Accepted values are: {allowed_varieties}"
            logging.error(msg)
            raise ValueError(msg)

        if order_type not in allowed_order_types:
            msg = f"Invalid order_type: '{order_type}'. Accepted values are: {allowed_order_types}"
            logging.error(msg)
            raise ValueError(msg)

        if product not in allowed_products:
            msg = f"Invalid product: '{product}'. Accepted values are: {allowed_products}"
            logging.error(msg)
            raise ValueError(msg)

        if validity not in allowed_validity:
            msg = f"Invalid validity: '{validity}'. Accepted values are: {allowed_validity}"
            logging.error(msg)
            raise ValueError(msg)

        try:
            order_id = self.kite.place_order(
                tradingsymbol=trading_symbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=quantity,
                variety=variety,
                order_type=order_type,
                product=product,
                validity=validity
            )
            logging.info("Order placed. ID is: {}".format(order_id))
            return order_id
        except Exception as e:
            logging.exception("Order placement failed")
            raise

    def order_history(self):
        """
        Retrieves the order history from the Kite session.

        Returns:
            orders (list/dict): The order history data as returned by KiteConnect.
        """
        if not self.kite:
            logging.error("Session is not initialized. Cannot retrieve order history.")
            raise ValueError("Session is not initialized")
        try:
            orders = self.kite.orders()
            logging.info("Order history retrieved successfully.")
            return orders
        except Exception as e:
            logging.exception("Failed to retrieve order history")
            raise

    def cancel_order(self, order_id, variety='regular'):
        """
        Cancels an order given its order ID.

        Parameters:
            order_id (str): The ID of the order to cancel.
            variety (str): The variety of the order (default 'regular').

        Returns:
            cancel_response (dict): The response from the cancel order API call.
        """
        if not self.kite:
            logging.error("Session is not initialized. Cannot cancel order.")
            raise ValueError("Session is not initialized")
        try:
            cancel_response = self.kite.cancel_order(order_id=order_id, variety=variety)
            logging.info("Order cancellation response: {}".format(cancel_response))
            return cancel_response
        except Exception as e:
            logging.exception("Order cancellation failed")
            raise
