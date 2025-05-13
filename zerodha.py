# ------------------Import Libraries -------------#
import re
import logging
from typing import List

import numpy as np
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
from yahooquery import Ticker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------------CONSTANTS------------------#
# API_KEY = ''  # Your Kite API key - ensure this is set
# CLIENT_ID = 'CCC807'  # Your client ID
TIME_LOWER_LIMIT = 15
TIME_UPPER_LIMIT = 60

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
                    (expiries > today + pd.Timedelta(days=TIME_LOWER_LIMIT)) &
                    (expiries < today + pd.Timedelta(days=TIME_UPPER_LIMIT))
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
                'category': categories
            })
            return symbols_df
        except Exception as e:
            logging.error(f"[extract_available_symbols] Failed to build symbols_df: {e}")
            return pd.DataFrame()

    def extract_options_chain(self, ticker: str) -> pd.DataFrame:
        """
        Extracts the top-5 ATM options chain for a given ticker.

        Parameters:
            ticker (str): Underlying symbol (e.g. 'NIFTY', 'BANKNIFTY', or any NSE stock code).

        Returns:
            pd.DataFrame: Merged DataFrame of call and put data for the top-5 ATM strikes.
        """
        # 1. Input validation
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("`ticker` must be a non-empty string.")

        # 2. Map to Yahoo Finance symbol and fetch spot price
        def _get_spot_price(symbol: str) -> float:
            """Return last close price via yfinance."""
            price_map = {
                'NIFTY': '^NSEI',
                'BANKNIFTY': '^NSEBANK',
                'FINNIFTY': 'NIFTY_FIN_SERVICE.NS',
                'MIDCPNIFTY': 'NIFTY_MIDCAP_100.NS',
                'NIFTYNXT50': '^NSMIDCP'
            }
            yf_symbol = price_map.get(symbol, f"{symbol}.NS")
            try:
                hist = yf.Ticker(yf_symbol).history(period="1d")
                return float(hist['Close'].iloc[0])
            except Exception as err:
                logging.error(f"Error fetching spot price for {yf_symbol}: {err}")
                raise RuntimeError(f"Failed to fetch spot price for {symbol}")

        spot_price = _get_spot_price(ticker)

        # 3. Filter available strikes and pick top-5 ATM
        df = self.options_df
        available = df[df['name'] == ticker]
        if available.empty:
            raise LookupError(f"No options data found for ticker '{ticker}'")
        strikes = available['strike'].unique()
        top_strikes = sorted(strikes, key=lambda s: abs(s - spot_price))[:11]
        strike_price_multiple = np.abs(top_strikes[1] - top_strikes[0])

        subset = available[available['strike'].isin(top_strikes)]
        calls = subset[subset['instrument_type'] == 'CE']
        puts = subset[subset['instrument_type'] == 'PE']

        # 4. Fetch live quotes from Kite
        tokens = subset['instrument_token'].astype(str).tolist()
        try:
            raw_quotes = self.kite.quote(tokens)
        except Exception as err:
            logging.error(f"Kite quote API error: {err}")
            raise RuntimeError("Failed to fetch live quotes")

        # 5. Build rows in a list for performance
        rows: List[dict] = []
        for strike in top_strikes:
            call_slice = calls[calls['strike'] == strike]
            put_slice = puts[puts['strike'] == strike]

            for df_slice, side in ((call_slice, "CE"), (put_slice, "PE")):
                for _, opt in df_slice.iterrows():
                    tok = str(opt['instrument_token'])
                    try:
                        q = raw_quotes[tok]
                        depth = q.get('depth', {})
                        ohlc = q.get('ohlc', {})
                        rows.append({
                            "instrument_type": side,
                            "underlying_last": spot_price,
                            "ticker_symbol": ticker,
                            "strike": strike,
                            "expiry": opt['expiry'],
                            f"{side.lower()}_token": tok,
                            f"{side.lower()}_volume": q.get('volume'),
                            f"{side.lower()}_bid_qty": depth['buy'][0]['quantity'],
                            f"{side.lower()}_bid_price": depth['buy'][0]['price'],
                            f"{side.lower()}_ask_price": depth['sell'][0]['price'],
                            f"{side.lower()}_ask_qty": depth['sell'][0]['quantity'],
                            f"{side.lower()}_oi": q.get('oi'),
                            f"{side.lower()}_open": ohlc.get('open'),
                            f"{side.lower()}_high": ohlc.get('high'),
                            f"{side.lower()}_low": ohlc.get('low'),
                            f"{side.lower()}_close": ohlc.get('close'),
                            f"trading_symbol_{side.lower()}": opt['tradingsymbol']
                        })
                    except (KeyError, IndexError) as e:
                        logging.warning(f"Missing data for token {tok}: {e}")
                        continue

        # 6. Final DataFrame and merge CE/PE columns
        result_df = pd.DataFrame(rows)
        # wide-format: one row per strike, CE & PE side-by-side
        try:
            # 6a. Merge the stacked versions
            ce_df = result_df[result_df['instrument_type'] == 'CE'][[
                'ticker_symbol', 'underlying_last', 'strike', 'expiry',
                'ce_token', 'ce_bid_qty', 'ce_bid_price', 'ce_ask_qty', 'ce_ask_price',
                'ce_oi', 'ce_open', 'ce_high', 'ce_low', 'ce_close', 'trading_symbol_ce'
            ]]

            pe_df = result_df[result_df['instrument_type'] == 'PE'][[
                'strike', 'expiry', 'ticker_symbol', 'underlying_last',
                'pe_token', 'pe_bid_qty', 'pe_bid_price', 'pe_ask_qty', 'pe_ask_price',
                'pe_oi', 'pe_open', 'pe_high', 'pe_low', 'pe_close', 'trading_symbol_pe'
            ]]

            # 6b) Merge on the true keys
            final_df = pd.merge(
                ce_df,
                pe_df,
                on=['strike', 'expiry', 'ticker_symbol', 'underlying_last'],
                suffixes=('_ce', '_pe'),
                how='outer'
            )
            final_df['mid_ce'] = (final_df['ce_bid_price'] + final_df['ce_ask_price']) / 2
            final_df['mid_pe'] = (final_df['pe_bid_price'] + final_df['pe_ask_price']) / 2
            final_df['bid_ask_spread_ce'] = final_df['ce_ask_price'] - final_df['ce_bid_price']
            final_df['bid_ask_spread_pe'] = final_df['pe_ask_price'] - final_df['pe_bid_price']
            final_df['bid_ask_pct_ce'] = final_df['bid_ask_spread_ce'] / final_df['mid_ce']
            final_df['bid_ask_pct_pe'] = final_df['bid_ask_spread_pe'] / final_df['mid_pe']
            final_df['atm_strike'] = strike_price_multiple * (
                round(final_df['underlying_last'] / strike_price_multiple))

            # 6c) Liquidity filters
            final_df.dropna(inplace=True)
            # final_df = final_df[
            #     (final_df['ce_oi'] >= 500) &
            #     (final_df['pe_oi'] >= 500) &
            #     (final_df['bid_ask_pct_ce'] <= 0.1) &
            #     (final_df['bid_ask_pct_pe'] <= 0.1)
            #     ].reset_index(drop=True)

            self.option_chain = final_df
            return final_df
        except Exception:
            # fallback: just return the stacked view
            final_df = result_df
            final_df.dropna(inplace=True)
            # final_df = final_df[
            #     (final_df['ce_oi'] >= 500) &
            #     (final_df['pe_oi'] >= 500)
            #     ].reset_index(drop=True)
            self.option_chain = final_df
            return final_df

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
