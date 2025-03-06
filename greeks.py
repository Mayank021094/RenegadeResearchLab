# ------------------Import Libraries -------------#
from arch import arch_model
import pandas as pd
import numpy as np
import math
from extract_options_data import ExtractOptionsData  # Note: Duplicate imports removed
from scipy.optimize import minimize, brentq
import scipy.optimize as opt
from scipy.stats import norm
import scipy.stats as si
import warnings
from scipy.interpolate import CubicSpline
from extract_options_chain import ExtractOptionsChain
from extract_rf import extract_risk_free_rate

warnings.filterwarnings('ignore')
import datetime

# ---------------------CONSTANTS------------------#
DEFAULT_TRADING_PERIODS = 252  # Typical number of trading days in a year
DEFAULT_WINDOW = 30  # Default rolling window size for volatility calculations


# --------------------MAIN CODE-------------------#
class Extract_Greeks:
    def __init__(self, S, K, imp_vol, rf, maturity, option_type, q=0, current_date=None):
        """
        Initialize the option parameters and time to maturity.

        Parameters:
            S: Underlying asset price.
            K: Strike price.
            imp_vol: Implied volatility (annualized).
            rf: DataFrame or dict containing risk-free rate data.
                (Note: If rf values are in percentage, consider converting them by dividing by 100.)
            maturity: Option expiration date (as a datetime.date or a compatible string).
            option_type: 'CE' for call, 'PE' for put.
            q: Dividend yield (default is 0).
            current_date: Valuation date (defaults to today's date if not provided).
        """
        self.S = S
        self.K = K
        self.imp_vol = imp_vol
        self.r = np.mean(rf['MIBOR Rate (%)'])  # Make sure these are in decimal form if needed.
        self.option_type = option_type

        if current_date is None:
            current_date = datetime.date.today()

        # Calculate time to maturity:
        # t1: trading days converted to years (used for volatility scaling).
        trading_days = np.busday_count(current_date, maturity)
        self.t1 = trading_days / DEFAULT_TRADING_PERIODS

        # t2: calendar days converted to years (used for discounting).
        calendar_days = np.busday_count(current_date, maturity, weekmask='1111111')
        self.t2 = calendar_days / 365

        self.q = q

    def delta(self):
        """
        Calculate the option delta using the Black-Scholes formula.
        Uses t2 for drift and discounting, and t1 for volatility scaling.
        """
        # Compute d1 using t2 for the drift term and t1 for volatility.
        d1 = (np.log(self.S / self.K) + (self.r - self.q) * self.t2 + 0.5 * self.imp_vol ** 2 * self.t1) / \
             (self.imp_vol * np.sqrt(self.t1))

        if self.option_type == 'CE':
            # Call delta: exp(-q*t2) * N(d1)
            delta = np.exp(-self.q * self.t2) * si.norm.cdf(d1)
        elif self.option_type == 'PE':
            # Put delta: exp(-q*t2) * (N(d1) - 1) which is equivalent to -exp(-q*t2)*N(-d1)
            delta = np.exp(-self.q * self.t2) * (si.norm.cdf(d1) - 1)
        else:
            raise ValueError("option_type must be 'CE' (call) or 'PE' (put)")

        return delta

    def gamma(self):
        """
        Calculate the option gamma using the Black-Scholes formula.
        Gamma is identical for calls and puts.
        """
        # d1 remains the same as in delta.
        d1 = (np.log(self.S / self.K) + (self.r - self.q) * self.t2 + 0.5 * self.imp_vol ** 2 * self.t1) / \
             (self.imp_vol * np.sqrt(self.t1))

        # Gamma: exp(-q*t2)*phi(d1) / (S * sigma * sqrt(t1))
        gamma = (si.norm.pdf(d1) * np.exp(-self.q * self.t2)) / (self.S * self.imp_vol * np.sqrt(self.t1))

        return gamma

    def vega(self):
        """
        Calculate the option vega, which measures sensitivity to changes in implied volatility.
        Note: Here, we use t1 for the volatility scaling (consistent with d1).
        """
        d1 = (np.log(self.S / self.K) + (self.r - self.q) * self.t2 + 0.5 * self.imp_vol ** 2 * self.t1) / \
             (self.imp_vol * np.sqrt(self.t1))

        # Vega: S * exp(-q*t2)*phi(d1)*sqrt(t1)
        vega = self.S * np.exp(-self.q * self.t2) * np.sqrt(self.t1) * si.norm.pdf(d1)

        return vega

    def rho(self):
        """
        Calculate the option rho, which measures sensitivity to changes in the risk-free rate.
        Uses t2 for discounting.
        """
        d1 = (np.log(self.S / self.K) + (self.r - self.q) * self.t2 + 0.5 * self.imp_vol ** 2 * self.t1) / \
             (self.imp_vol * np.sqrt(self.t1))
        d2 = d1 - self.imp_vol * np.sqrt(self.t1)

        if self.option_type == 'CE':
            # Call rho: K * t2 * exp(-r*t2) * N(d2)
            rho = self.K * self.t2 * np.exp(-self.r * self.t2) * si.norm.cdf(d2)
        elif self.option_type == 'PE':
            # Put rho: -K * t2 * exp(-r*t2) * N(-d2)
            rho = -self.K * self.t2 * np.exp(-self.r * self.t2) * si.norm.cdf(-d2)
        else:
            raise ValueError("option_type must be 'CE' (call) or 'PE' (put)")

        return rho

    def theta(self):
        """
        Calculate the option theta, which measures the sensitivity of the option price
        to the passage of time (time decay).
        Uses t1 for the volatility scaling in the first term and t2 for discounting.
        """
        d1 = (np.log(self.S / self.K) + (self.r - self.q) * self.t2 + 0.5 * self.imp_vol ** 2 * self.t1) / \
             (self.imp_vol * np.sqrt(self.t1))
        d2 = d1 - self.imp_vol * np.sqrt(self.t1)

        # First term: time decay due to volatility
        # Use t1 in the denominator for volatility scaling and exp(-q*t2) for dividend discounting.
        term_1 = self.S * si.norm.pdf(d1) * self.imp_vol * np.exp(-self.q * self.t2) / (2 * np.sqrt(self.t1))

        if self.option_type == 'CE':
            # Call theta: -term_1 - rK exp(-r*t2)*N(d2) + qS exp(-q*t2)*N(d1)
            term_2 = self.q * self.S * si.norm.cdf(d1) * np.exp(-self.q * self.t2)
            term_3 = self.r * self.K * np.exp(-self.r * self.t2) * si.norm.cdf(d2)
            theta = -term_1 - term_3 + term_2
        elif self.option_type == 'PE':
            # Put theta: -term_1 + rK exp(-r*t2)*N(-d2) - qS exp(-q*t2)*N(-d1)
            term_2 = self.q * self.S * si.norm.cdf(-d1) * np.exp(-self.q * self.t2)
            term_3 = self.r * self.K * np.exp(-self.r * self.t2) * si.norm.cdf(-d2)
            theta = -term_1 + term_3 - term_2
        else:
            raise ValueError("option_type must be 'CE' (call) or 'PE' (put)")

        return theta
