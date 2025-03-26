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
GAMMA_RISK_AVERSION = 2
BROKERAGE = 0.03 * 1.18 / 100
STT = 0.02 / 100
NSE = 0.00173 / 100
STAMP = 0.002 / 100


# --------------------MAIN CODE-------------------#
class Extract_Greeks:
    def __init__(self, K, imp_vol, rf, maturity, option_category, q=0, current_date=None):
        """
        Initialize the option parameters and time to maturity.

        Parameters:
            K: Strike price.
            imp_vol: Implied volatility (annualized).
            rf: DataFrame or dict containing risk-free rate data.
                (Note: If rf values are in percentage, consider converting them by dividing by 100.)
            maturity: Option expiration date (as a datetime.date or a compatible string).
            option_category: 'CE' for call, 'PE' for put.
            q: Dividend yield (default is 0).
            current_date: Valuation date (defaults to today's date if not provided).
        """
        self.K = K
        self.imp_vol = imp_vol
        self.r = np.mean(rf['MIBOR Rate (%)'])  # Make sure these are in decimal form if needed.
        self.option_category = option_category

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

    def delta(self, S):
        """
        Calculate the option delta using the Black-Scholes formula.
        Uses t2 for drift and discounting, and t1 for volatility scaling.
        """
        # Compute d1 using t2 for the drift term and t1 for volatility.
        d1 = (np.log(S / self.K) + (self.r - self.q) * self.t2 + 0.5 * self.imp_vol ** 2 * self.t1) / \
             (self.imp_vol * np.sqrt(self.t1))

        if self.option_category == 'CE':
            # Call delta: exp(-q*t2) * N(d1)
            delta = np.exp(-self.q * self.t2) * si.norm.cdf(d1)
        elif self.option_category == 'PE':
            # Put delta: exp(-q*t2) * (N(d1) - 1) which is equivalent to -exp(-q*t2)*N(-d1)
            delta = np.exp(-self.q * self.t2) * (si.norm.cdf(d1) - 1)
        else:
            raise ValueError("option_category must be 'CE' (call) or 'PE' (put)")

        return delta

    def gamma(self, S):
        """
        Calculate the option gamma using the Black-Scholes formula.
        Gamma is identical for calls and puts.
        """
        # d1 remains the same as in delta.
        d1 = (np.log(S / self.K) + (self.r - self.q) * self.t2 + 0.5 * self.imp_vol ** 2 * self.t1) / \
             (self.imp_vol * np.sqrt(self.t1))

        # Gamma: exp(-q*t2)*phi(d1) / (S * sigma * sqrt(t1))
        gamma = (si.norm.pdf(d1) * np.exp(-self.q * self.t2)) / (S * self.imp_vol * np.sqrt(self.t1))

        return gamma

    def vega(self, S):
        """
        Calculate the option vega, which measures sensitivity to changes in implied volatility.
        Note: Here, we use t1 for the volatility scaling (consistent with d1).
        """
        d1 = (np.log(S / self.K) + (self.r - self.q) * self.t2 + 0.5 * self.imp_vol ** 2 * self.t1) / \
             (self.imp_vol * np.sqrt(self.t1))

        # Vega: S * exp(-q*t2)*phi(d1)*sqrt(t1)
        vega = S * np.exp(-self.q * self.t2) * np.sqrt(self.t1) * si.norm.pdf(d1)

        return vega

    def rho(self, S):
        """
        Calculate the option rho, which measures sensitivity to changes in the risk-free rate.
        Uses t2 for discounting.
        """
        d1 = (np.log(S / self.K) + (self.r - self.q) * self.t2 + 0.5 * self.imp_vol ** 2 * self.t1) / \
             (self.imp_vol * np.sqrt(self.t1))
        d2 = d1 - self.imp_vol * np.sqrt(self.t1)

        if self.option_category == 'CE':
            # Call rho: K * t2 * exp(-r*t2) * N(d2)
            rho = self.K * self.t2 * np.exp(-self.r * self.t2) * si.norm.cdf(d2)
        elif self.option_category == 'PE':
            # Put rho: -K * t2 * exp(-r*t2) * N(-d2)
            rho = -self.K * self.t2 * np.exp(-self.r * self.t2) * si.norm.cdf(-d2)
        else:
            raise ValueError("option_category must be 'CE' (call) or 'PE' (put)")

        return rho

    def theta(self, S):
        """
        Calculate the option theta, which measures the sensitivity of the option price
        to the passage of time (time decay).
        Uses t1 for the volatility scaling in the first term and t2 for discounting.
        """
        d1 = (np.log(S / self.K) + (self.r - self.q) * self.t2 + 0.5 * self.imp_vol ** 2 * self.t1) / \
             (self.imp_vol * np.sqrt(self.t1))
        d2 = d1 - self.imp_vol * np.sqrt(self.t1)

        # First term: time decay due to volatility
        # Use t1 in the denominator for volatility scaling and exp(-q*t2) for dividend discounting.
        term_1 = S * si.norm.pdf(d1) * self.imp_vol * np.exp(-self.q * self.t2) / (2 * np.sqrt(self.t1))

        if self.option_category == 'CE':
            # Call theta: -term_1 - rK exp(-r*t2)*N(d2) + qS exp(-q*t2)*N(d1)
            term_2 = self.q * S * si.norm.cdf(d1) * np.exp(-self.q * self.t2)
            term_3 = self.r * self.K * np.exp(-self.r * self.t2) * si.norm.cdf(d2)
            theta = -term_1 - term_3 + term_2
        elif self.option_category == 'PE':
            # Put theta: -term_1 + rK exp(-r*t2)*N(-d2) - qS exp(-q*t2)*N(-d1)
            term_2 = self.q * S * si.norm.cdf(-d1) * np.exp(-self.q * self.t2)
            term_3 = self.r * self.K * np.exp(-self.r * self.t2) * si.norm.cdf(-d2)
            theta = -term_1 + term_3 - term_2
        else:
            raise ValueError("option_category must be 'CE' (call) or 'PE' (put)")

        return theta

    def zakamouline_delta_band(self, S, spread=0, option_gamma=0):
        """
        Compute Zakamouline's hedging band boundaries using specific formulas.

        The formulas used are:
            H0 = λ / (γ * S * σ^2 * T)
            H1 = 1.12 * λ^0.31 * T^0.05 * ( e^(-rT) / σ )^0.25 * ( |Γ| / γ )^0.5
            k  = -4.76 * λ^0.78 * T^(-0.02) * ( e^(-rT)/σ )^0.25 * ( γ * S^2 * |Γ| )^0.15

        The adjusted volatility is then:
            σ_m^2 = σ^2 (1 - k)

        The Black–Scholes delta is computed using σ_m, and the hedging band is:
            lower_bound = delta(σ_m) - (H0 + H1)
            upper_bound = delta(σ_m) + (H0 + H1)

        Note:
            This function depends on several attributes and global constants:
              - self.imp_vol: Implied volatility (σ)
              - self.t1: A time parameter (likely time to expiration)
              - self.t2: Time to maturity (T) used in exponential terms
              - self.r: Risk-free rate (r)
              - self.K: Strike price (K)
              - self.q: Dividend yield (q)
              - self.option_category: Option category ('CE' for call or 'PE' for put)
              - Global constants: BROKERAGE, STT, NSE, STAMP, GAMMA_RISK_AVERSION
            Ensure these are defined and set appropriately in your code.

        Parameters:
            S          : Underlying price.
            spread     : Transaction cost spread, which is added to other fixed cost components.
            option_gamma : (Γ) Option gamma. If zero or not provided, standard BS gamma is assumed.

        Returns:
            A tuple (lower_bound, upper_bound) representing the boundaries of the hedging band.
        """
        # Combine the spread with other cost components to compute λ (transaction cost parameter)
        lam = spread + BROKERAGE + STT + NSE + STAMP

        # 1) Compute H0 using the formula: H0 = λ / (γ * S * σ^2 * t1)
        H0 = lam / (GAMMA_RISK_AVERSION * S * self.imp_vol ** 2 * self.t1)

        # 2) Compute H1 using the formula:
        #    H1 = 1.12 * λ^0.31 * t2^0.05 * ( e^(-r*t2)/σ )^0.25 * ( |Γ| / γ )^0.5
        term_1 = lam ** 0.31
        term_2 = self.t2 ** 0.05
        term_3 = (np.exp(-self.r * self.t2) / self.imp_vol) ** 0.25
        term_4 = (abs(option_gamma) / GAMMA_RISK_AVERSION) ** 0.5
        H1 = 1.12 * term_1 * term_2 * term_3 * term_4

        # 3) Compute k using the formula:
        #    k = -4.76 * λ^0.78 * t2^(-0.02) * ( e^(-r*t2)/σ )^0.25 * (γ * S^2 * |Γ| )^0.15
        k_part_1 = lam ** 0.78
        k_part_2 = self.t2 ** (-0.02)
        k_part_3 = (np.exp(-self.r * self.t2) / self.imp_vol) ** 0.25
        k_part_4 = (GAMMA_RISK_AVERSION * (S ** 2) * abs(option_gamma)) ** 0.15
        # Fixed the factor to -5.76 to match the docstring
        k_val = -4.76 * k_part_1 * k_part_2 * k_part_3 * k_part_4

        # 4) Calculate the adjusted volatility:
        #    σ_m^2 = σ^2 (1 - k)
        sigma_m_sq = self.imp_vol ** 2 * (1.0 - k_val)
        if sigma_m_sq <= 0:
            raise ValueError(f"Adjusted volatility^2 is non-positive (1 - k = {1 - k_val}). Check parameters.")
        sigma_m = np.sqrt(sigma_m_sq)

        # 5) Compute the Black–Scholes delta using adjusted volatility σ_m.
        #    d1 = [ln(S/K) + (r - q)*t2 + 0.5*σ_m^2*t1] / (σ_m * sqrt(t1))
        d1 = (np.log(S / self.K) + (self.r - self.q) * self.t2 + 0.5 * sigma_m ** 2 * self.t1) / (
                sigma_m * np.sqrt(self.t1))

        # Calculate delta based on the option category.
        if self.option_category == 'CE':
            # Call option delta: exp(-q*t2) * N(d1)
            delta_m = np.exp(-self.q * self.t2) * si.norm.cdf(d1)
        elif self.option_category == 'PE':
            # Put option delta: exp(-q*t2) * (N(d1) - 1)
            delta_m = np.exp(-self.q * self.t2) * (si.norm.cdf(d1) - 1)
        else:
            raise ValueError("option_category must be 'CE' (call) or 'PE' (put)")

        # 6) Determine the hedging band boundaries.
        # The band is defined by subtracting and adding (H0 + H1) from the delta.
        band_halfwidth = H0 + H1
        lower_bound = delta_m - band_halfwidth
        upper_bound = delta_m + band_halfwidth

        return lower_bound, upper_bound
