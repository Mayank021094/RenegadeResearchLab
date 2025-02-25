# ------------------Import Libraries -------------#
import pandas as pd
import arch
import numpy as np
from numpy.linalg import LinAlgError
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from scipy.stats import probplot, moment
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import t, expon
import pymc as pm
import warnings
import statistics
import arviz as az
warnings.filterwarnings('ignore')

# ---------------------CONSTANTS------------------#

# --------------------MAIN CODE-------------------#

class Expected_Mean:

    def __init__(self, prices_df):
        self.prices = prices_df
        self.returns = np.log(self.prices).diff().dropna()

    def arima(self):
        train_size = int(0.5 * len(self.returns))
        ret_predictions = {}

        for column in self.returns.columns:
            returns_series = self.returns[column].mul(100)
            results = {}

            # Step 1: Evaluate all (p, q) combinations
            for p in range(1, 3):
                for q in range(1, 3):
                    aic, bic = [], []
                    convergence_error = 0
                    stationarity_error = 0
                    y_pred = []

                    # Rolling window forecast
                    for T in range(train_size, len(returns_series)):
                        train_set = returns_series[T - train_size: T]

                        # Ensure train_set is valid
                        if len(train_set) < 2:
                            print(f"Insufficient data for p={p}, q={q} at T={T}")
                            continue

                        try:
                            # Fit ARIMA model
                            model = ARIMA(endog=train_set, order=(p, 0, q)).fit()
                            forecast = model.forecast(steps=1)  # Get one-step forecast
                            y_pred.append(forecast[0])  # Append forecasted value
                            aic.append(model.aic)  # Append AIC
                            bic.append(model.bic)  # Append BIC
                            #print(y_pred[-1])
                        except LinAlgError:
                            # Handle convergence error
                            convergence_error += 1
                            print(f"LinAlgError for p={p}, q={q} at T={T}")
                            continue
                        except ValueError:
                            # Handle stationarity error
                            stationarity_error += 1
                            print(f"ValueError for p={p}, q={q} at T={T}")
                            continue

                    # Skip if no predictions were made
                    if not y_pred:
                        print(f"No predictions for p={p}, q={q}. Skipping.")
                        continue

                    # Align y_true with y_pred
                    y_true_series = returns_series.iloc[train_size:train_size + len(y_pred)]
                    print(pd.DataFrame({'y_pred':y_pred, 'y_true':y_true_series}))
                    # Calculate evaluation metrics
                    rmse = np.sqrt(mean_squared_error(y_true=y_true_series, y_pred=y_pred))
                    mean_aic = np.mean(aic) if aic else np.inf
                    mean_bic = np.mean(bic) if bic else np.inf

                    # Store results
                    results[(p, q)] = {
                        'rmse': rmse,  # Root Mean Squared Error
                        'aic': mean_aic,  # Average AIC
                        'bic': mean_bic,  # Average BIC
                        'convergence_error': convergence_error,  # Count of convergence errors
                        'stationarity_error': stationarity_error  # Count of stationarity errors
                    }

            print(pd.DataFrame.from_dict(results))
            # Step 2: Select the best (p, q) combination
            # Criteria: Minimize RMSE, AIC, BIC, and avoid errors
            best_pq = min(
                results.keys(),
                key=lambda k: (
                    results[k]['rmse'],  # Prioritize RMSE
                    results[k]['aic'],  # Then prioritize AIC
                    results[k]['bic'],  # Then prioritize BIC
                    results[k]['convergence_error'],  # Minimize convergence errors
                    results[k]['stationarity_error']  # Minimize stationarity errors
                )
            )
            best_p, best_q = best_pq

            # Step 3: Refit the model with the best (p, q) on the entire series
            best_model = ARIMA(returns_series, order=(best_p, 0, best_q)).fit()

            # Step 4: Forecast expected returns
            forecast = best_model.forecast(steps=len(self.returns) - train_size)

            # Step 5: Store the forecasted returns
            ret_predictions[column] = forecast.values[0]

        # Create a DataFrame from the predictions
        self.exp_ret_df = pd.DataFrame.from_dict(ret_predictions, orient='index', columns=['expected_returns'])

        # Return the expected returns DataFrame
        return self.exp_ret_df

    def bayesian_dist(self):
        """
        Bayesian modeling of returns using a Student's T-distribution.

        This method estimates the prior distributions for mean, standard deviation,
        and degrees of freedom (nu), fits an exponential distribution to nu values,
        and constructs a Bayesian model.
        """
        # Step 1: Estimate mean and standard deviation priors from historical returns
        mean_prior = self.returns.mean()  # Prior for mean of returns
        std_prior = self.returns.std()  # Prior for standard deviation of returns

        # Step 2: Calculate rolling estimates of degrees of freedom (nu) and std deviation
        window_size = 52  # Use a rolling window of 52 data points (e.g., 1 year of weekly data)
        nu_values = []
        sigma_values = []

        for i in range(len(self.returns) - window_size):
            # Extract a rolling window of returns
            window = self.returns[i:i + window_size]

            # Fit a Student's T-distribution to the window and collect nu
            t_params = t.fit(window)
            nu_values.append(t_params[0])  # Degrees of freedom

            # Collect standard deviation of the window
            sigma_values.append(window.std()[0])

        # Step 3: Remove outliers from nu_values
        nu_values = np.array(nu_values)
        q1 = np.percentile(nu_values, 25)  # First quartile
        q3 = np.percentile(nu_values, 75)  # Third quartile
        iqr = q3 - q1  # Interquartile range

        # Define lower and upper bounds for outliers
        lower_bound = q1 - 2 * iqr
        upper_bound = q3 + 2 * iqr

        # Filter out nu outliers
        cleaned_nu_values = nu_values[(nu_values >= lower_bound) & (nu_values <= upper_bound)]

        # Step 4: Fit an Exponential distribution to the cleaned nu values
        beta, loc = expon.fit(cleaned_nu_values)  # Ensure loc=0 for meaningful scale parameter

        # Step 5: Construct the Bayesian model using PyMC
        with pm.Model() as expected_returns_model:
            # Prior for mean: Normal distribution with mean_prior and std_prior
            mean = pm.Normal('mean', mu=mean_prior, sigma=std_prior)

            # Prior for standard deviation: Half-Normal with standard deviation of rolling sigmas
            std = pm.HalfNormal('std', sigma=statistics.stdev(sigma_values))

            # Prior for degrees of freedom (nu - 2): Exponential with rate 1 / beta
            nu_minus_two = pm.Exponential('nu_minus_two', scale=beta)
            nu = pm.Deterministic('nu', nu_minus_two + 2)  # Ensure nu > 2 for validity in T-distribution

            # Likelihood: Student's T-distribution for returns
            returns = pm.StudentT('returns', nu=nu, mu=mean, sigma=std, observed=self.returns)

            # Sampling from the posterior
            trace = pm.sample(2000, return_inferencedata=True)  # Adjust sample size as needed

            # Extract posterior samples for the mean
            mean_posterior_samples = trace.posterior['mean'].values

            # Compute the expected mean (posterior mean)
            expected_mean = np.mean(mean_posterior_samples)

            print(expected_mean)











