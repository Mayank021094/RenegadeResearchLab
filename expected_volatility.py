# ------------------Import Libraries -------------#
from arch import arch_model
from mgarch import MGARCH
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from sklearn.metrics import mean_squared_error

# ---------------------CONSTANTS------------------#


# --------------------MAIN CODE-------------------#

class Expected_Volatility:
    def __init__(self, prices_df):

        self.prices = prices_df
        # Initialize Returns
        self.returns = np.log(self.prices).diff().dropna()
        self.vol = None
        self.cov = None
        # Test if the returns series is stationary
        test_results = self.test_unit_root(self.returns)

    def mgarch(self):

        model = MGARCH(dist='t')
        fitted_parameters = model.fit(self.returns.mul(100))
        self.cov = model.predict()
        return self.cov

    def garch(self):
        trainsize = int(0.5 * len(self.returns))
        T = len(self.returns)
        vol_predictions = {}
        for column in self.returns.columns:
            returns_series = self.returns[column].mul(100)
            results = {}
            for p in range(2, 4):
                for q in range(1, 3):
                    result = []
                    for s, t in enumerate(range(trainsize, T - 1)):
                        train_set = returns_series.iloc[s: t]
                        test_set = returns_series.iloc[t + 1]  # 1-step ahead forecast
                        model = arch_model(y=train_set, vol="GARCH", p=p, q=q, dist='t').fit(disp='off')
                        forecast = model.forecast(horizon=1)
                        mu = forecast.mean.iloc[-1, 0]
                        var = forecast.variance.iloc[-1, 0]
                        result.append([(test_set - mu) ** 2, var])
                    df = pd.DataFrame(result, columns=['y_true', 'y_pred'])
                    results[(p, q)] = np.sqrt(mean_squared_error(df.y_true, df.y_pred))
            # Find the best p, q combination with the minimum RMSE
            best_p_q = min(results, key=results.get)  # This gets the (p, q) with the minimum RMSE
            best_p, best_q = best_p_q
            # print(results)

            # Use the best p, q values to fit the final GARCH model
            train_set = returns_series[:-1]
            model = arch_model(y=train_set, vol='GARCH', p=best_p, q=best_q, dist='t').fit(disp='off')
            forecast = model.forecast(horizon=1)
            # Extract predicted variance (volatility) from the final model
            vol_predictions[column] = np.sqrt(forecast.variance.iloc[-1, 0])
            # print(column, ':', np.sqrt(forecast.variance.iloc[-1, 0]))
        self.vol = pd.DataFrame.from_dict(vol_predictions, orient='index', columns=['expected_volatility'])
        return self.vol

    # ----------------------Test to check Stationary Time Series-----------#
    def test_unit_root(self, df):
        return df.apply(lambda x: f'{pd.Series(adfuller(x)).iloc[1]:.2%}').to_frame('p-value')
