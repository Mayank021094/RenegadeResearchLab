# ------------------Import Libraries -------------#
import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma
import pandas as pd


# --------------------MAIN CODE-------------------#
class MGARCH:
    def __init__(self, dist='norm'):
        """
        Initialize the MGARCH model.

        Parameters:
        dist (str): Distribution type for the model ('norm' or 't').
        """
        if dist in ['norm', 't']:
            self.dist = dist
        else:
            raise ValueError("Invalid distribution. Use 'norm' or 't'.")

    def fit(self, returns):
        """
        Fit the MGARCH model to the provided multivariate returns.

        Parameters:
        returns (numpy.ndarray): 2D array of multivariate returns (T x N).

        Returns:
        dict: Fitted model parameters.
        """
        self.ret_df = returns
        self.returns = np.asarray(returns)
        self.T, self.N = self.returns.shape

        if self.N < 2:
            raise ValueError("Input must be multivariate (at least two columns).")

        self.mean = self.returns.mean(axis=0)
        self.returns -= self.mean

        # Initialize covariance matrix
        self.Q_bar = np.cov(self.returns.T)

        # Optimize the log-likelihood
        if self.dist == 'norm':
            res = minimize(self._loglike_norm, (0.01, 0.94), bounds=[(1e-6, 1), (1e-6, 1)])
            self.a, self.b = res.x
            return {'a': self.a, 'b': self.b, 'distribution': self.dist}
        elif self.dist == 't':
            res = minimize(self._loglike_t, (0.01, 0.94, 8),
                           bounds=[(1e-6, 1), (1e-6, 1), (3, None)])
            self.a, self.b, self.dof = res.x
            return {'a': self.a, 'b': self.b, 'dof': self.dof, 'distribution': self.dist}

    def predict(self, steps=1):
        """
        Predict the covariance matrix for a given number of steps ahead.

        Parameters:
        steps (int): Number of steps ahead to forecast.

        Returns:
        numpy.ndarray: Forecasted covariance matrix for the next step.
        """
        if not hasattr(self, 'a'):
            raise ValueError("Model must be fitted before predicting.")

        Q_t = np.copy(self.Q_bar)
        H_t = np.zeros((self.N, self.N))

        for _ in range(steps):
            residual = self.returns[-1]
            et = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t)))) @ residual
            Q_t = (1 - self.a - self.b) * self.Q_bar + self.a * np.outer(et, et) + self.b * Q_t
            R_t = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t)))) @ Q_t @ np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t))))
            H_t = np.diag(np.std(self.returns, axis=0)) @ R_t @ np.diag(np.std(self.returns, axis=0))

        cov_matrix = pd.DataFrame(H_t, index=self.ret_df.columns.tolist(), columns=self.ret_df.columns.tolist())
        return cov_matrix

    def _loglike_norm(self, params):
        """
        Log-likelihood function for MGARCH with normal distribution.

        Parameters:
        params (tuple): Model parameters (a, b).

        Returns:
        float: Negative log-likelihood value.
        """
        a, b = params
        Q_t = np.copy(self.Q_bar)
        loglike = 0

        for t in range(self.T):
            residual = self.returns[t]
            Q_t = (1 - a - b) * self.Q_bar + a * np.outer(residual, residual) + b * Q_t
            R_t = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t)))) @ Q_t @ np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t))))
            H_t = np.diag(np.std(self.returns, axis=0)) @ R_t @ np.diag(np.std(self.returns, axis=0))
            loglike += self.N * np.log(2 * np.pi) + np.log(np.linalg.det(H_t)) + residual.T @ np.linalg.inv(
                H_t) @ residual

        return 0.5 * loglike

    def _loglike_t(self, params):
        """
        Log-likelihood function for MGARCH with t-distribution.

        Parameters:
        params (tuple): Model parameters (a, b, degrees of freedom).

        Returns:
        float: Negative log-likelihood value.
        """
        a, b, dof = params
        Q_t = np.copy(self.Q_bar)
        loglike = 0

        for t in range(self.T):
            residual = self.returns[t]
            Q_t = (1 - a - b) * self.Q_bar + a * np.outer(residual, residual) + b * Q_t
            R_t = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t)))) @ Q_t @ np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t))))
            H_t = np.diag(np.std(self.returns, axis=0)) @ R_t @ np.diag(np.std(self.returns, axis=0))

            term1 = gamma((dof + self.N) / 2) / (
                        gamma(dof / 2) * (dof - 2) ** (self.N / 2) * np.sqrt(np.linalg.det(H_t)))
            term2 = (1 + residual.T @ np.linalg.inv(H_t) @ residual / (dof - 2)) ** (-(dof + self.N) / 2)
            loglike += -np.log(term1 * term2)

        return -loglike
