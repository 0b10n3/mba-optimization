from datetime import datetime
from typing import List
from scipy.optimize import minimize

from portfolio import Portfolio, PortfolioStrategy, DataProvider


import numpy as np

class OneOverNStrategy(PortfolioStrategy):


    @property
    def name(self) -> str:
        return "1/N Portfolio (Equal Weights)"

    def calculate(self, date: datetime, assets: List[str], data_provider: DataProvider) -> Portfolio:
        num_assets = len(assets)
        if num_assets == 0:
            return Portfolio(date=date, strategy_name=self.name, weights={})

        weight = 1.0 / num_assets
        weights = {asset: weight for asset in assets}

        return Portfolio(date=date, strategy_name=self.name, weights=weights)


class MarkowitzStrategy(PortfolioStrategy):
    """
    Implements the classical Mean-Variance Optimization to find the portfolio
    with the maximum Sharpe Ratio (the tangency portfolio).
    """

    def __init__(self, lookback_days: int = 252, risk_free_rate: float = 0.02):
        self._lookback_days = lookback_days
        self._risk_free_rate = risk_free_rate

    @property
    def name(self) -> str:
        return f"Markowitz MVO (Lookback: {self._lookback_days}d)"

    def calculate(self, date: datetime, assets: List[str], data_provider: DataProvider) -> Portfolio:
        # 1. Fetch historical data for the available assets
        price_history = data_provider.get_price_history(assets, date, self._lookback_days)

        # After fetching, the assets we can actually use are the columns of the dataframe
        # This handles cases where some assets had no data and were dropped.
        valid_assets = price_history.columns.tolist()

        if len(valid_assets) < 2:  # Need at least 2 assets for covariance
            raise ValueError("Not enough assets with complete historical data for MVO.")

        # 2. Calculate expected returns and covariance matrix
        log_returns = np.log(price_history / price_history.shift(1)).dropna()
        if log_returns.empty:
            raise ValueError("Could not compute valid returns from price history.")

        # Annualize returns and covariance
        mu = log_returns.mean() * 252
        cov_matrix = log_returns.cov() * 252
        num_assets = len(valid_assets)

        # 3. Define the objective function (negative Sharpe Ratio)
        def negative_sharpe_ratio(weights):
            p_return = np.sum(mu * weights)
            p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if p_volatility == 0:
                return 0  # Avoid division by zero
            return - (p_return - self._risk_free_rate) / p_volatility

        # 4. Define constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))

        # 5. Set initial guess
        initial_weights = np.array([1. / num_assets] * num_assets)

        # 6. Run the optimization
        result = minimize(
            fun=negative_sharpe_ratio,
            x0=initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            # If optimization fails, fall back to an equal-weight portfolio for robustness
            # In a real-world scenario, this failure should be logged.
            fallback_weights = {asset: 1.0 / num_assets for asset in valid_assets}
            return Portfolio(date=date, strategy_name=self.name, weights=fallback_weights)

        # 7. Extract and return the optimal weights
        optimal_weights = dict(zip(valid_assets, result.x))
        return Portfolio(date=date, strategy_name=self.name, weights=optimal_weights)



class NaiveRiskParityStrategy(PortfolioStrategy):


    def __init__(self, lookback_days: int = 252):
        self._lookback_days = lookback_days

    @property
    def name(self) -> str:
        return f"Naive Risk Parity (Lookback: {self._lookback_days}d)"

    def calculate(self, date: datetime, assets: List[str], data_provider: DataProvider) -> Portfolio:

        price_history = data_provider.get_price_history(assets, date, self._lookback_days)
        if price_history.shape[0] < 2:
            raise ValueError("Not enough historical data to calculate volatility.")

        log_returns = np.log(price_history / price_history.shift(1)).dropna()

        if log_returns.empty:
            raise ValueError("Could not compute valid returns for all assets.")

        volatility = log_returns.std() * np.sqrt(252)  # Annualized volatility


        inverse_volatility = 1 / volatility
        sum_inverse_vol = inverse_volatility.sum()

        if sum_inverse_vol == 0:
            raise ValueError("Cannot calculate weights, sum of inverse volatility is zero.")

        normalized_weights = inverse_volatility / sum_inverse_vol

        return Portfolio(date=date, strategy_name=self.name, weights=normalized_weights.to_dict())
