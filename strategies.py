from datetime import datetime
from typing import List

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


    def __init__(self, lookback_days: int = 252):
        self._lookback_days = lookback_days

    @property
    def name(self) -> str:
        return f"Markowitz MVO (Lookback: {self._lookback_days}d)"

    def calculate(self, date: datetime, assets: List[str], data_provider: DataProvider) -> Portfolio:

        price_history = data_provider.get_price_history(assets, date, self._lookback_days)
        if price_history.shape[0] < 2:
            raise ValueError("Not enough historical data to calculate returns.")

        log_returns = np.log(price_history / price_history.shift(1)).dropna()

        if log_returns.empty or log_returns.shape[1] != len(assets):
            raise ValueError("Could not compute valid returns for all assets.")


        print("      (Note: Markowitz using simplified placeholder logic)")
        num_assets = len(assets)
        weights = {asset: 1.0 / num_assets for asset in assets}  # Fallback to 1/N

        return Portfolio(date=date, strategy_name=self.name, weights=weights)


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
