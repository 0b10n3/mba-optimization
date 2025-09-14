from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
from datetime import datetime

from logger import logger



@dataclass
class Portfolio:

    date: datetime
    strategy_name: str
    weights: Dict[str, float] = field(default_factory=dict)
    goal_threshold: Optional[float] = None


class PortfolioStrategy(ABC):


    @property
    @abstractmethod
    def name(self) -> str:

        pass

    @abstractmethod
    def calculate(
            self, date: datetime, assets: List[str], data_provider: 'DataProvider'
    ) -> Portfolio:

        pass


class Observer(ABC):


    @abstractmethod
    def update(self, subject: Any, event_data: Dict[str, Any]):

        pass


class Subject(ABC):


    @abstractmethod
    def attach(self, observer: Observer):

        pass

    @abstractmethod
    def detach(self, observer: Observer):

        pass

    @abstractmethod
    def notify(self, event_data: Dict[str, Any]):

        pass


class DataProvider:
    """
    Encapsulates the data source (DataFrame) and provides clean methods
    to access the data required by strategies and analysis.
    """

    def __init__(self, data_frame: pd.DataFrame):
        if not isinstance(data_frame.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")
        self._df = data_frame.sort_index()

        # Asset data processing
        self._pivot_prices = self._df.pivot(columns='cod_negociacao', values='preco_ultimo').ffill()
        self._daily_returns = self._pivot_prices.pct_change().dropna(how='all')

        # Benchmark data processing
        # Since factors are the same for a given day, we can group by index and take the first value.
        daily_factors = self._df.groupby(self._df.index).first()
        self._selic_factors = daily_factors['fator_diario_selic'].fillna(0)
        self._ipca_factors = daily_factors['fator_diario_ipca'].fillna(0)
        self._ibovespa_factors = daily_factors['fator_diario_ibovespa'].fillna(0) + 1.0

        # Create an equal-weighted benchmark from all assets in the universe
        self._benchmark_returns = self._daily_returns.mean(axis=1)

    def get_assets_for_date(self, date: datetime) -> List[str]:
        """Returns a list of unique asset codes available on a specific date."""
        if date in self._df.index:
            return sorted(self._df.loc[self._df.index == date, 'cod_negociacao'].unique().tolist())
        return []

    def get_price_history(self, assets: List[str], end_date: datetime, window_days: int) -> pd.DataFrame:
        """Returns a DataFrame of historical prices for a list of assets."""
        start_date = end_date - pd.Timedelta(days=window_days)
        # Drop columns with any NaN values, as they can't be used in covariance calculations
        return self._pivot_prices.loc[start_date:end_date, assets].dropna(axis=1, how='any')

    def get_daily_returns(self, start_date: datetime, end_date: datetime, assets: List[str]) -> pd.DataFrame:
        """Returns a DataFrame of daily returns for a given period and assets."""
        return self._daily_returns.loc[start_date:end_date, assets]

    @property
    def all_dates(self) -> pd.DatetimeIndex:
        """Returns all unique dates in the data."""
        return self._df.index.unique()

    @property
    def benchmark_returns(self) -> pd.Series:
        """Returns the daily returns of an equal-weighted market benchmark."""
        return self._benchmark_returns

    @property
    def selic_cumulative_returns(self) -> pd.Series:
        """Returns the cumulative returns of the SELIC factor."""
        return self._selic_factors.cumprod()

    @property
    def ipca_cumulative_returns(self) -> pd.Series:
        """Returns the cumulative returns of the IPCA factor."""
        return self._ipca_factors.cumprod()

    @property
    def ibovespa_cumulative_returns(self) -> pd.Series:
        """Returns the cumulative returns of the IBOVESPA factor."""
        return self._ibovespa_factors.cumprod()

class DateIterator(Subject):


    def __init__(self, dates: List[datetime]):
        self._dates = sorted(dates)
        self._observers: List[Observer] = []

    def attach(self, observer: Observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: Observer):
        self._observers.remove(observer)

    def notify(self, event_data: Dict[str, Any]):
        for observer in self._observers:
            observer.update(self, event_data)

    def run(self):

        last_quarter = -1
        for date in self._dates:
            current_quarter = (date.month - 1) // 3 + 1
            if current_quarter != last_quarter:
                # This is the first day we've seen in a new quarter.
                # Fire the event.
                self.notify({'event_type': 'NEW_QUARTER', 'date': date})
                last_quarter = current_quarter


class PortfolioRebalancer(Observer):


    def __init__(self, data_provider: DataProvider, strategies: List[PortfolioStrategy]):
        self._data_provider = data_provider
        self._strategies = strategies
        self.generated_portfolios: List[Portfolio] = []

    def update(self, subject: Any, event_data: Dict[str, Any]):

        event_type = event_data.get('event_type')
        if event_type == 'NEW_QUARTER':
            date = event_data.get('date')
            logger.info(f"\n--- Rebalancing Event Triggered on {date.strftime('%Y-%m-%d')} ---")

            assets = self._data_provider.get_assets_for_date(date)
            if not assets:
                logger.info(f"No assets available on {date.strftime('%Y-%m-%d')}. Skipping.")
                return

            logger.info(f"Available assets for allocation: {assets}")

            for strategy in self._strategies:
                logger.info(f"Calculating portfolio for strategy: '{strategy.name}'...")
                try:
                    portfolio = strategy.calculate(date, assets, self._data_provider)
                    self.generated_portfolios.append(portfolio)
                    logger.info(f"Successfully generated portfolio for '{strategy.name}'.")
                except Exception as e:
                    logger.error(f"ERROR: Could not calculate portfolio for '{strategy.name}'. Reason: {e}")

