from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
from datetime import datetime



@dataclass
class Portfolio:

    date: datetime
    strategy_name: str
    weights: Dict[str, float] = field(default_factory=dict)


class PortfolioStrategy(ABC):


    @property
    @abstractmethod
    def name(self) -> str:
        """A user-friendly name for the strategy."""
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
    to access the data required by strategies.
    """

    def __init__(self, data_frame: pd.DataFrame):
        """
        Initializes the DataProvider.

        Args:
            data_frame: A pandas DataFrame with a datetime index and columns
                        including 'cod_negociacao' and 'preco_ultimo'.
        """
        if not isinstance(data_frame.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")
        self._df = data_frame

    def get_assets_for_date(self, date: datetime) -> List[str]:
        """Returns a list of unique asset codes available on a specific date."""
        if date in self._df.index:
            return sorted(self._df.loc[self._df.index == date, 'cod_negociacao'].unique().tolist())
        return []

    def get_price_history(self, assets: List[str], end_date: datetime, window_days: int) -> pd.DataFrame:
        """
        Returns a DataFrame of historical prices for a list of assets over a
        given lookback window.
        """
        start_date = end_date - pd.Timedelta(days=window_days)

        # Filter data for the relevant date range and assets
        history_df = self._df[
            (self._df.index >= start_date) &
            (self._df.index <= end_date) &
            (self._df['cod_negociacao'].isin(assets))
            ]

        # Pivot to get assets as columns and dates as index
        price_history = history_df.pivot(columns='cod_negociacao', values='preco_ultimo')

        # Forward-fill to handle non-trading days and missing values
        price_history = price_history.ffill()

        return price_history


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
            print(f"\n--- Rebalancing Event Triggered on {date.strftime('%Y-%m-%d')} ---")

            assets = self._data_provider.get_assets_for_date(date)
            if not assets:
                print(f"No assets available on {date.strftime('%Y-%m-%d')}. Skipping.")
                return

            print(f"Available assets for allocation: {assets}")

            for strategy in self._strategies:
                print(f"Calculating portfolio for strategy: '{strategy.name}'...")
                try:
                    portfolio = strategy.calculate(date, assets, self._data_provider)
                    self.generated_portfolios.append(portfolio)
                    print(f"Successfully generated portfolio for '{strategy.name}'.")
                except Exception as e:
                    print(f"ERROR: Could not calculate portfolio for '{strategy.name}'. Reason: {e}")

