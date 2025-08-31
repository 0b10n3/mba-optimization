import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


from portfolio import Portfolio, DataProvider


class PerformanceAnalyzer:
    """
    Calculates and visualizes performance metrics for a set of strategies.
    """

    def __init__(self, portfolios: List[Portfolio], data_provider: DataProvider, risk_free_rate: float = 0.0):
        self._portfolios = sorted(portfolios, key=lambda p: (p.strategy_name, p.date))
        self._data_provider = data_provider
        self._risk_free_rate = risk_free_rate
        self._benchmark_returns = data_provider.benchmark_returns
        self._strategy_returns = self._calculate_all_strategy_returns()

        # Fetch benchmark cumulative returns
        self._selic_cum_returns = data_provider.selic_cumulative_returns
        self._ipca_cum_returns = data_provider.ipca_cumulative_returns
        self._ibovespa_cum_returns = data_provider.ibovespa_cumulative_returns

    def _calculate_all_strategy_returns(self) -> pd.DataFrame:
        """
        Calculates the daily return series for each strategy by stitching
        together the returns from each rebalancing period.
        """
        all_returns = {}
        strategy_names = sorted(list(set(p.strategy_name for p in self._portfolios)))

        for name in strategy_names:
            strategy_portfolios = [p for p in self._portfolios if p.strategy_name == name]
            period_returns_list = []

            for i, p in enumerate(strategy_portfolios):
                start_date = p.date
                if i + 1 < len(strategy_portfolios):
                    end_date = strategy_portfolios[i + 1].date - pd.Timedelta(days=1)
                else:
                    end_date = self._data_provider.all_dates.max()

                weights = pd.Series(p.weights)
                if weights.empty: continue

                asset_returns = self._data_provider.get_daily_returns(start_date, end_date, list(weights.index))
                portfolio_daily_returns = asset_returns.dot(weights)
                period_returns_list.append(portfolio_daily_returns)

            if period_returns_list:
                all_returns[name] = pd.concat(period_returns_list)

        return pd.DataFrame(all_returns).dropna(how='all')

    def calculate_metrics(self) -> pd.DataFrame:
        """Calculates key performance metrics for each strategy."""
        metrics = {}
        # Align benchmark returns with strategy returns for correct calculations
        aligned_benchmark = self._benchmark_returns.reindex(self._strategy_returns.index).ffill()

        for name, returns in self._strategy_returns.items():
            if returns.empty: continue

            # --- Standard Metrics ---
            total_return = (1 + returns).prod() - 1
            annual_volatility = returns.std() * np.sqrt(252)
            excess_returns = returns - (self._risk_free_rate / 252)
            sharpe_ratio = (excess_returns.mean() * 252) / annual_volatility if annual_volatility > 0 else 0

            # --- Max Drawdown ---
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()

            # --- Sortino Ratio ---
            downside_returns = returns[returns < 0]  # Using 0 as the target return
            downside_std = downside_returns.std() * np.sqrt(252)
            sortino_ratio = ((
                                         returns.mean() * 252) - self._risk_free_rate) / downside_std if downside_std > 0 else np.inf

            # --- Jensen's Alpha ---
            comparison_df = pd.DataFrame({'strategy': returns, 'benchmark': aligned_benchmark}).dropna()
            cov_matrix = comparison_df.cov() * 252
            beta = cov_matrix.loc['strategy', 'benchmark'] / cov_matrix.loc['benchmark', 'benchmark']
            portfolio_annual_return = returns.mean() * 252
            benchmark_annual_return = aligned_benchmark.mean() * 252
            expected_return_capm = self._risk_free_rate + beta * (benchmark_annual_return - self._risk_free_rate)
            jensens_alpha = portfolio_annual_return - expected_return_capm

            metrics[name] = {
                "Total Return": total_return,
                "Annualized Volatility": annual_volatility,
                "Sharpe Ratio": sharpe_ratio,
                "Sortino Ratio": sortino_ratio,
                "Jensen's Alpha": jensens_alpha,
                "Max Drawdown": max_drawdown
            }

        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
        # Reorder columns for logical presentation
        return metrics_df[[
            "Total Return", "Annualized Volatility", "Sharpe Ratio",
            "Sortino Ratio", "Jensen's Alpha", "Max Drawdown"
        ]]

    def plot_cumulative_returns(self):
        """Plots the cumulative returns (equity curve) for all strategies and benchmarks."""
        if self._strategy_returns.empty:
            print("No returns data to plot.")
            return

        # Calculate cumulative returns for strategies
        strategy_cum_returns = (1 + self._strategy_returns).cumprod()

        # Combine all series to be plotted, ensuring they align on the same index
        all_series = strategy_cum_returns.copy()

        # Normalize benchmarks to start at 1 on the first day of the strategy returns
        start_date = strategy_cum_returns.index.min()

        selic_norm = self._selic_cum_returns / self._selic_cum_returns.loc[start_date]
        ipca_norm = self._ipca_cum_returns / self._ipca_cum_returns.loc[start_date]
        ibov_norm = self._ibovespa_cum_returns / self._ibovespa_cum_returns.loc[start_date]

        all_series['SELIC'] = selic_norm
        all_series['IPCA'] = ipca_norm
        all_series['IBOVESPA'] = ibov_norm

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot strategies with solid lines
        strategy_cum_returns.plot(ax=ax, linewidth=2)

        # Plot benchmarks with dashed lines
        all_series[['SELIC', 'IPCA', 'IBOVESPA']].plot(ax=ax, linestyle='--', linewidth=1.5)

        ax.set_title('Strategy and Benchmark Performance Comparison', fontsize=16)
        ax.set_ylabel('Cumulative Return (Normalized)')
        ax.set_xlabel('Date')
        ax.grid(True)
        ax.legend(title='Strategies & Benchmarks')

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}x'))

        plt.tight_layout()
        plt.show()