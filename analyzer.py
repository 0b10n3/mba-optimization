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
        self._strategy_returns, self._goal_thresholds = self._calculate_all_strategy_returns()

        # Fetch benchmark cumulative returns
        self._selic_cum_returns = data_provider.selic_cumulative_returns
        self._ipca_cum_returns = data_provider.ipca_cumulative_returns
        self._ibovespa_cum_returns = data_provider.ibovespa_cumulative_returns

    def _calculate_all_strategy_returns(self) -> (pd.DataFrame, Dict[str, float]):
        """
        Calculates the daily return series for each strategy and stores goal thresholds.
        """
        all_returns = {}
        goal_thresholds = {}
        strategy_names = sorted(list(set(p.strategy_name for p in self._portfolios)))

        for name in strategy_names:
            strategy_portfolios = [p for p in self._portfolios if p.strategy_name == name]
            period_returns_list = []

            # Store goal threshold if it exists for this strategy
            if strategy_portfolios and strategy_portfolios[0].goal_threshold is not None:
                goal_thresholds[name] = strategy_portfolios[0].goal_threshold

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

        return pd.DataFrame(all_returns).dropna(how='all'), goal_thresholds

    def calculate_metrics(self) -> pd.DataFrame:
        """Calculates key performance metrics for each strategy."""
        metrics = {}
        aligned_benchmark = self._benchmark_returns.reindex(self._strategy_returns.index).ffill()

        for name, returns in self._strategy_returns.items():
            if returns.empty: continue

            annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
            annual_volatility = returns.std() * np.sqrt(252)
            excess_returns = returns - (self._risk_free_rate / 252)
            sharpe_ratio = (excess_returns.mean() * 252) / annual_volatility if annual_volatility > 0 else 0

            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()

            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (annualized_return - self._risk_free_rate) / downside_std if downside_std > 0 else np.inf

            comparison_df = pd.DataFrame({'strategy': returns, 'benchmark': aligned_benchmark}).dropna()
            cov_matrix = comparison_df.cov() * 252
            beta = cov_matrix.loc['strategy', 'benchmark'] / cov_matrix.loc['benchmark', 'benchmark']
            benchmark_annual_return = (1 + aligned_benchmark).prod() ** (252 / len(aligned_benchmark)) - 1
            expected_return_capm = self._risk_free_rate + beta * (benchmark_annual_return - self._risk_free_rate)
            jensens_alpha = annualized_return - expected_return_capm

            # GBI-specific metric: Probability of Success
            success_prob = np.nan
            if name in self._goal_thresholds:
                goal_return = self._goal_thresholds[name]
                # Resample returns to yearly to compare with annual goal
                yearly_returns = (1 + returns).resample('A').prod() - 1
                if not yearly_returns.empty:
                    success_prob = (yearly_returns >= goal_return).mean()

            metrics[name] = {
                "Total Return": cumulative_returns.iloc[-1] - 1,
                "Annualized Volatility": annual_volatility,
                "Sharpe Ratio": sharpe_ratio,
                "Sortino Ratio": sortino_ratio,
                "Jensen's Alpha": jensens_alpha,
                "Max Drawdown": max_drawdown,
                "Success Probability": success_prob,
            }

        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
        return metrics_df[[
            "Total Return", "Annualized Volatility", "Sharpe Ratio",
            "Sortino Ratio", "Jensen's Alpha", "Max Drawdown", "Success Probability"
        ]]

    def plot_cumulative_returns(self):
        """Plots the cumulative returns (equity curve) for all strategies and benchmarks."""
        if self._strategy_returns.empty:
            print("No returns data to plot.")
            return

        strategy_cum_returns = (1 + self._strategy_returns).cumprod()
        all_series = strategy_cum_returns.copy()
        start_date = strategy_cum_returns.index.min()

        # Ensure benchmarks are available before trying to normalize
        if not self._selic_cum_returns.empty and start_date in self._selic_cum_returns.index:
            selic_norm = self._selic_cum_returns / self._selic_cum_returns.loc[start_date]
            all_series['SELIC'] = selic_norm

        if not self._ipca_cum_returns.empty and start_date in self._ipca_cum_returns.index:
            ipca_norm = self._ipca_cum_returns / self._ipca_cum_returns.loc[start_date]
            all_series['IPCA'] = ipca_norm

        if not self._ibovespa_cum_returns.empty and start_date in self._ibovespa_cum_returns.index:
            ibov_norm = self._ibovespa_cum_returns / self._ibovespa_cum_returns.loc[start_date]
            all_series['IBOVESPA'] = ibov_norm

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 8))

        strategy_cum_returns.plot(ax=ax, linewidth=2)

        benchmarks_to_plot = [b for b in ['SELIC', 'IPCA', 'IBOVESPA'] if b in all_series.columns]
        if benchmarks_to_plot:
            all_series[benchmarks_to_plot].plot(ax=ax, linestyle='--', linewidth=1.5)

        ax.set_title('Strategy and Benchmark Performance Comparison', fontsize=16)
        ax.set_ylabel('Cumulative Return (Normalized)')
        ax.set_xlabel('Date')
        ax.grid(True)
        ax.legend(title='Strategies & Benchmarks')

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}x'))

        plt.tight_layout()
        plt.show()

    def export_analysis_data(self, metrics_filename: str, returns_filename: str):
        """
        Exports the performance metrics summary and the daily cumulative returns
        to separate CSV files for further analysis.
        """
        # 1. Export Performance Metrics Summary
        metrics_df = self.calculate_metrics()
        try:
            metrics_df.to_csv(metrics_filename)
            print(f"\nPerformance metrics successfully exported to '{metrics_filename}'")
        except Exception as e:
            print(f"\nError exporting metrics to CSV: {e}")

        # 2. Prepare and Export Cumulative Returns Time Series
        if self._strategy_returns.empty:
            print("No cumulative returns data to export.")
            return

        strategy_cum_returns = (1 + self._strategy_returns).cumprod()
        all_series = strategy_cum_returns.copy()
        start_date = strategy_cum_returns.index.min()

        # Normalize and add benchmarks to the dataframe
        if not self._selic_cum_returns.empty and start_date in self._selic_cum_returns.index:
            selic_norm = self._selic_cum_returns / self._selic_cum_returns.loc[start_date]
            all_series['SELIC'] = selic_norm

        if not self._ipca_cum_returns.empty and start_date in self._ipca_cum_returns.index:
            ipca_norm = self._ipca_cum_returns / self._ipca_cum_returns.loc[start_date]
            all_series['IPCA'] = ipca_norm

        if not self._ibovespa_cum_returns.empty and start_date in self._ibovespa_cum_returns.index:
            ibov_norm = self._ibovespa_cum_returns / self._ibovespa_cum_returns.loc[start_date]
            all_series['IBOVESPA'] = ibov_norm

        try:
            all_series.to_csv(returns_filename)
            print(f"Cumulative returns data successfully exported to '{returns_filename}'")
        except Exception as e:
            print(f"\nError exporting cumulative returns to CSV: {e}")
