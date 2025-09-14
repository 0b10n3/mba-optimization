from pathlib import Path

from fn import (
    get_etfs_tickers,
    get_ipca,
    get_selic,
    get_unique_values_from_column,
    merge_bdis,
    merge_reference_index,
    read_ibovespa_index,
    plot_etf_evolution
)

from portfolio import (
    DataProvider,
    DateIterator,
    PortfolioRebalancer
)
from strategies import (
    OneOverNStrategy,
    MarkowitzStrategy,
    NaiveRiskParityStrategy,
    Goal,
    GoalBasedStrategy
)

from logger import logger
from analyzer import PerformanceAnalyzer

etfs_tickers = get_etfs_tickers(path=Path('data/FundosListados.csv'))

tickers = get_unique_values_from_column(
    df=etfs_tickers, column_name='Codigo Negociacao'
)

merged_data = merge_bdis(tickers=tickers)


selic_data = get_selic()
ipca_data = get_ipca()
ibovespa_index = read_ibovespa_index()

merged_data = merge_reference_index(merged_data, selic_data, 'selic')
merged_data = merge_reference_index(merged_data, ipca_data, 'ipca')
merged_data = merge_reference_index(
    merged_data, ibovespa_index, 'ibovespa', daily_factor=False
)

plot_etf_evolution(merged_data=merged_data)


def main():
    """
    Main function to initialize and run the portfolio backtesting application.
    """
    print("Setting up the portfolio backtesting application...")

    # 1. Load Data
    data_provider = DataProvider(merged_data)
    print(
        f"Data loaded with {len(merged_data)} records from {merged_data.index.min().strftime('%Y-%m-%d')} to {merged_data.index.max().strftime('%Y-%m-%d')}.")

    # 2. Define Goal for GBI Strategy
    investor_goal = Goal(
        name="Aposentadoria Confortável",
        threshold_return=0.10,  # Meta de retorno anual de 10%
        max_failure_prob=0.15  # Aceita no máximo 15% de chance de não atingir a meta
    )

    # 3. Initialize Strategies
    list_of_strategies = [
        OneOverNStrategy(),
        MarkowitzStrategy(lookback_days=120),  # Shorter lookback for mock data
        NaiveRiskParityStrategy(lookback_days=120),
        GoalBasedStrategy(goal=investor_goal, lookback_days=120)
    ]
    print("\nInitialized the following strategies:")
    for s in list_of_strategies:
        print(f"- {s.name}")

    # 4. Initialize Core Components
    portfolio_rebalancer = PortfolioRebalancer(data_provider, list_of_strategies)
    all_dates = sorted(merged_data.index.unique())
    date_iterator = DateIterator(all_dates)

    # 5. Wire up the Observer Pattern
    date_iterator.attach(portfolio_rebalancer)
    print("\nPortfolioRebalancer is now observing DateIterator.")

    # 6. Run the Simulation
    print("\nStarting simulation...")
    date_iterator.run()
    print("\nSimulation finished.")

    # 7. Analyze and Display Results
    print(f"\n--- Performance Analysis ---")
    if not portfolio_rebalancer.generated_portfolios:
        print("No portfolios were generated. Cannot run analysis.")
        return

    # Use a risk-free rate of 2% for Sharpe Ratio calculation
    risk_free_rate = 0.06
    analyzer = PerformanceAnalyzer(
        portfolio_rebalancer.generated_portfolios,
        data_provider,
        risk_free_rate
    )

    # Calculate and display metrics
    metrics_df = analyzer.calculate_metrics()
    print("\nPerformance Metrics Summary:")
    print(metrics_df.to_string(formatters={
        'Total Return': '{:,.2%}'.format,
        'Annualized Volatility': '{:,.2%}'.format,
        'Sharpe Ratio': '{:,.2f}'.format,
        'Sortino Ratio': '{:,.2f}'.format,
        "Jensen's Alpha": '{:,.3f}'.format,
        'Max Drawdown': '{:,.2%}'.format,
        'Success Probability': '{:,.2%}'.format,
    }))

    # Plot the results
    print("\nGenerating comparison plots...")
    analyzer.plot_cumulative_returns()

    # 8. Export results to CSV
    print("\nExporting analysis to CSV files...")
    analyzer.export_analysis_data(
        metrics_filename="resultados_metricas.csv",
        returns_filename="resultados_retornos_acumulados.csv"
    )


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
