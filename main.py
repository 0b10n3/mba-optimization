from pathlib import Path

from fn import (
    get_etfs_tickers,
    get_ipca,
    get_selic,
    get_unique_values_from_column,
    merge_bdis,
    merge_reference_index,
    read_ibovespa_index,
)

from portfolio import (
    DataProvider,
    DateIterator,
    PortfolioRebalancer
)
from strategies import (
    OneOverNStrategy,
    MarkowitzStrategy,
    NaiveRiskParityStrategy
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


def main():


    data_provider = DataProvider(merged_data)
    logger.info(f"Data loaded with {len(merged_data)} records.")

    list_of_strategies = [
        OneOverNStrategy(),
        MarkowitzStrategy(lookback_days=252),
        NaiveRiskParityStrategy(lookback_days=252)
    ]
    logger.info("Initialized the following strategies:")
    for s in list_of_strategies:
        logger.info(f"- {s.name}")

    portfolio_rebalancer = PortfolioRebalancer(data_provider, list_of_strategies)

    all_dates = sorted(merged_data.index.unique())
    date_iterator = DateIterator(all_dates)

    date_iterator.attach(portfolio_rebalancer)
    logger.info("\nPortfolioRebalancer is now observing DateIterator.")

    logger.info("Starting simulation...")
    date_iterator.run()
    logger.info("\nSimulation finished.")

    logger.info(f"\n--- Performance Analysis ---")
    if not portfolio_rebalancer.generated_portfolios:
        logger.warning("No portfolios were generated. Cannot run analysis.")
        return

    # Use a risk-free rate of 2% for Sharpe Ratio calculation
    risk_free_rate = 0.02
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
        'Max Drawdown': '{:,.2%}'.format
    }))

    # Plot the results
    print("\nGenerating comparison plots...")
    analyzer.plot_cumulative_returns()

    print("Plots generated successfully. Check the 'plots' directory.")


if __name__ == "__main__":
    main()
