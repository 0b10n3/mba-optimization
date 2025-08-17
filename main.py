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

from logger import logger

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


logger.info('Stop Here')



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


if __name__ == "__main__":
    main()
