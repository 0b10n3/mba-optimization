from pathlib import Path

from fn import (get_etfs_tickers,
                get_unique_values_from_column,
                merge_bdis,
                get_ipca,
                get_selic,
                merge_reference_index,
                read_ibovespa_index
                )

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
merged_data = merge_reference_index(merged_data, ibovespa_index, 'ibovespa', daily_factor=False)


print('Stop Here')
