from pathlib import Path

from matplotlib import pyplot as plt

from fn import (get_etfs_tickers,
                get_unique_values_from_column,
                merge_bdis,
                get_ipca,
                get_selic,
                merge_reference_index
                )

etfs_tickers = get_etfs_tickers(path=Path('data/FundosListados.csv'))

tickers = get_unique_values_from_column(
    df=etfs_tickers, column_name='Codigo Negociacao'
)

merged_data = merge_bdis(tickers=tickers)

# Plots the evelution of the number of ETFs over time
#count_etfs = merged_data.groupby(level=0).size()
# count_etfs.plot(kind='line', title='Quantidade de ETFs por Data')
# plt.show()

selic_data = get_selic()
ipca_data = get_ipca()

merged_data = merge_reference_index(merged_data, selic_data, 'selic')
merged_data = merge_reference_index(merged_data, ipca_data, 'ipca')

print('Stop Here')
