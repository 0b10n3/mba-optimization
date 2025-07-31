from pathlib import Path

from matplotlib import pyplot as plt

from fn import (get_etfs_tickers,
                get_unique_values_from_column,
                merge_bdis,
                get_selic,
                merge_reference_index
                )

etfs_tickers = get_etfs_tickers(path=Path('data/FundosListados.csv'))

tickers = get_unique_values_from_column(
    df=etfs_tickers, column_name='Codigo Negociacao'
)

merged_data = merge_bdis(tickers=tickers)

count_etfs = merged_data.groupby(level=0).size()

count_etfs.plot(kind='line', title='Quantidade de ETFs por Data')
plt.show()

selic_data = get_selic()

full_data = merge_reference_index(merged_data, selic_data)

print('Stop Here')
