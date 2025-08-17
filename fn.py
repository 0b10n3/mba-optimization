import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import requests
from matplotlib import pyplot as plt

from logger import logger

SELIC_URL = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.4189/dados'
IPCA_URL = "http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='PRECOS12_IPCA12')"


def get_etfs_tickers(path: Path) -> Union[pd.DataFrame, None]:
    try:
        logger.info(f'Reading ETFs tickers and metadata from {str(path)}.')
        etfs_tickers = pd.read_csv(path)
        if etfs_tickers.empty:
            logger.warning(
                'No data in {str(path)}. Returning empty dataframe.'
            )
            return pd.DataFrame()
    except FileNotFoundError as _err:
        logger.error(
            f'File not found: {str(path)}. '
            f'Please check the path and try again.'
        )
        return None

    return etfs_tickers


def get_unique_values_from_column(
    df: pd.DataFrame, column_name: str
) -> List[str]:
    if column_name not in df.columns:
        logger.error('Colunm not found in DataFrame.')
        return []

    unique_values = df[column_name].unique().tolist()
    logger.info('Returning unique values from column: ' + column_name)
    return unique_values


def merge_bdis(
    tickers: List[str], years: range = range(2014, 2025)
) -> Union[pd.DataFrame, None]:
    full_hist_path = Path('data/full_hist.csv')

    if full_hist_path.exists():
        print(f'Loading existing merged data from {full_hist_path}')
        return pd.read_csv(
            full_hist_path, index_col='data_pregao', parse_dates=True
        )

    yearly_dfs = []

    for year in years:
        file_path = Path(f'data/COTAHIST_A{year}.TXT')
        if not file_path.exists():
            logger.warning(
                f'File {file_path} does not exist. Skipping year {year}.'
            )
            continue

        _year_data = parse_cothist(file_path)

        if _year_data.empty or 'cod_negociacao' not in _year_data.columns:
            logger.warning(f'No valid data in {file_path} for year {year}.')
            continue

        _year_data_filtered = _year_data[
            _year_data['cod_negociacao'].isin(tickers)
        ].copy()

        if _year_data_filtered.empty:
            continue

        _year_data_filtered['data_pregao'] = pd.to_datetime(
            _year_data_filtered['data_pregao'], format='%Y%m%d'
        )

        yearly_dfs.append(_year_data_filtered)

    if not yearly_dfs:
        logger.warning('No data found for the specified tickers and years.')
        return pd.DataFrame()

    concat_df = pd.concat(yearly_dfs, ignore_index=True)

    columns_to_drop = [
        'tipo_registro',
        'cod_bdi',
        'nome_resumido',
        'especificacao_papel',
        'prazo_termo',
        'moeda_referencia',
        'preco_exercicio',
        'indicador_correcao',
        'data_vencimento',
        'preco_exercicio_pontos',
        'cod_isin',
        'num_distribuicao',
        'fator_cotacao',
    ]
    concat_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    concat_df.set_index('data_pregao', inplace=True)
    concat_df.sort_index(inplace=True)

    logger.info(
        f'Data merged successfully for years {years[0]} to {years[-1]}.'
    )

    logger.info(f'Saving merged data to {full_hist_path}')
    concat_df.to_csv(full_hist_path)

    return concat_df


def parse_cothist(file_path: str) -> pd.DataFrame:
    col_specs_and_names: Dict[str, Tuple[int, int]] = {
        'tipo_registro': (0, 2),
        'data_pregao': (2, 10),
        'cod_bdi': (10, 12),
        'cod_negociacao': (12, 24),
        'tipo_mercado': (24, 27),
        'nome_resumido': (27, 39),
        'especificacao_papel': (39, 49),
        'prazo_termo': (49, 52),
        'moeda_referencia': (52, 56),
        'preco_abertura': (56, 69),
        'preco_maximo': (69, 82),
        'preco_minimo': (82, 95),
        'preco_medio': (95, 108),
        'preco_ultimo': (108, 121),
        'preco_oferta_compra': (121, 134),
        'preco_oferta_venda': (134, 147),
        'num_negocios': (147, 152),
        'qtd_titulos_negociados': (152, 170),
        'vol_total_negociado': (170, 188),
        'preco_exercicio': (188, 201),
        'indicador_correcao': (201, 202),
        'data_vencimento': (202, 210),
        'fator_cotacao': (210, 217),
        'preco_exercicio_pontos': (217, 230),
        'cod_isin': (230, 242),
        'num_distribuicao': (242, 245),
    }

    col_specs: List[Tuple[int, int]] = list(col_specs_and_names.values())
    col_names: List[str] = list(col_specs_and_names.keys())

    try:
        df = pd.read_fwf(
            file_path,
            colspecs=col_specs,
            names=col_names,
            skiprows=1,
            skipfooter=1,
            encoding='latin-1',
        )
    except FileNotFoundError:
        logger.error(f'File not found: {file_path}')
        return pd.DataFrame()
    except Exception:
        logger.error(f'Error reading file: {file_path}')
        return pd.DataFrame()

    df = df[df['tipo_registro'] == 1].copy()
    df.drop('tipo_registro', axis=1, inplace=True)

    df['data_pregao'] = pd.to_datetime(df['data_pregao'], format='%Y%m%d')

    df['data_vencimento'] = pd.to_datetime(
        df['data_vencimento'], format='%Y%m%d', errors='coerce'
    )

    price_cols = [
        'preco_abertura',
        'preco_maximo',
        'preco_minimo',
        'preco_medio',
        'preco_ultimo',
        'preco_oferta_compra',
        'preco_oferta_venda',
        'vol_total_negociado',
        'preco_exercicio',
    ]

    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') / 100

    integer_cols = ['num_negocios', 'qtd_titulos_negociados', 'fator_cotacao']
    for col in integer_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    string_cols = [
        'cod_negociacao',
        'nome_resumido',
        'especificacao_papel',
        'moeda_referencia',
        'cod_isin',
    ]
    for col in string_cols:
        try:
            df[col] = df[col].str.strip()
        except Exception as _err:
            logger.error(f'Error processing {col}: {_err}')
            df[col] = pd.NA

    return df


def get_selic(
    url: str = SELIC_URL, full_hist_path=Path('data/full_hist_selic.csv')
) -> pd.DataFrame:
    if full_hist_path.exists():
        logger.info(f'Loading existing merged data from {full_hist_path}')
        return pd.read_csv(full_hist_path, index_col='data', parse_dates=True)

    try:
        logger.info(f'Retrieving SELIC data from {url}.')
        selic_df = pd.read_json(url)
        if selic_df.empty:
            logger.warning('No data retrieved from SELIC API.')
            return pd.DataFrame()
    except Exception as _err:
        logger.error(f'Error retrieving SELIC data: {_err}')
        return pd.DataFrame()

    selic_df['data'] = pd.to_datetime(selic_df['data'], format='%d/%m/%Y')
    selic_df.set_index('data', inplace=True)
    selic_df.rename(columns={'valor': 'selic'}, inplace=True)

    selic_df.to_csv(full_hist_path)

    return selic_df


def get_ipca(
    url: str = IPCA_URL, full_hist_path: Path = Path('data/full_hist_ipca.csv')
) -> pd.DataFrame:
    if full_hist_path.exists():
        logger.info(f'Loading existing merged data from:  {full_hist_path}')
        return pd.read_csv(full_hist_path, index_col='data', parse_dates=True)

    try:
        response = requests.get(url)
        response.raise_for_status()

        json_data = response.json()

        value_list = json_data.get('value')

        if not value_list:
            logger.warning('No data retrieved from IPCA API.')
            return pd.DataFrame()

        for item in value_list:
            item['VALDATA'] = item['VALDATA'][:10]

        ipca_df = pd.DataFrame(value_list)

        if ipca_df.empty:
            logger.warning('No data retrieved from IPCA API.')
            return pd.DataFrame()

    except requests.exceptions.RequestException as _err:
        logger.error(f'Error on retrieve data from IPCA API: {_err}')
        return pd.DataFrame()
    except KeyError:
        logger.error("Erro: key 'value' not found.")
        return pd.DataFrame()

    ipca_df = ipca_df[['VALDATA', 'VALVALOR']]

    ipca_df.rename(
        columns={'VALDATA': 'data', 'VALVALOR': 'ipca'}, inplace=True
    )

    ipca_df['data'] = pd.to_datetime(ipca_df['data'])

    ipca_df.set_index('data', inplace=True)

    monthly_variation = ipca_df['ipca'].pct_change()

    annualized_rate = (((1 + monthly_variation) ** 12) - 1) * 100

    ipca_df['ipca'] = annualized_rate

    ipca_df.dropna(inplace=True)

    full_hist_path.parent.mkdir(parents=True, exist_ok=True)
    ipca_df.to_csv(full_hist_path)

    return ipca_df


def merge_reference_index(
    merged_data,
    index_data: pd.DataFrame,
    index_name: str,
    daily_factor: bool = True,
) -> pd.DataFrame:
    if daily_factor:
        index_data[f'fator_diario_{index_name}'] = (
            1 + index_data[index_name] / 100
        ) ** (1 / 252)

    merged_data_reset = merged_data.reset_index()
    index_data_reset = index_data.reset_index()

    merged_data_reset.sort_values('data_pregao', inplace=True)
    index_data_reset.sort_values('data', inplace=True)

    final_df = pd.merge_asof(
        merged_data_reset,
        index_data_reset[['data', f'fator_diario_{index_name}']],
        left_on='data_pregao',
        right_on='data',
        direction='backward',
    )

    final_df.set_index('data_pregao', inplace=True)
    final_df = final_df.drop(columns='data')

    return final_df


def read_ibovespa_index(start_year=2014, end_year=2024):
    all_yearly_data = []

    month_map = {
        'Jan': 1,
        'Fev': 2,
        'Mar': 3,
        'Abr': 4,
        'Mai': 5,
        'Jun': 6,
        'Jul': 7,
        'Ago': 8,
        'Set': 9,
        'Out': 10,
        'Nov': 11,
        'Dez': 12,
    }

    for year in range(start_year, end_year + 1):
        filename = f'data/ibovespa_index/IBOVESPA_{year}.csv'

        if not os.path.exists(filename):
            logger.warning(f"Warning: File '{filename}' not found. Skipping.")
            continue

        try:
            df_year = pd.read_csv(
                filename,
                sep=';',
                skiprows=1,
                index_col='Dia',
                encoding='latin-1',
            )

            s_unpivoted = df_year.stack()

            df_unpivoted = s_unpivoted.reset_index()
            df_unpivoted.columns = ['day', 'month_abbr', 'close']

            df_unpivoted['month'] = df_unpivoted['month_abbr'].map(month_map)
            df_unpivoted['year'] = year

            df_unpivoted['data'] = pd.to_datetime(
                df_unpivoted[['year', 'month', 'day']], errors='coerce'
            )

            df_unpivoted.dropna(subset=['data'], inplace=True)

            df_unpivoted['close'] = (
                df_unpivoted['close']
                .astype(str)
                .str.replace('.', '', regex=False)
                .str.replace(',', '.', regex=False)
                .astype(float)
            )

            all_yearly_data.append(df_unpivoted[['data', 'close']])

        except Exception as e:
            logger.error(f'An error occurred while processing {filename}: {e}')

    if not all_yearly_data:
        logger.warning('No data was processed. Returning an empty DataFrame.')
        return pd.DataFrame(columns=['close', 'fator_diario_ibovespa'])

    final_df = pd.concat(all_yearly_data, ignore_index=True)

    final_df.sort_values('data', inplace=True)

    final_df.set_index('data', inplace=True)

    final_df['fator_diario_ibovespa'] = final_df['close'].pct_change()

    return final_df


def etfs_count(merged_data):
    count_etfs = merged_data.groupby(level=0).size()
    count_etfs.plot(kind='line', title='Quantidade de ETFs por Data')
    plt.show()

    return count_etfs
