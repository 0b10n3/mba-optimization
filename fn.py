import pandas as pd
from typing import List, Union, Tuple, Dict
from logger import logger
from pathlib import Path


def get_etfs_tickers(path: Path) -> Union[pd.DataFrame, None]:

    try:
        logger.info(f"Reading ETFs tickers and metadata from {str(path)}.")
        etfs_tickers = pd.read_csv(path)
        if etfs_tickers.empty:
            logger.warning("No data in {str(path)}. Returning empty dataframe.")
            return pd.DataFrame()
    except FileNotFoundError as _err:
        logger.error(f"File not found: {str(path)}. Please check the path and try again.")
        return None

    return etfs_tickers


def get_unique_values_from_column(df: pd.DataFrame, column_name: str) -> List[str]:

      if column_name not in df.columns:
        logger.error('Colunm not found in DataFrame.')
        return []

      unique_values = df[column_name].unique().tolist()
      logger.info('Returning unique values from column: ' + column_name)
      return unique_values


def merge_bdis(tickers: List[str], years: range = range(2014, 2025)) -> Union[pd.DataFrame, None]:

    full_hist_path = Path('data/full_hist.csv')

    if full_hist_path.exists():
        print(f"Loading existing merged data from {full_hist_path}")
        return pd.read_csv(
            full_hist_path,
            index_col='data_pregao',
            parse_dates=True
        )

    yearly_dfs = []

    for year in years:
        file_path = Path(f'data/COTAHIST_A{year}.TXT')
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist. Skipping year {year}.")
            continue

        _year_data = parse_cothist(file_path)
        if _year_data.empty or 'cod_negociacao' not in _year_data.columns:
            logger.warning(f"No valid data in {file_path} for year {year}.")
            continue

        _year_data_filtered = _year_data[_year_data['cod_negociacao'].isin(tickers)].copy()

        if _year_data_filtered.empty:
            continue

        _year_data_filtered['data_pregao'] = pd.to_datetime(
            _year_data_filtered['data_pregao'], format='%Y%m%d'
        )
        
        yearly_dfs.append(_year_data_filtered)

    if not yearly_dfs:
        logger.warning("No data found for the specified tickers and years.")
        return pd.DataFrame()

    concat_df = pd.concat(yearly_dfs, ignore_index=True)

    columns_to_drop = [
        'tipo_registro', 'cod_bdi', 'nome_resumido', 'especificacao_papel',
        'prazo_termo', 'moeda_referencia', 'preco_exercicio', 'indicador_correcao',
        'data_vencimento', 'preco_exercicio_pontos', 'cod_isin', 'num_distribuicao',
        'fator_cotacao'
    ]
    concat_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    concat_df.set_index('data_pregao', inplace=True)
    concat_df.sort_index(inplace=True)

    logger.info(f"Data merged successfully for years {years[0]} to {years[-1]}.")

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
            encoding='latin-1'
        )
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading file: {file_path}")
        return pd.DataFrame()

    df = df[df['tipo_registro'] == 1].copy()
    df.drop('tipo_registro', axis=1, inplace=True)

    df['data_pregao'] = pd.to_datetime(df['data_pregao'], format='%Y%m%d')

    df['data_vencimento'] = pd.to_datetime(df['data_vencimento'], format='%Y%m%d', errors='coerce')

    price_cols = [
        'preco_abertura', 'preco_maximo', 'preco_minimo', 'preco_medio',
        'preco_ultimo', 'preco_oferta_compra', 'preco_oferta_venda',
        'vol_total_negociado', 'preco_exercicio'
    ]

    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') / 100

    integer_cols = [
        'num_negocios', 'qtd_titulos_negociados', 'fator_cotacao'
    ]
    for col in integer_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    string_cols = [
        'cod_negociacao', 'nome_resumido',
        'especificacao_papel', 'moeda_referencia', 'cod_isin'
    ]
    for col in string_cols:
        try:
            df[col] = df[col].str.strip()
        except Exception as _err:
            print(f"Erro ao processar a coluna {col}: {_err}")
            df[col] = pd.NA

    return df
