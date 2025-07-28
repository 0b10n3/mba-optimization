import pandas as pd
import yfinance as yf
from typing import List
from logger import log


def get_market_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Busca dados históricos de preços de fechamento ajustados do Yahoo Finance.

    Args:
        tickers (List[str]): Lista de tickers dos ativos.
        start_date (str): Data de início no formato 'YYYY-MM-DD'.
        end_date (str): Data de fim no formato 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: DataFrame com os preços de fechamento ajustados,
                      onde cada coluna corresponde a um ticker.
                      Retorna um DataFrame vazio em caso de erro.
    """
    log.info(f"Buscando dados de mercado para {len(tickers)} ativos de {start_date} a {end_date}.")
    try:

        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)


        if len(tickers) == 1:
            prices = data[['Close']]
            prices.columns = tickers
        else:
            prices = data['Close']

        if prices.empty:
            log.warning("O DataFrame de preços retornado pelo yfinance está vazio.")
            return pd.DataFrame()


        missing_tickers = set(tickers) - set(prices.columns)
        if missing_tickers:
            log.warning(f"Não foi possível obter dados para todos os tickers. Ausentes: {', '.join(missing_tickers)}")
            return pd.DataFrame()

        prices.dropna(inplace=True)
        log.info(f"Dados obtidos com sucesso. {len(prices)} pregões válidos encontrados.")
        return prices

    except Exception as e:
        log.error(f"Ocorreu um erro ao buscar os dados de mercado: {e}")
        return pd.DataFrame()
    
    

def get_stock_date_range(tickers):
    """
    Get the minimum and maximum available dates for historical stock data from Yahoo Finance.

    :param tickers: A string or a list of strings representing the stock ticker(s).
    :return: A pandas DataFrame with tickers as index and 'Min Date' and 'Max Date' as columns.
             Returns an empty DataFrame if no data is found or an error occurs.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    elif not isinstance(tickers, list):
        logger.error("Tickers must be a string or a list of strings.")
        return pd.DataFrame()

    date_ranges = {}

    for ticker_symbol in tickers:
        try:
            ticker_obj = yf.Ticker(ticker_symbol)

            hist = ticker_obj.history(period="max", auto_adjust=False)

            if hist.empty:
                logger.warning(f"No historical data found for {ticker_symbol}.")
                date_ranges[ticker_symbol] = {'Min Date': pd.NaT, 'Max Date': pd.NaT}
                continue

            min_date = hist.index.min()
            max_date = hist.index.max()

            if isinstance(min_date, pd.Timestamp) and min_date.tzinfo is not None:
                min_date = min_date.tz_localize(None)
            if isinstance(max_date, pd.Timestamp) and max_date.tzinfo is not None:
                max_date = max_date.tz_localize(None)


            date_ranges[ticker_symbol] = {'Min Date': min_date, 'Max Date': max_date}
            logger.info(f"Date range for {ticker_symbol}: Min Date - {min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else 'N/A'}, Max Date - {max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else 'N/A'}")

        except Exception as e:
            logger.error(f"Error fetching data for {ticker_symbol}: {e}")
            date_ranges[ticker_symbol] = {'Min Date': pd.NaT, 'Max Date': pd.NaT}

    if not date_ranges:
        return pd.DataFrame()

    result_df = pd.DataFrame.from_dict(date_ranges, orient='index')
    return result_df

import pandas as pd
from typing import Dict, Tuple, List

def parse_cothist(file_path: str) -> pd.DataFrame:
    """
    Analisa um arquivo de cotações históricas da B3 (formato COTAHIST) e o converte
    em um DataFrame do pandas.

    Args:
        file_path: O caminho para o arquivo COTAHIST.AAAA.TXT.

    Returns:
        Um DataFrame do pandas com os dados das cotações, com tipos de dados
        corretamente formatados.
    """
    # Especificação das colunas com base no layout do arquivo "REGISTRO - 01"
    # (posição inicial, posição final)
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

    # Extrai as especificações de largura e os nomes das colunas
    col_specs: List[Tuple[int, int]] = list(col_specs_and_names.values())
    col_names: List[str] = list(col_specs_and_names.keys())

    try:
        # Lê o arquivo de largura fixa, ignorando o header e o footer
        df = pd.read_fwf(
            file_path,
            colspecs=col_specs,
            names=col_names,
            skiprows=1,      # Pula o registro de header (tipo "00")
            skipfooter=1,    # Pula o registro de trailer (tipo "99")
            encoding='latin-1' # Codificação comum para esses arquivos
        )
    except FileNotFoundError:
        print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo: {e}")
        return pd.DataFrame()

    # --- Limpeza e Conversão de Tipos de Dados ---

    # Filtra apenas os registros de cotações (tipo "01")
    df = df[df['tipo_registro'] == 1].copy()
    df.drop('tipo_registro', axis=1, inplace=True)


    # Converte colunas de data
    df['data_pregao'] = pd.to_datetime(df['data_pregao'], format='%Y%m%d')
    # Para a data de vencimento, erros são convertidos para NaT (Not a Time)
    df['data_vencimento'] = pd.to_datetime(df['data_vencimento'], format='%Y%m%d', errors='coerce')

    # Colunas que representam preços/valores com 2 casas decimais
    price_cols = [
        'preco_abertura', 'preco_maximo', 'preco_minimo', 'preco_medio',
        'preco_ultimo', 'preco_oferta_compra', 'preco_oferta_venda',
        'vol_total_negociado', 'preco_exercicio'
    ]

    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') / 100

    # Colunas que são inteiros
    integer_cols = [
        'num_negocios', 'qtd_titulos_negociados', 'fator_cotacao'
    ]
    for col in integer_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')


    # Colunas de texto (string) para remover espaços em branco
    string_cols = [
        'cod_bdi', 'cod_negociacao', 'nome_resumido',
        'especificacao_papel', 'moeda_referencia', 'cod_isin'
    ]
    for col in string_cols:
        df[col] = df[col].str.strip()


    print(f"Arquivo '{file_path}' processado com sucesso.")
    print(f"Total de {len(df)} registros de cotações carregados.")

    return df

if __name__ == "__main__":
    import logging

    # Configura o logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Exemplo de uso
    logger.info("Iniciando a coleta de dados de mercado...")

    # df = pd.read_csv('data/FundosListados.csv')
    #
    # df["Codigo Negociacao"] = df["Codigo Negociacao"].astype(str) + ".SA"
    #
    #
    # tickers = df['Codigo Negociacao'].tolist()
    #
    # date_range = get_stock_date_range(tickers)

    file_to_process = 'data/COTAHIST_A2015.TXT'

    try:

        cota_df = parse_cothist(file_to_process)

        if not cota_df.empty:
            print("\n--- Primeiras 5 linhas do DataFrame ---")
            print(cota_df.head())
            print("\n--- Informações do DataFrame ---")
            cota_df.info()
            print("\n--- Estatísticas Descritivas ---")
            print(cota_df.describe())

    except Exception as e:
        print(f"Não foi possível criar ou processar o arquivo de exemplo: {e}")


    print("Stop Here")