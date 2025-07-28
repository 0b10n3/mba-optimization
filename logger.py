# logger.py
"""
Configuração do sistema de logging para o projeto.

Este módulo centraliza a configuração do logger, permitindo que outros
módulos (main, fn, strategies) importem e utilizem uma instância
consistente para registrar mensagens de informação, avisos e erros.
"""
import logging
import sys

def setup_logger():
    """
    Configura e retorna uma instância do logger.

    O logger é configurado para exibir mensagens no console com um formato
    padronizado, incluindo data, nível da mensagem e o conteúdo.

    Returns:
        logging.Logger: Instância do logger configurado.
    """
    # Cria um logger com o nome 'portfolio_research'.
    logger = logging.getLogger('portfolio_research')
    logger.setLevel(logging.INFO) # Define o nível mínimo de log a ser exibido.

    # Evita adicionar múltiplos handlers se a função for chamada mais de uma vez.
    if not logger.handlers:
        # Cria um handler para direcionar os logs para a saída padrão (console).
        stream_handler = logging.StreamHandler(sys.stdout)

        # Define o formato das mensagens de log.
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        stream_handler.setFormatter(formatter)

        # Adiciona o handler ao logger.
        logger.addHandler(stream_handler)

    return logger

# Instancia o logger para ser importado por outros módulos.
log = setup_logger()

