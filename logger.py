import logging


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'CRITICAL': '\033[95m',
        'RESET': '\033[0m',
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        message = super().format(record)
        return f'{log_color}{message}{self.COLORS["RESET"]}'


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = ColoredFormatter(
    '{asctime} - {levelname} - {message}',
    style='{',
    datefmt='%Y-%m-%d %H:%M',
)

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
