"""Create custom logger to be used across modules"""
import logging

LOG_LEVEL = logging.DEBUG


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors.
    based off: https://stackoverflow.com/a/56944256
    """

    grey = "\033[38m"
    green = "\033[32m"
    yellow = "\033[33m"
    red = "\033[31m"
    bold_red = "\033[31;1m"
    reset = "\033[0m"
    FORMAT = "%(asctime)s : %(name)s : %(levelname)s : %(message)s"

    FORMATS = {
        logging.DEBUG: grey + FORMAT + reset,
        logging.INFO: green + FORMAT + reset,
        logging.WARNING: yellow + FORMAT + reset,
        logging.ERROR: red + FORMAT + reset,
        logging.CRITICAL: bold_red + FORMAT + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def new_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Create new logger to be used across modules"""
    logger = logging.getLogger(name)
    formatter = CustomFormatter()
    stream = logging.StreamHandler()

    stream.setFormatter(formatter)
    logger.setLevel(level)

    logger.addHandler(stream)

    return logger
