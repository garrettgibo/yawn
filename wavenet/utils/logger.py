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
    FORMAT = "%(asctime)s  {color}[%(levelname)s]{padding}\033[0m  %(message)s"

    FORMATS = {
        logging.DEBUG: FORMAT.format(color=grey, padding=" " * 3),
        logging.INFO: FORMAT.format(color=green, padding=" " * 4),
        logging.WARNING: FORMAT.format(color=yellow, padding=" "),
        logging.ERROR: FORMAT.format(color=red, padding=" " * 3),
        logging.CRITICAL: FORMAT.format(color=bold_red, padding=" "),
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

    # Start up Message
    logger.info("=" * 60)
    logger.info("Initiating %s", name)

    return logger
