"""Create global logger to be used across modules"""
import logging
from colorlog import ColoredFormatter

LOG_LEVEL = logging.DEBUG
LOGFORMAT = "%(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"

logging.root.setLevel(LOG_LEVEL)

formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()

stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)

logger = logging.getLogger(__name__)

logger.setLevel(LOG_LEVEL)
logger.addHandler(stream)
