"""
Consistent logger setup for all pipeline modules.

Call get_logger(__name__) at the top of each module. Idempotent: calling
twice with the same name returns the same configured logger.
"""

import logging
import sys
from pathlib import Path


_DEFAULT_FORMAT = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def get_logger(name, level=logging.INFO, log_file=None):
    """
    Return a configured logger.

    Parameters
    ----------
    name : str
        Typically __name__ of the calling module.
    level : int
        Logging level (default INFO).
    log_file : str or Path or None
        If provided, also write to this file.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger