import functools
import logging
from typing import Callable


def set_logger(log_level: str = "INFO") -> None:
    level = getattr(logging, str(log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def log_decorator(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info("Start %s", func.__name__)
        out = func(*args, **kwargs)
        logging.info("Finish %s", func.__name__)
        return out

    return wrapper
