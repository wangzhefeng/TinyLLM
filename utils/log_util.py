"""Console and daily rotating file logging with lazy filesystem initialization."""

import logging
import os
import re
import sys
import threading
from logging import handlers
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
LOG_NAME = os.environ.get("LOG_NAME") or "default"
LOG_DIR = ROOT_PATH / "logs" / LOG_NAME
LOG_LEVEL = os.environ.get("SERVICE_LOG_LEVEL", "INFO")
_LOCK = threading.Lock()
_LOGGER = None


def get_logger():
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    with _LOCK:
        if _LOGGER is not None:
            return _LOGGER
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
        )
        stream_handler = logging.StreamHandler(stream=sys.stderr)
        stream_handler.setLevel(LOG_LEVEL)
        stream_handler.setFormatter(formatter)

        log_dir = ROOT_PATH / "logs" / (os.environ.get("LOG_NAME") or "default")
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = handlers.TimedRotatingFileHandler(
            filename=log_dir / "service",
            when="MIDNIGHT",
            interval=1,
            backupCount=10,
            encoding="utf-8",
        )
        file_handler.suffix = "%Y-%m-%d.log"
        file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)

        logger = logging.getLogger("tinyllm")
        logger.handlers.clear()
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        logger.setLevel(LOG_LEVEL)
        logger.propagate = False
        _LOGGER = logger
        return _LOGGER


class _LazyLogger:
    def __getattr__(self, name):
        return getattr(get_logger(), name)


logger = _LazyLogger()
