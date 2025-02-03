import logging
import os
import sys
from logging import handlers
import re

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOG_LEVEL = os.environ.get("SERVICE_LOG_LEVEL", "INFO")

default_formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
)
stream_handler = logging.StreamHandler(stream=sys.stderr)
stream_handler.setLevel(LOG_LEVEL)
stream_handler.setFormatter(default_formatter)

log_path = os.path.join(f"{ROOT_PATH}/logs", "service")
# 设置动态删除日志文件周期为365天
time_rotating_file_handler = handlers.TimedRotatingFileHandler(
    filename=log_path, when="MIDNIGHT", interval=1, backupCount=10
)
time_rotating_file_handler.suffix = "%Y-%m-%d.log"
# 需要注意的是suffix和extMatch一定要匹配的上，如果不匹配，过期日志不会被删除。
time_rotating_file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
time_rotating_file_handler.setLevel(LOG_LEVEL)
time_rotating_file_handler.setFormatter(default_formatter)

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.addHandler(time_rotating_file_handler)

logger.setLevel(LOG_LEVEL)
logger.propagate = False
