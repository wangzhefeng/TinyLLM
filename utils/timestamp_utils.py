# -*- coding: utf-8 -*-

# ***************************************************
# * File        : timestamp_utils.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-19
# * Version     : 0.1.031901
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import pytz
from typing import Union
from datetime import datetime

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def to_unix_time(dt: Union[str, datetime], dt_format: str = '%Y-%m-%d %H:%M:%S') -> int:
    """
    timestamp to unix
    """
    epoch = datetime.fromtimestamp(0)
    if isinstance(dt, str):
        dt = datetime.strptime(dt, dt_format)
    unix_time = int((dt - epoch).total_seconds())

    return unix_time


def from_unix_time(unix_time: int, tz: str = None) -> datetime:
    """
    将 Unix 时间戳转换为指定时区的日期时间格式

    Args:
        unix_time (int): 需要转化的 unix 时间戳
        time_zone (str): 时区. Defaults to "Asia/Shanghai".

    Returns:
        datetime: datetime 格式
    """
    if tz:
        local_tz = pytz.timezone(tz)
        timestamp = datetime \
            .fromtimestamp(int(unix_time)) \
            .replace(tzinfo = pytz.utc) \
            .astimezone(local_tz)
    else:
        timestamp = datetime.fromtimestamp(int(unix_time))

    return timestamp


def align_timestamp(timestamp: int, time_zone: str = "Asia/Shanghai", resolution: str = "5s") -> int:
    """
    Align the UTC `timestamp` to the `resolution` with respect to `time_zone`.

    Args:
        timestamp (Number): 需校准时间戳
        time_zone (str, optional): 时区. Defaults to "Asia/Shanghai".
        resolution (str, optional): 需调整聚合度. Defaults to "5s".

    Raises:
        ValueError: 聚合度输入不正确

    Returns:
        int: 对齐后的 Unix 时间戳

    Example:
    >>> _align_timestamp(1503497069, "America/Chicago", resolution="1s")
    1503497069
    >>> _align_timestamp(1503497069, "UTC", resolution="5s")
    1503497065
    >>> _align_timestamp(1503497069, "Europe/Moscow", resolution="10s")
    1503497060
    >>> _align_timestamp(1503497069, "Europe/London", resolution="15s")
    1503497055
    >>> _align_timestamp(1503497069, "Europe/London", resolution="15sec")
    1503497055
    >>> _align_timestamp(1503497069, "Asia/Shanghai", resolution="1min")
    1503497040
    >>> _align_timestamp(1503497069, "Africa/Cairo", resolution="5min")
    1503496800
    >>> _align_timestamp(1503497069, "Europe/Brussels", resolution="10min")
    1503496800
    >>> _align_timestamp(1503497069, "Asia/Jerusalem", resolution="15min")
    1503496800
    >>> _align_timestamp(1503497069, "Asia/Calcutta", resolution="1h")
    1503495000
    >>> _align_timestamp(1503497069, "America/New_York", resolution="1h")
    1503496800
    >>> _align_timestamp(1503497069, "America/Los_Angeles", resolution="12h")
    1503471600
    >>> _align_timestamp(1503497069, "Australia/Sydney", resolution="1d")
    1503496800
    """
    tz = pytz.timezone(time_zone)
    if resolution is None:
        return timestamp
    elif resolution == "1s":
        return int(timestamp)
    elif resolution == "5s":
        return int(timestamp / 5) * 5
    elif resolution == "10s":
        return int(timestamp / 10) * 10
    elif resolution == "15s":
        return int(timestamp / 15) * 15
    elif resolution == "1min":
        return int(timestamp / 60) * 60
    elif resolution == "5min":
        return int(timestamp / 300) * 300
    elif resolution == "10min":
        return int(timestamp / 600) * 600
    elif resolution == "15min":
        return int(timestamp / 900) * 900
    elif resolution == "1h":
        dt = datetime.fromtimestamp(timestamp, tz = tz).replace(minute = 0, second = 0, microsecond = 0)
        return int((dt - datetime(1970, 1, 1, tzinfo=pytz.utc)).total_seconds())
    elif resolution == "6h":
        dt = datetime.fromtimestamp(timestamp, tz = tz).replace(minute = 0, second = 0, microsecond = 0)
        dt = dt.replace(hour = int(dt.hour / 6) * 6)
        return int((dt - datetime(1970, 1, 1, tzinfo = pytz.utc)).total_seconds())
    elif resolution == "8h":
        dt = datetime.fromtimestamp(timestamp, tz = tz).replace(minute = 0, second = 0, microsecond = 0)
        dt = dt.replace(hour = int(dt.hour / 8) * 8)
        return int((dt - datetime(1970, 1, 1, tzinfo = pytz.utc)).total_seconds())
    elif resolution == "12h":
        dt = datetime.fromtimestamp(timestamp, tz = tz).replace(minute = 0, second = 0, microsecond = 0)
        dt = dt.replace(hour = int(dt.hour / 12) * 12)
        return int((dt - datetime(1970, 1, 1, tzinfo = pytz.utc)).total_seconds())
    elif resolution == "1d":
        dt = datetime.fromtimestamp(timestamp, tz = tz).replace(hour = 0, minute = 0, second = 0, microsecond = 0)
        return int((dt - datetime(1970, 1, 1, tzinfo = pytz.utc)).total_seconds())
    elif resolution == "1mo":
        dt = datetime.fromtimestamp(timestamp, tz = tz).replace(day = 1, hour = 0, minute = 0, second = 0, microsecond = 0)
        return int((dt - datetime(1970, 1, 1, tzinfo = pytz.utc)).total_seconds())
    elif resolution == "1y":
        dt = datetime.fromtimestamp(timestamp, tz = tz).replace(month = 1, day = 1, hour = 0, minute = 0, second = 0, microsecond = 0)
        return int((dt - datetime(1970, 1, 1, tzinfo = pytz.utc)).total_seconds())
    else:
        raise ValueError("Invalid resolution: %s" % resolution)






# 测试代码 main 函数
def main():
    timestamp_str = "2020-06-03 09:41:44"
    timestamp_format = '%Y-%m-%d %H:%M:%S'
    
    timestamp = datetime.strptime(timestamp_str, timestamp_format)

    res = to_unix_time(timestamp_str)
    print(res)

    res = to_unix_time(timestamp)
    print(res)
    # -----------
    unix = 1591148504
    res = from_unix_time(unix)
    print(res)
    
    res = from_unix_time(unix, tz = "UTC")
    print(res)

    res = from_unix_time(unix, tz = "Asia/Shanghai")
    print(res)

if __name__ == "__main__":
    main()
