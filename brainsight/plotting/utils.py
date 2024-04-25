from datetime import datetime, timedelta

import numpy as np


def str_to_ms(string: str) -> int:
    """Convert a string timestamp into miliseconds."""
    start = datetime.strptime("", "")
    delta = datetime.strptime(string, "%H:%M:%S") - start
    return int(delta.total_seconds() * 1000)


def ms_to_str(ms: int) -> str:
    """Convert a string timestamp into miliseconds."""
    return str(timedelta(milliseconds=int(ms))).split(".")[0]


def nanpow2db(y):
    """Power to dB conversion"""
    y[y == 0] = np.nan
    return 10 * np.log10(y)
