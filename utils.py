import datetime as dt

from numpy import dtype
from sqlalchemy import NVARCHAR, FLOAT


class Timer:
    """
    for stopwatch
    """

    def __init__(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        print('Time taken: %s' % (dt.datetime.now() - self.start_dt))


def convert_dtypes(dtypes):
    """
    For conversions of data types in db
    :param dtypes:
    :return:
    """
    for k, v in dtypes.items():
        if v == dtype('float64'):
            dtypes[k] = FLOAT()
        elif v == dtype('O'):
            dtypes[k] = NVARCHAR(length=255)
    return dtypes


def get_diff_intervals(old_l, old_r, new_l, new_r):
    """
    return intervals that not covered in old one
    If they are not intersected, make it consecutive
    :param old_l:
    :param old_r:
    :param new_l:
    :param new_r:
    :return:
    """
    intervals = []
    if new_l < old_l:
        intervals.append([new_l, old_l - 1])
    if new_r > old_r:
        intervals.append([old_r + 1, new_r])
    return intervals
