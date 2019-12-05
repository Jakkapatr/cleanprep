import sys
import datetime
from abc import ABC

sys.path.append('..')

import pandas as pd

from utils import prep_assist as prep


class EventLog(pd.DataFrame, ABC):
    def __init__(self, *args, **kwargs):
        # Popping out unrelated arguments to pd.DataFrame so it doesn't freak out.
        key_col = kwargs.pop("key_col", None)
        tms_col = kwargs.pop("tms_col", None)
        field_cols = kwargs.pop("field_cols", None)
        num_cols = kwargs.pop("num_cols", None)
        super(EventLog, self).__init__(*args, **kwargs)

        # Well but i want to assign the value
        self.key_col = key_col
        self.tms_col = tms_col
        self.field_cols = field_cols
        self.num_cols = num_cols
        if tms_col:
            self[tms_col] = pd.to_datetime(self[tms_col])

    def get_pivot(self, commands):
        return prep.get_pivot(self, commands)

    # TODO: Add more analytics stuffs into it


class UpdateLog:  # TODO: Convert this part to be pandas_subclass like
    def __init__(self, df, element_col, datetime_col, val_col):

        df[datetime_col] = pd.to_datetime(df[datetime_col])
        self.df = df
        self.element_col = element_col
        self.datetime_col = datetime_col
        self.val_col = val_col
        self.start_col_nm = 'start_' + datetime_col
        self.end_col_nm = 'end_' + datetime_col
        self.daily_log = pd.DataFrame()
        self.range_df = None

    def change_log_to_range(self) -> pd.DataFrame:
        out = self.df.copy()
        out = out.sort_values(self.datetime_col, ascending=False)
        if type(self.val_col) == list:
            out['dup_flg'] = True
            for col in self.val_col:
                out['dup_flg'] = out['dup_flg'] & out.groupby(self.element_col)[col].apply(lambda x: x == x.shift(-1))
        else:
            out['dup_flg'] = out.groupby(self.element_col)[self.val_col].apply(lambda x: x == x.shift(-1))
        out['dup_flg'] = out['dup_flg'].fillna(False)
        out = out[~out['dup_flg']]
        out.drop('dup_flg', axis=1, inplace=True)
        out['diff'] = out.groupby(self.element_col)[self.datetime_col].diff()
        out.rename(columns={self.datetime_col: self.start_col_nm}, inplace=True)
        out[self.end_col_nm] = out[self.start_col_nm] - out['diff']
        out[self.end_col_nm] = out[self.end_col_nm].fillna(datetime.datetime(2222, 12, 31))
        out.drop('diff', axis=1, inplace=True)
        return out


class MultiDumpLog:
    def __init__(self, df, element_col, datetime_col, val_col):
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        self.df = df
        self.element_col = element_col
        self.datetime_col = datetime_col
        self.val_col = val_col
        self.start_col_nm = 'start_' + datetime_col
        self.end_col_nm = 'end_' + datetime_col
        self.daily_log = pd.DataFrame()
        self.range_df = None

    def change_log_to_range(self) -> pd.DataFrame:

        # Logic -1. If product has different value in val_col, ignore
        # Logic 0. If value column is null, it'll be assumed missing
        # Logic 1. If there's information from last week but not this week, end date will be date of this week
        # Logic 2. If there's information starting from this week, start date will be date of last week
        dt_lst = [datetime.datetime(2019, 1, 1)] + list(set(self.df[self.datetime_col])) + [
            datetime.datetime(2222, 12, 31)]
        dt_lst = sorted(dt_lst)

        # Create Batch Date Mapper
        dt_lw, dt_nw = {}, {}
        for i in range(len(dt_lst) - 1):
            dt_nw[dt_lst[i]] = dt_lst[i + 1]
        for i in range(1, len(dt_lst)):
            dt_lw[dt_lst[i]] = dt_lst[i - 1]

        # Drop Unnecessary Duplicates
        out = self.df.copy().sort_values(self.datetime_col)
        out = out.sort_values(self.element_col)

        if type(self.val_col) == list:
            out['dup_flg'] = True
            for col in self.val_col:
                out['dup_flg'] = out['dup_flg'] & out.groupby(self.element_col)[col].apply(lambda x: x == x.shift(1))
        else:
            out['dup_flg'] = out.groupby(self.element_col)[self.val_col].apply(lambda x: x == x.shift(1))

        out['dup_flg'] = out['dup_flg'].fillna(False)
        out = out[~out['dup_flg']]
        out.drop('dup_flg', axis=1, inplace=True)

        # Check conditions logic
        out['_last_week'] = out.groupby(self.element_col)[self.datetime_col].apply(lambda x: x.shift(1))
        out['_next_week'] = out.groupby(self.element_col)[self.datetime_col].apply(lambda x: x.shift(-1))
        out['lw_null'] = (out['_last_week'].isnull()).astype(int)
        out['nw_null'] = (out['_next_week'].isnull()).astype(int)
        # Fulfill logic
        out[self.start_col_nm] = out.apply(
            lambda x: dt_lw[x[self.datetime_col]] if x['lw_null'] else x[self.datetime_col], axis=1)
        out[self.end_col_nm] = out.apply(lambda x: x['_next_week'] if x['nw_null'] else dt_nw[x[self.datetime_col]],
                                         axis=1)
        out.drop(['_last_week', '_next_week', 'lw_null', 'nw_null'], axis=1, inplace=True)
        # Date Adjustments
        out['diff'] = out.groupby(self.element_col)[self.start_col_nm].diff()
        # out[self.end_col_nm] = out[self.start_col_nm] + out['diff']
        out[self.end_col_nm] = out[self.end_col_nm].fillna(datetime.datetime(2222, 12, 31))
        out = out[out[self.end_col_nm] != out[self.start_col_nm]]
        out.drop(['diff', self.datetime_col], axis=1, inplace=True)
        return out.drop_duplicates()

    def generate_daily_value(self, start_date, end_date):
        self.range_df = self.change_log_to_range()
        for d in range((end_date - start_date).days + 1):
            day = start_date + datetime.timedelta(days=d)

            tmp = self.range_df[(self.range_df[self.start_col_nm] <= day) & (
                    self.range_df[self.end_col_nm] > day)][[self.element_col, self.val_col]]
            tmp[self.datetime_col] = day
            self.daily_log = self.daily_log.append(tmp)

        return self.daily_log