import os
import datetime

import numpy as np
import pandas as pd

from utils import prep_assist as prep


class EventLog(pd.DataFrame):
    @property
    def _constructor(self):
        return EventLog

    # Referencing from https://dev.to/pj_trainor/extending-the-pandas-dataframe-133l
    # def __init__(self,data, key_col, tms_col, field_cols, num_cols,*args, **kwargs):

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

    def get_pivots(self):
        return self.group


class UpdateLogsTable:
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

    def change_log_to_range(self):
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

    def generate_daily_value(self, start_date, end_date):
        self.range_df = self.change_log_to_range()
        for d in range((end_date - start_date).days + 1):
            day = start_date + datetime.timedelta(days=d)

            tmp = self.range_df[(self.range_df[self.start_col_nm] <= day) & (
                self.range_df[self.end_col_nm] > day)][[self.element_col, self.val_col]]
            tmp[self.datetime_col] = day
            self.daily_log = self.daily_log.append(tmp)
        return self.daily_log
