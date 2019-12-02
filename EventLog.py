import os

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

