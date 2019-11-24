import os
from typing import Dict, Union, List

import numpy as np
import pandas as pd


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

    def test(self):
        return self.group


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def groupby_test(df: pd.DataFrame,
                 index_col: Union[str, List[str]],
                 val_col: Union[str, List[str]],
                 field_cols: Union[str, List[str]],
                 agg:Union[str, List[str]])-> pd.DataFrame:
    return pd.pivot_table(df, values = val_col, index = index_col, columns = field_cols, aggfunc=agg)


def get_pivot(df: pd.DataFrame,
              index_col: str,
              commands: List[Dict[str,Union[str,List[str]]]]) -> pd.DataFrame:
    out = pd.DataFrame(df[index_col].unique()).set_index(index_col)
    for cmd in commands:
        grp = groupby_test(df,
                        index_col=index_col,
                        val_col=cmd['val'],
                        field_cols=cmd['col'],
                        agg=cmd['agg']
                        )
        grp.columns = ['_'.join([str(c) for c in lst]) for lst in grp.columns]
        out = out.join(grp, how='left')
    return out
