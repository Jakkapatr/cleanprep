import re

import numpy as np
import pandas as pd
from scipy import stats

from typing import Dict, Union, List


def add_datepart(df: pd.DataFrame, fldname: str, drop: bool =True):
    '''Add a bunch of datetime-related features to a dataframe
    Generated Features list of that column:
        'Year', 'Month', 'Week', 'Day', 'Dayofweek',
        'Dayofyear', 'Is_month_end', 'Is_month_start',
        'Is_quarter_end', 'Is_quarter_start', 'Is_year_end',
        'Is_year_start'
    Args:
    - df (pd.DataFrame) : DataFrame that
    - fldname (string) : Column name of datetime column to generate feature
    - drop (boolean) : If True, the function will drop fldname column from df

    Returns:
    This function is not returning any parameters.
    '''
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
              'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt,n.lower())
    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop: df.drop(fldname, axis=1, inplace=True)


def reduce_mem_usage(df, use_float16=False, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    col: str
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2    
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def add_isnull_cols(df, col_lst):
    for col in col_lst:
        df[col+'_null']  = df[col].isnull().astype(np.int8)
    return df


def interpolate_values(df, group_col, tms_col):
    df = df.sort_values(tms_col)
    df = df.groupby(group_col).apply(lambda g: g.interpolate(method='bfill')).reset_index()
    return


def add_lag_feature(df, group_by, group_cols, window=3):
    rolled = df.groupby(group_by)[group_cols].rolling(window=window, min_periods=0)
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    lag_max = rolled.max().reset_index().astype(np.float16)
    lag_min = rolled.min().reset_index().astype(np.float16)
    lag_std = rolled.std().reset_index().astype(np.float16)
    for col in group_cols:
        df[f'{col}_mean_lag{window}'] = lag_mean[col]
        df[f'{col}_max_lag{window}'] = lag_max[col]
        df[f'{col}_min_lag{window}'] = lag_min[col]
        df[f'{col}_std_lag{window}'] = lag_std[col]
    return


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
    return pd.pivot_table(df, values=val_col, index=index_col, columns=field_cols, aggfunc=agg)


def get_pivot(df: pd.DataFrame,
              index_col: str,
              commands: List[Dict[str,Union[str,List[str]]]]) -> pd.DataFrame:
    out = pd.DataFrame(df[index_col].unique()).set_index(index_col)
    for cmd in commands:
        grp = groupby_test(df=df, index_col=index_col, val_col=cmd['val'], field_cols=cmd['col'], agg=cmd['agg'])
        grp.columns = ['_'.join([str(c) for c in lst]) for lst in grp.columns]
        out = out.join(grp, how='left')
    return out