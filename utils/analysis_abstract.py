import pprint

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from scipy.fftpack import fft


class AnalysisUtils:
    @classmethod
    def coverage(cls, data):
        return data.apply(
            lambda v: pd.Series([v.dtype, 1.0 - float(v.isnull().sum()) / len(v), v.isnull().sum(), v.nunique()],
                                ['Type', 'Coverage', 'Total NA\'s', '#Unique']), axis=0)

    @classmethod
    def summary(cls, data):
        def inner(v):
            return pd.Series([v.min(), np.nanpercentile(v, 25), v.mean(), v.median(), np.nanpercentile(v, 75), v.max()],
                             ['min', 'Q1', 'mean', 'median', 'Q3', 'max'])

        def inner_datetime(v):
            return pd.Series([v.min(), v.max()], ['min', 'max'])

        def summarize_categorical_series(series, output_na=True):
            cnt = series.value_counts()
            num_na = series.isnull().sum()
            head = cnt.head()
            others_cnt = cnt.iloc[len(head):].sum()

            print(head)

            if others_cnt > 0:
                print('Others: {}'.format(others_cnt))

            if num_na > 0:
                print('NA: {}'.format(num_na))

        NUMERIC_TYPES = ['uint8', 'int8', 'uint16', 'uint32', 'int32', 'int64', 'float64']

        # Take care of input of Type "Series".
        if isinstance(data, pd.core.series.Series):
            num_na = data.isnull().sum()

            if data.dtype in NUMERIC_TYPES:
                print(inner(data))
            elif data.dtype in ['<M8[ns]']:
                print(inner_datetime(data))
            else:
                summarize_categorical_series(data, False)

            print("NA: {} (from {}; {})".format(num_na, len(data), float(num_na) / len(data)))

            return

        print('Shape: {}'.format(data.shape))
        print('Coverage:')
        print(AnalysisUtils.coverage(data))

        # Summarize numeric columns
        numeric_idx = [x in NUMERIC_TYPES for x in data.dtypes]
        if sum(numeric_idx) > 0:
            numeric_cols = list(data.dtypes[numeric_idx].index)
            rv = data[numeric_cols].apply(inner)
            print('\n{}'.format(rv))

        # Summarize datetime columns
        datetime_idx = [x in ['<M8[ns]'] for x in data.dtypes]
        if sum(datetime_idx) > 0:
            datetime_cols = list(data.dtypes[datetime_idx].index)
            rv = data[datetime_cols].apply(inner_datetime)
            print('\n{}'.format(rv))

        for c in data.columns:
            print('\n' + c)
            summarize_categorical_series(data[c])


class DateService:

    @staticmethod
    def month_number(s, cutoff_date):
        stamp = pd.Timestamp(cutoff_date)

        return 12 * (stamp.year - s.dt.year) + (stamp.month - s.dt.month)

    @staticmethod
    def week_number(s, cutoff_date):
        stamp = pd.Timestamp(cutoff_date)

        return np.ceil((stamp - s).dt.days / 7).astype(int)


class TsOp:

    @staticmethod
    def argmin(row):
        return len(row) - 1 - np.argmin(row)

    @staticmethod
    def argmax(row):
        return len(row) - 1 - np.argmax(row)

    @staticmethod
    def autocorrelation(row, lag=1):
        n = len(row)

        if n < lag + 2:
            return 0

        return stats.pearsonr(row[lag:], row[:n - lag])[0]

    @staticmethod
    def delta(row, lag=1, relative=0):
        if lag < len(row):
            diff = row[-1] - row[-1 - lag]

            if relative == 0:
                return diff
            else:
                base = abs(row[-1] + row[-1 - lag])

                if base > 0.0:
                    return diff / base
                else:
                    return 2e9

        return 0

    @staticmethod
    def fourier(row, f=0):
        n = len(row)

        if n < 3:
            return 0

        return np.real(fft(row))[f]

    @staticmethod
    def ts_rank(df):
        return df.rank(axis=1).iloc[:, -1]

    @staticmethod
    def ts_zscore(df):
        return (df.iloc[:, -1] - df.mean(axis=1)) / df.std(axis=1)

    @staticmethod
    def up_down_ratio(row):
        if len(row) == 0:
            return 0

        up = (row[1:] > row[:-1]).sum()
        down = (row[1:] < row[:-1]).sum()

        return up / (down + 1)


class TJUtils:
    @classmethod
    def cont_effect(cls, data, predictor, label, target_value=1, bins=10, quantile_cut=True):
        if data[predictor].nunique() < bins:
            TJUtils.discrete_effect(data, predictor, label, target_value)
            return

        if quantile_cut:
            g = data.groupby(pd.qcut(data[predictor], q=bins, duplicates='drop'))
        else:
            g = data.groupby(pd.cut(data[predictor], bins=bins))

        avg = g[label].mean()
        cnt = g[label].size()

        sns.pointplot(x=list(range(len(avg))), y=avg, ci=None)

        x = plt.gca().axes.get_xlim()
        benchmark = table(data[label])[target_value]
        plt.plot(x, len(x) * [benchmark], color='orange', linestyle='--')

        print(pd.DataFrame({'bin': range(len(avg)),
                            'signal': avg,
                            'diff': avg - benchmark,
                            'count': cnt,
                            'frac': cnt / cnt.sum()},
                           columns=['bin', 'signal', 'diff', 'count', 'frac']))

    @classmethod
    def discrete_effect(cls, data, predictor, label, target_value=1, **kwargs):
        g = data.groupby(predictor)
        avg = g[label].mean()
        cnt = g[label].size()

        sns.barplot(x=predictor, y=label, data=data, ci=None, order=avg.index, **kwargs)

        benchmark = table(data[label])[target_value]

        # Add a benchmark line
        x = plt.gca().axes.get_xlim()
        plt.plot(x, len(x) * [benchmark], color='orange', linestyle='--')

        print(pd.DataFrame({'signal': avg, 'diff': avg - benchmark, 'count': cnt, 'frac': cnt / cnt.sum()},
                           columns=['signal', 'diff', 'count', 'frac']))

    @classmethod
    def drop_const_columns(cls, data):
        mask = data.nunique() == 1
        all_cols = data.columns.values
        to_drop = all_cols[mask]

        data.drop(to_drop, axis=1, inplace=True)

    @classmethod
    def hist(cls, series, figsize=(15, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        sns.distplot(series, kde=False, ax=ax)


def percentile(series):
    return pd.Series([np.nanpercentile(series, x) for x in range(101)], list(range(101)))


def reduce_mem_usage(df, use_float16=False, verbose=True):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

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

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem,
                                                                              100 * (start_mem - end_mem) / start_mem))
    return df


def autocorrelation_columns(df, columns, lookback=1):
    n = len(columns)
    rows = len(df)

    if n < lookback + 2:
        return np.zeros(rows)

    reshaped = [[df[c].iloc[i] for c in columns] for i in range(rows)]
    rv = np.zeros(rows)

    for i in range(rows):
        v = reshaped[i]

        rv[i] = stats.pearsonr(v[lookback:], v[:n - lookback])[0]

    return rv


def fourier_columns(df, columns, f=0):
    n = len(columns)
    rows = len(df)

    if n < 3:
        return np.zeros(rows)

    reshaped = [[df[c].iloc[i] for c in columns] for i in range(rows)]
    rv = np.zeros(rows)

    for i in range(rows):
        v = reshaped[i]

        rv[i] = np.real(fft(v))[f]

    return rv


def summary(data):
    AnalysisUtils.summary(data)


def table(series, normalize=True):
    if isinstance(series, np.ndarray):
        series = pd.Series(series)

    return series.value_counts(normalize=normalize)