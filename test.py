import logging

def log(func):
    """
    Log what function is called
    """
    def wrap_log(*args, **kwargs):
        name = func.__name__
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # add file handler
        fh = logging.FileHandler("%s.log" % name)
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(fmt)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        logger.info("Running function: %s" % name)
        result = func(*args, **kwargs)
        logger.info("Result: %s" % result)
        return func
    return wrap_log

@log
def double_function(a):
    """
    Double the input parameter
    """
    return a*2

if __name__ == "__main__":
    value = double_function(2)


class DecoratorTest(object):
    """
    Test regular method vs @classmethod vs @staticmethod
    """

    def __init__(self):
        """Constructor"""
        pass

    def doubler(self, x):
        """"""
        print("running doubler")
        return x*2

    @classmethod
    def class_tripler(klass, x):
        """"""
        print("running tripler: %s" % klass)
        return x*3

    @staticmethod
    def static_quad(x):
        """"""
        print("running quad")
        return x*4

if __name__ == "__main__":
    decor = DecoratorTest()
    print(decor.doubler(5))
    print(decor.class_tripler(3))
    print(DecoratorTest.class_tripler(3))
    print(DecoratorTest.static_quad(2))
    print(decor.static_quad(3))

    print(decor.doubler)
    print(decor.class_tripler)
    print(decor.static_quad)


class Person(object):
    """"""

    def __init__(self, first_name, last_name):
        """Constructor"""
        self.first_name = first_name
        self.last_name = last_name

    @property
    def full_name(self):
        """
        Return the full name
        """
        return "%s %s" % (self.first_name, self.last_name)


from decimal import Decimal

class Fees(object):
    """"""

    def __init__(self):
        """Constructor"""
        self._fee = None

    @property
    def fee(self):
        """
        The fee property - the getter
        """
        return self._fee

    @fee.setter
    def fee(self, value):
        """
        The setter of the fee property
        """
        if isinstance(value, str):
            self._fee = Decimal(value)
        elif isinstance(value, Decimal):
            self._fee = value

if __name__ == "__main__":
    f = Fees()


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