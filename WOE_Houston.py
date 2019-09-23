import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
import string
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.externals import joblib 
import gc


max_bin = 20
force_bin = 3


# define a binning function
def mono_bin(Y, X, n = max_bin):
    
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin         
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
        d2 = d1.groupby('Bucket', as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3=d3.reset_index(drop=True)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.sum().EVENT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.EVENT_RATE/d3.NON_EVENT_RATE)
    d3["IV"] = (d3.EVENT_RATE-d3.NON_EVENT_RATE)*np.log(d3.EVENT_RATE/d3.NON_EVENT_RATE)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    
    return(d3)

def char_bin(Y, X):
        
    df1 = pd.DataFrame({"X": X, "Y": Y})
    df2 = df1.groupby('X',as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.groups
    d3["MAX_VALUE"] = df2.groups
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y     
    d3["EVENT_RATE"] = d3.EVENT/d3.sum().EVENT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.EVENT_RATE/d3.NON_EVENT_RATE)
    d3["IV"] = (d3.EVENT_RATE-d3.NON_EVENT_RATE)*np.log(d3.EVENT_RATE/d3.NON_EVENT_RATE)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    
    return(d3)

def data_vars(df1, target):
    
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]
    
    x = df1.dtypes.index
    count = -1
    
    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i            
                count = count + 1
                
            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv,ignore_index=True)
    
    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    
    iv['STRENGTH'] = 'useless'
    iv.loc[iv['IV'] >= 0.02, 'STRENGTH'] = 'weak'
    iv.loc[iv['IV'] >= 0.1, 'STRENGTH'] = 'medium'
    iv.loc[iv['IV'] >= 0.3, 'STRENGTH'] = 'strong'
    iv.loc[iv['IV'] > 0.5, 'STRENGTH'] = 'suspicious'
    
    iv = iv.reset_index()
    
    return(iv_df,iv)


class BaseTransformer(BaseEstimator, TransformerMixin):
    """
    Base class.
    """
    def __init__(self, columns=None, suffix='_base'):
        self.columns = columns
        self.suffix = suffix
        self.scaler = None
    
    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns.tolist()
        self.scaler.fit(X[self.columns])
    
    def transform(self, X, y=None):
        return None
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    
    def save(self, path):
        joblib.dump((self.columns, self.suffix, self.scaler), path)
    
    def load(self, path):
        self.columns, self.suffix, self.scaler = joblib.load(path)


class WoeEncoder(BaseTransformer):
    """
    Weight of Evidence Encoding.
    """
    def __init__(self, columns=None, suffix="_woe"):
        self.columns = columns
        self.suffix = suffix
        self.iv_df = pd.DataFrame()
        self.iv = pd.DataFrame()
        
    def fit(self, df, y=None):
        if self.columns is None:
            self.columns = df.columns.tolist()
        
        iv_df, iv = data_vars(df[self.columns], y)
        
        self.iv_df = iv_df
        self.iv = iv
    
    def transform(self, df):
        for col in self.columns:
            LBs = self.iv_df[self.iv_df['VAR_NAME'] == col]['MIN_VALUE'].tolist()
            UBs = self.iv_df[self.iv_df['VAR_NAME'] == col]['MAX_VALUE'].tolist()
            
            criteria = [df[col].between(LBs[i], UBs[i]) for i in range(len(LBs))]
            values = self.iv_df[self.iv_df['VAR_NAME'] == col]['WOE'].tolist()
            default = self.iv_df[self.iv_df['VAR_NAME'] == col][self.iv_df['MIN_VALUE'].isna()]['WOE'].mean()

            df[col + self.suffix] = np.select(criteria, values, default)

            del LBs; del UBs; del criteria; del values; del default; gc.collect()
            
        return df
    
    def fit_transform(self, df, y):
        self.fit(df, y)
        return self.transform(df)