import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

'''
These are helper functions for univariate and multivariate exploration of the Exoplanet Dataset
'''

# DATA DIGNITY -------------------------------------------------------------

def column_null_analyzer(df):
    dict_list = []
    for col in df.columns:
        col_nulls = df[col].isnull().sum()
        col_null_percent = col_nulls/len(df.index)
        dict_list.append({'':col,'num_rows_missing':col_nulls,'pct_rows_missing':col_null_percent})
    return pd.DataFrame(dict_list).set_index('').sort_values(by='num_rows_missing', ascending=False)

# UNIVARIATE DISTRIBUTIONS

def univariate(df):
num_cols = [col for col in df.columns if df[col].dtype != 'object']
cat_cols = [col for col in df.columns if df[col].dtype == 'object']

for i in num_cols:
    print(i)
    plt.figure(figsize = (10,5))
    plt.subplot(121)
    sns.boxplot(y=df[i].values)
    plt.subplot(122)
    sns.histplot(x=df[i])
    plt.show()
    print(df[i].value_counts())
    print('\n-----\n')