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
    '''
    Creates list of nulls and their percentage of rows
    '''
    dict_list = []
    for col in df.columns:
        col_nulls = df[col].isnull().sum()
        col_null_percent = col_nulls/len(df.index)
        dict_list.append({'':col,'num_rows_missing':col_nulls,'pct_rows_missing':col_null_percent})
    return pd.DataFrame(dict_list).set_index('').sort_values(by='num_rows_missing', ascending=False)

# UNIVARIATE DISTRIBUTIONS -------------------------------------------------

def univariate(df):
    '''
    Plots distribution of all numerical columns
    '''
    num_cols = [col for col in df.columns if df[col].dtype != 'object']

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

# BI/MULTI-VARIATE ---------------------------------------------------------

def plot_variable_pairs(train):
    '''
    Takes in a cleaned and split (but not scaled or encoded) training dataset 
    and outputs a heatmap and linear regression lines based on correlations. 
    '''
    # Create lists of categorical and continuous numerical columns
    num_cols = [col for col in train.columns if train[col].dtype != 'object']
    cat_cols = [col for col in train.columns if train[col].dtype == 'object']
    
    # Create a correlation matrix from the continuous numerical columns
    df_num_cols = train[num_cols]
    corr = df_num_cols.corr()

    # Pass correlation matrix on to sns heatmap
    plt.figure(figsize=(12,12))
    sns.heatmap(corr, annot=True, cmap="flare", mask=np.triu(corr))
    plt.show()
    
    # Create lm plots for all numerical data
    combos = list(combinations(num_cols,2))
    for i in combos:
        sns.lmplot(x=i[0],y=i[1],data=train, hue=cat_cols[0])
        plt.show()

def plot_categorical_and_continuous_vars(train):
    '''
    Takes in a cleaned and split (but not scaled or encoded) training dataset 
    and outputs charts showing distributions for each of the categorical variables.
    '''
    # Create lists of categorical and continuous numerical columns
    num_cols = [col for col in train.columns if train[col].dtype != 'object']
    cat_cols = [col for col in train.columns if train[col].dtype == 'object']
    
    # Create 3x side-by-side categorial to continous numeric plots
    for i in num_cols:
        plt.figure(figsize = (18,6))
        plt.subplot(1,3,1)
        sns.boxplot(data = train, x=cat_cols[0], y=i)    
        plt.subplot(1,3,2)
        sns.violinplot(data = train, x=cat_cols[0], y=i)
        plt.subplot(1,3,3)
        sns.barplot(data = train, x=cat_cols[0], y=i)
        plt.show()

def plot_numerical_against_target(train,target):
    '''
    Plots all unscaled numerical features against target.
    '''
    num_cols = [col for col in train.columns if train[col].dtype != 'object']

    for i in num_cols:
        sns.relplot(data = train, x=i, y=target)
        plt.show()