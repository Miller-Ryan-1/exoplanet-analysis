import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split

def outlier_destroyer(df,k,cols_to_remove):
    '''
    Finds the outliers for each column using k * IQR; for many dataframes it's lower bound becomes zero
    Then it removes the rows which contain any of these outlier values (above or below the bounds)
    '''
    #start_size = df.shape[0]
    outlier_cutoffs = {}
    # Take out the features that I do not want to remove 'outliers' from: basically non-continuous features
    for col in df.drop(columns = cols_to_remove).columns:
        q1 = df[col].quantile(.25)
        q3 = df[col].quantile(.75)
        iqr = (q3 - q1)
        if q1-(k * iqr) < 0:
            low_outlier = 0
        else:
            low_outlier = q1 - (k * iqr)
        high_outlier = q3 + (k * iqr)

        outlier_cutoffs[col] = {}
        outlier_cutoffs[col]['upper_bound'] = high_outlier
        outlier_cutoffs[col]['lower_bound'] = low_outlier
    
    for col in df.drop(columns = cols_to_remove).columns:
        
        col_upper_bound = outlier_cutoffs[col]['upper_bound']
        col_lower_bound = outlier_cutoffs[col]['lower_bound']

        #remove rows with an outlier in that column
        df = df[df[col] <= col_upper_bound]
        df = df[df[col] >= col_lower_bound]
    #end_size = df.shape[0]
    #print(f'Removed {start_size - end_size} rows, or {(100*(start_size-end_size)/start_size):.2f}%')
    return df


def splitter(df, target = 'None', train_split_1 = .8, train_split_2 = .7, random_state = 123):
    '''
    Splits a dataset into train, validate and test dataframes.
    Optional target, with default splits of 56% 'Train' (80% * 70%), 20% 'Test', 24% Validate (80% * 30%)
    Defailt random seed/state of 123
    '''
    if target == 'None':
        train, test = train_test_split(df, train_size = train_split_1, random_state = random_state)
        train, validate = train_test_split(train, train_size = train_split_2, random_state = random_state)
        print(f'Train = {train.shape[0]} rows ({100*(train_split_1*train_split_2):.1f}%) | Validate = {validate.shape[0]} rows ({100*(train_split_1*(1-train_split_2)):.1f}%) | Test = {test.shape[0]} rows ({100*(1-train_split_1):.1f}%)')
        print('You did not stratify.  If looking to stratify, ensure to add argument: "target = variable to stratify on".')
        return train, validate, test
    else: 
        train, test = train_test_split(df, train_size = train_split_1, random_state = random_state, stratify = df[target])
        train, validate = train_test_split(train, train_size = train_split_2, random_state = random_state, stratify = train[target])
        print(f'Train = {train.shape[0]} rows ({100*(train_split_1*train_split_2):.1f}%) | Validate = {validate.shape[0]} rows ({100*(train_split_1*(1-train_split_2)):.1f}%) | Test = {test.shape[0]} rows ({100*(1-train_split_1):.1f}%)')
        return train, validate, test    