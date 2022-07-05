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

def initial_clean(df):
    '''
    Cleans the dataset before the split for EDA
    '''
    # Create Columns/Features
    df['planet'] = df.hostname + '-' + df.pl_letter
    df['multistar'] = np.where(df.sy_snum > 1, 1,0)

    # Filter Columns
    df = df[(df.discoverymethod == 'Transit') | (df.discoverymethod == 'Radial Velocity')] # Drops about 2% of rows, most of which had lots of nulls
    df = df[df.pl_controv_flag == 0] # Drops the controversial planets which are small enough best not to deal with

    # Drop Columns
    df = df.drop(columns = ['discoverymethod', # Indicate drop reasons below
                            'pl_controv_flag',
                            'sy_snum',
                            'cb_flag',
                            'hostname',
                            'glat',
                            'glon',
                            'sy_gaiamag',
                            'pl_ratdor',
                            'pl_dens',
                            'pl_masse',
                            'st_logg',
                            'rowupdate', # Useful only to impute some of the observation values, not for further analysis
                            'disc_year'])

    # Rename Columns
    df = df.rename(columns={'sy_pnum':'num_planets_in_sys','pl_orbper':'orbital_period','pl_rade':'y','st_teff':'star_temp','st_met':'metallicity','st_lum':'luminosity','st_age':'star_age','st_mass':'star_mass','st_dens':'star_density','st_rad':'star_radius','sy_dist':'star_distance_from_earth'})

    # Create the new df *Note: This is what you would replace with the loop/function to impute values
    df = df.groupby(df.planet).mean()

    # Change Datatypes
    df = df.astype({'num_planets_in_sys':'int','multistar':'int'})

    # Add back in any lost columns
    df['discovery_order'] = df.index.str[-1].map({'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9})

    # Encode the target (into Cat and 1/0)
    df['y'] = np.where(df['y'] > .8, np.where(df['y']< 1.15,'Earthlike','Not-Earthlike'),'Not-Earthlike')
    df['y_encoded'] = np.where(df['y'] == 'Earthlike',1,0)

    # Nuke the last of the nulls
    df = df.dropna()

    return df

def final_clean(train,validate,test):
    '''
    Cleans the Dataset after EDA for modeling
    '''
    X_lt_train = train[['luminosity','star_temp']]
    X_lt_validate = validate[['luminosity','star_temp']]
    X_lt_test = test[['luminosity','star_temp']]

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X_lt_train)

    train['lt_cluster'] = kmeans.predict(X_lt_train)
    validate['lt_cluster'] = kmeans.predict(X_lt_validate)
    test['lt_cluster'] = kmeans.predict(X_lt_test)

    train = train.drop(columns = ['luminosity','star_temp','star_mass','star_radius','star_distance_from_earth'])
    validate = validate.drop(columns = ['luminosity','star_temp','star_mass','star_radius','star_distance_from_earth'])
    test = test.drop(columns = ['luminosity','star_temp','star_mass','star_radius','star_distance_from_earth'])

    scale_cols = ['orbital_period','star_density']

    scaler = MinMaxScaler()
    scaler.fit(train[scale_cols])

    train_scaled_features = pd.DataFrame(scaler.transform(train[scale_cols]),columns=['scaled_orb_period','scaled_star_density']).set_index([train.index.values])
    validate_scaled_features = pd.DataFrame(scaler.transform(validate[scale_cols]),columns=['scaled_orb_period','scaled_star_density']).set_index([validate.index.values])
    test_scaled_features = pd.DataFrame(scaler.transform(test[scale_cols]),columns=['scaled_orb_period','scaled_star_density']).set_index([test.index.values])
   
    train_scaled = train.merge(train_scaled_features, left_index = True, right_index = True).drop(columns = scale_cols)
    validate_scaled = validate.merge(validate_scaled_features, left_index = True, right_index = True).drop(columns = scale_cols)
    test_scaled = test.merge(test_scaled_features, left_index = True, right_index = True).drop(columns = scale_cols)

    X_train = train_scaled.drop(columns=['y','y_encoded'])
    y_train = train_scaled.y

    X_validate = validate_scaled.drop(columns=['y','y_encoded'])
    y_validate = validate_scaled.y

    X_test = test_scaled.drop(columns=['y','y_encoded'])
    y_test = test_scaled.y


    return train, X_train, y_train, X_validate, y_validate, X_test, y_test
