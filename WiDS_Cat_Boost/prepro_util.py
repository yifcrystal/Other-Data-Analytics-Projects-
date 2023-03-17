import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

from catboost import CatBoostRegressor
from catboost import Pool, CatBoostClassifier
import xgboost as xgb


### set index
def set_index_and_to_dt(dataset):
    dataset = dataset.set_index('index')
    dataset['startdate'] = pd.to_datetime(dataset['startdate'] , format = '%m/%d/%y')
    print('index and datetime set')
    return dataset

### categorical data encoding ("climate regions")
def encode_categorical_data(dataset):
    encode = preprocessing.LabelEncoder()
    dataset['climateregions__climateregion'] = encode.fit_transform(dataset['climateregions__climateregion'])
    print('categorical data encoded')
    return dataset

### location data handling (round "lat" and "lon")
def location_data_handle(dataset , round_to):
    dataset.loc[: ,'lat'] = round(dataset.loc[:,'lat'], round_to)
    dataset.loc[: , 'lon'] = round(dataset.loc[: , 'lon'] , round_to)
    print('location data handled')
    return dataset


### calculate missing value with mean
def na_imputer(dataset):
    df = dataset.copy()
    df = df.sort_values(by = ['lat', 'lon' , 'startdate'])
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    for col in dataset.columns:
        if dataset[col].dtypes != "O":
            if dataset[col].dtypes != np.dtype('<M8[ns]'):            
                df[col] = imputer.fit_transform(dataset.loc[ : , col].values.reshape(-1 ,1))
        else:
            continue
    return df

### handle na
def handle_na(dataset , fill_type):
    df = dataset.copy()
    if fill_type == 'f':
        df = dataset.sort_values(['lat' , 'lon' , 'startdate']).ffill()
    elif fill_type == 'b':
        df = dataset.sort_values(['lat' , 'lon' , 'startdate']).ffill()
    else:
        if fill_type == 'mean':
            df = na_imputer(dataset)
        else:
            return df
    return df

### handle datatime (column startdate)
def handle_datetime(dataset):
    df = dataset.copy()
    df['year'] = dataset['startdate'].dt.year
    df['month'] = dataset['startdate'].dt.month
    df['day'] = dataset['startdate'].dt.dayofyear
    print('datetime handled')
    return df

### split predictor and predicted
def x_y_split(dataset , target_column):
    x = dataset[[col for col in dataset.columns if col != target_column]]
    y = dataset[target_column]
    return x , y



### preprocess data function
def preprocess_data(dataset , round_to_which_digit , fill_type , target_column):
    
    temp_df = dataset.copy()
    temp_df = set_index_and_to_dt(temp_df) ### set index and convert datetime
    
    try:
        temp_df , y = x_y_split(temp_df , target_column)
    except:
        pass
    
    #temp_df = encode_categorical_data(temp_df) ### categorical data encoding
    temp_df = location_data_handle(temp_df , round_to_which_digit) ### longtitude and latitude handling
    temp_df = handle_na(temp_df , fill_type) ### handle na
    temp_df = handle_datetime(temp_df) ### create year, month, day
    
    temp_df = temp_df.drop(['startdate'] , axis = 1) ### drop processed column
    
    try:
        temp_df = temp_df.merge(y , left_index = True, right_index = True) ### merge y value back
    except:
        pass
    return temp_df


def PCA_transform(dataset , variance , target):
    
    ### split predictor and predicted
    try:
        x , y = x_y_split(dataset , target)
    except:
        x = dataset

    ### scale
    sc = preprocessing.StandardScaler()
    scaled_x = pd.DataFrame(sc.fit_transform(x) , columns = x.columns)

    ### PCA
    pca_model = PCA(n_components = variance , svd_solver = 'full')
    transformed = pca_model.fit_transform(scaled_x)
    
    ### make dataframe with new components
    columns = []
    for i in range(1 , transformed.shape[1] + 1):
        columns.append(f"component {i}")
    transformed_df = pd.DataFrame(transformed , columns = columns)

    try:
        ### combine transformed x with y
        transformed_df[target] = y
    except:
        pass    
    
    return transformed_df
