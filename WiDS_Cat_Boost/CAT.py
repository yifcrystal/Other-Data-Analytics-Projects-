import pandas as pd
import numpy as np
import datetime as dt
import prepro_util
import os

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import catboost
from catboost import CatBoostRegressor
from catboost import Pool, CatBoostClassifier



### read train data
train = pd.read_csv('../../../../Desktop/wids/train_data.csv')

### Data preprocessing


### target column
target = 'contest-tmp2m-14d__tmp2m'


### preprocess train data
pre_train = prepro_util.preprocess_data(train , 4 , "mean" , target)


### Split data

x_train , x_test , y_train , y_test = train_test_split(pre_train[[col for col in pre_train.columns if col != target]],\
                                                       pre_train[target] , test_size = 0.3)


### CAT model training

def cat_boost(depth , rate , x_train , y_train):
### train the model - CatBoost
    model = CatBoostRegressor(cat_features = ['climateregions__climateregion'],
                                  max_depth = depth,
                                  n_estimators = 20000,
                                  eval_metric = 'RMSE',
                                  learning_rate = rate,
                                  verbose = 1,
                                  random_seed = 0).fit(x_train, y_train)
    return model


try:
     os.mkdir('models')
except:
     pass

### containers
#model_list = []
#spec_list = []
testing_list = []

### loops
for depth in range(3 , 11): ### max depth adjustment
    for rate in range(5 , 9): ### learning rate adjustment
        ### model making
        temp_model = cat_boost(depth, rate / 100 ,\
                               x_train, y_train)
        ### prediciton
        ### use RMSE to evaluate
        y_pred = temp_model.predict(x_test)
        mse = mean_squared_error(y_pred, y_test)        
        
        ### store result
        test_string = f"max_d_{depth}_learn_r_{rate}"
        
        ### save model
        temp_model.save_model(f"./models/CATboost_{test_string}.json")
        #model_list.append(temp_model)
        #spec_list.append({"spec" : test_string})
        
        testing_list.append([f"{test_string}" , mse])
        

### save model
# for model_index in range(len(model_list)):
#     model_list[model_index].save_model(f"./models/CATboost_{spec_list[model_index]['spec']}.json")
    
    
### RMSE test result
df = pd.DataFrame(testing_list , columns = ['model_name' , 'RMSE'])
df.to_csv("model_test_result.csv" , index = False)

