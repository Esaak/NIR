from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error  
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import optuna

from perform_visualization import perform_eda, perform_eda_short, performance_visualizations

import os
import re

class TargetPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, near_zero = 1e-4):
        self.near_zero = near_zero

    def fit(self, data: pd.DataFrame, *args):
        self.columns = data.columns
        self.std_y_bias = np.min(data["c_std_y"][data["c_std_y"] > 0])
        self.std_z_bias = np.min(data["c_std_z"][data["c_std_z"] > 0])
        self.mean_y_mean = np.mean(data["c_mean_y"])
        self.mean_z_mean = np.mean(data["c_mean_z"])
        self.mean_y_std = np.std(data["c_mean_y"])
        self.mean_z_std = np.std(data["c_mean_z"])

        return self

    def transform(self, data: pd.DataFrame):
        # data["c_std_y"] -= self.std_y_bias 
        # data["c_std_z"] -= self.std_z_bias
        
        data["c_std_y"] = np.log1p(data["c_std_y"])
        data["c_std_z"] = np.log1p(data["c_std_z"])

        

        # data["c_mean_y"] -= self.mean_y_mean        
        # mean_y_sign =  np.sign(data["c_mean_y"])
        # self.mean_y_sign = mean_y_sign
        # data["c_mean_y"] = np.abs(data["c_mean_y"])        
        # data["c_mean_y"] = np.log(np.abs(data["c_mean_y"]))
        # self.mean_y_log_max = np.max(data["c_mean_y"]) 
        # data["c_mean_y"] -= self.mean_y_log_max
        # data["c_mean_y"] = data["c_mean_y"] * mean_y_sign

        # data["c_mean_z"] -= self.mean_z_mean
        # mean_z_sign =  np.sign(data["c_mean_z"])
        # self.mean_z_sign = mean_z_sign
        # data["c_mean_z"] = np.log(np.abs(data["c_mean_z"]))
        # self.mean_z_log_max = np.max(data["c_mean_z"]) 
        # data["c_mean_z"] -= self.mean_z_log_max
        # data["c_mean_z"] = data["c_mean_z"] * mean_z_sign
        data["c_mean_z"] = (data["c_mean_z"] - self.mean_z_mean)/self.mean_z_std

        return data
    
    def inverse_transform(self, data: pd.DataFrame):
        data["c_std_y"] = np.expm1(data["c_std_y"])
        data["c_std_z"] = np.expm1(data["c_std_z"])
        # data["c_std_y"] += self.std_y_bias
        # data["c_std_z"] += self.std_z_bia

        # data["c_mean_y"] *= self.mean_y_sign
        # data["c_mean_y"] += self.mean_y_log_max
        # data["c_mean_y"] = np.exp(data["c_mean_y"])
        # data["c_mean_y"] *= self.mean_y_sign
        # data["c_mean_z"] *= self.mean_z_sign
        # data["c_mean_z"] += self.mean_z_log_max
        # data["c_mean_z"] = np.exp(data["c_mean_z"])
        # data["c_mean_z"] *= self.mean_z_sign
        
        # data["c_mean_y"] += self.mean_y_mean
        # data["c_mean_z"] += self.mean_z_mean
        data["c_mean_z"] = data["c_mean_z"] * self.mean_z_std + self.mean_z_mean
        return data

random_seed = 42
early_stopping_round = 100
stat_path = "/app/nse/ml"
pattern = re.compile(r'output_*\d')
folder_paths =[]
for folder_name in os.listdir(stat_path):
    if pattern.match(folder_name):
        folder_paths.append(folder_name)
filename_features = "features_full.csv"
filename_target = "target_full.csv"
X = pd.DataFrame()
y = pd.DataFrame()
for folder in folder_paths:
    X_tmp = pd.read_csv("/app/nse/ml" + "/" + folder + "/" + filename_features)
    y_tmp = pd.read_csv("/app/nse/ml" + "/" + folder + "/" + filename_target)
    
    is_unnamed = pd.isna(X_tmp.columns[0]) or str(X_tmp.columns[0]).startswith('Unnamed:')
    is_unnamed_y = pd.isna(y_tmp.columns[0]) or str(y_tmp.columns[0]).startswith('Unnamed:')
    if is_unnamed:
        X_tmp = X_tmp.drop(X_tmp.columns[0], axis=1)
    if is_unnamed_y:
        y_tmp = y_tmp.drop(y_tmp.columns[0], axis=1)
    
    if X_tmp.columns[0] != "y":
        col_y = np.ones(X_tmp.shape[0]) * 1000
        X_tmp.insert(0, "y", col_y)
    
    print(X_tmp.shape, y_tmp.shape)
    X = pd.concat([X, X_tmp], axis = 0)
    y = pd.concat([y, y_tmp], axis = 0)
X.reset_index(inplace=True, drop=True)
y.reset_index(inplace=True, drop=True)
X.shape, y.shape
mask = (y["c_std_y"] != 0) & (y["c_std_z"] != 0)
X = X[mask]
y = y[mask]

X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=random_seed)
X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.1, random_state=random_seed) 
y_train_1 = y_train.copy()
target_pr = TargetPreprocessor()
target_pr.fit_transform(y_train_1)

y_pred ={}
for target in ["c_mean_y", "c_mean_z"]:
    
    mlp_regressor = MLPRegressor()
    mlp_regressor.fit(X_train.copy(), y_train_1[target].copy())
    y_pred_tmp = mlp_regressor.predict(X_test.copy())
    y_pred[target] = y_pred_tmp 

for target in ["c_std_y", "c_std_z"]:
    
    mlp_regressor = MLPRegressor()
    
    mlp_regressor.fit(X_train.copy(), y_train_1[target].copy())
    y_pred_tmp = mlp_regressor.predict(X_test.copy())
    y_pred[target] = y_pred_tmp

target_pr.inverse_transform(y_pred)
performance_visualizations(y_pred, y_test)