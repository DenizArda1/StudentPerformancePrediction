import os
import sys
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_obj(filepath,obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)

        with open(filepath,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def hyperparameter_tuning():
    param_grid = {
        "DecisionTreeRegressor": {
            "criterion": ["squared_error", "friedman_mse","absolute_error","poisson"],
            #"max_features": ["sqrt", "log2"],
            "splitter": ["best","random"]
        },
        "RandomForestRegressor": {
            "n_estimators": [8,16,32,64,128],
            "max_features": ["sqrt", "log2",None],
            "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"]
        },
        "KNeighborsRegressor": {
            "n_neighbors": [5,10,15],
            "weights": ["uniform","distance"],
            "algorithm": ["auto","ball_tree","kd_tree","brute"]
        },
        "XGBRegressor": {
            "learning_rate": [0.1,0.01,0.05],
            "n_estimators": [8,16,32,64]
        },
        "CatBoostRegressor": {
            "learning_rate": [0.1,0.01,0.05],
            "iterations": [10,30,50],
            "depth":[6,8,10,12,14]
        },
        "AdaBoostRegressor": {
            "learning_rate": [0.1,0.01,0.05],
            "n_estimators": [8,16,32,64],
            "loss":['linear','square','exponential']
        },
        "SVR": {
            "kernel": ["linear","poly","rbf"],
            "C": [1,5,10],
            "degree": [2,3]
        }
    }
    return param_grid

def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        params = hyperparameter_tuning()
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = params.get(model_name,{})
            gs = GridSearchCV(estimator=model,param_grid=para,n_jobs=-1,cv=3)

            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[model_name] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)

def load_obj(filepath):
    try:
        with open(filepath,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)