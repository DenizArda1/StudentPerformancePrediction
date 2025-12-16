import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

import shap
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting train and test data")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],train_arr[:,-1],
                test_arr[:,:-1],test_arr[:,-1]
            )
            models = {
                "Ridge": Ridge(), #Used Ridge instead of Linear Regression to get high accuracy from SHAP library
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "SVR": SVR()
            }
            model_report : dict=evaluate_models(X_train=X_train,y_train=y_train,
                                               X_test=X_test,y_test=y_test,models=models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found",sys)

            logging.info("Best model found")
            save_obj(filepath=self.model_trainer_config.trained_model_file_path,obj=best_model)

            logging.info("Creating SHAP explainer for interpretability")
            try:
                explainer = shap.Explainer(best_model)
            except:
                explainer = shap.KernelExplainer(best_model.predict, X_train[:100])

            explainer_path = os.path.join("artifacts","explainer.pkl")
            save_obj(filepath=explainer_path,obj=explainer)

            predictions = best_model.predict(X_test)
            r2 = r2_score(y_test,predictions)
            return r2

        except Exception as e:
            raise CustomException(e,sys)
