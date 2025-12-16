import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_columns = ['writing score','reading score']
            categorical_columns = ['gender','race/ethnicity','parental level of education','lunch','test preparation course']

            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            logging.info("Numerical columns scaled")
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
                    ('onehot', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical columns encoded")
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", numerical_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ],
                remainder='passthrough'
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            logging.info("test_data and train_data loaded")

            preprocessing_obj = self.get_data_transformer_obj()
            target_column_name = 'math score'
            numerical_columns = ['writing score', 'reading score']
            input_feature_train_data = train_data.drop(columns=[target_column_name],axis=1)
            target_feature_train_data = train_data[target_column_name]

            input_feature_test_data = test_data.drop(columns=[target_column_name],axis=1)
            target_feature_test_data = test_data[target_column_name]

            logging.info("Applying preprocessor on training data and testing data")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_data)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_data)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_data)]

            logging.info("Saving preprocessed data")
            save_obj(filepath=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)


