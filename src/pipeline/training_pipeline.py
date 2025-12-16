import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from src.exception import CustomException

class TrainingPipeline:
    def __init__(self):
        pass
    def start_training(self):
        try:
            obj = DataIngestion()
            train_data_path,test_data_path = obj.initiate_data_ingestion()

            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_data_path,
                test_data_path
            )
            model_trainer = ModelTrainer()
            r2 = model_trainer.initiate_model_trainer(train_arr, test_arr)
            return r2
        except Exception as e:
            raise CustomException(e,sys)