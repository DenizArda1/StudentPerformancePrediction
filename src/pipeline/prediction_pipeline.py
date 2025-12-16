import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_obj
from src.schema import StudentPerformanceSchema
from pydantic import ValidationError

class PredictionPipeline:
    def __init__(self):
        pass
    def prediction(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            explainer_path = 'artifacts/explainer.pkl'
            model = load_obj(filepath=model_path)
            preprocessor = load_obj(filepath=preprocessor_path)
            explainer = load_obj(filepath=explainer_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            #Prediction should be between 0 and 100
            preds = np.clip(preds, 0, 100)

            shap_values = explainer(data_scaled)
            vals = shap_values.values[0]
            max_impact_index = np.argmax(np.abs(vals))

            impact_factor = ""
            if max_impact_index == 0:
                val = features['writing score'].values[0]
                impact_factor = f"Writing Score ({val})"
            elif max_impact_index == 1:
                val = features['reading score'].values[0]
                impact_factor = f"Reading Score ({val})"
            else:
                all_features = ['writing score', 'reading score']
                try:
                    cat_encoder = preprocessor.named_transformers_['categorical_pipeline'].named_steps['onehot']
                    cat_features = cat_encoder.get_feature_names_out()
                    all_features.extend(cat_features)
                    raw_feature_name = all_features[max_impact_index]

                    #x0 -> gender, x1->race ...
                    feature_map = {
                        "x0": "gender",
                        "x1": "race/ethnicity",
                        "x2": "parental level of education",
                        "x3": "lunch",
                        "x4": "test preparation course"
                    }

                    readable_map = {
                        "gender": "Gender",
                        "race/ethnicity": "Race/Ethnicity",
                        "parental level of education": "Parental Level of Education",
                        "lunch": "Lunch",
                        "test preparation course": "Preparation Course"
                    }

                    # "x0_female" -> "x0" , "female".
                    prefix = raw_feature_name.split('_')[0]

                    if prefix in feature_map:
                        col_name_in_df = feature_map[prefix]
                        readable_name = readable_map.get(col_name_in_df, col_name_in_df)

                        user_value = features[col_name_in_df].values[0]
                        impact_factor = f"{readable_name}: {user_value}"
                    else:
                        impact_factor = raw_feature_name

                except:
                    impact_factor = "Other Factors.."

            return preds, impact_factor

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,gender:str,race_ethnicity:str,parental_level_of_education:str,lunch:str,test_preparation_course:str,
                 reading_score:float,writing_score:float):
        try:
            validated_data = StudentPerformanceSchema(
                gender=gender, #type: ignore
                race_ethnicity=race_ethnicity, #type: ignore
                parental_level_of_education=parental_level_of_education, #type: ignore
                lunch=lunch, #type: ignore
                test_preparation_course=test_preparation_course, #type: ignore
                reading_score=int(reading_score),
                writing_score=int(writing_score)
            )
            self.gender = validated_data.gender
            self.race_ethnicity = validated_data.race_ethnicity
            self.parental_level_of_education = validated_data.parental_level_of_education
            self.lunch = validated_data.lunch
            self.test_preparation_course = validated_data.test_preparation_course
            self.reading_score = validated_data.reading_score
            self.writing_score = validated_data.writing_score
        except ValidationError as e:
            error_messages = "; ".join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
            raise CustomException(f"Data Validation Error: {error_messages}", sys)
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'gender': [self.gender],
                'race/ethnicity': [self.race_ethnicity],
                'parental level of education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test preparation course': [self.test_preparation_course],
                'reading score': [self.reading_score],
                'writing score': [self.writing_score]
            }

            return pd.DataFrame(data=custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)