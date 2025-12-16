import pytest
import os
import pandas as pd
from src.pipeline.prediction_pipeline import PredictionPipeline

ARTIFACTS_EXIST = os.path.exists("artifacts/model.pkl") and os.path.exists("artifacts/preprocessor.pkl")

@pytest.mark.skipif(not ARTIFACTS_EXIST, reason="Artifacts not found")
def test_prediction_pipeline():
    input_data = {
        'gender': ['female'],
        'race/ethnicity': ['group B'],
        'parental level of education': ["bachelor's degree"],
        'lunch': ['standard'],
        'test preparation course': ['none'],
        'reading score': [72],
        'writing score': [74]
    }
    df = pd.DataFrame(input_data)
    pipeline = PredictionPipeline()
    try:
        prediction, explanation = pipeline.prediction(df)
        assert isinstance(prediction[0],(int,float))
        assert 0 <= prediction[0] <= 100
        assert isinstance(explanation,str)
        assert len(explanation) > 0
    except Exception as e:
        pytest.fail(str(e))

