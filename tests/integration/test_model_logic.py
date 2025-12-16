import pytest
import pandas as pd
from src.pipeline.prediction_pipeline import PredictionPipeline

def test_model_monotonicity():
    pipeline = PredictionPipeline()

    #Student 1: Low Reading Score
    df_low = pd.DataFrame({
        'gender': ['male'],
        'race/ethnicity': ['group A'],
        'parental level of education': ["high school"],
        'lunch': ['standard'],
        'test preparation course': ['none'],
        'reading score': [50],
        'writing score': [50]
    })

    #Student 2: High Reading Score
    df_high = df_low.copy()
    df_high['reading score'] = 90
    df_high['writing score'] = 90

    pred_low, _ = pipeline.prediction(df_low)
    pred_high, _ = pipeline.prediction(df_high)
    if pred_low[0] > pred_high[0]:
        pytest.fail(
            f"A logic error has detected\n"
            f"Expected: Higher note >= Lower note\n"
            f"Received: Low {pred_low[0]} > High {pred_high[0]}\n"
            f"Difference: {pred_high[0] - pred_low[0]} point"
        )
