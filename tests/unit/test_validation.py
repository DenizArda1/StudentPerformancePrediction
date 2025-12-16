import pytest
from src.pipeline.prediction_pipeline import CustomData
from src.exception import CustomException

#Successful Case
def test_custom_data_valid_input():
    data = CustomData(
        gender="male",
        race_ethnicity="group A",
        parental_level_of_education="high school",
        lunch="standard",
        test_preparation_course="none",
        reading_score=75,
        writing_score=80
    )
    assert data.gender == "male"
    assert data.reading_score == 75
    assert data.writing_score == 80

#Parametrized Testing
@pytest.mark.parametrize("invalid_reading,invalid_writing,expected_msg",[
    (101, 80, "reading_score"),
    (-1, 80, "reading_score"),
    (80, 105, "writing_score"),
    (80, -10, "writing_score")
])
def test_score_boundaries(invalid_reading, invalid_writing, expected_msg):
    with pytest.raises(CustomException) as e:
        CustomData(
            gender="male",
            race_ethnicity="group B",
            parental_level_of_education="some college",
            lunch="standard",
            test_preparation_course="none",
            reading_score=invalid_reading,
            writing_score=invalid_writing
        )
    assert expected_msg in str(e.value)

@pytest.mark.parametrize("field, invalid_value", [
    ("gender", "robot"),
    ("lunch", "burger_king"),
    ("race_ethnicity", "group Z"),
    ("test_preparation_course", "half_completed")
])
def test_invalid_categories(field, invalid_value):
    kwargs = {
        "gender": "male",
        "race_ethnicity": "group A",
        "parental_level_of_education": "high school",
        "lunch": "standard",
        "test_preparation_course": "none",
        "reading_score": 50,
        "writing_score": 50
    }

    kwargs[field] = invalid_value
    with pytest.raises(CustomException) as e:
        CustomData(**kwargs)
    assert "Data Validation Error" in str(e.value)
