import pytest
from app import app as flask_app

@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Student Performance Indicator" in response.data

def test_prediction_endpoint(client):
    form_data = {
        'gender': 'female',
        'ethnicity': 'group B',
        'parental_level_of_education': "bachelor's degree",
        'lunch': 'standard',
        'test_preparation_course': 'none',
        'reading_score': 72,
        'writing_score': 74
    }
    response = client.post("/predictdata", data=form_data)
    assert response.status_code == 200
    assert b"Estimated Score" in response.data