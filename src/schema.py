from typing import Literal
from pydantic import BaseModel, Field, field_validator


class StudentPerformanceSchema(BaseModel):
    gender: Literal['male', 'female']
    race_ethnicity: Literal['group A', 'group B', 'group C', 'group D', 'group E']
    parental_level_of_education: Literal[
        "bachelor's degree", "some college", "master's degree",
        "associate's degree", "high school", "some high school"
    ]
    lunch: Literal['standard', 'free/reduced']
    test_preparation_course: Literal['none', 'completed']

    reading_score: int = Field(..., ge=0, le=100, description="Writing score should be between 0 and 100.")
    writing_score: int = Field(...,ge=0, le=100, description="Reading score should be between 0 and 100.")

    @field_validator('reading_score', 'writing_score')
    @classmethod
    def check_score_range(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Score should be between 0 and 100')
        return v