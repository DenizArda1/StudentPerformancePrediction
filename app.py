from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.pipeline.prediction_pipeline import PredictionPipeline,CustomData
from src.pipeline.training_pipeline import TrainingPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        prediction_pipeline = PredictionPipeline()
        results, explanation = prediction_pipeline.prediction(pred_df)
        return render_template('home.html',results=results[0],explanation=explanation)

@app.route('/train',methods=['GET','POST'])
def train_model():
    try:
        pipeline = TrainingPipeline()
        score = pipeline.start_training()
        return render_template('train.html',message=f"Model Trained Successfully",score=score)
    except Exception as e:
        return render_template('train.html',error=f"Error Occurred while Training: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)