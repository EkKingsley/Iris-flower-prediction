#Using Flask in python as web api to accept data,
# and predict the type of iris flower

#import necessary packages
from flask import Flask, request, render_template
from flask_wtf import FlaskForm
import joblib
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from wtforms import FloatField, SubmitField
from wtforms.validators import DataRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = '2ac5a75f4c6fa7283d4b3b0a7fa0f93f'

#load model
model = joblib.load('knn_model.pkl')

#create web form according to the data inputs needed
class InputForm(FlaskForm):
    sepalL = FloatField('Sepal Length', validators=[DataRequired()])
    sepalW = FloatField('Sepal Width ', validators=[DataRequired()])
    petalL = FloatField('Petal Length', validators=[DataRequired()])
    petalW = FloatField('Petal Width', validators=[DataRequired()])
    submit = SubmitField('Predict')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = InputForm()
    if request.method == 'POST':
        sepalL = form.sepalL.data
        sepalW = form.sepalW.data
        petalL = form.petalL.data
        petalW = form.petalW.data

        cols = ['Sepal Length (CM)', 'Sepal Width (CM)', 'Petal Length (CM)',
                'Petal Width (CM)']

        data = [[sepalL, sepalW, petalL, petalW]]

        data = pd.DataFrame(data, columns=cols)

        #predict on the data
        predict = "Setosa" if model.predict(data)[0] == 0 else "Versicolor" if model.predict(data)[0] == 1 else "Virginica"



        return render_template('iris.html', title="Predict Iris Flower Type", form=form, predict=predict)

    return render_template('iris.html', title='Predict Iris Flower Type', form=form)


if __name__ == '__main__':
    app.run(port=3000, debug=True)