#import Flask
from flask import Flask, render_template, request
from textblob import TextBlob
import numpy as np
import pandas as pd
import pickle
import re 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        #get form data
        words = request.form.get('words')

        data = [{'tweets':words}]
        #call preprocessDataAndPredict and Pass inputs
        prediction = preprocessDataAndPredict(data)
        #pass prediction to template
        return render_template('predict.html', prediction = prediction)
    pass

def preprocessDataAndPredict(data):
    test_data=pd.DataFrame(data) #store inputs in dataframe

    test_data['tweets'],_=pd.factorize(test_data['tweets'])

    with open('my_classifier.pickle', 'rb') as file:  
        trained_model = pickle.load(file)

    prediction = trained_model.classify(test_data)    
    return prediction

    
    pass

if __name__ == '__main__':
    app.run(debug=True)

