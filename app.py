

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import pickle
import joblib
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from flask import Flask, render_template,request



import pickle
model=pickle.load(open('spamclassifier_MnB.pkl','rb')) #rb means it import the file in the read binary mode

vectorizer=pickle.load(open('vectorizer.pkl','rb')) 
#vectorizer converts the text data to the numerical data

app=Flask(__name__)

# @app.route('/',methods=)
@app.route("/",methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        if request.method=='POST':
            messg=str(request.form['mesg'])
            transformed=vectorizer.transform([messg])
            transformed_data=transformed.toarray()
            pred=model.predict(transformed_data)
            output=str(pred[0])
            return render_template('result.html',prediction=f'{output}')
    except Exception as e:
        return render_template('result.html',prediction=f"Error: {str(e)}")









if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)
