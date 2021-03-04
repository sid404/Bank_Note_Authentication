# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 00:03:57 2021

@author: SID
"""
from flask import Flask,request,render_template
import jsonify
import pandas as pd
import pickle

app=Flask(__name__)
pickle_file=open('bank_note_classifier.pkl','rb')
classifier=pickle.load(pickle_file)

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        variance=float(request.form['variance'])
        skewness=float(request.form['skewness'])
        curtosis=float(request.form['curtosis'])
        entropy=float(request.form['entropy'])
        pred=classifier.predict([[variance,skewness,curtosis,entropy]])
        if pred==0:
            return render_template('index.html',prediction_text="It is not a legitimate note")
        else:
            return render_template('index.html',prediction_text="It is a legitimate note")
    else:
        return render_template('index.html')
    
if __name__=="__main__":
    app.debug=True
    app.run()