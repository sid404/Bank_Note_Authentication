# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 23:49:42 2021

@author: SID
"""
import pandas as pd
import numpy as np

df=pd.read_csv("BankNote_Authentication.csv")

print(df.isnull().sum())

X=df.iloc[:,:-1]
Y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,Y_train)
pred=rf.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(Y_test,pred))
print("\n")
print(classification_report(Y_test,pred))

import pickle
filename="bank_note_classifier.pkl"
pickle.dump(rf,open(filename,'wb'))