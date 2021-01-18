# -*- coding: utf-8 -*-
#importing module
#MODULE IMPORT
import numpy as np
import pandas as pd
import pickle
#import seaborn as sns
#import sklearn
#creatind Dataframe
attr = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']  
file = pd.read_csv('processed.cleveland.csv',header=None)
file = np.array(file)
df = pd.DataFrame(data=file,columns=attr)
#raplacing '?'with nan
def replace1(x):
    if x=='?':
        return np.nan
    else:
        return x
df = df.applymap(replace1)
df['ca'].fillna(1.5,inplace=True)
df['thal'].fillna(3.0,inplace=True)
#replace 1,2,3->0 1 in target for binary prediction
def rep_target(x):
    if x==0:
        return 0
    else:
        return 1
df['target'] = df['target'].apply(lambda x:rep_target(x))

#scaling dataset
X = df.drop('target',axis=1)
y = df['target']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
#print(help(scaler))
X = scaler.transform(X)
#splitting dataset into Train test:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Fitting And Training Model
from sklearn.ensemble import RandomForestClassifier
RFC_Model  =RandomForestClassifier(n_estimators=50)
RFC_Model.fit(X_train,y_train)
y_pred = RFC_Model.predict(X_test)


#evaluation
#from sklearn.metrics import confusion_matrix
#print("confusion matrix \n",confusion_matrix(y_test,y_pred))
#serializing and saving the model
dbfile = open('examplePickle.pkl', 'wb')
pickle.dump(RFC_Model, dbfile)
dbfile.close() 