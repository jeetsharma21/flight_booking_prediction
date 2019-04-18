#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:36:00 2018

@author: I849589
"""
#importing main libraries to be used
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

#last 5 records in combined csv file is from unlabeled searches csv for which prediction is required 
df = pd.read_csv('labeled_searches_combined.csv', header=0)

#Dropping records having blanks/incomplete data
df.drop(df[df.arr_airport == '-' ].index, inplace=True )
df.drop(df[df.arr_city == '16' ].index, inplace=True )
df.drop(df[df.platform == 'customer_service' ].index, inplace=True )
df.drop(df[df.platform == 'hello' ].index, inplace=True )
df.drop(df[df.days_till_dep < 0 ].index, inplace=True )

#due to redundant info (can be derived information based on airport codes) which in turn will have higher entropy dropping some columns
df = df.drop(['dep_city', 'dep_state', 'dep_country', 'arr_city', 'arr_state', 'arr_country'], axis=1)

#dropping all the nan value rows 
df.dropna(inplace=True) 

#generating dummies encoding for categorical feature columns
data2 = pd.get_dummies(df, columns =[ 'dep_airport', 'arr_airport', 'platform', 'browser'])

z = data2.shape[0] - 5

#Splitting the combined data set into validation(data to predict) and X(training and test) 
#Splitting into validation,training and test dataframes
X3 = data2.iloc[z:,0:2]
X4 = data2.iloc[z:,3:]
X_val = X3.assign(**X4)

X1 = data2.iloc[:-5,0:2]
X2 = data2.iloc[:-5,3:]
X = X1.assign(**X2)
y = data2.iloc[:-5,2]

#splitting the X frame to train test and split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#scaler = preprocessing.StandardScaler().fit(X_train)
#X_train_scaled = scaler.transform(X_train)

#build the logistic regression model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


### accuracy for test data
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

### predict booking on validation data
y_pred = classifier.predict(X_val)
print(y_pred)