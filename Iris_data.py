# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:32:49 2020

@author: Piyush Mourya
Project- IRIS DATASET
"""

# Importing required libraries
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Read the dataset
data = pd.read_csv('iris.csv')
print(data.head())

print('\n\nColumn Names\n\n')
print(data.columns)

# Label encode the target variable
encode = LabelEncoder()
data.Species = encode.fit_transform(data.Species)

print(data.head())

# Splitting data into training and test set
train, test = train_test_split(data,test_size=0.2,random_state=0)

print('shape of train data : ',train.shape)
print('shape of testing data',test.shape)

# seperating the target and independent variables
train_x = train.drop(columns=['Species'],axis=1)
train_y = train['Species']

test_x = test.drop(columns=['Species'],axis=1)
test_y = test['Species']

# Creating the object of the model
model = LogisticRegression()
model.fit(train_x,train_y)
predict = model.predict(test_x)

print('Predicted Values on Test Data',encode.inverse_transform(predict))

print('\n\nAccuracy score on test data : \n\n' )
print(accuracy_score(test_y,predict))