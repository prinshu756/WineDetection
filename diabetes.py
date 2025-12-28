import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import StandardScaler

#data collection and processing
diabetes_data = pd.read_csv('D:/Machine_Learning/diabetes.csv')
X = diabetes_data.drop(columns='Outcome', axis=1)
Y = diabetes_data['Outcome']

#Standardizing the data

scaler = StandardScaler()
X_new = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
print(Y.shape, Y_train.shape, Y_test.shape) 