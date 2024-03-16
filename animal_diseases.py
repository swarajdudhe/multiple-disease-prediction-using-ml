

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('animal_dis_me.csv')

# print first 5 rows of the dataset
heart_data.head()

# print last 5 rows of the dataset
heart_data.tail()

# number of rows and columns in the dataset
heart_data.shape

# getting some info about the data
heart_data.info()

# checking for missing values
heart_data.isnull().sum()

# statistical measures about the data
heart_data.describe()

# checking the distribution of Target Variable
heart_data['prognosis'].value_counts()

"""Splitting the Features and Target"""

#storing data in x and labels in y 
X = heart_data.drop(columns='prognosis', axis=1)
Y = heart_data['prognosis']

# printing data i.e. x
print(X)

# printing labels in y
print(Y)

"""Splitting the Data into Training data & Test Data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# printing the shape of traing data and labels and testing data and labels
print(X.shape, X_train.shape, X_test.shape)

"""Model Training

# logistic regression
"""

#importing algorithm
from sklearn.linear_model import LogisticRegression
model_animal = LogisticRegression()

# training the LogisticRegression model with Training data
model_animal.fit(X_train, Y_train)

"""Model Evaluation"""

# accuracy on training data
X_train_prediction = model_animal.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

#printing accuracy of training data
print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model_animal.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# printing accuracy of testing data
print('Accuracy on Test data : ', test_data_accuracy)


# import pickle
# pickle.dump(model_animal,open('animal_disease.pkl','wb'))

# 