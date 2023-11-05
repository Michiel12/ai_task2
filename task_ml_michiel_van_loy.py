import streamlit as st

import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn import metrics

st.title('Diabetes Prediction App')

# read and split data
diabete_df = pd.read_csv('./resources/diabetes_data_upload.csv', sep=",")

# print information about data
diabete_df.info()

# print the number of null values
print(diabete_df.isnull().sum())

feature_cols = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis','muscle stiffness', 'Alopecia', 'Obesity']

# split dataset in features and target variable
X = diabete_df[feature_cols]
y = diabete_df['class']

# split data in training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# transform data using One Hot Encoding 
ce_oh = ce.OneHotEncoder(cols = feature_cols[1:])
X_cat_oh = ce_oh.fit_transform(X)
X_cat_oh_test = ce_oh.fit_transform(X_test)



# Decision Three Classifier
from sklearn.tree import DecisionTreeClassifier

# fit the classifier
## build the models
clf = DecisionTreeClassifier(criterion = "entropy")
## train the classifiers
clf = clf.fit(X_cat_oh, y)

# create predictions
prediction = clf.predict(X_cat_oh_test)
# Calculate and print the accuracy
accuracy = metrics.accuracy_score(y_test, prediction)
print("Accuracy:", accuracy)



# Support Vector Machine
from sklearn.svm import SVC

# Create an SVM classifier
clf = SVC(kernel='linear')  # You can choose different kernels like 'linear', 'rbf', 'poly', etc.

# Train the classifier on the training data
clf.fit(X_cat_oh, y)

# Make predictions on the test data
prediction = clf.predict(X_cat_oh_test)

# Calculate and print the accuracy of the classifier
accuracy = metrics.accuracy_score(y_test, prediction)
print("Accuracy:", accuracy)



# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create a KNN classifier with k=4 (you can choose the value of k)
clf = KNeighborsClassifier(n_neighbors=4)

# Train the classifier on the training data
clf.fit(X_cat_oh, y)

# Make predictions on the test data
prediction = clf.predict(X_cat_oh_test)

# Calculate and print the accuracy of the classifier
accuracy = metrics.accuracy_score(y_test, prediction)
print("Accuracy:", accuracy)
