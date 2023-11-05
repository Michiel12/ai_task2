import streamlit as st

import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn import metrics

st.title('Diabetes Prediction App')

# read and split data
diabete_df = pd.read_csv('diabetes_data_upload.csv', sep=",")

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


def DTC():
    # Decision Tree Classifier
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(X_cat_oh, y)
    prediction = clf.predict(X_cat_oh_test)
    accuracy = metrics.accuracy_score(y_test, prediction)
    
    st.write("Accuracy:", accuracy)
    st.write("test")

def SVM():
    # Support Vector Machine
    from sklearn.svm import SVC
    clf = SVC(kernel='linear')
    clf.fit(X_cat_oh, y)
    prediction = clf.predict(X_cat_oh_test)
    accuracy = metrics.accuracy_score(y_test, prediction)
    
    st.write("Accuracy:", accuracy)
    st.write("test 2")

st.button(label='test', on_click=DTC())
st.button(label='test 2', on_click=SVM())

if st.button('test'):
    st.write('test string')


# Create buttons for different prediction methods
method = st.radio("Select a prediction method:", ("Decision Tree", "Support Vector Machine", "K-Nearest Neighbors"))

if method == "Decision Tree":
    # Decision Tree Classifier
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(X_cat_oh, y)
    prediction = clf.predict(X_cat_oh_test)
    accuracy = metrics.accuracy_score(y_test, prediction)

    st.write("Accuracy:", accuracy)
    st.write("Explanation: Decision Tree is a non-linear classification model that creates a tree-like structure to make predictions based on the feature values.")

elif method == "Support Vector Machine":
    # Support Vector Machine
    from sklearn.svm import SVC
    clf = SVC(kernel='linear')
    clf.fit(X_cat_oh, y)
    prediction = clf.predict(X_cat_oh_test)
    accuracy = metrics.accuracy_score(y_test, prediction)

    st.write("Accuracy:", accuracy)
    st.write("Explanation: Support Vector Machine is a linear classifier that aims to find the hyperplane that best separates the data into different classes.")

elif method == "K-Nearest Neighbors":
    # K-Nearest Neighbors
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=4)
    clf.fit(X_cat_oh, y)
    prediction = clf.predict(X_cat_oh_test)
    accuracy = metrics.accuracy_score(y_test, prediction)

    st.write("Accuracy:", accuracy)
    st.write("Explanation: K-Nearest Neighbors is a non-parametric classification algorithm that classifies data points based on the majority class of their k-nearest neighbors.")

st.info("Please select a prediction method to see the accuracy and explanation.")
