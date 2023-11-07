import streamlit as st

import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn import metrics

st.title('Diabetes Prediction App')

# read and split data
diabetes_df = pd.read_csv('diabetes_data_upload.csv', sep=",")

feature_cols = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis','muscle stiffness', 'Alopecia', 'Obesity']

# split dataset in features and target variable
X = diabetes_df[feature_cols]
y = diabetes_df['class']

# split data in training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# transform data using One Hot Encoding 
ce_oh = ce.OneHotEncoder(cols = feature_cols[1:])
X_cat_oh = ce_oh.fit_transform(X)
X_cat_oh_test = ce_oh.fit_transform(X_test)


# Create buttons for different prediction methods
method = st.radio("Select a prediction method:", ("Decision Tree Classifier", "Support Vector Machine", "K-Nearest Neighbors"))

if method == "Decision Tree Classifier":
    # Decision Tree Classifier
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(X_cat_oh, y)
    prediction = clf.predict(X_cat_oh_test)
    accuracy = metrics.accuracy_score(y_test, prediction)
    confusion = metrics.confusion_matrix(y_test, prediction)
    
    st.write("Accuracy:", accuracy)
    st.write("Confusion Matrix:")
    st.write("[[True Negative (TN)  False Positive (FP)]")
    st.write(" [False Negative (FN)  True Positive (TP)]]")
    st.write(confusion)
    st.write("The Decision Tree Classifier begins with selecting the features. These features will be used to ask questions. These question are for example about the age, is the person older or younger then 30 years, and these question will form a tree. This will continue until it goes trough all features or until a maximum depth is met. The values of the prediction will go trough the tree to get a prediction.")

elif method == "Support Vector Machine":
    # Support Vector Machine
    from sklearn.svm import SVC
    clf = SVC(kernel='linear')
    clf.fit(X_cat_oh, y)
    prediction = clf.predict(X_cat_oh_test)
    accuracy = metrics.accuracy_score(y_test, prediction)
    confusion = metrics.confusion_matrix(y_test, prediction)

    st.write("Accuracy:", accuracy)
    st.write("Confusion Matrix:")
    st.write("[[True Negative (TN)  False Positive (FP)]")
    st.write(" [False Negative (FN)  True Positive (TP)]]")
    st.write(confusion)
    st.write("In Linear Support Vector Machine you start of with for exmaple two categories, cats and dogs. In this categorie SVM will try to find a line to plit the data into two groups. This line is called the decision boundry. This line will be based on special points, these points are called support vectors and they make sure the line takes the best route. The line will try to have as much distance between the support vectors, it wants the largest margin. Like expected will SVM first have to train itself but afterwards it can tell the difference between both groups.")

elif method == "K-Nearest Neighbors":
    # K-Nearest Neighbors
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=4)
    clf.fit(X_cat_oh, y)
    prediction = clf.predict(X_cat_oh_test)
    accuracy = metrics.accuracy_score(y_test, prediction)
    confusion = metrics.confusion_matrix(y_test, prediction)

    st.write("Accuracy:", accuracy)
    st.write("Confusion Matrix:")
    st.write("[[True Negative (TN)  False Positive (FP)]")
    st.write(" [False Negative (FN)  True Positive (TP)]]")
    st.write(confusion)
    st.write("For K-Nearest Neighbors we have data than we want to plit in two group, red and blue. These colors will be represented on a map where each house will have a color. When KNN wants to predict the color of a new house it will look at his neighbors. The amount of neighbors KNN will look at (in my casue 3), will decide what color the new house will get. It will choose the color that occurs the most in the neighbors. The amount of neighbors to look at can be different for each cenario, here are some of the number with their accuracy: *1 = 0.81*, *2 = 0.88*, *3 = 0.85*, *4 = 0.89* and *5 = 0.88*. In my case 4 is the ideal amount of neighbors to look at.")
