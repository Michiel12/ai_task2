import streamlit as st

import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn import metrics

st.title('AI Task 2: Diabetes Prediction App - Michiel Van Loy r0889624')

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
    
    st.header("Decision Tree Classifier")
    st.subheader("Explanation")
    st.write("The Decision Tree Classifier begins with selecting the features. These features will be used to ask questions. These questions are for example about the age, is the person older or younger than 30 years, and these question will form a tree. This will continue until it goes through all features or until a maximum depth is met. The values of the prediction will go through the tree to get a prediction.")
    st.write("As you can see, this algorithm has an accuracy of around 0.75 - 0.85. Also, the decision matrix shows that this algorithm sometimes has a lot of false positive values, this means that the prediction was positive when it should have been negative.")
    st.subheader("Accuracy")
    st.write("Accuracy:", accuracy)
    st.write("Confusion Matrix:")
    st.write("[[True Negative  False Positive]")
    st.write(" [False Negative  True Positive]]")
    st.write(confusion)

elif method == "Support Vector Machine":
    # Support Vector Machine
    from sklearn.svm import SVC
    clf = SVC(kernel='linear')
    clf.fit(X_cat_oh, y)
    prediction = clf.predict(X_cat_oh_test)
    accuracy = metrics.accuracy_score(y_test, prediction)
    confusion = metrics.confusion_matrix(y_test, prediction)

    st.subheader("Explanation")
    st.write("In Linear Support Vector Machine, you start off with for example two categories, cats and dogs. Within this categories SVM will try to find a line to split the data into two groups. This line is called the decision boundary. This line will be based on special points, these points are called support vectors and they make sure the line takes the best route. The line will try to have as much distance between the support vectors, it wants the largest margin. Like expected will SVM first have to train itself but afterwards it can tell the difference between both groups.")
    st.write("As you can see, the accuracy of SVM is not as good, but it is a lot more consistent. When we look at the confusion matrix, you can see that also this algorithm has a lot of false positive values. Luckily it is better to have false positives than false negatives.")
    st.subheader("Accuracy")
    st.write("Accuracy:", accuracy)
    st.write("Confusion Matrix:")
    st.write("[[True Negative  False Positive]")
    st.write(" [False Negative  True Positive]]")
    st.write(confusion)

elif method == "K-Nearest Neighbors":
    # K-Nearest Neighbors
    from sklearn.neighbors import KNeighborsClassifier
    
    # Dropdown for selecting the number of neighbors
    k_neighbors = st.selectbox("Select the number of neighbors (K):", list(range(1, 11)))

    clf = KNeighborsClassifier(n_neighbors=k_neighbors)
    clf.fit(X_cat_oh, y)
    prediction = clf.predict(X_cat_oh_test)
    accuracy = metrics.accuracy_score(y_test, prediction)
    confusion = metrics.confusion_matrix(y_test, prediction)

    st.subheader("Explanation")
    st.write("Also for K-Nearest Neighbors we have data that we want to split in two group, red and blue. These colors will be represented on a map, where each house will have a color. When KNN wants to predict the color of a new house it will look at his neighbors. The amount of neighbors KNN will look at (in my case 4), will decide what color the new house will get. It will choose the color that occurs the most in the neighbors. The amount of neighbors to look at can be different for each scenario, here are some of the number with their accuracy: *1 = 0.81*, *2 = 0.88*, *3 = 0.85*, *4 = 0.89* and *5 = 0.88*. In my case, 4 is the ideal amount of neighbors to look at.")
    st.write("Here you can see that KNN has the best accuracy. This is also because I can choose the amount of neighbors to look at myself so I can choose the best number. Also the confusion matrix has the best output of them all, it has the most right answers, the only downside is that KNN has the highest amount of false negative values.")
    st.subheader("Accuracy")
    st.write("Accuracy:", accuracy)
    st.write("Confusion Matrix:")
    st.write("[[True Negative  False Positive]")
    st.write(" [False Negative  True Positive]]")
    st.write(confusion)
