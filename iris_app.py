import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using 'map()'.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
score = svc_model.score(X_train, y_train)

# creating a Random Forest classifier model
rf_clf=RandomForestClassifier(n_jobs=-1,n_estimators = 100)
rf_clf.fit(X_train,y_train)

#create a Logistic Regression model
log_reg=LogisticRegression(n_jobs=-1)
log_reg.fit(X_train,y_train)



@st.cache()
def prediction(model,SepalLength,SepalWidth,PetalLength,PetalWidth):

  species=model.predict([[SepalLength,SepalWidth,PetalLength,PetalWidth]])
  species=species[0]

  if species== 0:
    return "Iris-setosa"
  elif species==1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor"

#title to the app

st.sidebar.title("Iris Flower Prediction Application")

#slider for sepal and petal lenght and width
## Add 4 sliders and store the value returned by them in 4 separate variables.

s_length=st.sidebar.slider("Sepal Length",float(iris_df['SepalLengthCm'].min()),float(iris_df['SepalLengthCm'].max()))
s_width=st.sidebar.slider("Sepal Width",float(iris_df['SepalWidthCm'].min()),float(iris_df['SepalWidthCm'].max()))
p_length=st.sidebar.slider("Petal Lenght",float(iris_df['PetalLengthCm'].min()),float(iris_df['PetalLengthCm'].max()))
p_width=st.sidebar.slider("Petal width",float(iris_df['PetalWidthCm'].min()),float(iris_df['PetalWidthCm'].max()))

# When 'Predict' button is pushed, the 'prediction()' function must be called 
# and the value returned by it must be stored in a variable, say 'species_type'. 
# Print the value of 'species_type' and 'score' variable using the 'st.write()' function.
classifier=st.sidebar.selectbox('Classifier',('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))

if st.button("Predict"):
  if classifier=="Support Vector Machine":
    species_type=prediction(svc_model,s_length,s_width,p_length,p_width)
    score=svc_model.score(X_train,y_train)
   
  elif classifier=="Logistic Regression":
    species_type=prediction(log_reg,s_length,s_width,p_length,p_width)
    score=log_reg.score(X_train,y_train)

  elif classifier=="Random Forest Classifier":
    species_type=prediction(rf_clf,s_length,s_width,p_length,p_width)
    score=rf_clf.score(X_train,y_train)

  st.write("Species predicted:",species_type)
  st.write("Accuracy score of this model is:",score)



