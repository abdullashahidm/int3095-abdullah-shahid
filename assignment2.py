#--- Assignment 2 ---#

#--- Imports ---#

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

#--- Method(s) ---#

def seperator():
    print("******************************"*5)

#--- Task 1 ---#  Load dataset & seperate features from target

diabetes_data=pd.read_csv("diabetes.csv")

seperator()
print(diabetes_data.describe())
seperator()

are_diabetic=diabetes_data[diabetes_data["Outcome"]==1]
not_diabetic=diabetes_data[diabetes_data["Outcome"]==0]
print(len(are_diabetic))
seperator()
print(len(not_diabetic))

#--- Task 2 ---# Detect missing values hiden as 0 & replace with median of columns

seperator()
print((diabetes_data==0).sum())
seperator()

to_replace_columns=["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]   
diabetes_data[to_replace_columns]=diabetes_data[to_replace_columns].replace(0, np.nan)
print(diabetes_data.isnull().sum())
seperator()

medians=diabetes_data[to_replace_columns].median()
diabetes_data[to_replace_columns]=diabetes_data[to_replace_columns].fillna(medians)
#Use median because it is unaffected by outlier or boundary values as compared with using mean
print(diabetes_data.describe())

#--- Task 3 ---# Split the data & train three models

X=diabetes_data[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
Y=diabetes_data["Outcome"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True,stratify=Y)

seperator()
print(f"Length of X_train is {len(X_train)}")
print(f"Length of X_test is {len(X_test)}")
print(f"Length of Y_train is {len(Y_train)}")
print(f"Length of Y_test is {len(Y_test)}")

seperator()
print("Training KNNN model")
knn_model=KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train,Y_train)

seperator()
print("Training Decision Tree Model")
dec_tree_model=DecisionTreeClassifier(random_state=42)
dec_tree_model.fit(X_train,Y_train)

seperator()
print("Training Logistic Regression model")
logistic_reg_model=LogisticRegression(solver="lbfgs",random_state=42,max_iter=500,)
logistic_reg_model.fit(X_train,Y_train)

