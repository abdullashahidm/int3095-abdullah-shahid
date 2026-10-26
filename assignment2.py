#--- Assignment 2 ---#

#--- Imports ---#

import numpy as np
import pandas as pd
import matplotlib as mt
import sklearn

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
