#--- Assignment 2 ---#

#--- Imports ---#
import pandas as pd
import matplotlib as mt
import sklearn

#--- Task 1 ---#
diabetes_data=pd.read_csv("diabetes.csv")

print("******************************"*5)
print(diabetes_data.describe())
print("******************************"*5)

are_diabetic=diabetes_data[diabetes_data["Outcome"]==1]
not_diabetic=diabetes_data[diabetes_data["Outcome"]==0]
print(len(are_diabetic))
print("******************************"*5)
print(len(not_diabetic))

#--- Task 2 ---#

