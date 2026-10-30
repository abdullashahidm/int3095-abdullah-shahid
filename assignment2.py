#--- Assignment 2 ---#

#--- Imports ---#

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

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
diabetes_data[to_replace_columns]=diabetes_data[to_replace_columns].replace(0,np.nan)
print(diabetes_data.isnull().sum())
seperator()

medians=diabetes_data[to_replace_columns].median()
diabetes_data[to_replace_columns]=diabetes_data[to_replace_columns].fillna(medians)
#Use median because it is unaffected by outlier or boundary values as compared with using mean
print(diabetes_data.describe())

#--- Task 3 ---# Split the data & train three models

unscaled_accuracies={}

X=diabetes_data[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
Y=diabetes_data["Outcome"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True,stratify=Y)

seperator()
print(f"Length of X_train is {len(X_train)}")
print(f"Length of X_test is {len(X_test)}")
print(f"Length of Y_train is {len(Y_train)}")
print(f"Length of Y_test is {len(Y_test)}")

seperator()
print("Training KNN model")
knn_model=KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train,Y_train)
predictions=knn_model.predict(X_test)
accuracy=accuracy_score(Y_test,predictions)
unscaled_accuracies["KNN Model"]=accuracy

print("Training Decision Tree Model")
dec_tree_model=DecisionTreeClassifier(random_state=42)
dec_tree_model.fit(X_train,Y_train)
predictions=dec_tree_model.predict(X_test)
accuracy=accuracy_score(Y_test,predictions)
unscaled_accuracies["Decision Tree Model"]=accuracy

print("Training Logistic Regression model")
logistic_reg_model=LogisticRegression(random_state=42,max_iter=1000)
logistic_reg_model.fit(X_train,Y_train)
predictions=logistic_reg_model.predict(X_test)
accuracy=accuracy_score(Y_test,predictions)
unscaled_accuracies["Logistic Regression Model"]=accuracy

seperator()
for model_name,accuracy_rating in unscaled_accuracies.items():
    print(f"Unscaled data {model_name} has accuracy {accuracy_rating:.4f}")

#--- Task 4 ---# Scale the features with a pipline

scaled_accuracies={}

pipelines={
    "KNN":make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=3)),
    "Decision Tree":make_pipeline(StandardScaler(),DecisionTreeClassifier(random_state=42)),
    "Logistic Regression":make_pipeline(StandardScaler(),LogisticRegression(random_state=42,max_iter=1000))
}

seperator()
for model_name,model_pipeline in pipelines.items():
    model_pipeline.fit(X_train,Y_train)
    print(f"Trained model {model_name} on scaled data")
    predictions=model_pipeline.predict(X_test)
    accuracy=accuracy_score(Y_test,predictions)
    scaled_accuracies[model_name]=accuracy

seperator()
for model_name,acc_score in scaled_accuracies.items():
    print(f"Scaled data {model_name} model has accuracy {acc_score:.4f}")

comparison_table=pd.DataFrame({
    "Unscaled accuracy":list(unscaled_accuracies.values()),
    "Scaled accuracy":list(scaled_accuracies.values())
},index=["KNN","Dec. tree","Logistic reg."])
seperator()
print(comparison_table)
#Saving this comparison table to print later for Task 6
original_comparison_table=comparison_table
seperator()

#As KNN relies on calculating physical distance between points so scaling points changes that distance between so changes KNN
#Since Decision Tree model relies on logical checks on data, logic checks output should not change if all data is scaled in the same way
#Scaler ensures no Data Leakage, because the Mean & S.d. used are only of the Training data set and not the entire data set

#--- Task 5 ---# Compare the models

models=["KNN","Decision tree","Logistic regression"]

unscaled_vals=[accuracy*100 for accuracy in unscaled_accuracies.values()]
scaled_vals=[accuracy*100 for accuracy in scaled_accuracies.values()]

x=np.arange(len(models))
width=0.35

fig,ax=plt.subplots(figsize=(10, 6))

rects1=ax.bar(x-width/2,unscaled_vals,width,label="Unscaled",color="blue")
rects2=ax.bar(x+width/2,scaled_vals,width,label="Scaled",color="purple")

ax.set_ylabel("Accuracy(%)")
ax.set_title("Scaled & unscaled data Model accuracies")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 100)  
ax.legend()

ax.bar_label(rects1,fmt="%.1f%%",padding=3)
ax.bar_label(rects2,fmt="%.1f%%",padding=3)

plt.tight_layout()
plt.savefig("accuracy-comparison.png")
plt.show()
seperator()

best=pipelines["KNN"]
Y_pred=best.predict(X_test)
print("Classifications report for best performing (KNN) model:\n")
print(classification_report(Y_test,Y_pred,target_names=["Not Diabetic","Diabetic"]))
seperator()

#--- Task 6 ---# Reflection

print("Q1: Which model was most accurate on the test set, and by how much?")
print("Ans: KNN model was the most accurate on scaled data at 72.1% (3.9% better than Decision tree & 1.3% better than Logistic regression). And as for unscaled data, Logistic regression was the most accurate with 70.1% compared to 68.2% on the other two\n")
seperator()

print("Q2: The models scored close to 100% on Iris. Why are they much less accurate here?")
print("Ans: As the flower's features are distinct from one another, but in the case of detecting diabetes, some features may be similar for two people even if one person has Diabetes and on does not. Thus, these sorts of feature overlaps make the model less accurate\n")
seperator()

print("Q3: 65% of patients are not diabetic. What accuracy would a model get by always predicting 'not diabetic'? Are your models much better than that?")
print("Ans: That model would get 65% accuracy, whereas as mine one is more accurate at 72.1%. Thus, my model is slightly better, but perhaps not ideal\n")
seperator()

print("Q4: Change n_neighbors to 5, 9 and 15. How does the accuracy of your scaled KNN pipeline change?")
for _ in [5,9,15]:
    pipeline=make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=_))
    pipeline.fit(X_train,Y_train)
    predicted_X_values=pipeline.predict(X_test)
    acc=accuracy_score(Y_test,predicted_X_values)
    print(f"{_} neighbors has Accuracy: {acc:.4f}")
print("For 5 neighbors, the model accuracy increases to 0.75 which far higher than 0.72 and this indicates that 5 neighbors is more suitable. This decreases slightly for 9 & 12 neighbors, but it still ends up being more accurate than 3 neighbors\n")
seperator()

print("Q5: A doctor wants to use one model for screening. Which would you choose, and why?")
print("Ans: Due to the medical context, rather the maximum accuracy, we would prefer highest recall for Diabetic patients\n")

dec_tree=pipelines["Decision Tree"]
Y_pred=dec_tree.predict(X_test)
print("Classifications report for Decision Tree model:\n")
print(classification_report(Y_test,Y_pred,target_names=["Not Diabetic","Diabetic"]))

log_reg=pipelines["Logistic Regression"]
Y_pred=log_reg.predict(X_test)
print("Classifications report for Logistic regression model:\n")
print(classification_report(Y_test,Y_pred,target_names=["Not Diabetic","Diabetic"]))

print("When we compare these reports for the KNN model printed above, we can see that the KNN model has the highest recall for Diabetic patients (which is 0.61 as compared to 0.48 for Decision tree & 0.5 for Logistic regression. Hence, the scaled data KNN model is the most recommended choice)\n")
seperator()

print("Q6: Re-run your three models on the original data without cleaning. Did handling the missing values change the accuracies? Suggest a reason.")
print("Ans: In order to answer that, we can re-run our above code, but this type ommitting the data cleaning step\n")

uncleaned_data=pd.read_csv("diabetes.csv")
X_uncleaned=uncleaned_data[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
Y_uncleaned=uncleaned_data["Outcome"]

X_uncleaned_training,X_uncleaned_testing,Y_uncleaned_training,Y_uncleaned_testing=train_test_split(X_uncleaned,Y_uncleaned,test_size=0.2,random_state=42,stratify=Y_uncleaned)

uncleaned_scaled_accuracies={}
for model_name,pipe in pipelines.items():
    pipe.fit(X_uncleaned_training,Y_uncleaned_training)
    acc_score=accuracy_score(Y_uncleaned_testing,pipe.predict(X_uncleaned_testing))
    uncleaned_scaled_accuracies[model_name]=acc_score

for unclean_model,uncleaned_model_accuracy in uncleaned_scaled_accuracies.items():
    print(f"{unclean_model} model on uncleaned & scaled data has accuracy {uncleaned_model_accuracy:.4f}")

print()
print(original_comparison_table)
print()

print("Comparing model accuracies for scaled data shows that for KNN model, accuracy drops when using uncleaned data from 72 to 69. This is due to the changed physical distances between uncleaned (zeros) points")