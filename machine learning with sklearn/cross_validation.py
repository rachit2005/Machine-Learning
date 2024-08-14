# it is a technique for validationg the model efficiency by training it on the subset of input data
# and testing on previously unseen subset of the input data 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("50_Startups.csv")

x = dataset[["R&D Spend"]]
y = dataset["Profit"]

new_data = dataset.head(10)

x_new = new_data[["R&D Spend"]]
y_new = new_data["Profit"]

from sklearn.model_selection import LeaveOneOut , LeavePOut , KFold , StratifiedKFold

# cross validation by leave one out method

# loo = LeaveOneOut() 
# for train , test in loo.split(x,y):
#     print(train , test)



# cross validation by using leave p out

# lpo = LeavePOut(p=2) 
# for train , test in lpo.split(x_new , y_new):
#     print(train , test)




# cross validation using k fold method

# kfold_cv = KFold(n_splits=2)

# for train , test in kfold_cv.split(x,y):
#     print(train , test)




# cross validation using satisfied k fold method is used for only classification not for regression

# new_dataset = pd.read_csv("Placement.csv")

# x_new1 = new_dataset[["CGPA"]]
# y_new1 = new_dataset["Placement"]

# skfold = StratifiedKFold(n_splits=5) 
# for train , test in skfold.split(x_new1,y_new1):
#     print(train , test)


# checking the prediction capabilities of dataset
# cross_val_score is a powerful tool for evaluating machine learning models by providing cross-validated accuracy scores. 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score , cross_val_predict

p = cross_val_score(estimator=LinearRegression() ,X=x ,y=y, cv=KFold(n_splits=10)) #cv--> Determines the cross-validation splitting strategy , you can also give number to train your model that many time 
p.sort()
print(p) #prints the score
y_pred = cross_val_predict(estimator=LinearRegression() , X=x , y=y , cv=KFold(n_splits=10)) #it will train and test the model automatically
print(y_pred)


plt.scatter(dataset["R&D Spend"] , dataset["Profit"])
plt.plot(dataset["R&D Spend"] ,y_pred)
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.show()

