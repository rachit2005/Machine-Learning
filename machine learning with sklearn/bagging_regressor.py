import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("Placement.csv")

x = dataset[["CGPA"]]
y = dataset["Placement"]

from sklearn.model_selection import train_test_split
xtrain , xtest  , ytrain , ytest = train_test_split(x,y,test_size=0.2 , random_state=42)


# bagging classifiers
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# meta estimators 
bg = BaggingRegressor(estimator=DecisionTreeRegressor() , n_estimators=30)
bg.fit(xtrain,ytrain)

print(bg.score(xtest , ytest))

# Random Forest Classifier
rf = RandomForestRegressor(n_estimators=100)
rf.fit(xtrain,ytrain)

print(rf.score(xtest , ytest))