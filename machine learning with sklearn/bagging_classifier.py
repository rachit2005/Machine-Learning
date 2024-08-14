import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons

x,y = make_moons(n_samples=2500 , noise=0.2)

df = {"x1" : x[:,0] , "x2" : x[:,1] , "output": y}

dataset = pd.DataFrame(df)

x_a = dataset[["x1" , "x2"]]
y_a = dataset["output"]

from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain , ytest = train_test_split(x_a , y_a , test_size=0.2 , random_state=42)

# bagging classifiers

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# meta estimators 

bg = BaggingClassifier(estimator=SVC() , n_estimators=50)
bg.fit(xtrain, ytrain)

print(bg.score(xtest , ytest))

# Random Forest Classifier 

rf = RandomForestClassifier(n_estimators=100 ) #its base estimator is decision tree classifier
rf.fit(xtrain,ytrain)

print(rf.score(xtest , ytest))
