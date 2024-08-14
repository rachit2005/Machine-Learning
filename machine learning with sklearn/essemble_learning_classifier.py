# essemble learning in ml combine the insights obtained from multiple learning models to facilates accurate and improved decisions
# it is basically a technique using multiple ml models at once to find best prediction

# it is of two types:
# 1)--> bagging
# 2) --> boosting

# boosting essemble learning

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import seaborn as sns

x,y = make_moons(n_samples=2500, noise=0.2)

df = pd.DataFrame(data={"data1"  :x[:,0] , "data2": x[:,1] , "output" : y})

# sns.scatterplot(x="data1" , y="data2" , hue="output",data=df)
# plt.show()

x_a = df[["data1" , "data2"]]
y_a = df["output"]

from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain , ytest = train_test_split(x_a , y_a, test_size=0.2 , random_state=42)

# creating multiple model 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# decision tree model 
dt = DecisionTreeClassifier()
dt.fit(xtrain,ytrain)
dt_score = dt.score(xtest , ytest)


# support vector regressor 
sv = SVC()
sv.fit(xtrain , ytrain)
svc_score = sv.score(xtest , ytest)

#guassian nb
gnb = GaussianNB()
gnb.fit(xtrain , ytrain)
gnb_score = gnb.score(xtest , ytest)


# now using essemble learning
from sklearn.ensemble import VotingClassifier
li = [("dt1" , DecisionTreeClassifier()) , ("sv1" , SVC()) ,("gnb1" , GaussianNB())]
vc = VotingClassifier(estimators=li) #this is average voting and if we add weights on it then it become average weighted voting 
vc.fit(xtrain,ytrain)

# print(vc.score(xtest,ytest) , vc.score(xtrain,ytrain))

# max voting 
predict = {"dt" : dt.predict(xtest) , "svc" : sv.predict(xtest) , "gnb" : gnb.predict(xtest ) , "vc" : vc.predict(xtest )}
pr = pd.DataFrame(predict)
print(pr)

