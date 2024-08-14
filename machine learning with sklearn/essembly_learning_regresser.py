import pandas as pd

dataset = pd.read_csv("Placement.csv")

x = dataset[["CGPA"]]
y = dataset["Placement"]

from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain , ytest = train_test_split(x,y , test_size=0.2 , random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

lr = LinearRegression()
sv = SVR()
dt = DecisionTreeRegressor()

# decision tree regression 
dt.fit(xtrain, ytrain)
dt_score = dt.score(xtest , ytest)

# linear regression 
lr.fit(xtrain, ytrain)
lr_score = lr.score(xtest , ytest)


# support vector regression 
sv.fit(xtrain, ytrain)
sv_score = sv.score(xtest , ytest)

print(dt_score , lr_score , sv_score)

# now essembly learning regression 
from sklearn.ensemble import VotingRegressor
estimators = [("dt1" , DecisionTreeRegressor()) , ("lr1" , LinearRegression()) , ("sv1" , SVR())]
vr = VotingRegressor(estimators=estimators) 
vr.fit(xtrain , ytrain)
print(vr.score(xtest , ytest)) #gives the average score
print(vr.predict(xtest)) #gives the average predict answer

