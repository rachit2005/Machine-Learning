import pandas as pd

dataset = pd.read_csv("50_Startups.csv")
x = dataset[["R&D Spend"]]
y = dataset["Profit"]

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=5 , criterion="absolute_error" , splitter="best")
dt.fit(x_train , y_train)

# finding the best params for training the model  by using grid search cv
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
df = {"criterion" : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'] ,
       "splitter" : ['best', 'random'] , "max_depth" : [i for i in range(2,20)]}

# gd = GridSearchCV(DecisionTreeRegressor() , param_grid=df)
# gd.fit(x_train , y_train) #training the grid to find the best params
# print(gd.best_params_) # printing the best parameter for training the model
# print(gd.best_score_) # printing the best score that we can get

# print(dt.score(x_test , y_test))

# finding the best params for training the model  by using random search cv
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
df = {"criterion" : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'] ,
       "splitter" : ['best', 'random'] , "max_depth" : [i for i in range(2,20)]}

rd = RandomizedSearchCV(DecisionTreeRegressor() , param_distributions=df , n_iter=20) # n_iter = chose how many combination do you want to use 
rd.fit(x_train , y_train) #training the grid to find the best params
# print(rd.best_params_) # printing the best parameter for training the model
# print(rd.best_score_) # printing the best score that we can get

print(dt.score(x_test , y_test))
