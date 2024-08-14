from sklearn.metrics import mean_absolute_error , mean_squared_error
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("BostonHousing.csv")

# plt.figure(figsize=(10,10))
# sns.heatmap(data= dataset.corr() , annot=True)
# plt.show()

x = dataset.iloc[: , :-1]
y = dataset["housing_price"]

# ******************************************************* scaling data **************************************************************8

# Data scaling is the process of transforming the values of the features of a dataset till they are within a specific range, 
# e.g. 0 to 1 or -1 to 1. This is to ensure that no single feature dominates the distance calculations in an algorithm,
# and can help to improve the performance of the algorithm

sc = StandardScaler()
sc.fit(x)
x = pd.DataFrame(sc.transform(x) , columns=x.columns)

x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.35 , random_state=42)

from sklearn.linear_model import LinearRegression , Lasso , Ridge

# *********************************linear regression **********************************

lr = LinearRegression()
lr.fit(x_train , y_train)
# print(lr.score(x_test , y_test))
print(f"mean squared error of linear {mean_squared_error(y_test , lr.predict(x_test))}")
print(f"mean absolute error of linear {mean_squared_error(y_test , lr.predict(x_test))}")

# plt.bar(x.columns  , lr.coef_)
# plt.title("linear regression")
# plt.show()

# *********************************LASSO REGULARIZATION **********************************
# Lasso is a modification of linear regression

la = Lasso(alpha=0) # alpha means the penalty 
la.fit(x_train , y_train)
# print(la.score(x_test,y_test))

# plt.bar(x.columns  , la.coef_)
# plt.title("lasso regression")
# plt.show()

print(f"mean squared error of lasso {mean_squared_error(y_test , la.predict(x_test))}")
print(f"mean absolute error of lasso {mean_absolute_error(y_test , la.predict(x_test))}")

# *********************************RIDGE REGULARIZATION **********************************
# ridge is also modification of linear regression

ra = Ridge(alpha=0.5) # alpha means the penalty 
ra.fit(x_train , y_train)
# print(ra.score(x_test,y_test))

# plt.bar(x.columns  , ra.coef_)
# plt.title("Ridge regression")
# plt.show()

print(f"mean squared error of ridge {mean_squared_error(y_test , ra.predict(x_test))}")
print(f"mean absolute error of ridge {mean_absolute_error(y_test , ra.predict(x_test))}")