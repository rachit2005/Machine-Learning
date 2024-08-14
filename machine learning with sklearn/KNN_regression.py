import pandas as pd

dataset = pd.read_csv("Social_Network_Ads.csv")

x = dataset[["Age" , "EstimatedSalary"]]
y = dataset["Purchased"]

# splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.2 , random_state=10)

# training the model 
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors=15)

knr.fit(x_train , y_train)

print(knr.score(x_test,y_test))
print(knr.score(x_train,y_train))
