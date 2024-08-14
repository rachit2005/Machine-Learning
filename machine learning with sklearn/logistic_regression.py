# if the data is linearly seperable data then only the logistic regression can be applied

# it if of three types :
# 1) binomial --> 1 or 0 , True or False , pass or fail
# 2) multinomial -->  data is seperable more than two types
# 3) ordinal --> data is seperable in a order like low ,medium and high


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression , LogisticRegression

dataset = pd.read_csv("Social_Network_Ads.csv")
dataset.drop(columns=["Gender" , "User ID" , "EstimatedSalary"] , inplace=True) #Age  Purchased

lr = LinearRegression()
lo = LogisticRegression()

x = dataset[["Age"]] 
y = dataset["Purchased"]

x_train , x_test , y_train , y_test = train_test_split(x, y , test_size=0.2 , random_state=42)
lr.fit(x_train , y_train)
lo.fit(x_train , y_train)

print(lr.score(x_test , y_test))
print(lo.score(x_test , y_test))
print(lo.predict([[40]]))

sns.scatterplot(x = "Age" , y="Purchased" , data=dataset)
sns.lineplot(x=dataset["Age"] , y=lo.predict(x) , color = "black")
plt.show()
