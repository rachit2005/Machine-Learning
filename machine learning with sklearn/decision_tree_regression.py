import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree

dataset = pd.read_csv("50_Startups.csv")

input = dataset[["Administration" , "Marketing Spend"]]
output = dataset["Profit"]

# sns.pairplot(data=dataset)
# plt.show()

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(input, output , test_size=0.2 , random_state=42)

dr = DecisionTreeRegressor(criterion="absolute_error")
dr.fit(x_train,y_train)
print(dr.score(x_test,y_test))
print(dr.score(x_train,y_train))

# plot_tree(dr)
# plt.show()