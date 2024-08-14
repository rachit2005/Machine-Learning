# k - nearest neihgbour algo can be used for regression and classification both but mostly used for the classification 
# KNN a non-parametric algo which means it does not make any assumption on underlying data 
# it is also called lazy learner algorithm 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

dataset = pd.read_csv("Social_Network_Ads.csv")

x = dataset[["Age" , "EstimatedSalary"]]
y = dataset["Purchased"]

# sns.scatterplot(x="Age" , y="EstimatedSalary" , data=dataset , hue="Purchased")
# plt.show()

# scalling the data for model training
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x)

x = pd.DataFrame(sc.transform (x) , columns=x.columns)

# splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2 , random_state=42)

# training the model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train , y_train)

# print(knn.score(x_test , y_test))
# print(knn.score(x_train , y_train))

plot_decision_regions(x.to_numpy() , y.to_numpy() , clf=knn)
plt.show()
