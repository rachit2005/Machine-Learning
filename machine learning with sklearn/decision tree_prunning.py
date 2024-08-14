# prunning --> removing of sub node 
# if the model is overfittiong then only prunning is prefered

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Social_Network_Ads.csv")
x = data[['Age', "EstimatedSalary"]]
y = data["Purchased"]

# scalling the data so that all data contributes in machine learning equally
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(x)
x = pd.DataFrame(ss.transform(x) , columns=x.columns)

# splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)

# making the decision tree cclassifier model ---> cause the purchaed data is in format of (0,1)
# this is pre prunning
from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier(max_depth=3) #max depth means how much deeper does decision tree you want to make , it doesn't count the first node 
dt.fit(x_train , y_train)

# if both are close to each other then the model is not over fitted
print(dt.score(x_test,y_test)) 
print(dt.score(x_train,y_train))

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x.to_numpy() , y.to_numpy() , clf=dt)
plt.show()

# plotting the tree
# from sklearn.tree import plot_tree
# plt.figure(figsize=(50,50))
# plot_tree(decision_tree=dt)
# plt.savefig("tree_entropy.jpg")
# plt.show()

# best method to find max depth 

# the i for which minimum difference of train data and test data is best for max depth
'''
for i in range(1,20):
    dt2 = DecisionTreeClassifier(max_depth=i)
    dt2.fit(x_train,y_train)

    print(f"data of {i} ---> {dt2.score(x_train,y_train) - dt2.score(x_test,y_test)}")
'''