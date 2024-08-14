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
from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train , y_train)

# plotting the tree
from sklearn.tree import plot_tree
plt.figure(figsize=(50,50))
plot_tree(decision_tree=dt)
# plt.savefig("tree_entropy.jpg")
plt.show()

# plotting the graph of decision tree 
# from mlxtend.plotting import plot_decision_regions
# plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=dt)
# plt.show()

