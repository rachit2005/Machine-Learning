# it is used to group the unlablled dataset into a cluster and known as hierarchical cluster analysis (HCA)
# it is developed in the form of tree structure and this structure is known as the dendrogram
# this type of clustering works with linear data 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv(r"https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
dataset.drop("variety" , inplace=True, axis=1)

# to see type of data 
# sns.pairplot(data=dataset)
# plt.show()

# make the dendogram to find the no of cluster/groups 
import scipy.cluster.hierarchy as sc

# sc.dendrogram(sc.linkage(dataset , method="single" , metric="euclidean"))
# plt.show()

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=2,linkage="single")
dataset["predict"] = ac.fit_predict(dataset)

sns.pairplot(data=dataset ,hue= "predict")
plt.show()
