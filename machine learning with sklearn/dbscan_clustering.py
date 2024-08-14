# dbscan --> density - based spacial clustering of application with noise,
# the clusters found by dbscan can be any shape which deals with some special cases that other method cannot.
# it is used for clustering of non linear data 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# importing an dataset from sklearn itself
from sklearn.datasets import make_moons

x,y = make_moons(n_samples=250 , noise=0.05)
df = {"data1": x[:,0] , "data2" : x[:,1]}

dataset = pd.DataFrame(df)

# sns.pairplot(data=dataset)
# plt.show()

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.2 , metric="euclidean" , min_samples=5)
dataset["predict"] = dbscan.fit_predict(dataset)

sns.scatterplot(x="data1" , y="data2" , data=dataset, hue="predict")
plt.show()