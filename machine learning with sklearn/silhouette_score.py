# Grouping unlabeled examples is called clustering 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
dataset.drop("variety" , inplace=True, axis=1)

#silhouette score is a metric used to evaluate the quality of clustering performed by K-means
# it is used to find the best number of clusters/groups
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
ss = []
no_cluster = []

for i in range(2,21):
    km = KMeans(n_clusters=i , init="k-means++") #kmeans++ helps to find the best cluster or grouped 
    km.fit(dataset)
    ss.append(silhouette_score(dataset , metric="euclidean" , labels=km.labels_ )) #gives the value of silhouette score
    no_cluster.append(i) 

plt.plot(no_cluster , ss)
plt.xlabel("number of cluster") 
plt.ylabel("silhouette score")
plt.xticks(no_cluster)
plt.show()
