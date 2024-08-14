# Grouping unlabeled examples is called clustering 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
dataset.drop("variety" , inplace=True, axis=1)

# K Means Clustering Using the Elbow Method
# WCSS is the sum of the squared distance between each point and the centroid in a cluster.

from sklearn.cluster import KMeans
wcss = []

for i in range(2,21):
    km = KMeans(n_clusters=i , init="k-means++") #kmeans++ helps to find the best cluster or grouped 
    km.fit(dataset)
    wcss.append(km.inertia_) #gives the value of wcss


plt.plot([i for i in range(2,21)] , wcss , marker = "o")
plt.grid(visible=True)
plt.ylabel("wcss")
plt.xlabel("no of cluster")
plt.show()

# youll will find the prefect cluster number by observing the graph || we are using elbow method

org_data = pd.read_csv(r"https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")

kmn = KMeans(n_clusters=3)
dataset["predict"] = kmn.fit_predict(dataset)

sns.pairplot(data=org_data , hue="variety")
plt.show()