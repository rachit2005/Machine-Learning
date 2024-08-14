# it is one of the most popular supervised learning algorithm used for classification and regression

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

dataset = pd.read_csv("Placement.csv")

x = dataset[["CGPA" , "IQ"]]
y = dataset["Placement"]

sns.scatterplot(x="CGPA" , y="IQ" , hue="Placement" , data=dataset)
plt.show()

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.2 , random_state=42)

from sklearn.svm import SVC

svc = SVC(kernel="linear") #using kernel(linear) method then applying linear SV machine learning Model
svc.fit(x_train , y_train)

plot_decision_regions(X=x.to_numpy() , y=y.to_numpy() , clf=svc)
plt.show()