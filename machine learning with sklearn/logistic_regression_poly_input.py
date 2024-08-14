import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=2)
dataset = pd.read_csv("Social_Network_Ads.csv")
dataset.drop(["Gender" , "User ID"] , inplace=True , axis=1)

x = dataset.iloc[:,:-1]
y = dataset["Purchased"]

# sns.scatterplot(x="EstimatedSalary" , y="Age" , data=dataset , hue="Purchased")
# plt.show()

pf.fit(x)
x = pd.DataFrame(pf.transform(x))

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)

lr = LogisticRegression()
lr.fit(x_train , y_train)

print(lr.score(x_test,y_test)*100)

# plot_decision_regions(X=x.to_numpy() , y=y.to_numpy() , clf=lr)
# plt.show()