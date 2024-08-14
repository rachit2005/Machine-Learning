# it is bayes condition probability 
# naive bayes is of three types:
# 1) Gaussian --> when data is normal distribution
# 2) multinomial --> when data is discrete like text data
# 3) bernouli --> when data is in boolean form like (1,0)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Placement.csv")
x = dataset[["CGPA" , "IQ"]]
y = dataset["Placement"]

# sns.kdeplot(data=dataset["IQ"])
# plt.show()

x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.2 , random_state=42)

from sklearn.naive_bayes import GaussianNB , MultinomialNB , BernoulliNB

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(x_train , y_train)
mnb.fit(x_train , y_train)
bnb.fit(x_train , y_train)

# print(gnb.score(x_test , y_test))
# print(mnb.score(x_test , y_test))
# print(bnb.score(x_test , y_test))

plot_decision_regions(x.to_numpy() ,y.to_numpy() , clf=gnb)
# plot_decision_regions(x.to_numpy() ,y.to_numpy() , clf=mnb)
# plot_decision_regions(x.to_numpy() ,y.to_numpy() , clf=bnb)
plt.show()
