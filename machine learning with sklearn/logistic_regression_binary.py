import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions

# Logistic regression is a supervised machine learning algorithm that accomplishes binary classification tasks by predicting the probability 
# of an outcome, event, or observation.
# The model delivers a binary or dichotomous outcome limited to two possible outcomes: yes/no, 0/1, or true/false.

lr = LogisticRegression()
dataset = pd.read_csv("Placement.csv")

x = dataset[["CGPA" , "IQ"]]
y = dataset["Placement"]


x_train , x_test , y_train  , y_test = train_test_split(x , y , random_state=42 , test_size=0.2)
lr.fit(x_train,y_train)


plot_decision_regions(X=x.to_numpy() , y=y.to_numpy() , clf=lr)
plt.show()
