# confusion matrix is used to analys your classification/regression model by seeing in which situation the model is failed or passed
# it is also known as error matrix

# accuracy = (true_positive + True_negetive)/total no of inputs

# plz watch confusion matrix again!!!!!!!!!!!!!!!!!!!!!!!


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv("Placement.csv")
x = dataset[["CGPA" , "IQ"]]
y = dataset["Placement"]

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)

lr = LogisticRegression()
lr.fit(x_train ,y_train)

# ************************************************MAKING OF CONFUSION MATRIX************************************************

from sklearn.metrics import confusion_matrix , precision_score , recall_score , f1_score

cf = confusion_matrix(y_test , lr.predict(x_test))
'''
shows matrix as :  [tp , fp]
                   [fn , tn]

'''
sns.heatmap(cf , annot=True)
plt.show()

pr = precision_score(y_test , lr.predict(x_test)) #shows how precise the model is. higher the precision value lower is the "False positive value"
recal = recall_score(y_test , lr.predict(x_test)) # higher the recall value lower is the "False negetive value" , this also known as sensitivity
f1 = f1_score(y_test , lr.predict(x_test))