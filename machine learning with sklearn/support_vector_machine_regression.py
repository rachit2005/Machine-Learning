import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

dataset = pd.read_csv("50_Startups.csv")

x1_input = dataset[["R&D Spend"]]
y1_input = dataset["Profit"]

x_train , x_test , y_train , y_test = train_test_split(x1_input,y1_input ,test_size=0.2 , random_state=42)

sv = SVR(kernel="linear")
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
sv.fit(x_train,y_train)

print(sv.score(x_test , y_test))
print(sv.score(x_train , y_train))

sns.scatterplot(x= dataset["R&D Spend"], y=dataset["Profit"]  )
plt.plot(dataset["R&D Spend"] , sv.predict(x1_input) , color = "black")
plt.show()
