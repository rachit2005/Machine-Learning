import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv("med.csv")
# columns:
# age     sex    bmi  children smoker     region     charges

x = dataset[["children" , "age" , "bmi" , "charges"]]
y = dataset["region"]

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.3 , random_state=42)

# ***********************************************************OVR METHOD************************************************************
# This method creates one logistic regression model for each class against all other classes

lr = LogisticRegression(multi_class="ovr")
lr.fit(x_train , y_train)
# print(lr.score(x_test , y_test))

# ***********************************************************multinomial METHOD************************************************************

lr2 = LogisticRegression(multi_class="multinomial")
lr2.fit(x_train , y_train)
print(lr2.score(x_test , y_test))


# sns.pairplot(data=dataset , hue="region")
# plt.show()
