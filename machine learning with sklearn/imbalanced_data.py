import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv("Social_Network_Ads.csv")
# print(dataset["Purchased"].value_counts())

x = dataset[["Age" , "EstimatedSalary"]]
y = dataset["Purchased"]

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)

lr = LogisticRegression()
# lr.fit(x_train , y_train) #as data was imbalanced so the model is train baised

#*************************************************************** balancing the data *********************************************************

# 1) method  = under sampling---> reduce the amount of large data to the amount of small data

'''

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler()
ru_x , ru_y = rus.fit_resample(x,y) #thats how sampling of data will happen 

print(ru_y.value_counts()) #shows the amount of data is balanced (decrease the amount of large data)

lr.fit(ru_x , ru_y) #now the model is not biased
print(lr.predict([[35,20000]]))

'''

# 2) method = over smapling --> increase the amount of small data to the amount of large data

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()
ro_x , ro_y  = ros.fit_resample(x , y)

print(ro_y.value_counts()) #shows the amount of data is balanced (increase the amount of small data)

lr.fit(ro_x , ro_y)
print(lr.score(x_test , y_test))

