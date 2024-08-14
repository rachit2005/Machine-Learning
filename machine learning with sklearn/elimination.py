import pandas as pd

dataset = pd.read_csv("C:\\Users\\RACHIT\\Downloads\\BostonHousing.csv")
'''
crim    zn  indus  chas    nox     rm   age     dis  rad  tax  ptratio       b  lstat  housing_price
'''

input_data = dataset.iloc[:,:-1] # rows and columns

output_data = dataset["housing_price"]

from sklearn.model_selection import train_test_split
x_train ,x_text , y_train , y_test = train_test_split(input_data,output_data , test_size=0.25)
