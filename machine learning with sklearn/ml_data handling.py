import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



df = pd.read_excel("example.xls")
# df.dropna(inplace=True , axis=0) #axis = 0 means index and axis = 1 means columns

# print(df.fillna(method = "ffill"))
# df["Gender"].fillna(df["Gender"].mode()[0] , inplace=True)
# print(df)

# print(df.isnull().sum()) # shows total null values in respective columns
# print(df.isnull().sum().sum()) # gives total null values in whole data frame 
# print((df.isnull().sum() / df.shape[0]) *100) # giving percentage null values in a column 
# print(df.shape) # gives a tuple with (no of rows , no of columns)

# print((df.isnull().sum().sum() / (df.shape[0] *df.shape[1])) * 100) # giving percentage null value in whole dataframe

# print(df.notnull().sum().sum())

# print(df.select_dtypes(include="object").columns) #giving name of columns that have data type of "object" in a list form
 
# filling every column's null values with its mode
# for column in df.select_dtypes(include="object").columns:
#     df[column].fillna(df[column].mode()[0] , inplace=True)

# print(df)

# sns.heatmap(df.isnull())
# plt.show()

#----------------------------------how to fill null value using scikit learn---------------------------------

from sklearn.impute import SimpleImputer
si = SimpleImputer(strategy="most_frequent")  #Univariate imputer for completing missing values with simple strategies.
# si = SimpleImputer(strategy="most_frequent")  # filling null values by the most frequent data
ar = si.fit_transform(df[df.select_dtypes(include="object").columns])
df2 = pd.DataFrame(data=ar , columns= df.select_dtypes(include="object").columns)
print(df2)
