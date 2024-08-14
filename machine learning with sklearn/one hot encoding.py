# one hot encoding ---> can be used when number of data is minimum eg- (yes ,no) or (male ,female)

import pandas as pd
import sklearn.impute as si
from sklearn.preprocessing import OneHotEncoder 

df = pd.read_excel("example.xls")
df.bfill(inplace=True)

en_dum = df[["Gender" , "Married"]]  #wants to encode gender and married ****send the columns names in a list even if it is one**************
# print(pd.get_dummies(en_dum))

ohe = OneHotEncoder(drop="first")
print(ohe)
ar = ohe.fit_transform(en_dum).toarray()
df2 = pd.DataFrame(data=ar , columns=[ "Gender_male"  , "marreid_yes"])
print(df2)