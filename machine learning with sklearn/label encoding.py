import pandas as pd


df = pd.read_excel("example.xls")
df["full_name"] = df[["First Name" , "Last Name"]].agg(" ".join, axis=1) # joining two columns along thier rows

# ------------------------------- LABEL ENCODING-------------------------------------------

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# df["First Name"] = le.fit_transform(df["First Name"])
# print(df)

# ------------------------ORDINAL ENCODING ------------------------------------------------

from sklearn.preprocessing import OrdinalEncoder

df.bfill(inplace=True )
ord_data = [['United States' ,'Great Britain' ,'France']] # assigns values as [0,1,2,...] respectively
oe = OrdinalEncoder(categories=ord_data)
df["Country_en_by_sklearn"] = oe.fit_transform(df[["Country"]])
print(df)


# by map function
ord_data2 = {"United States" : 0 , 'Great Britain':1 , 'France':2}
df["Country_en_by_map"] = df["Country"].map(ord_data2)
print(df)