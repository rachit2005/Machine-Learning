# Association rule mining could be used to identify relationships between items that are frequently purchased together. For example,
# the rule "If a customer buys bread, they are also likely to buy milk" is an association rule that could be mined from this data set.

# it is of treee types :
# 1) --> APRIORI ALGORITHM
# 2) --> ECLAT ALGORITHM
# 3) --> FP GROWTH ALGO

import pandas as pd
import numpy as np
import collections

dataset = pd.read_csv("groceries.csv")

# we are removing the nan value from the dataframe and making it in a form of list 

market = []
for i in range(0,dataset.shape[0]):
    cus = []
    for j in dataset.columns:
        if type(dataset[j][i]) == str:
            cus.append(dataset[j][i])

    market.append(cus)

# here market is list of lists .so,
# we are converting the market in to a one list 

items_list = [] #list of all items bought

for i in market:
    for j in i:
        items_list.append(j)


p = collections.Counter(items_list)
d = {"Item name" : p.keys() , "values" : p.values()}

item_data = pd.DataFrame(d).sort_values(by=["values"] , ascending=False)

# go to this website to understand whats happening here "https://rasbt.github.io/mlxtend/user_guide/preprocessing/TransactionEncoder/"
from mlxtend.preprocessing.transactionencoder import TransactionEncoder
tr = TransactionEncoder()
tr.fit(market)

# ********************************applying the apriori alogrithm********************************************

# # making the dataframe of boolean form to find the frequencies of which items has been bought person by person
# df_new = pd.DataFrame(tr.transform(market) , columns=tr.columns_)\
# # applying the apriori algorithm
# from mlxtend.frequent_patterns import apriori
# # max_len : Maximum length of the itemsets generated.
# # min_support --> A float between 0 and 1 for minumum support of the itemsets returned
# print(apriori(df_new , min_support=0.05 , use_colnames=True , max_len=3 ).sort_values(by=["support"] , ascending=False)) #printing the repeated product from most to least
# # returns the pandas dataframe 

# ********************************applying the fp growth algorithm *******************************************

# df = pd.DataFrame(data=tr.transform(market) , columns= tr.columns_)
# from mlxtend.frequent_patterns import fpgrowth

# print(fpgrowth(df=df, min_support=0.05 , use_colnames=True , max_len=3).sort_values(by="support" , ascending=False)) #showing the most repeited items

