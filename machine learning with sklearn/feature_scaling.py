import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_set = pd.read_csv("med.csv")

# sns.displot(data=data_set["charges"] , kde = True)
# plt.show()

#************************************************SCALING**************************************************************

#1) standardlising the data --->  a method of feature scaling in which data values are rescaled to fit the distribution between 0 and 1 using mean 
#                                 and standard deviation as the base to find specific values

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# scaler.fit_transform(data_set[["charges"]])
# array = scaler.transform(data_set[["charges"]])

# data_set["charges_scaler"] = array

# plt.title("before scaling")
# sns.displot(data=data_set["charges_scaler"] , kde=True)

# plt.title("after scaling")
# sns.displot(data=data_set["charges"] , kde=True)

# plt.show()

# 2) normalisation the data --> a scaling method where values are shifted and rescaled to maintain their ranges between 0 and 1

# from sklearn.preprocessing import MinMaxScaler
# ms = MinMaxScaler()
# ms.fit(data_set[["charges"]])
# data_set["charges_normal"] = ms.transform(data_set[["charges"]])


# sns.displot(data_set["charges_normal"] , kde = True)
# plt.show()

# *****************************************************************END**************************************************************

# data = {"name" : ["rachit" , "harry" , "ron" , "draco" , "rachit" , "ron"] , "eng" : [8,7,9,6,8,9] , "hindi" : [12,23,34,45,12,34]}
# data_set2 = pd.DataFrame(data)
# print(data_set2.drop_duplicates(keep="last"))

