import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_excel("example.xls")
dataset.ffill(inplace=True)

# replacing and changing the data and its data type

dataset["Id"].replace(to_replace="23423+" , value="3" , inplace=True )
dataset["Id"] = dataset["Id"].astype(dtype="int64")

# function transformer

q1 = dataset["charges"].quantile(0.25)
q3 = dataset["charges"].quantile(0.75)

iqr = q3 - q1
min_r = q1 - (1.5*iqr)
max_r = q3 + (1.5*iqr)

new_dataset= dataset[dataset["charges"] <= max_r]


from sklearn.preprocessing import FunctionTransformer
import numpy as np


# ft = FunctionTransformer(func= np.log1p)
# ft.fit(dataset[["charges"]])
# dataset["charges_transform"] = ft.transform(dataset[["charges"]])


# sns.displot(data=dataset["charges_transform"] , kde = True)
# plt.title("charges transformed")

# sns.displot(data=dataset["charges"] , kde = True)
# plt.title("charges")

# plt.show()



ft = FunctionTransformer(func= lambda x:x*2)
ft.fit(dataset[["charges"]])
dataset["charges_transform"] = ft.transform(dataset[["charges"]])

sns.displot(data=dataset["charges_transform"] , kde = True)
plt.title("charges transformed")

sns.displot(data=dataset["charges"] , kde = True)
plt.title("charges")

plt.show()