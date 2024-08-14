import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("med.csv")
data.bfill(inplace= True)
print(data["charges"])

# l = pd.DataFrame(data={"age":[5,4,6,7,3,10000]})
# now the 100000 is an outlier


#*********************************************** removing the outlier *************************************************************************8
# 1) IQR method-->

# q1 = data["charges"].quantile(0.25)
# q3 = data["charges"].quantile(0.75)

# IQR = q3 - q1
# min_range = q1- (1.5*IQR)
# max_range = q3+ (1.5*IQR)
# print(min_range , max_range)

# new_dataset = data[data["charges"]<=max_range]
# print(new_dataset)

# plt.figure(figsize=(10,5))
# # sns.boxplot(data=data , x="charges")
# sns.boxplot(data=new_dataset , x="charges")
# plt.show()




# 2) -------- z score

# z = (x-mean)/standard deviation
# min_range = (data["charges"].mean() - 3*data["charges"].std())
# max_range = (data["charges"].mean() + 3*data["charges"].std())

# new_dataset = data[data["charges"] <= max_range]

# z_score = (data["charges"] - data["charges"].mean())/data["charges"].std()
# data["z_score"] = z_score

# print(f"z score {data[data['z_score'] < 3].shape} and by max range {new_dataset.shape} have removed same number of outliers")

# sns.boxplot(data=data["charges"])
# plt.figure(figsize=(15,5))
# sns.boxplot(data=data , x="charges")
# plt.show()