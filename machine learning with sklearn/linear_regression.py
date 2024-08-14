import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# **************************LINEAR REGRESSION****************************************

# question----> find salary on the basis of only the YearsExperience? 

# Answer --> 
# since , salary is continuous so regression analysis will be applied and since only one input (experience) is given to find output(salary) ,
# then linear regression will be applied.

# from sklearn.linear_model import LinearRegression   

# dataset = pd.read_csv("Salary_Data.csv")

# input_data = dataset[["YearsExperience"]]
# output_data = dataset[["Salary"]]

# x_train , x_test , y_train , y_test = train_test_split(input_data , output_data , test_size=0.2, random_state=42) 
# #above ,we are spliting the data into training and testing data sets

# lr = LinearRegression() # we are creating an object
# lr.fit(x_train , y_train) # in fit the data provided will find the best value for linear relation (ie y=mx+c)
# print(lr.predict([[10]])) # output is 39343 so the pridicted value should be close to output 
# print(lr.score(x_test,y_test) * 100) #gives percentage score 
# print(lr.coef_) #gives the value of slope
# print(lr.intercept_) #gives the value of y intercept (ie c)


# sns.scatterplot(x="YearsExperience" , y="Salary" , data=dataset , markers="D" , legend="full")
# plt.plot(input_data , lr.predict(input_data) , color = "black" , label="original")
# plt.legend(["original data" , "pridict line"]) # will show dot and line respectively to the code
# plt.show() 



# ******************************************** MULTIPLE  LINEAR REGRESSION ********************************************

# it has more than one input to predict the only one output for example- fundamental analysis of company
# gives an equation like    y = m1x1 + m2x2 ... ,where x1 ,x2 .. are different inputs and this equation is plane

# question --> find medical charges on the basis of age , bmi
# answer ->

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

# dataset = pd.read_csv("50_Startups.csv")

# x1_input = dataset[["R&D Spend" , "Administration"]]
# y1_input = dataset[["Profit"]]

# lr = LinearRegression()
# x_train , x_test , y_train , y_test = train_test_split(x1_input , y1_input , test_size=0.2 , random_state=42)
# lr.fit(X=x_train , y=y_train)

# print(lr.score(X=x_test , y= y_test) * 100)
# y_predict = lr.predict(x_test)

# plt.scatter(y_test , y_predict ) #take input as (x axis , y axis)
# plt.xlabel("original")
# plt.ylabel("pridicted")

# plt.show()

# ******************************************** POLYNOMIAL  LINEAR REGRESSION ********************************************

# it also has more than one input to give output but does not form a linear equation
# it forms a polynomial equation hence the name (ie y = c + m1x + m2x^2 ... mnx^n) thus forms a curve not a straight line

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

url = "https://gist.githubusercontent.com/RobotOptimist/cc82e87e7d2104e58711b7c846a9e220/raw/69ca02a46df0be8c210ebbded3106c5d9956a4bc/Position_Salaries.csv"

dataset = pd.read_csv(url)

x = dataset[["Level"]]
y = dataset[["Salary"]]

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

pr = PolynomialFeatures(degree=2) # keep giving degree till most acurate score is not achieved
pr.fit(x)
x = pr.transform(x) # transformed to polynomial 

x_train , x_test , y_train , y_test= train_test_split(x,y,random_state=42 , test_size=0.2)

lr = LinearRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test)*100)

plt.scatter(dataset["Level"] , dataset["Salary"])
plt.plot(dataset["Level"] ,lr.predict(x))
plt.xlabel("level")
plt.ylabel("salary")
plt.show()
