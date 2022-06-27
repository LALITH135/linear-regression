
#Building a prediction model for Salary_hike

import pandas as pd
# import data
df = pd.read_csv('E:\\data science\\ASSIGNMENTS\\ASSIGNMENTS\\l.r assig\\Salary_Data.csv')
df.shape
list(df)
df.head()
df.columns
df.corr()
df.describe()
df.info()

import matplotlib.pyplot as plt

df.plot.scatter(x = 'YearsExperience',y = 'Salary')
# by scatter plot we find,there is a positive relationship b/w x&y 

# histogram
df['YearsExperience'].hist()
df['Salary'].hist()

# bar graph
t1 = df['YearsExperience'].value_counts()
t1.plot(kind="bar")
t1 = df['Salary'].value_counts()
t1.plot(kind="bar")


import seaborn as sns    

# pair plot
sns.pairplot(df,diag_kind='kde')
plt.show

# distrubution plot
sns.distplot(df['YearsExperience'], bins = 10, kde = True)
plt.show
sns.distplot(df['Salary'], bins = 10, kde = True)
plt.show

# regression plot
sns.regplot(x='YearsExperience',y='Salary',data=df,color='green')
plt.show


#split the vairbales in x&y

x = df['YearsExperience']
y = df['Salary']

import numpy as np
# converting 1D arrary to 2D array

x=x[:,np.newaxis]
x.ndim
x

from sklearn.linear_model import LinearRegression
# FITTING X&Y
lm=LinearRegression()
lm.fit(x,y)

# FOR INTERCEPT
lm.intercept_

# FOR COEFFICIENT
lm.coef_

# PREDICTING THE VALUES
y_pred=lm.predict(x)
y_pred

# FOR MEAN SQUARE ERROR
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y,y_pred)
mse

# FOR ROOT MEAN SQUARE ERROR
rmse = np.sqrt(mse)
rmse

# FOR MEAN ABSOULUTE ERROR.
from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y, y_pred)
mae

# ROOT MEAN SQUARE LOG ERROR
from sklearn.metrics import mean_squared_log_error
rmsle=np.sqrt(mean_squared_log_error(y, y_pred))
rmsle

# FOR R2 SCORE
from sklearn.metrics import mean_squared_error,r2_score
r2 = r2_score(y,y_pred)
print('r2score:',r2.round(4))

# For preparing linear regression stats model we need to import the statsmodels.formula.api
# data transformations

import statsmodels.formula.api as smf

# for square root
sq = smf.ols('np.sqrt(Salary)~np.sqrt(YearsExperience)',data=df).fit()
sq.params
sq.summary()

# for cube root transformation
cbr = smf.ols('np.cbrt(Salary)~np.cbrt(YearsExperience)',data=df).fit()
cbr.params
cbr.summary()

# for log transformation
log = smf.ols('np.log(Salary)~np.log(YearsExperience)',data=df).fit()
log.params
log.summary()

#===================================================================




































































