
# Predicting delivery time using sorting time 

import pandas as pd

df = pd.read_csv('E:\\data science\\ASSIGNMENTS\\ASSIGNMENTS\\l.r assig\\delivery_time.csv')
df.shape
df.head()
list(df)
df.columns
df.corr()
df.describe()
df.info()

# scatter plot
df.plot.scatter(x = 'Sorting Time',y = 'Delivery Time')

#split the vairbales in x&y

x = df['Sorting Time']
y = df['Delivery Time']

# histogram
df['Sorting Time'].hist()
df['Delivery Time'].hist()

# bar graph
t1 = df['Sorting Time'].value_counts()
t1.plot(kind="bar")
t1 = df['Delivery Time'].value_counts()
t1.plot(kind="bar")

import matplotlib.pyplot as plt
import seaborn as sns    
# pair plot
sns.pairplot(df,diag_kind='kde')
plt.show

# distrubution plot
sns.distplot(df['Delivery Time'], bins = 10, kde = True)
plt.show
sns.distplot(df['Sorting Time'], bins = 10, kde = True)
plt.show

# regression plot
sns.regplot(x='Sorting Time',y='Delivery Time',data=df,color='green')
plt.show


import numpy as np
# converting 1D arrary to 2D array
x=x[:,np.newaxis]
x.ndim
x

from sklearn.linear_model import LinearRegression
# FIT X&Y
lm=LinearRegression()
lm.fit(x,y)

# for intercept
lm.intercept_
# for coefficenet
lm.coef_

y_pred=lm.predict(x)
y_pred

# FOR M.S.E
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y,y_pred)
mse

# FOR R.M.S.E
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

from sklearn.metrics import mean_squared_error,r2_score
# FOR R2 SCORE
r2 = r2_score(y,y_pred)
print('r2score:',r2.round(4))

# For preparing linear regression stats model we need to import the statsmodels.formula.api
# data transformations

import statsmodels.formula.api as smf

# for square root
sq = smf.ols('np.sqrt(x)~np.sqrt(y)',df).fit()
sq.params
sq.summary()

# for cube root transformation
cbr = smf.ols('np.cbrt(x)~np.cbrt(y)',df).fit()
cbr.params
cbr.summary()

# for log transformation
log = smf.ols('np.log(x)~np.log(y)',df).fit()
log.params
log.summary()

#=====================================================================














