#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score


# In[2]:


df = pd.read_csv('ford.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


df.columns


# In[8]:


plt.figure(figsize=(12,10))
sns.countplot(y='model', data=df)
plt.title('Model types')
plt.show()


# In[9]:


sns.countplot(x='transmission', data=df)
plt.title('Transmission Types')
plt.show()


# In[10]:


sns.countplot(x='fuelType', data=df)
plt.title('Fuel Types')
plt.show()


# In[11]:


print(df['model'].value_counts())
print("\n\n")
print(df['transmission'].value_counts())
print("\n\n")
print(df['fuelType'].value_counts())


# In[12]:


fuelType = df['fuelType']
transmission = df['transmission']
price = df['price']
fig, axes = plt.subplots(1,2, figsize=(15,5), sharey=True)
fig.suptitle('Visualizing categorical data columns')
sns.barplot(x=fuelType, y=price, ax=axes[0])
sns.barplot(x=transmission, y=price, ax = axes[1])


# In[13]:


df.replace({'transmission':{'Manual':0, 'Automatic':1, 'Semi-Auto':2}}, inplace=True)
df.replace({'fuelType':{'Petrol':0, 'Diesel':1, 'Hybrid':2, 'Electric':3, 'Other':4}}, inplace=True)


# In[14]:


df = df.drop("model", axis=1)
df.head()


# In[15]:


plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True)
plt.title('Corrlealtion between the columns')
plt.show()


# In[16]:


df.corr()['price'].sort_values()


# In[17]:


fig = plt.figure(figsize=(7,5))
plt.title('Correlation between year and price')
sns.regplot(x='price', y='year', data=df)


# In[18]:


X = df.drop('price', axis=1)
y = df['price']


# In[19]:


print("Shape of X is :", X.shape)
print("Shape of y is :", y.shape)


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[21]:


print("Shape of X_train is: ", X_train.shape)
print("Shape of y_train is: ", y_train.shape)
print("Shape of X_test is: ", X_test.shape)
print("Shape of y_test is: ", y_test.shape)


# In[22]:


scaler = StandardScaler()


# In[23]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[24]:


linreg = LinearRegression()
linreg.fit(X_train, y_train)
linreg_pred = linreg.predict(X_test)


# In[25]:


linreg_mae = mean_absolute_error(y_test, linreg_pred)
linreg_r2 = r2_score(y_test, linreg_pred)
print("MAE of linear regression model is:", linreg_mae)
print("R2 score of linear regression model is:", linreg_r2)


# In[26]:


linreg_score = cross_val_score(linreg, X_test, y_test, cv=4)
print("Linear Regression model accuracy is: {}".format(linreg_score.mean()*100))


# In[27]:


dtree = DecisionTreeRegressor()
dtree.fit(X_train, y_train)
dtree_pred = dtree.predict(X_test)


# In[28]:


dtree_mae = mean_absolute_error(y_test, dtree_pred)
dtree_r2 = r2_score(y_test, dtree_pred)
print("MAE of decision tree model is:", dtree_mae)
print("R2 score of decision tree model is:", dtree_r2)


# In[29]:


dtree_score = cross_val_score(dtree, X_test, y_test, cv=4)
print("Decision Tree model accuracy is: {}".format(dtree_score.mean()*100))


# In[31]:


xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)


# In[32]:


xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)
print("MAE of xgboost model is:", xgb_mae)
print("R2 score of xgboost model is:", xgb_r2)


# In[33]:


xgb_score = cross_val_score(xgb, X_test, y_test, cv=4)
print("Decision Tree model accuracy is: {}".format(xgb_score.mean()*100))


# In[34]:


df.columns


# In[35]:


df.head()


# In[36]:


data = {'year':2017, 'transmission':1, 'mileage':15944, 'fuelType':0, 'tax':150, 'mpg':57.7,
       'engineSize':1.0}
index= [0]
new_df = pd.DataFrame(data, index)
new_df


# In[37]:


new_pred = xgb.predict(new_df)
print("The car price for the new data is: ", new_pred)

