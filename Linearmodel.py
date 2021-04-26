#!/usr/bin/env python
# coding: utf-8

# 

# ## Imports
# ** Import pandas, numpy, matplotlib,and seaborn. Then set %matplotlib inline 
# (You'll import sklearn as you need it.)**

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# 

# In[3]:


customers=pd.read_csv('Ecommerce Customers')


# 

# In[4]:


customers.head()


# In[6]:


customers.info()


# In[7]:


customers.describe()


# 

# In[13]:


sns.jointplot(data=customers, x="Time on Website", y="Yearly Amount Spent")


# In[281]:





# ** Do the same but with the Time on App column instead. **

# In[14]:


sns.jointplot(data=customers, x="Time on Website", y="Time on App")


# 

# In[18]:


sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)


# 

# In[19]:


sns.pairplot(customers)


# 

# In[285]:


lenght of membership


# 

# In[20]:


sns.lmplot(x='Yearly Amount Spent',y='Length of Membership',data=customers)


# 

# In[21]:


customers.columns


# In[48]:


X=customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y=customers['Yearly Amount Spent']


# 

# In[49]:


from sklearn.model_selection import train_test_split


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# 

# In[51]:


from sklearn.linear_model import LinearRegression


# 

# In[52]:


lm=LinearRegression()


# 

# In[53]:


lm.fit(X_train,y_train)


# 

# In[63]:


print('Coefficients: \n' , lm.coef_)


# 

# In[66]:


prediction=lm.predict(X_test)


# 

# In[67]:


plt.scatter(y_test,prediction)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# 

# In[69]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# 

# In[70]:


sns.displot(y_test-prediction)


# 

# In[74]:


coefficients=pd.DataFrame(lm.coef_,X.columns)
coefficients.name=['Coeffieints']
coefficients


# ** How can you interpret these coefficients? **

# Interpreting the coefficients:
# 
# Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
# Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
# Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
# Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.

# **Do you think the company should focus more on their mobile app or on their website?**

# This is tricky, there are two ways to think about this: Develop the Website to catch up to the performance of the mobile app, or develop the app more since that is what is working better. This sort of answer really depends on the other factors going on at the company, you would probably want to explore the relationship between Length of Membership and the App or the Website before coming to a conclusion!

# 
