#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


engagement = pd.read_csv('takehome_user_engagement.csv')
df = pd.read_csv('takehome_users.csv', encoding = 'latin')


# In[3]:


engagement.head()


# In[4]:


df.head()


# In[6]:


engage = engagement.groupby('user_id').filter(lambda x: len(x) >= 3)


# In[7]:


engage.reset_index(drop=True, inplace = True) 


# In[8]:


active = 0
active_users = []
for i in range(len(engage)-2):
    user = engage['user_id'][i] 
    if user != active and user == engage['user_id'][i+2]: 
        st = pd.Timestamp(engage['time_stamp'][i]) 
        et = st + pd.Timedelta('7D') 
        if st < pd.Timestamp(engage['time_stamp'][i+1]) < et and st < pd.Timestamp(engage['time_stamp'][i+2]) < et:
            active_users.append(user) 
            active = user 


# In[9]:


len(active_users)


# In[10]:


y = pd.Series(np.random.randn(len(df)))
n = 0
for i in range(len(df)):
    if df['object_id'][i] == active_users[n]:
        y[i] = 1
        n = n+1
        if n > len(active_users)-1:
            n = n -1
    else:
        y[i] = 0
y.head()


# In[11]:


df1 = pd.DataFrame(y,columns = ['active_users'])


# In[12]:


df = pd.concat([df,df1], axis = 1)


# In[13]:


df.head()


# In[14]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(['ORG_INVITE','GUEST_INVITE','PERSONAL_PROJECTS','SIGNUP','SIGNUP_GOOGLE_AUTH'])


# In[15]:


creation = le.transform(df['creation_source'])


# In[16]:


df3 = pd.DataFrame(creation,columns = ['creation'])


# In[17]:


df = pd.concat([df,df3],axis=1)


# In[18]:


df.isnull().sum()


# In[19]:


3177/len(df["last_session_creation_time"])*100


# In[20]:


df.head()


# In[21]:


df = df[df.last_session_creation_time.notnull()]
df = df[df.invited_by_user_id.notnull()]


# In[24]:


y = df['active_users']
X = df[['creation','last_session_creation_time','opted_in_to_mailing_list','enabled_for_marketing_drip','org_id','invited_by_user_id']]


# In[25]:


from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)


# In[26]:


from sklearn.tree import DecisionTreeClassifier

# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth 
dt = DecisionTreeClassifier()

# Fit dt to the training set
dt.fit(X_train, y_train) # it will ask all possible questions, compute the information gain and choose the best split

# Predict test set labels
y_pred = dt.predict(X_test)
y_pred


# In[27]:


from sklearn.metrics import accuracy_score, roc_auc_score, plot_roc_curve
#we compute the eval metric on test/validation set only primarily

# Predict test set labels
y_pred = dt.predict(X_test) 

# Compute test set accuracy
acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))
acc = roc_auc_score(y_test, y_pred)
print("Test set auc: {:.2f}".format(acc))
plot_roc_curve(dt, X_test, y_test)


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from matplotlib import pyplot
dt = DecisionTreeClassifier(max_depth = 2)
# Fit dt to the training set
dt.fit(X_train, y_train)
importance = dt.feature_importances_
# pyplot.bar([x for x in range(len(importance))], importance)
list(zip(importance,X_test.columns)) # it calculates the feature importances based on IG


# Conclusion
# 
# From the above data we found that the Decission Tree is the only successfull machine learning model with the "area under curve - receiver operator characterstics" (AUROC) value as 
# 0.85. By this ML Model we can conclude that **"last_session_creation_time"** has maximum importance.
# 
# 
