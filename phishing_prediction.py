#!/usr/bin/env python
# coding: utf-8

# In[25]:


#importing all the necessary library.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.metrics import accuracy_score


# In[6]:


df=pd.read_csv(r"C:\Users\tusha\Downloads\Phishing Data - Phishing Data.csv")
df


# In[7]:


df.shape


# In[8]:


df.columns


# In[9]:


df.info()


# In[10]:


# checking if there is any NULL values
df.isnull().sum()


# In[11]:


#plotting the distribution of the features
df.hist(bins=50,figsize=(12,12))
plt.show()


# In[12]:


#visualizing the correlation with respect to result.
plt.figure(figsize=(8, 12))
heat=sns.heatmap(df.corr()[['Result']].sort_values(by='Result',ascending=False),vmin=-1,vmax=1,annot=True)
heat.set_title('Correlation w.r.t Result',fontdict={'fontsize':16},pad=18)


# As we can see that the Column 'SSfinal_State' is least correlated and 'Domain_registration_length' is higly correlated.
# 

# In[13]:


#descirbe
df.describe()


# In[14]:


#Splitting the Design matrix and predicate.
y=df['Result']
X=df.drop('Result',axis=1)


# In[15]:


X.shape,y.shape


# In[19]:


#splitting dataset into test and train with 70-30 split.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, random_state = 12)
X_train.shape, X_test.shape


# ## As we can see that dataset is Classification dataset as it is predicting wheather the site is phishing site(1) or not(0)
# ## So we will use Logistic Regression.
# 
# 
# 
# 

# ### LOGISTIC REGRESION - it is a basemodel for all classification. it  is very efficient when we have only to types of classes to predict and also when the data is linearly seperable. And it will perform good with our dataset. it uses sigmoid function for the threshold.
# 

# In[57]:


#Logistic resgression model.
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(random_state=0,max_iter=100)
model.fit(X_train,y_train)



# In[58]:


#predicting train and test dataset.
y_train_pred= model.predict(X_train)
y_test_pred=model.predict(X_test)


# ### Using Confusion Matrix for evaluation

# In[69]:


from sklearn.metrics import confusion_matrix
confuse_matrix= confusion_matrix(y_test,y_test_pred)
print(confuse_matrix)


# In[75]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_test_pred))


# In[76]:


sns.heatmap(confuse_matrix/np.sum(confuse_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
plt.ylabel('Actuals')
plt.xlabel('Predictions')
plt.show()


# In[78]:


from sklearn.metrics import accuracy_score
acc_score=accuracy_score(y_test,y_test_pred)
acc_score


# #### ROC CURVE FOR THE MODEL
# 

# In[79]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test , y_test_pred)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# #### AOC CURVE FOR THE MODEL

# In[80]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test , y_test_pred)


plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.fill_between(fpr, tpr)
plt.tight_layout()
plt.show()


# In[ ]:




