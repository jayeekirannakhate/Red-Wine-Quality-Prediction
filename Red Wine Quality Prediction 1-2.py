#!/usr/bin/env python
# coding: utf-8

# # Red Wine Quality Prediction
# 
# ##  Problem Statement:
# 
# The dataset is related to red and white variants of the Portuguese "Vinho Verde" wine. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
# 
# 
# This dataset can be viewed as classification task. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.
# 
# ## Attribute Information
# 
# Input variables (based on physicochemical tests):
# 1 - fixed acidity
# 2 - volatile acidity
# 3 - citric acid
# 4 - residual sugar
# 5 - chlorides
# 6 - free sulfur dioxide
# 7 - total sulfur dioxide
# 8 - density
# 9 - pH
# 10 - sulphates
# 11 - alcohol
# Output variable (based on sensory data):
# 12 - quality (score between 0 and 10)
# 
# What might be an interesting thing to do, is to set an arbitrary cutoff for your dependent variable (wine quality) at e.g. 7 or higher getting classified as 'good/1' and the remainder as 'not good/0'.
# This allows you to practice with hyper parameter tuning on e.g. decision tree algorithms looking at the ROC curve and the AUC value.
# 
# You need to build a classification model. 
# 
# ## Inspiration
# 
# Use machine learning to determine which physiochemical properties make a wine 'good'!
# 
# 
# 
# ### Downlaod Files:
# https://github.com/dsrscientist/DSData/blob/master/winequality-red.csv
# 

# ### To set a raw data from github

# In[1]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[2]:


# importing require library
import pandas as pd # for data wrangling purpose
import numpy as np # Basic computation library
import seaborn as sns # For Visualization 
import matplotlib.pyplot as plt # ploting package
from sklearn.model_selection import train_test_split


# ### Importing a data

# In[3]:


#importing data
df=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv')


# In[4]:


df.head()


# In[5]:


#data shape
df.shape


# In[6]:


df.info()


# In[7]:


# cheaking the null values
sns.heatmap(df.isnull())
plt.title("Null values")
plt.show()


# In[8]:


df.isnull().sum()


# * No null value in data set

# # Data Analysis and visulaization

# In[9]:


# statsical summary
df.describe()


# In[10]:


df['quality'].value_counts()


# In[11]:


plt.figure(figsize=(5,5))
labels = '5','6','7','4','8','3'
fig, ax = plt.subplots()
ax.pie(df['quality'].value_counts(),labels = labels,radius =3 ,autopct = '%1.1f%%', shadow=True,)
plt.show()


# * We have 6 different type of wine quality samples
# * Majority of samples are of quality 5 and 6 
# * 3 is bad quality
# * 7 and 8 is in good quality category

# In[12]:


df.columns


# In[13]:


#fixed acidity vs qulity
plot = plt.figure(figsize=(6,6))
sns.barplot(x= 'quality', y = 'fixed acidity', data = df)
plt.show()


# * fixed acidity is almost same in all quality of wine

# In[14]:


#acidity vs qulity
plot = plt.figure(figsize=(6,6))
sns.barplot(x= 'quality', y = 'volatile acidity', data = df)
plt.show()


# * volatile acidity available in bad quality wine more that good one
# * volatile acidity is more the quality of wine is less

# In[15]:


#citric acid vs qulity
plot = plt.figure(figsize=(6,6))
sns.barplot(x= 'quality', y = 'citric acid', data = df)
plt.show()


# * This is opoosite to volatile acidity
# * If the citric acid content low the qulity of wine is low
# * and if the citric acid high the qulity of wine is also high

# In[16]:


#residual sugar vs qulity
plot = plt.figure(figsize=(6,6))
sns.barplot(x= 'quality', y = 'residual sugar', data = df)
plt.show()


# * residual sugar is almost same in all quality of wine

# In[17]:


#chlorides vs qulity
plot = plt.figure(figsize=(6,6))
sns.barplot(x= 'quality', y = 'chlorides', data = df)
plt.show()


# * The high chlorides the low wine quality is

# In[18]:


#free sulfur dioxide vs qulity
plot = plt.figure(figsize=(6,6))
sns.barplot(x= 'quality', y = 'free sulfur dioxide', data = df)
plt.show()


# In[19]:


#total sulfur dioxide vs qulity
plot = plt.figure(figsize=(6,6))
sns.barplot(x= 'quality', y = 'total sulfur dioxide', data = df)
plt.show()


# * total sulfur dioxide used more in averge qualite wine

# In[20]:


#density vs qulity
plot = plt.figure(figsize=(6,6))
sns.barplot(x= 'quality', y = 'density', data = df)
plt.show()


# * Density is similar in every type of wine

# In[21]:


#pH vs qulity
plot = plt.figure(figsize=(6,6))
sns.barplot(x= 'quality', y = 'pH', data = df)
plt.show()


# In[22]:


#sulphates vs qulity
plot = plt.figure(figsize=(6,6))
sns.barplot(x= 'quality', y = 'sulphates', data = df)
plt.show()


# * The more sulphates the more quality is

# In[23]:


#alcohol vs qulity
plot = plt.figure(figsize=(6,6))
sns.barplot(x= 'quality', y = 'alcohol', data = df)
plt.show()


# In[24]:


df.hist(figsize=(10,10),bins=50)
plt.show()


# ## constructing a heatmap to undestand the correlation

# In[25]:


correlation = df.corr()


# In[26]:


plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar= True, square=True, fmt='.1f',annot =True, annot_kws={'size':8}, cmap = 'Reds')


# ## data preprocessing
# 

# In[27]:


# separate the data and label
X = df.drop("quality",axis=1)


# In[28]:


X


# ## Feature Selection
# 

# In[29]:


df['g_quality'] = [1 if x >= 7 else 0 for x in df['quality']]
X = df.drop(['quality','g_quality'], axis = 1)
Y = df['g_quality']


# In[30]:


df['g_quality'].value_counts()


# In[31]:


X


# In[32]:


Y


# ## Splitting Dataset
# 

# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=5)


# In[34]:


print(Y.shape, y_train.shape, y_test.shape)


# ## Model Training
# ### random forest classifier
# 
# 
# Random forest is one of the versatile algorithm in machine learning. It is believed that random forest is one of the best algorithm we can see in ML. Because it avoids bias, at the same time it is doesn't depend on only one model but uses multiple models to make any decision, that is the reason random forest is the favourite algorithm of any data scientist.
# 
# Random Forests are a combination of tree predictors where each tree depends on the values of a random vector sampled independently with the same distribution for all trees in the forest. The basic principle is that a group of “weak learners” can come together to form a “strong learner”.

# In[35]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[36]:


model = RandomForestClassifier()


# In[38]:


model.fit(X_train, y_train)


# In[39]:


model.score(X_test,y_test)


# ### Building a predictive system

# In[40]:


input_data = (8.5,0.28,0.56,1.8,0.092,35.0,103.0,0.9969,3.3,0.75,10.5)

# changing the input data to nummpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we predicting the lable for one value
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
    print('Good Quality wine')
    
else:
    print('bad Quality wine')


# In[41]:


input_data = (7.9,0.43,0.21,1.6,0.106,10.0,37.0,0.9966,3.17,0.91,9.5)

# changing the input data to nummpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we predicting the lable for one value
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
    print('Good Quality wine')
    
else:
    print('bad Quality wine')


# In[ ]:





# In[ ]:




