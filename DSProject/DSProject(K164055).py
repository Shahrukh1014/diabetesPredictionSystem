#!/usr/bin/env python
# coding: utf-8

# # Importing Important Libraries.

# In[227]:


from scipy import optimize
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


# # Importing Data  

# In[228]:


data = pd.read_csv('diabetes.csv')


# # Visualizing Data

# In[229]:


print (data.shape)


# In[230]:


print (data.describe())


# # Defining Functions for Mean and Mode.

# In[231]:


def chkColumnForVal(col_name,value):
    print (col_name)
    rowcount=0
    out_array=[]
    for t in df[col_name]:
        if(t<value):
            out_array.append(rowcount)
        rowcount=rowcount+1
    return len(out_array)


# In[232]:


def cal_mmm(col_name):
    mean = df[col_name].mean()
    mode = df[col_name].mode()
    #median = df[col_name].median
    mmm_array=[mean,mode]
    return mmm_array


# # Data into Dataframe 

# In[233]:


df = DataFrame.from_csv('diabetes.csv', header = 0, sep = ',' ,index_col = None)


# # Data Cleaning Process From here.

# # 1) Checking for Null Values (if any)

# In[234]:


print (df.isnull().sum())
#this shows thr is no null values in any of the column, but I've replaced the null values before in the code below.


# # Checking for Outliers

# In[235]:


df.boxplot(figsize=(12,8))


# In[236]:


df.hist(figsize=(10, 8))


# In[237]:


df.plot()


# In[238]:


ax1 = df.plot.scatter(x="Pregnancies", y="Outcome")
ax1 = df.plot.scatter(x="Insulin", y="Outcome")
ax1 = df.plot.scatter(x="BloodPressure", y="Outcome")
ax1 = df.plot.scatter(x="Glucose", y="Outcome")
ax1 = df.plot.scatter(x="SkinThickness", y="Outcome")
ax1 = df.plot.scatter(x="BMI", y="Outcome")
ax1 = df.plot.scatter(x="Age", y="Outcome")


# In[239]:


#filling missing values from each column to their mean value

#df['Glucose'].fillna(cal_mmm("Glucose")[0], inplace=True)
#df['BloodPressure'].fillna(cal_mmm("BloodPressure")[0], inplace=True)
#df['SkinThickness'].fillna(cal_mmm("SkinThickness")[0], inplace=True)
#df['BMI'].fillna(cal_mmm("BMI")[0], inplace=True)
#df['DiabetesPedigreeFunction'].fillna(cal_mmm("DiabetesPedigreeFunction")[0], inplace=True)

#we can ignore this part if no null values are thr! 


# # Replacing Zeros with Mean Values

# In[240]:


#filling zero values from each column to their mean value

df['Glucose']=df.Glucose.mask(data.Glucose == 0,cal_mmm("Glucose")[0])
df['BloodPressure']=df.BloodPressure.mask(data.BloodPressure == 0,cal_mmm("BloodPressure")[0])
df['SkinThickness']=df.SkinThickness.mask(data.SkinThickness == 0,cal_mmm("SkinThickness")[0])
df['Insulin']=df.Insulin.mask(data.Insulin == 0,cal_mmm("Insulin")[0])
df['BMI']=df.BMI.mask(data.BMI == 0,cal_mmm("BMI")[0])
df['DiabetesPedigreeFunction']=df.DiabetesPedigreeFunction.mask(data.DiabetesPedigreeFunction == 0,cal_mmm("DiabetesPedigreeFunction")[0])


# # Spliting Data into Training and Testing Dataset

# In[242]:


#y = df["Outcome"]
X = df.values[:, 0:8]
Y = df.values[:,8]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# # Model 1: Using Gini.

# In[243]:


clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
clf_gini.fit(X_train, y_train)
print(clf_gini)


# In[244]:


y_pred = clf_gini.predict(X_test)
print(y_pred)


# # Accuracy Using Gini

# In[207]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred)*100)


# # Model 2: Using KNN.

# In[245]:


from sklearn.neighbors import KNeighborsClassifier


# In[246]:


neigh = KNeighborsClassifier(n_neighbors=20)


# In[247]:


neigh.fit(X_train, y_train)


# In[248]:


pred = neigh.predict(X_test)
print(pred)


# # Accuracy Using KNN.

# In[249]:


print(accuracy_score(y_test,pred)*100)


# # Model 3: Using Naives Byes

# In[253]:


from sklearn.naive_bayes import GaussianNB
import numpy as np


# In[254]:


model = GaussianNB()


# In[255]:


model.fit(X_train, y_train)


# In[256]:


predicted= model.predict(X_test)
print(predicted)


# # Accuracy Using Naive Byes

# In[258]:


print(accuracy_score(y_test,predicted)*100)


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




