#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import the necessary libraries
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# read the csv diabetedataset
pima = pd.read_csv("diabetes.csv")
pima


# In[ ]:


# show the last 10 records of the dataset
pima.tail(10)


# In[ ]:


# show the first 10 records of the dataset

pima.head(10)


# In[ ]:


# Find the dimensions of the pima dataframe using method .ndim
pima.ndim


# In[ ]:


#Pima dataframe has 768 rows and 9 columns. The first element shows the number of rows in the data and the second element shows the number of columns in the data.
pima.shape 


# In[ ]:


# Find the size of pima dataframe
pima.size


# In[ ]:


# Get the datatypes of the variables in the dataset using info() function
pima.info()


# In[ ]:


# check for any missing values using 
pima.isnull().values.any()


# In[ ]:


# Obtain the summary statistics using .describe() function of all variables present in all rows and all columns except the last column using .iloc() function

pima.iloc[:,0:8].describe()


# In[ ]:


# plot a distribution plot for variable BloodPressure

sns.displot(pima['BloodPressure'], kind='kde')
plt.show()


# In[ ]:


# Obtain the BMI of the person having the highest Glucose
pima[pima['Glucose']==pima['Glucose'].max()]['BMI']


# In[ ]:


# Obtain the mean, median and mode of BMI

m1 = pima['BMI'].mean()  # mean
print(m1)
m2 = pima['BMI'].median()  # median
print(m2)
m3 = pima['BMI'].mode()[0]  # mode
print(m3)


# In[ ]:


# Obtain the number of women with glucose levels above mean level of glucose

pima[pima['Glucose']>pima['Glucose'].mean()].shape[0]


# In[ ]:


# obtain the number of women with blood pressure equals to the median blood pressure and less than the median BMI

pima[(pima['BloodPressure']==pima['BloodPressure'].median()) & (pima['BMI']<pima['BMI'].median())]


# In[ ]:


# Create a pairplot for the variables 'Glucose', 'SkinThickness', and 'DiabetesPedigreeFunction'
sns.pairplot(data=pima,vars=['Glucose', 'SkinThickness', 'DiabetesPedigreeFunction'], hue='Outcome')
plt.show()


# In[ ]:


# Plot the scatterplot between 'Glucose' and 'Insulin'

sns.scatterplot(x='Glucose',y='Insulin',data=pima)
plt.show()


# In[ ]:


# Plot the boxplot for the 'Age' variable

plt.boxplot(pima['Age'])

plt.title('Boxplot of Age')
plt.ylabel('Age')
plt.show()


# In[ ]:


# Plot histograms for the 'Age' variable to understand the number of women in different age groups who have diabetes 

plt.hist(pima[pima['Outcome']==1]['Age'], bins = 5)
plt.title('Distribution of Age for Women who has Diabetes')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


# Plot histograms for the 'Age' variable to understand the number of women in different age groups who do not have diabetes

plt.hist(pima[pima['Outcome']==0]['Age'], bins = 5)
plt.title('Distribution of Age for Women who do not have Diabetes')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


# Find the interquartile range of all the variables

Q1 = pima.quantile(0.25)
Q3 = pima.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[ ]:


# Find the correlation matrix
corr_matrix = pima.iloc[:,0:8].corr()

corr_matrix


# In[ ]:


# plot the matrix

plt.figure(figsize=(8,8))
sns.heatmap(corr_matrix, annot = True)

# display the plot
plt.show()

