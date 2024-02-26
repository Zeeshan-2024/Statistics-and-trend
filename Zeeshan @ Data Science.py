#!/usr/bin/env python
# coding: utf-8

# # Importing Module

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# # Dataset Loading

# In[2]:


# Loading the dataset
dataset = pd.read_csv("diabetes.csv")


# # Display basic information about the dataset

# In[3]:


dataset.head(5)


# In[4]:


dataset.info()


# # Checking for missing values

# In[5]:


print("Missing values:\n", dataset.isnull().sum())


# ## Histogram Plot

# In[6]:


def plot_histogram():
    plt.figure(figsize=(8, 6))
    plt.hist(dataset['Age'], bins=20, color='skyblue')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()
plot_histogram()


# ## Scatter Plot

# In[7]:


def plot_scatter():
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Glucose', y='Insulin', data=dataset, color='orange')
    plt.title('Glucose vs. Insulin')
    plt.xlabel('Glucose')
    plt.ylabel('Insulin')
    plt.show()
plot_scatter()


# ## Correlation Matrix

# In[22]:


def plot_heatmap():
    plt.figure(figsize=(10, 8))
    correlation_matrix = dataset.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
plot_heatmap()


# ## Descriptive Statistics

# In[9]:


# Calculation of statistics (describe and corr)
describe_stats = dataset.describe()
correlation_stats = dataset.corr()


# In[10]:


describe_stats


# In[11]:


correlation_stats


# # Model Evaluation

# ## Dataset splitting into train and test

# In[12]:


# Spliting the dataset into features (X) and target variable (y)
X = dataset.drop(columns=['Outcome'])
y = dataset['Outcome']


# In[13]:


# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Logistic Regression

# In[14]:


base_model = LogisticRegression()
base_model.fit(X_train, y_train)


# In[15]:


# Making predictions on the test set (20% data)
y_pred = base_model.predict(X_test)


# In[16]:


# Evaluating the performance of the base model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[17]:


# classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# # Random Forest Classifier

# In[18]:


random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)


# In[19]:


# Making predictions on the test set (20%)
y_pred = random_forest_model.predict(X_test)


# In[20]:


# Evaluating the performance of the Random Forest model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[21]:


# Printing classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:




