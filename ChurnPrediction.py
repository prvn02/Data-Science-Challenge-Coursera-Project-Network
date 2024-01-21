#!/usr/bin/env python
# coding: utf-8

# ![COUR_IPO.png](attachment:COUR_IPO.png)

# # Welcome to the Data Science Coding Challange!
# 
# Test your skills in a real-world coding challenge. Coding Challenges provide CS & DS Coding Competitions with Prizes and achievement badges!
# 
# CS & DS learners want to be challenged as a way to evaluate if they’re job ready. So, why not create fun challenges and give winners something truly valuable such as complimentary access to select Data Science courses, or the ability to receive an achievement badge on their Coursera Skills Profile - highlighting their performance to recruiters.

# ## Introduction
# 
# In this challenge, you'll get the opportunity to tackle one of the most industry-relevant maching learning problems with a unique dataset that will put your modeling skills to the test. Subscription services are leveraged by companies across many industries, from fitness to video streaming to retail. One of the primary objectives of companies with subscription services is to decrease churn and ensure that users are retained as subscribers. In order to do this efficiently and systematically, many companies employ machine learning to predict which users are at the highest risk of churn, so that proper interventions can be effectively deployed to the right audience.
# 
# In this challenge, we will be tackling the churn prediction problem on a very unique and interesting group of subscribers on a video streaming service! 
# 
# Imagine that you are a new data scientist at this video streaming company and you are tasked with building a model that can predict which existing subscribers will continue their subscriptions for another month. We have provided a dataset that is a sample of subscriptions that were initiated in 2021, all snapshotted at a particular date before the subscription was cancelled. Subscription cancellation can happen for a multitude of reasons, including:
# * the customer completes all content they were interested in, and no longer need the subscription
# * the customer finds themselves to be too busy and cancels their subscription until a later time
# * the customer determines that the streaming service is not the best fit for them, so they cancel and look for something better suited
# 
# Regardless the reason, this video streaming company has a vested interest in understanding the likelihood of each individual customer to churn in their subscription so that resources can be allocated appropriately to support customers. In this challenge, you will use your machine learning toolkit to do just that!

# ## Understanding the Datasets

# ### Train vs. Test
# In this competition, you’ll gain access to two datasets that are samples of past subscriptions of a video streaming platform that contain information about the customer, the customers streaming preferences, and their activity in the subscription thus far. One dataset is titled `train.csv` and the other is titled `test.csv`.
# 
# `train.csv` contains 70% of the overall sample (243,787 subscriptions to be exact) and importantly, will reveal whether or not the subscription was continued into the next month (the “ground truth”).
# 
# The `test.csv` dataset contains the exact same information about the remaining segment of the overall sample (104,480 subscriptions to be exact), but does not disclose the “ground truth” for each subscription. It’s your job to predict this outcome!
# 
# Using the patterns you find in the `train.csv` data, predict whether the subscriptions in `test.csv` will be continued for another month, or not.

# ### Dataset descriptions
# Both `train.csv` and `test.csv` contain one row for each unique subscription. For each subscription, a single observation (`CustomerID`) is included during which the subscription was active. 
# 
# In addition to this identifier column, the `train.csv` dataset also contains the target label for the task, a binary column `Churn`.
# 
# Besides that column, both datasets have an identical set of features that can be used to train your model to make predictions. Below you can see descriptions of each feature. Familiarize yourself with them so that you can harness them most effectively for this machine learning task!

# In[28]:


import pandas as pd
data_descriptions = pd.read_csv('data_descriptions.csv')
pd.set_option('display.max_colwidth', None)
data_descriptions


# ## How to Submit your Predictions to Coursera
# Submission Format:
# 
# In this notebook you should follow the steps below to explore the data, train a model using the data in `train.csv`, and then score your model using the data in `test.csv`. Your final submission should be a dataframe (call it `prediction_df` with two columns and exactly 104,480 rows (plus a header row). The first column should be `CustomerID` so that we know which prediction belongs to which observation. The second column should be called `predicted_probability` and should be a numeric column representing the __likellihood that the subscription will churn__.
# 
# Your submission will show an error if you have extra columns (beyond `CustomerID` and `predicted_probability`) or extra rows. The order of the rows does not matter.
# 
# The naming convention of the dataframe and columns are critical for our autograding, so please make sure to use the exact naming conventions of `prediction_df` with column names `CustomerID` and `predicted_probability`!
# 
# To determine your final score, we will compare your `predicted_probability` predictions to the source of truth labels for the observations in `test.csv` and calculate the [ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html). We choose this metric because we not only want to be able to predict which subscriptions will be retained, but also want a well-calibrated likelihood score that can be used to target interventions and support most accurately.

# ## Import Python Modules
# 
# First, import the primary modules that will be used in this project. Remember as this is an open-ended project please feel free to make use of any of your favorite libraries that you feel may be useful for this challenge. For example some of the following popular packages may be useful:
# 
# - pandas
# - numpy
# - Scipy
# - Scikit-learn
# - keras
# - maplotlib
# - seaborn
# - etc, etc

# In[2]:


# Import required packages

# Data packages
import pandas as pd
import numpy as np

# Machine Learning / Classification packages
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

# Visualization Packages
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Import any other packages you may want to use
# Additional Machine Learning packages
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Evaluation metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Visualization packages
import matplotlib.pyplot as plt
import seaborn as sns


# ## Load the Data
# 
# Let's start by loading the dataset `train.csv` into a dataframe `train_df`, and `test.csv` into a dataframe `test_df` and display the shape of the dataframes.

# In[5]:


train_df = pd.read_csv("train.csv")
print('train_df Shape:', train_df.shape)
train_df.head()


# In[6]:


test_df = pd.read_csv("test.csv")
print('test_df Shape:', test_df.shape)
test_df.head()


# ## Explore, Clean, Validate, and Visualize the Data (optional)
# 
# Feel free to explore, clean, validate, and visualize the data however you see fit for this competition to help determine or optimize your predictive model. Please note - the final autograding will only be on the accuracy of the `prediction_df` predictions.

# In[20]:


# Load the training data
train_df = pd.read_csv('train.csv')

# Load the testing data
test_df = pd.read_csv('test.csv')

# Display the shape of the dataframes
print("Training Data Shape:", train_df.shape)
print("Testing Data Shape:", test_df.shape)

# Display the first few rows of the training data
print("\nTraining Data Preview:")
print(train_df.head())

# Display basic statistics of the training data
print("\nTraining Data Statistics:")
print(train_df.describe())

# Check for missing values in the training data
print("\nMissing Values in Training Data:")
print(train_df.isnull().sum())

# Visualize the distribution of the target variable ('Churn')
sns.countplot(x='Churn', data=train_df)
plt.title('Distribution of Churn in Training Data')
plt.show()


# ## Make predictions (required)
# 
# Remember you should create a dataframe named `prediction_df` with exactly 104,480 entries plus a header row attempting to predict the likelihood of churn for subscriptions in `test_df`. Your submission will throw an error if you have extra columns (beyond `CustomerID` and `predicted_probaility`) or extra rows.
# 
# The file should have exactly 2 columns:
# `CustomerID` (sorted in any order)
# `predicted_probability` (contains your numeric predicted probabilities between 0 and 1, e.g. from `estimator.predict_proba(X, y)[:, 1]`)
# 
# The naming convention of the dataframe and columns are critical for our autograding, so please make sure to use the exact naming conventions of `prediction_df` with column names `CustomerID` and `predicted_probability`!

# ### Example prediction submission:
# 
# The code below is a very naive prediction method that simply predicts churn using a Dummy Classifier. This is used as just an example showing the submission format required. Please change/alter/delete this code below and create your own improved prediction methods for generating `prediction_df`.

# **PLEASE CHANGE CODE BELOW TO IMPLEMENT YOUR OWN PREDICTIONS**

# In[21]:


# Assuming 'Churn' is the target variable, and you want to drop 'CustomerID'
# Adjust the features accordingly based on your dataset
features = train_df.drop(['CustomerID', 'Churn'], axis=1)

# Identify categorical columns
categorical_columns = features.select_dtypes(include=['object']).columns.tolist()

# One-hot encode categorical columns
features = pd.get_dummies(features, columns=categorical_columns)

# Instantiate and fit the Logistic Regression model
logreg_model = LogisticRegression(random_state=42)
logreg_model.fit(features, train_df.Churn)


# In[22]:


# Assuming 'CustomerID' needs to be dropped before making predictions
# Adjust the features accordingly based on your dataset
test_features = test_df.drop(['CustomerID'], axis=1)

# One-hot encode categorical columns in the test data (if necessary)
test_features = pd.get_dummies(test_features, columns=categorical_columns, drop_first=True)

# Ensure that the test data has the same columns as the training data
missing_features = set(features.columns) - set(test_features.columns)
for feature in missing_features:
    test_features[feature] = 0  # Add missing features with zero values

# Use the logistic regression model to make predictions on the test data
predicted_probability = logreg_model.predict_proba(test_features)[:, 1]


# In[23]:


# Combine predictions with label column into a dataframe
prediction_df = pd.DataFrame({'CustomerID': test_df['CustomerID'],
                               'predicted_probability': predicted_probability})


# In[24]:


# Assuming 'CustomerID' needs to be dropped before making predictions
# Adjust the features accordingly based on your dataset
test_features = test_df.drop(['CustomerID'], axis=1)

# One-hot encode categorical columns in the test data (if necessary)
test_features = pd.get_dummies(test_features, columns=categorical_columns, drop_first=True)

# Ensure that the test data has the same columns as the training data
missing_features = set(features.columns) - set(test_features.columns)
for feature in missing_features:
    test_features[feature] = 0  # Add missing features with zero values

# Use the logistic regression model to make predictions on the test data
predicted_probability = logreg_model.predict_proba(test_features)[:, 1]

# Combine predictions with label column into a dataframe
prediction_df = pd.DataFrame({'CustomerID': test_df['CustomerID'],
                               'predicted_probability': predicted_probability})

# View the shape and the first 10 rows of the prediction_df dataframe
print(prediction_df.shape)
print(prediction_df.head(10))


# **PLEASE CHANGE CODE ABOVE TO IMPLEMENT YOUR OWN PREDICTIONS**

# ## Final Tests - **IMPORTANT** - the cells below must be run prior to submission
# 
# Below are some tests to ensure your submission is in the correct format for autograding. The autograding process accepts a csv `prediction_submission.csv` which we will generate from our `prediction_df` below. Please run the tests below an ensure no assertion errors are thrown.

# In[25]:


prediction_df.to_csv("prediction_submission.csv", index=False)
submission = pd.read_csv("prediction_submission.csv")
assert isinstance(submission, pd.DataFrame), 'You should have a dataframe named prediction_df.'


# In[26]:


assert submission.columns[0] == 'CustomerID', 'The first column name should be CustomerID.'
assert submission.columns[1] == 'predicted_probability', 'The second column name should be predicted_probability.'


# In[27]:


assert submission.shape[0] == 104480, 'The dataframe prediction_df should have 104480 rows.'


# In[28]:


assert submission.shape[1] == 2, 'The dataframe prediction_df should have 2 columns.'


# In[15]:


# FINAL TEST CELLS - please make sure all of your code is above these test cells

## This cell calculates the auc score and is hidden. Submit Assignment to see AUC score.


# ## SUBMIT YOUR WORK!
# 
# Once we are happy with our `prediction_df` and `prediction_submission.csv` we can now submit for autograding! Submit by using the blue **Submit Assignment** at the top of your notebook. Don't worry if your initial submission isn't perfect as you have multiple submission attempts and will obtain some feedback after each submission!
