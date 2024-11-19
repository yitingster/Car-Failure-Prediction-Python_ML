# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:22:26 2023

@author: Yi Ting
"""

# =============================================================================
# Import packages
# =============================================================================
import numpy as np
import pandas as pd
import sqlite3
import seaborn as sns
import os

from src import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report

# =============================================================================
# 1. Connect to data
# =============================================================================

dbfile = "C:/Users/Yi Ting/Desktop/AIAP/data/failure.db"
con = sqlite3.connect(dbfile)

#curosr
cur = con.cursor()

#read table name
table_list = [a for a in cur.execute(
    "SELECT name FROM sqlite_master WHERE type = 'table'")
    ]

#output: failure
print(table_list)


# =============================================================================
# 2. Extract data to df, check src.py file imported
# =============================================================================

# select from 'noshow' as per table_list output-failure
df = pd.read_sql_query('SELECT * FROM failure', con)

#close connection
con.close()

#Display data to ensure extract success
df.head()

#check src.py file imported
print_test('sucess')


# =============================================================================
# 3. Data cleaning 
# [Same as task1 but focus only on preparing data for Machinie learning]
# 3.1 Factory: Need to ensure consistency in data entry  
# 3.2 Temperature: Need same unit of measurement
# =============================================================================

### 3.1 #### 
#Cleaning inconsistent data entry in 'Factory'.
df['Factory'] = df['Factory'].replace({
    'Newton, China':'New York, U.S', 
    'Seng Kang, China':'Shang Hai, China',  
    'Bedok, Germany':'Berlin, Germany'
    })

print("After cleaning data-inconsistency in Factory column, the unique values are: {}".format(str(set(df.Factory))))


### 3.2 #### 
#Clearn data inconsistency in Temperature column
#change farenheit to degree celcius 

#Split Temperature from unit of measurement
df[['Temperature2', 'Temperature_units']] = df[
    "Temperature"].str.split(" ", expand=True)

#Remove white space in the split Temperature2
df['Temperature2'].str.strip()
#change type to float
df['Temperature2'] = df['Temperature2'].astype(float)

#Remove °, easier to type
df['Temperature_units'] = df[
    'Temperature_units'].str[-1:]

#Changing Farenheit to Degree celcius by slicing 
#if measurement_unit is farenheit, apply formula from src 

#No change for degree celcius, use temerature2
celcius_condition = df['Temperature_units'] == 'C'
df.loc[celcius_condition,'Temperature']= df.Temperature2

#Farenheit, apply farenheit_to_celcius module
farenheit_condition = df['Temperature_units'] == 'F'
df.loc[farenheit_condition,'Temperature']= farenheit_to_celcius(
    df.Temperature2)

#round Temperature column to 1d.p.
df['Temperature'] = df['Temperature'].apply(
    lambda x: round(x,1))

#drop previously split column - Temperature2 and Temperature_unit
df.drop(['Temperature2', 'Temperature_units'], axis=1, inplace=True)


# =============================================================================
# 4. Feature engineering
# 4.1 Seperate model and year into new column
#     Change year to age of car, easier for machine learning
# 4.2 create column to indicate if there is failure
# 4.3 Dimension reduction - remove car ID
# Remove Failure_A, if all other failure type = 0 means A 
# =============================================================================

### 4.1 ###
#Separate year produced from model
df[['Model', 'Year_released']] = df["Model"].str.split(",", expand=True)

#change year released to age of car
df['Year_released'] = df['Year_released'].astype(int)
df["age"] = 2023 - df['Year_released'] 
df = df.drop(columns = 'Year_released')

### 4.2 ###
#failure column list
failure_col = [
    'Failure A', 'Failure B', 'Failure C', 
    'Failure D', 'Failure E'
    ]

#create a column indicating failure (Y/N)
#using module failure_identification from src.py
failure_identification(df, failure_col)

#drop failure type, not needed for ML pipeline
df = df.drop(columns='Failure_type')

### 4.3 ###
#remove car_ID
df = df.drop(columns=['Car ID', 'Failure A'])


# =============================================================================
# 5. Missing Data Handling
# Change missing membership type to undisclose
# not missing at random, all missing have failure 
# =============================================================================

#change missing value in membership to undisclose
df.loc[df[df['Membership'].isnull()].index , 'Membership'] = 'Undisclose' 


# =============================================================================
# 6. Outlier
# remove extreme outlier (230 degree) in Temperature
# RPM cannot be negative, remove since is 224/10081 = 0.02% of data
# =============================================================================

#Remove temperature 230 degree outlier
df = df.drop([4], axis = 0)

#remove RPM cannot be negative
df = df[df["RPM"]>0] 


# =============================================================================
# 7. Machine learning
# 7.1 Split data into training and testing set
# 7.2 Logistic regression
# 7.3 Decision Tree
# 7.4 Complement Naive Bayes
# =============================================================================

### 7.1 ###

#define predictor Y and features X
X = df.drop(['Failure'],axis=1)
y = df['Failure']

#Train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


### columns seperation by type ###

#columns as per type - numerical/categorical 
numerical = ['Temperature', 'RPM', 'Fuel consumption']
#split categorical column for one-hot/ordinal encoding
categorical_hot = ['Color', 'Factory']
categorical_oe = ['Usage', 'Membership', 'Model']

#machine learning pipline

#decision tree
dt = DecisionTreeClassifier(max_depth=4, 
                            min_samples_leaf=0.16, 
                            random_state=42, 
                            class_weight = 'balanced' 
                            )

classifiers = [
    LogisticRegression(class_weight='balanced'),
    dt,
    ComplementNB()
    ]


execute_pipeline(classifiers, Pipeline, 
                 X_train, y_train, 
                 X_test, y_test,
                 numerical,
                 categorical_hot,
                 categorical_oe
                 )














