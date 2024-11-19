# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:54:17 2023

@author: Yi Ting
"""

# =============================================================================
# Import packages
# =============================================================================
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
# =============================================================================
# Function to check file import success
# =============================================================================
def print_test(x):
    print("Src pyfile imported " + x)
    

# =============================================================================
# Data cleaning function
# =============================================================================
# Farenheit to degree celcius formula: (32°F − 32)*(5/9) = 0°C
def farenheit_to_celcius(x):
    return (x-32)*(5/9)      


# =============================================================================
# Feature Engineering function
# =============================================================================


#create a column indicating failure (Y/N); 
#and a column for the type of failure

def failure_identification(data, col):
    data['Failure'] = 0 #create new column and assume all N
    #create new column and assume no failure type
    data['Failure_type'] = "None" 
    #search all failure type and indicate Y if failure and type
    for i in col:
        fail_condition = (data[i] == 1)     
        data.loc[fail_condition, 'Failure'] = 1 #there is failure
        data.loc[fail_condition, 'Failure_type'] = i #failure type 
    return data



# =============================================================================
# Maching learning pipline classes
# =============================================================================

#Min-max normalising scaling for numeical feature
class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
    
    def fit(self, X, y=None):
        self.min = X[self.features].min()
        self.range = X[self.features].max()-self.min
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.features] = (X[self.features]-self.min)/self.range
        return X_transformed
  
#One-hot encoding for categorical feature
class HotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features, drop='first'):
        self.features = features
        self.drop = drop
    
    def fit(self, X, y=None):
        self.encoder = OneHotEncoder(sparse=False, drop=self.drop)
        self.encoder.fit(X[self.features])
        return self
    
    def transform(self, X):
        X_transformed = pd.concat([X.drop(columns=self.features).reset_index(drop=True), 
                                   pd.DataFrame(self.encoder.transform(X[self.features]), 
                                                columns=self.encoder.get_feature_names_out(self.features))],
                                  axis=1)
        return X_transformed

    
#Ordinal encoding for categorical feature    
class OrdEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features, drop='first'):
        self.features = features
        self.drop = drop
    
    def fit(self, X, y=None):
        self.encoder = OrdinalEncoder()
        self.encoder.fit(X[self.features])
        return self
    
    def transform(self, X):
        X_transformed = pd.concat([X.drop(columns=self.features).reset_index(drop=True), 
                                   pd.DataFrame(self.encoder.transform(X[self.features]), 
                                                columns=self.features)],
                                                axis=1)
        return X_transformed
    

# =============================================================================
# Maching learning pipline Execution
# =============================================================================

#Execute and evaluate pipeline

#evaluate functions - find ROC-AUC
def calculate_roc_auc(model_pipe, X, y):
    """Calculate roc auc score. 
    """
    y_proba = model_pipe.predict_proba(X)[:,1]
    return roc_auc_score(y, y_proba)
  
# Use model to predict test sample
def find_pred(model_pipe, X):
    y_result = model_pipe.predict(X)
    return y_result


def execute_pipeline(model, pipeline, X_train, y_train, X_test, y_test, num, cat_hot, cat_oe):
    #Execute scaling, encoding and select classifier
    for classifier in model:
        pipe = Pipeline([
            ('scaler', Scaler(num)),
            ('Hot_encoder', HotEncoder(cat_hot)),
            ('Ord_encoder', OrdEncoder(cat_oe)),
            ('model', classifier)
            ])
    
        #fit the model
        pipe.fit(X_train, y_train)
        
        #Use model to predict test sample
        y_pred = find_pred(pipe, X_test)


        #print evaluation metrics - Accuracy, Recall, F1-score and support
        print(classification_report(y_test, 
                                    y_pred, 
                                    target_names=['no fail', 'fail'],
                                    digits = 4
                                    ))
        #print overall accuracy
        print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
        #print ROC-AUC value
        print(f"Train ROC-AUC: {calculate_roc_auc(pipe, X_train, y_train):.4f}")
        print(f"Test ROC-AUC: {calculate_roc_auc(pipe, X_test, y_test):.4f}")    

# =============================================================================
# Maching learning - evaluation module
# =============================================================================



  