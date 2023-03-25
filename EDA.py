#!/usr/bin/env python
# coding: utf-8

# # Import Packages 

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
import warnings

from scipy import stats
from ipywidgets import interact
from scipy.stats.mstats import winsorize


# # Connect to Data

# In[2]:


# =============================================================================
# 1. Import data
# =============================================================================

file_path = "C:/Users/Yi Ting/Desktop/Studies/Github/Car_Failure/failure.xlsx"
df = pd.read_excel(file_path, header = 0)

#check dataframe
df.head()


# In[3]:


#Identify data types
df.dtypes


# From the display of data and datatypes, temperature can be classified as float but need to remove the unit of measurement.  
# The rest of the data types are correct.

# # Data Cleaning
# 
# Print unique value in each categorical data to check for data consistency.   
# Resuls: inconsistent data entry in "Factory"; missing data in "Membership"
# 

# In[4]:


# =============================================================================
# 3. Data cleaning
# =============================================================================

#3(i) Check for data inconsistency in categorical data - Model, Color, Factory, Usage, Fuel Consumption and Membership

categorical_data = ['Model', 'Color', 'Factory', 'Usage', 'Membership']

def check_data_entry(data, col):
    for i in col:
        print(i)
        print(set(df[i]))

check_data_entry(df,categorical_data)
#result: inconsistent data entry in "Factory", missing data in "Membership"

#Cleaning inconsistent data entry in 'Factory'.
df['Factory'] = df['Factory'].replace({
    'Newton, China':'New York, U.S', 
    'Seng Kang, China':'Shang Hai, China',  
    'Bedok, Germany':'Berlin, Germany'
    })


# In[5]:


print("After cleaning data-inconsistency in Factory column, the unique values are: {}".format(str(set(df.Factory))))


# Check for data inconsistency inconsistency in Temperature column. 
# Result: There are 2 units of measurement in the data- degree celcius and farenheit;  
# need to change Farenheit to degree celcius for data consistency in unit of measurement

# In[6]:


#3(ii) Data inconsistency in Temperature column- change farenheit to degree celcius. Change column to quantity data type. 

print(set(df["Temperature"]))

#Split Temperature from unit of measurement
df[['Temperature2', 'Temperature_units']] = df["Temperature"].str.split(" ", expand=True)

#Remove white space in the split Temperature2 and change type to float 
df['Temperature2'].str.strip()
df['Temperature2'] = df['Temperature2'].astype(float)

#Remove °, easier to type
df['Temperature_units'] = df['Temperature_units'].str[-1:]

# Farenheit to degree celcius formula: (32°F − 32)*(5/9) = 0°C
def farenheit_to_celcius(x):
    return (x-32)*(5/9)

#Changing Farenheit to Degree celcius by slicing: if measurement_unit is farenheit, apply farenheit to celcius formula 
celcius_condition = df['Temperature_units'] == 'C'
df.loc[celcius_condition,'Temperature']= df.Temperature2

farenheit_condition = df['Temperature_units'] == 'F'
df.loc[farenheit_condition,'Temperature']= farenheit_to_celcius(df.Temperature2)

#round Temperature column to 1d.p.
df['Temperature'] = df['Temperature'].apply(lambda x: round(x,1))

#drop previously split column - Temperature2 and Temperature_unit
df.drop(['Temperature2', 'Temperature_units'], axis=1, inplace=True)


# # Feature Engineering
# separate year from model
# create predictor column indicating failure (Y/N)  
# Create a column for type of failure for analysis and plots

# In[7]:


# =============================================================================
# 3. Feature engineering
# =============================================================================

#3(i) Separate year produced from model
df[['Model', 'Year_released']] = df["Model"].str.split(",", expand=True)



# In[8]:


#3(ii) create a predictor column indicating failure (Y/N) and a column for the type of failure for analysis and plots 

failure_col = ['Failure A', 'Failure B', 'Failure C', 'Failure D', 'Failure E']

def failure_identification(data, col):
    data['Failure'] = 'N' #create new column and assume all N
    #create new column and assume no failure type
    data['Failure_type'] = "None" 
    #search all failure type and indicate Y if failure and type
    for i in col:
        fail_condition = (data[i] == 1)     
        data.loc[fail_condition, 'Failure'] = "Y" #there is failure
        data.loc[fail_condition, 'Failure_type'] = i #failure type 
    return data


failure_identification(df, failure_col)




# # Missing data 
# Check if missing data is random against our primary variable of interest - Failure Y/N
# Result:   
# Data seems to be not missing at random. Undisclosed membership type all resulted in failure.  
# Filling the missing value with mode would likely skew the category type towards failure - Y   
# Thus, recommend to keep the missing data as a type by itself - undisclose  

# In[9]:


# =============================================================================
# 4. Missing data
# =============================================================================

df.isna().sum() #384 missing data in membership


# In[10]:


#Check if missing data is random against our primary variable of interest - Failure Y/N
#Create a new column to indicate if it is missing
df['Missing'] = 0
df.loc[df[df['Membership'].isnull()].index, 'Missing'] = 1
#check number of missing data against Failure
df[df['Missing']==1].groupby('Failure')['Missing'].count()


#change missing data as undisclose
df.loc[df[df['Membership'].isnull()].index , 'Membership'] = 'Undisclose' 


# # Outlier
# 
# Based on the result, there seem to be outlier in Temperature and RPM  from the mean and max/min value comparison.  
# From the boxplot, there is an extreme outlier in Temperature. Since it is only 1 row, it would not affect whole data and will be removed.   
# RPM cannot be negative, it would be remove since is 224/10081 = 0.02% of the data  
# 

# In[11]:


# =============================================================================
# 5. Outlier
# =============================================================================
df.describe()
#Based on the result, there seem to be outlier in Temperature and RPM  from the mean and max/min value comparison
#From the boxplot, there is an extreme outlier in Temperature. Since it is only 1 row, it would not affect whole data
#Thus, would be removed
#RPM cannot be negative, it would be remove since is 224/10081 = 0.02% of the data


# In[12]:


#function to detect outlier based on IQR
def find_outliers_IQR(df):
   q1 = df.quantile(0.25)
   q3 = df.quantile(0.75)
   IQR = q3-q1
   outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
   return outliers

#find number of outlier: temperature -930
continuous_col = ['Temperature', 'RPM', 'Fuel consumption']

#function to count number of outlier based on IQR

def outlier_count(data, col):
    for i in col:
        outliers = find_outliers_IQR(data[i])
        print('number of outliers in ' + str(i) + ":" + str(len(outliers)) )
       
  

#Result- Temperature: 930; RPM: 636; Fuel consumption: 578 
outlier_count(df, continuous_col)



# # Outlier - Distribution plot, histogram and probability plot
# 
# The results show a large number of outliers and skewed data
# Handling of outlier should be done separtely on train dataset to not affect prediction on real data
# 
# 

# In[13]:


#Graphical view of outlier in distribution plot, histogram and probability plot

for col in continuous_col:
    warnings.simplefilter('ignore') #hide distplot function suggestion warning
    plt.figure(figsize=(15,4))
    plt.subplot(131)
    sns.distplot(df[col], label="skew: " + str(np.round(df[col].skew(),2)))
    plt.legend()
    plt.subplot(132)
    sns.boxplot(df[col])
    plt.subplot(133)
    stats.probplot(df[col], plot=plt)
    plt.tight_layout()
    plt.show()
    


    


# # Heatmap - Check for correlation
# 
# result: low correlation between variables, no issue of multicollinearity

# In[14]:


#plot heatmat to see if there are correlation

corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)


# In[15]:


#Remove temperature 230 degree outlier
df = df.drop([4], axis = 0)

#remove RPM cannot be negative
df = df[df["RPM"]>0] 


# # Bar plot - Count failure (Y/N)
# 
# Barplot shows data imbalance - require balance dataset if applying machine learning. 

# In[16]:


# =============================================================================
# 4. imbalanced data 
# =============================================================================


sns.countplot(data = df, x = 'Failure')


# # Pairplot 
# 
# There are no relationship between the continuous variable.

# In[17]:


# =============================================================================
# 5. Exploratory Data Analysis (EDA)
# =============================================================================

col = ['Temperature', 'RPM', 'Fuel consumption']
sns.pairplot(df[col])

plt.show()


# # Interactive count plots
# 
# Insights:
# 
# 1. As new models come out, there are lesser data count 
# 2. White is the most popular colour for all three models
# 3. Majority of the models are produced in Shang Hai, China
# 
# 4. Is there a relationship between failure and high usage? 
# selecting column: Failure, hue: usage. It shows that no failure occur for all usage type proportionately.
# But high usage did has a higher proportion of failure 
# 
# 5. Failure type C occur the least while failure type B occur the most
# 
# 6. Having premium membership does not lower failure 
# possible reason: people are more careless with more repair service avaiable
# 
# 

# In[18]:


# =============================================================================
# 6. Interactive plots
# =============================================================================

categorical_col = [column for column in df.columns if df[column].dtypes == "object"]
categorical_col.remove('Car ID')

#column
dd1 = widgets.Dropdown(
    options=categorical_col,
    value=categorical_col[0],
    description='Select column'
    )

#hue
dd2 = widgets.Dropdown(
    options=categorical_col,
    value=categorical_col[0],
    description='Hue'
    )
ui = widgets.HBox([dd1,dd2])

def plot_countplot(column, hue):
    p = sns.countplot(data=df, x=column, hue=hue)
    if len(df[column].unique()) > 4: 
        p.tick_params(axis='x', rotation=90) #rotate for readability
    for i in p.patches:
        p.annotate('{:^.0f}'.format(i.get_height()), (i.get_x(), i.get_height()))

out = widgets.interactive_output(plot_countplot, {'column': dd1, 'hue': dd2})

display(ui, out)
plt.show()

