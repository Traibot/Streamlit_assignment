import streamlit as st
import os 


#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')


st.set_page_config(layout="wide")
# get the title of the page
st.title('Bike Sharing Demand')
st.markdown(''' We, team 4, have been hired by the city to build an interactive, insightful and complete report on the bike sharing demand in the cityfor the head of transportation services of the local government. 
The report will be used by the city to make decisions on how to improve the bike sharing system. ''')
st.markdown('''As part of the requirements, there are two big points:''')
st.markdown('''1. The city is looking for a deep analysis of the Bike-sharing service, to understand how the citizens are using the service in order to optimize it. ''')
st.markdown('''2. The city is looking for a prediction model that can predict total number of bicycle users on an hourly basis. It is said to to help with optimization of bike provisioning and will optimize the costs incurred from the outsourcing transportation company.''')


#Import data from repository
data = pd.read_csv("https://raw.githubusercontent.com/Traibot/Streamlit_assignment/main/bike-sharing_hourly.csv")

# get two tabs for the page EDA and ML 
tab1, tab2 = st.tabs(["EDA", "ML"])


# EDA tab
with tab1:
   # for any changement in the sidebar selection, the data will be updated
   st.title('PART I: Exploratory Data Analysis')
   # explain the analysis 
   st.markdown('''In this part, we will explore the data to understand the data and the relationship between the variables. We will also try to find some insights that can help us to build a better model. ''')

   # create a filter to select the column you want to see
   st.subheader('Select the column you want to see')
   column = st.multiselect('Column', data.columns)
   st.write(data[column])



# ML tab
with tab2:
   st.title('PART II: Machine Learning')
   # explain the analysis
   st.markdown('''In this part, we will build a model to predict the number of bike users. We will use the data from the previous part to build the model. ''')
   # get the head of the data 
   st.subheader('Head of the data')
   st.write(data.head())
   
   # look at the data types
   st.subheader('Data Types')
   st.write(data.dtypes)
   
   st.subheader("Map variables to definitions")
   st.markdown("To make the data more readable and for better interpretability, we map the categorical variables to the specified definitions")
   st.code('''data['season'] = data['season'].map({1:'spring', 2:'summer', 3:'fall', 4:'winter'})
   data['yr'] = data['yr'].map({0:2011, 1:2012})
   data['mnth'] = data['mnth'].map({1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'})
   data['holiday'] = data['holiday'].map({0:'No', 1:'Yes'})
   data['weekday'] = data['weekday'].map({0:'Sun', 1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu', 5:'Fri', 6:'Sat'})
   data['workingday'] = data['workingday'].map({0:'No', 1:'Yes'})
   data['weathersit'] = data['weathersit'].map({1:'Clear', 2:'Mist', 3:'Light Snow', 4:'Heavy Rain'})''')

   #Bin hr into 4 categories: Late Night/Early Morning, Morning, Afternoon/Evening, Night
   st.subheader("Bin hr into 4 categories: Late Night/Early Morning, Morning, Afternoon/Evening, Night")
   st.markdown("Since hr has 24 unique values, its better to bin this field")
   st.code('''def bin_hr(hr):
                  if hr >= 0 and hr < 6:
                     return 'Late Night/Early Morning'
                  elif hr >= 6 and hr < 12:
                     return 'Morning'
                  elif hr >= 12 and hr < 18:
                     return 'Afternoon/Evening'
                  else:
                     return 'Night'

               data['hr_cat'] = data['hr'].apply(bin_hr)''')

   # get the head of the data
   st.subheader('Head of the data')
   st.write(data.head())


   #Plot distribution of registered and casual users
   st.subheader("Plot distribution of registered and casual users")
   st.markdown("We want to check if there is any pronounced difference between registered and casual users. This will help us decide if we should build separate predictive models for this dataset or just build a single predictive model keeping cnt as the target variable")


   # upload the image
   st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/pic1.png?raw=true") 
   st.markdown(''' The distribution of users is right-skewed. This implies that a transformation might be needed to make the distribution more normal.
   For further clarity, we also check the proportion of casual users vs. registered users''')

   #Pie chart of registered and casual users
   st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/Pic2.png?raw=true") 
   

   st.title("Minimum Viable Model: Linear Regression")
   st.subheader('''To deal with the skewness, we decide to try transforming the target variable cnt
   Using *FunctionTransformer* from *sklearn*, we define a log transform for the target while defining the inverse transform (exponential) function for when we make the predictions''')
   st.code('''data['cnt'] = transformer.transform(data['cnt'].values.reshape(-1,1))
               data['registered'] = transformer.transform(data['registered'].values.reshape(-1,1))
               data['casual'] = transformer.transform(data['casual'].values.reshape(-1,1))''', language='python')

   st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/Pic3.png?raw=true") 


   st.markdown('''We decide to drop a few features carrying irrelevant (to predict total users) information or similar information as other features from the input features set: X
   temp gives the actual temperature of the day but users are more likely to make a decision based on the feeling temperature atemp
   seasonis just a categorised version of mnth and opting for it over the latter could lead our model to lose useful predictive power''')

   st.markdown("We dummy encode the categorical variables")
   st.markdown("- Extract categorical variables")
   st.markdown("- Dummy encode categorical variables")

   st.subheader("Using *Recursive Feature Elimination*, we also decide to extract the most important features for our MVM")

   st.markdown("Recursive Feature Selection ")
   st.code('''NUM_FEATURES = 5
   model = LinearRegression()
   rfe_stand = RFE(model, step=NUM_FEATURES)''', language='python')

   st.markdown('''Std Model Feature Ranking: [1 1 1 4 4 4 2 1 2 4 3 3 1 1 2 1 3 1 1 3 2 3 1 1 1 4 1 1 2]
   Standardized Model Score with selected features is: 0.707041 (0.000000)''')

   st.markdown('''Most important features (RFE): ['yr' 'atemp' 'hum' 'mnth_Jan' 'mnth_Nov' 'mnth_Oct' 'holiday_Yes'
   'weekday_Sat' 'weekday_Sun' 'workingday_Yes' 'weathersit_Heavy Rain'
   'weathersit_Light Snow' 'hr_cat_Late Night/Early Morning'
   'hr_cat_Morning']''')

   st.markdown('''We see that 'yr' 'atemp' 'hum' 'mnth_Jan' 'mnth_Nov' 'mnth_Oct' 'holiday_Yes' 'weekday_Sat' 'weekday_Sun' 'workingday_Yes' 'weathersit_Heavy Rain' 'weathersit_Light Snow' 'hr_cat_Late Night/Early Morning' 'hr_cat_Morning' are selected as the mopst important features''')

   st.markdown('''- We create a separate input set with the most important features''')
   st.markdown('''- Splitting data into train and test with X_imp''')
   st.markdown('''- We use *MinMaxScaler* to scale only the numerical features''')
   st.markdown('''- Fitting the model''')
   st.markdown('''- Plotting the predictions on the test data''')
   # get the pic1.png 
   st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/Pic4.png?raw=true")

   st.markdown('''- Inverse transforming y to calculate MAE''')
   st.markdown('''Train score (MAE):  85.66''')
   st.markdown('''Test score (MAE):  83.395''')





   
