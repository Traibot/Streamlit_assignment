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
import plotly.express as px

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

   st.markdown('''Next, we do some quick data quality check on the variables, verifying that:
* All data types are accurate for the fields
* There are no obvious outliers or erroneous data in the fields
* There are no nulls present in the entire dataset''')

   # look at the data types
   st.subheader('Data Types')
   st.write(data.dtypes)

   # get the data describe 
   st.subheader('Data Describe')
   st.write(data.describe())

   # get the data shape
   st.subheader('Data Shape')
   st.write(data.shape)

   # get the data null
   st.subheader('Data Null')
   st.write(data.isnull().sum())

   st.markdown('''Now we create a copy dataframe to obtain insights. Given the dataset and the questions that the administration of Washington D.C. has for us, we ran the following analysis to better understand customer usage. This includes understanding which conditions favor more participation and some ideas that could benefit potential marketing on behalf of the city.''')

   st.markdown('''# Features to study:
   * How many people use the service varying the atemp
   * Casual vs Registered varying by month (maybe some marketing analysis can be done here?)
   * Humidity vs usage (weather permitting)
   * Month with most 'ideal' days as established by a metric calculated (spin this as something to market a public bike race or something)
   * Histogram with most users per hour.
   * Weekday vs cnt (box plots, one per dow)
   * Cnt vs weather type in box plots''')

   st.subheader("Insight 1: Usage of service vs variation in feeling temperature")
   st.markdown('''First, we want to understand which conditions are more favorable for our users. This way we can understand what patterns might lead to maximum usage, as well as better forecasting of client surges in the event that we want to be mindful of our supply. In this case, we are looking for which (felt) temperatures tend to bring in more clients. We bin all felt temperatures into groups of 5 degrees (After denormalizing to use known measurements), then build a histogram to see what the curve is. Apparently, the most preferred temperature of our users is between 31 and 35 degrees Celsius to use the bike service.''')

   eda_df = data.copy(deep = True)

   # First, we denormalize the variable (assuming minimum of 0C° and maximum of 50C°)
   eda_df['atemp_denorm'] = [round(i*50) for i in eda_df['atemp']]

   # Then, we bin the felt temperatures
   eda_df['atemp_bins'] = pd.cut(x=eda_df['atemp_denorm'], bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                                 labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50'])


   # We continue by plotting how (perceived) temperature affects user count.
   atemp_data = eda_df[['cnt', 'atemp_bins']]
   atemp_data_hist = px.histogram(atemp_data, x='atemp_bins', y='cnt', category_orders=dict(atemp_bins=['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50']))
   atemp_data_hist
   
   st.subheader("Insight 2: Casual vs Registered users by month")

   st.markdown('''Up next, we compare usage month-over-month of our users, and we split it between those who use our system casually and those who are registered with us. This give us two insights:
   * There is a much larger proportion of registered users as opposed to casual ones
   * During high seasons there seems to be more registered users than casual ones from average.

   With user registration, we can better provide our services by being able to anonymously track each one across journeys. This would allow us to understand usage patterns better. Not to mention we can launch a marketing initiative to try to incentivize casual users to join the registry during those high season months, as they seem more prone to do so.''')

   # First, we create a month field to section by first of month (for both years separately):
   eda_df['month_date'] = [pd.to_datetime(str(i)[:8]+'01') for i in eda_df['dteday']]
   cas_reg = eda_df[['month_date', 'casual', 'registered']]

   # We add up the values per month
   cas_reg = cas_reg.groupby('month_date', as_index=False).sum()

   # We do a line plot to compare both behaviors over time.
   cas_reg_plot = px.line(cas_reg, x='month_date', y=['casual', 'registered'])
   cas_reg_plot
   

   st.subheader("Insight 3: Effect of humidity on everyday usage")
   st.markdown('''Is humidity a factor in usage? Do our customers think about this before getting on one of our bikes?
   With a correlation coefficient of -.09, the points out that no, the humidity of a given day is not a contributing factor to using our services. ''')

   # We filter all weather conditions that are logically less than ideal
   hum_use = eda_df[eda_df.weathersit <= 2]
   hum_use = hum_use[hum_use.season <=2]

   # We aggregate by day to obtain average humidity and sum of users for each day
   hum_use = hum_use.groupby('dteday').agg(cnt=('cnt', np.sum), hum=('hum', np.mean))

   # We run a correlation matrix on humidity and usage
   hum_use = hum_use[['cnt', 'hum']]
   hum_use.corr()

   # get the correlation matrix on humidity and usage 
   st.subheader('Correlation Matrix')
   # display it as a heatmap
   st.write(hum_use.corr())


   st.subheader("Insight 4: Which month has more 'ideal' days")

   st.markdown('''In order to capture more attention of the general public, we came up with the idea of holding public events to incentivize use of our platform and bikes. a 10k bikeathon would likely be a hit with our users, we believe. The issue with this is that we want to maximize the number of participants that day, and the best way to do so is to setting up an event on a day whose weather is ideal for bikers to join. Since we can't predict the exact weather of a day too much in advance, we identified a metric to establish what a 'good day' is, then count these throughout the years to see which month has the higher probability of giving us a 'good day' for a race.
   The metrics to count a day as good are:
   * Weather is clear, a little mist allowed
   * Felt temperature is between 25 and 35 Celcius, as per our past insight
   * Wind speed is under 25
   * It is not a working day

   Based on our findings, we conclude that the best months for an outdoor event to gather clients would be between June and July. However, data also points to September being acceptable if need be.''')

   # First we denormalize wind speed to use it with its normal metric.
   eda_df['windspeed_denorm'] = [round(i*67) for i in eda_df['windspeed']]

   # Next, we obtain the features we need to qualify a day as 'good', with an average per day.
   best_mo = eda_df.groupby(['dteday', 'month_date'], as_index=False).mean()
   best_mo = best_mo[['dteday', 'mnth', 'workingday', 'weathersit', 'atemp_denorm', 'windspeed_denorm']]

   # We define a function that checks if a given day in the dataset meets the criteria
   def good_day(best_mo):
      if ((best_mo['workingday'] < 0.1) and
      (best_mo['weathersit'] < 2.0) and
      (best_mo['atemp_denorm'] >= 25.0) and 
      (best_mo['atemp_denorm'] <= 35.0) and
      (best_mo['windspeed_denorm'] <= 25.0)):
         return 1
      else:
         return 0
   st.code('''def good_day(best_mo):
      if ((best_mo['workingday'] < 0.1) and
      (best_mo['weathersit'] < 2.0) and
      (best_mo['atemp_denorm'] >= 25.0) and 
      (best_mo['atemp_denorm'] <= 35.0) and
      (best_mo['windspeed_denorm'] <= 25.0)):
         return 1
      else:
         return 0''', language='python')
   # We apply the formula and obtain the aggregate of good days by month
   best_mo['good_day'] = best_mo.apply(good_day, axis=1)
   best_mo = best_mo.groupby('mnth', as_index=False).sum()
   best_mo['mnth'] = best_mo['mnth'].astype(int)

   # Lastly, we plot the graph
   best_mo_hist = px.bar(best_mo, x='mnth', y='good_day')
   best_mo_hist

   st.subheader("Insight 5: Users per hour")
   st.markdown('''By building a histogram that plots users by hour, we can see a clear bimodal curve. This shows that most users come to our services around 8 in the morning and around 5-6 in the afternoon. This makes perfect sense considering that those are the rush hour times. Perhaps our clients want to avoid car traffic, or they believe this is to be a healthier or greener alternative to driving. Either way, with this information at hand we can likely come up with some marketing scheme, where we give a subscription to users in exchange to reduced rates at peak times or something of the matter. ''')
   
   hour_users_hist = px.histogram(eda_df, x='hr', y='cnt')
   hour_users_hist

   st.subheader("Insight 6: Day of Week vs usage")
   st.markdown('''We wanted to better understand if a given day of week had more general use than another. For context, are our clients using our services more during leisure on the weekends, or is the service more used to commute to work? Turns out this is a little inconclusive, as the behavior between days doesn't vary by a large enough amount to be able to claim so. There seems to be some grater variance in use on the weekends; however, the means are close enough for us to be able to say that there is no discernable pattern across days of week.''')
   
   # First, we filter to only evaluate the Summer, which is our most active season.
   dow_use = eda_df[eda_df['season']==2]

   # We aggregate usage by day of week
   dow_use = dow_use.groupby(['dteday', 'weekday'], as_index=False).sum()

   # We display the box plot
   dow_use_box = px.box(dow_use, x='weekday', y='cnt')
   dow_use_box

   st.subheader("Insight 7: Weather type vs usage")
   st.markdown('''While this makes logical sense, we wanted to see by what amounts are our customers stopping using our services as weather gets progressively worse. As evidenced in the data in our graph below, there is very little participation when the weather is in a bad shape. However, it is interesting to note that misty days have a definitively smaller amount of customers than one with a fully cleared day. Mist doesn't exactly affect the biking experience, so perhaps this is psychological behavior. Maybe it would be an interesting proposition to study offering discounts in misty days so we can incentivize use instead of seeing the potential go to waste.''')

   # First we aggregate the count of users by day and weather conditions to see how they stack against each other
   weather_use = eda_df.groupby(['dteday', 'weathersit'], as_index=False).sum()

   # Lastly we plot the box plot.
   weather_use_box = px.box(weather_use, x='weathersit', y='cnt')
   weather_use_box









# ML tab
with tab2:
   st.title('PART II: Machine Learning')
   # explain the analysis
   st.markdown('''In this part, we will build a model to predict the number of bike users. We will use the data from the previous part to build the model. ''')
   # get the head of the data 
   st.subheader('Head of the data')
   
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
   st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/ML1.png?raw=true") 
   st.markdown(''' The distribution of users is right-skewed. This implies that a transformation might be needed to make the distribution more normal.
   For further clarity, we also check the proportion of casual users vs. registered users''')

   #Pie chart of registered and casual users
   st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/ML2.png?raw=true") 

   
   
   st.title("Deciding between temp and atemp, mnth and season")
   data2 = data.copy()
   data2 = data2.drop(['instant', 'dteday', 'hr'], axis = 1)
   cat_var = data2.select_dtypes(include = ['object']).columns
   data2 = pd.get_dummies(data2, columns = cat_var, drop_first = False)
   data3=data2.corr()['casual'].abs().sort_values(ascending = False)
   
   # get the data next to each other
   col1, col2 = st.columns(2)
   with col1:
      st.subheader("Correlation between casual users and other variables")
      st.write(data3)
   with col2:
      st.subheader("Correlation between registered users and other variables")
      data4=data2.corr()['registered'].abs().sort_values(ascending = False)
      st.write(data4)
   

   st.title("Minimum Viable Model: Random Forest Regression")
   st.markdown('''To deal with the skewness, we decide to try transforming the target variables casual and registered
 Using *FunctionTransformer* from *sklearn*, we define a log transform for the target while defining the inverse transform (exponential) function for when we make the predictions''')
   st.markdown('''Define FunctionTransformer from sklearn.preprocessing to transform data''')

   st.markdown('''We decide to drop a few features carrying irrelevant (to predict total users) information or similar information as other features from the input features set: X
   temp gives the actual temperature of the day and that has a bigger influence than atemp
   season is just a categorised version of mnth and opting for it could help our model perform better
   All this is verified by the above correlation analysis''')

   st.markdown("Create two X and y sets: X_cas and y_cas for casual users and X_reg and y_reg for registered users")
   st.code('''X_cas = data.drop(['instant', 'cnt', 'casual', 'registered', 'dteday', 'hr', 'atemp', 'mnth'], axis = 1)
   y_cas = data[['casual']]

   X_reg = data.drop(['instant', 'cnt', 'casual', 'registered', 'dteday', 'hr', 'atemp', 'mnth'], axis = 1)
   y_reg = data[['registered']]''', language='python')

   st.markdown("We dummy encode the categorical variables")

   st.markdown("Using *Recursive Feature Elimination*, we also decide to extract the most important features for our MVM")

   st.markdown("Recursive Feature Selection ")
   st.code('''NUM_FEATURES = 5
   model = RandomForestRegressor()
   rfe_stand = RFE(model, step=NUM_FEATURES)''', language='python')

   st.markdown('''
   * Cas: Std Model Feature Ranking: [1 1 1 1 2 4 1 4 3 4 4 3 3 3 1 4 3 1 1 1 1]
   * Reg: Std Model Feature Ranking: [1 1 1 1 2 4 1 4 3 4 4 3 3 3 1 4 3 1 1 1 1]
   * Cas: Standardized Model Score with selected features is: -10.168658 (0.000000)
   * Reg: Standardized Model Score with selected features is: 0.903461 (0.000000)''')

   st.markdown('''
   Cas: Most important features (RFE): ['yr' 'temp' 'hum' 'windspeed' 'season_winter' 'workingday_Yes' 'weathersit_Mist' 'hr_cat_Late Night/Early Morning' 'hr_cat_Morning' 'hr_cat_Night']
 
   Reg: Most important features (RFE): ['yr' 'temp' 'hum' 'windspeed' 'season_winter' 'workingday_Yes' 'weathersit_Mist' 'hr_cat_Late Night/Early Morning' 'hr_cat_Morning' 'hr_cat_Night']''')

   st.subheader('''
   Splitting data into train and test with X_imp''')

   st.markdown('''- Split data into train and test
   - ((13903, 10), (3476, 10), (13903, 1), (3476, 1))''')
   st.markdown('''
   - We use *MinMaxScaler* to scale only the numerical features
   - Running the model only with important features will help the runtime as this is just to set a baseline
   - Target transforming the target variables
   - Log transform y''')

   st.header("Plot log-transformed distribution of casual users")
   st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/ML3.png?raw=true") 

   st.header("Plot log-transformed distribution of registered users")
   st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/ML4.png?raw=true")

   st.markdown('''
   - Fitting the model''')

   st.markdown('''- Define Random Forest Regressor
   - RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=None, max_features='auto', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=100, n_jobs=None, oob_score=False,
                      random_state=42, verbose=0, warm_start=False)''')
   st.markdown('''- Inverse transforming y to calculate MAE''')
   st.markdown('''- Calculate MAE for casual users
   - Test score (MAE):  12.615''')
   st.markdown('''- Calculate MAE for registered users
   - Test score (MAE):  64.006''')

   st.markdown('''- Calculate MAE for total users
   - Calculate MAE for total users''')

   st.subheader("MVM: Hyperparameter tuning")
   st.markdown('''{'max_depth': 11, 'max_features': 'sqrt'}
   {'max_depth': 11, 'max_features': 'sqrt'}''')

   st.markdown('''- Define Random Forest Regressor with best parameters
   - RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=11, max_features='sqrt', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=100, n_jobs=None, oob_score=False,
                      random_state=42, verbose=0, warm_start=False)''')
   st.markdown('''- Calculate MAE for casual users
   - Test score (MAE):  13.393''')

   st.markdown('''- Calculate MAE for registered users
   - Test score (MAE):  63.199''')

   st.markdown('''- Calculate MAE for total users
   - Test score (MAE):  222.381''')

   st.header("Plot predicted vs actual")
   st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/ML5.png?raw=true")

   st.subheader("From the plot, we can see that the model is performing well for casual users when the actual values are greater than 2")
   st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/ML6.png?raw=true")


   # get this in two columns
   st.subheader("The same goes for registered users")
   col1, col2 = st.columns(2)
   with col1:
      st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/ML7.png?raw=true")

   with col2:
      st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/ML8.png?raw=true")

   st.markdown('''On comparing the two feature importance plots, we can see that the most important features for casual users are temp, humidity, and windspeed. For registered users, this feature importance is less pronounced. We can thus conclude that casual users seem to be taking more rides depending on the weather conditions, while registered users are more likely to take rides irrespective of the weather conditions.''')

   st.header("PyCaret: Predict *casual* and *registered* separately")
   st.markdown('''PyCaret is an easier and faster way to build machine learning models and will allow us to compare the performance of different models, keeping the observations from the MVM models in mind. We will use PyCaret to predict casual and registered users separately.''')

   st.markdown('''We will be testing our predictive models on the same test dataset in order to compare the results.
   We will also try to build a predictive model only for <em>cnt</em> and compare the results.''')

   st.code('''data = df.sample(frac=0.9, random_state=786)
   data_unseen = df.drop(data.index)

   data.reset_index(drop=True, inplace=True)
   data_unseen.reset_index(drop=True, inplace=True)

   print('Data for Modeling: ' + str(data.shape))
   print('Unseen Data For Predictions ' + str(data_unseen.shape))''', language='python')

   st.markdown('''Data for Modeling: (15641, 18)
   Unseen Data For Predictions (1738, 18)''')

   st.subheader("Casual users")
   st.markdown('''We do a similar test-train split as the MVM
    
   2. We remove outliers at 5%
      
   3. We ignore the same features as before
      
   4. Feature normalization and transformation is set to True
      
   5. Feature selections takes too long so we disable it
      
   6. We set thresholds for multicollinearity''')

   st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/ML9.png?raw=true")

   # get two columns
   st.subheader("Create")
   col1, col2 = st.columns(2)
   with col1:
      st.code('''model_cas = create_model('catboost', fold = 10, round = 2)''', language='python')
      st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/Catboost.png?raw=true")
   
   with col2:
      st.code('''model_cas_2 = create_model('rf', fold = 10, round = 2)''', language='python')
      st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/rf.png?raw=true")

   col1, col2 = st.columns(2)
   with col1:
      st.subheader("Tune")
      st.code('''tuned_cas = tune_model(model_cas, optimize = 'MAE', fold = 10, choose_better = True)''', language='python')
      st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/Tune.png?raw=true")
   
   with col2:
      st.subheader("Bagging")
      st.code('''bagged_cas = ensemble_model(model_cas, fold = 10, choose_better = True)''', language='python')
      st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/Bagging.png?raw=true")

   st.subheader("Stacking")
   col1, col2 = st.columns(2)
   with col1:
      st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/ML10.png?raw=true")

   with col2:
      st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/ML11.png?raw=true")

   st.markdown('''Predict on unseen data''')
   st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/Predict.png?raw=true")
   
   st.subheader("Registered users")
   st.markdown('''
   - Create
   - Tune
   - Bagging
   - Stacking''')

   # get two columns
   col1, col2 = st.columns(2)
   with col1:
      st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/ML12.png?raw=true")
   with col2:
      st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/ML14.png?raw=true")

   st.markdown('''Mean absolute error:  70.52''')

   st.markdown('''Splitting up the regression target to predict registered and casual users separately might make sense business-wise, depending on what the objective of city officials is. The MAE for registered users is worse than that of casual users and that can be attributed to the volume of registered users vs. casual users.
   However, this format of prediction massively increases the MAE of **total users** and thus we attempt to build a predictive model only for <em>cnt</em> as target to see if we can bring down the MAE and improve other metrics.''')

   st.header("PyCaret: Predict *cnt*")

   st.markdown('''
   - Create
   - Tune
   - Bagging
   - Stacking''')

   col1, col2 = st.columns(2)
   with col1:
      st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/ML15.png?raw=true")
   with col2:
      st.image("https://github.com/Traibot/Streamlit_assignment/blob/main/ML17.png?raw=true")


   st.title("Conclusion")
   st.markdown('''While the MAE is for predicting total users is only slightly worse than using a different predictive model for registered and casual users, this could keep flipping between the two approaches. However, the recommendation is to run two separate models as different factors influence registered users and casual users to bike in Washington D.C. 
   More information could certainly help the model, like costs, usage of other modes of transport in the city, location of bike stations, etc.''')


   








   
