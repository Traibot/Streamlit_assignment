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


   # plot the histogram
   fig = px.histogram(eda_df, x = 'atemp_bins', color = 'atemp_bins', title = 'Histogram of felt temperature')
   st.plotly_chart(fig)




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





   
