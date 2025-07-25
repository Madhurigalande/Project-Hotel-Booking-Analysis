


import numpy as np
from numpy import random
import pandas as pd
import os
from numpy.linalg import inv
import scipy
from scipy import stats
from scipy.stats import skew,kurtosis
import matplotlib
from matplotlib import pyplot as  plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import binom
pd.set_option('Display.max_columns',None)
from scipy.stats import expon
from scipy.stats import norm
from scipy.stats import t
import statsmodels
from statsmodels import stats
from statsmodels.stats import weightstats as ssw
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp


# # Data cleaning

# In[2]:


os.chdir(r'C:\\Users\\MADHURI\\Desktop\\CDAC\\Statistic_Sudip sir\\Program_S')


# ## Load Data

# In[3]:


df=pd.read_csv("hotel_bookings.csv")


# In[4]:


df.head()


# In[5]:


# View number of rows and columns,Understand data types of each column,Check non-null (non-missing) counts,memory usage
df.info()


# In[8]:


#Number of rows and columns
df.shape 


# In[9]:


#Checking null values


# In[10]:


df.isnull().sum()


# ## Check Unique values

# In[11]:


# check unique value in hotel column
df['hotel'].unique()


# In[14]:


#check unique column in arrival_date_year 
df['arrival_date_year'].unique()


# In[15]:


#check unique column is_canceled 
df['is_canceled'].unique()


# In[16]:


#check unique column in meal
df['meal'].unique()


# In[17]:


#check unique value in distribution_channel
df['distribution_channel'].unique()


# In[ ]:


Data descrption and solution:-
Four column have null values 
children-4 ,country-488,agent-16340,company-112593
if their are missing values more than 20% then remove that
1.if column data type is string then to fill null values by mode.
          # most frequent value
2.if column data type is numerical then fill null values by median   (df.column_name.mod)
     (df.column_name.median) Median is robust to outliers. i.e Median stays stable, even if extreme values are added
    #when outliers are present, because it ignores extreme values.


# ## Handle Missing Values

# In[ ]:


#Handling Missing values


# In[12]:


df=df.drop('company',axis=1)


# In[13]:


# To fill null values calculate median because this are numerical values so we have used median
df.agent.median()


# In[14]:


df['agent']=df.agent.fillna(14.0)


# In[15]:


df.head()


# In[16]:


df.shape


# In[17]:


# to fill null values calculate mode because this are string values so we have used mode and fill
df.country.mode()


# In[18]:


df['country']=df.country.fillna('PRT')


# In[19]:


df.head()


# ## Change  data type of column

# In[20]:


# convert float into interger and fill 0 at null values 
df['children'] = df['children'].fillna(0).astype(int)


# In[23]:


#Change Data Type
df['hotel'] = df.hotel.astype('string')
df['meal']=df.meal.astype('string')
df['country'] =df.country.astype('string')
df['market_segment']= df.market_segment.astype('string')
df['distribution_channel']=df.distribution_channel.astype('string')
df['reserved_room_type']=df.reserved_room_type.astype('string')                       
df['assigned_room_type']=df.assigned_room_type.astype('string')    
df['deposit_type']=df.deposit_type.astype('string')     
df['customer_type']=df.customer_type.astype('string')
df['reservation_status']=df.reservation_status.astype('string')


# In[21]:


df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])


# In[22]:


df['Date'] = pd.to_datetime(
    df['arrival_date_year'].astype(str) + '-' +
    df['arrival_date_month'].astype(str) + '-' +
    df['arrival_date_day_of_month'].astype(str),
    
)


# In[25]:


df = df.drop(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'], axis=1)
#optional


# In[24]:


df.Date


# In[25]:


# convert float into interger and fill 0 at null values 
df['children'] = df['children'].fillna(0).astype(int)


# In[26]:


df.shape


# ## create derived column

# #### Exploratory Data Analysis

# In[ ]:


# create derived columns


# In[27]:


# Create 'total_guests' column
df['total_guests'] = df['adults'] + df['children'] + df['babies']


# In[28]:


df['Total_Night']=df['stays_in_weekend_nights']+df['stays_in_week_nights']


# In[29]:


df.head()


# ## Handle duplicate value

# In[ ]:


#Handling Duplicate Values


# In[30]:


df.duplicated().sum()


# In[31]:


df = df.drop_duplicates()


# In[32]:


df.shape


# A boxplot helps you visualize the distribution of a numerical column (like adr) and identify:
# Median (middle line)-->Central value of adr
# Box (IQR)--->Spread of the middle 50% of data
# Whisker--Range of most of the data (not outliers)
# Outliers-->Unusually high or low adr values
# To check outliers we have plot boxplot

# In[33]:


#Insight: Revenue per night distribution and pricing outliers.
plt.figure(figsize=(8, 5))
sns.boxplot(y=df['adr'])
plt.title('Boxplot of ADR Average Daily Rate')
plt.xlabel('ADR')
plt.show()


# In[ ]:


Most daily rates are around ₹100
There is low variation for most guests
Extreme outliers exist above  ₹5000 — review needed
The data is right-skewed, with some high-paying bookings:-Most prices are on the lower end, but a few are very high.
most bookings are normal and consistent, but due to some unusal prices impact on analysis


# In[34]:


def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]


# In[35]:


df = remove_outliers_iqr(df, 'adr')


# In[36]:


sns.boxplot(y=df['adr'])


# In[37]:


df.describe()


# In[39]:


plt.figure(figsize=(8, 5))
sns.boxplot(y=df['lead_time'])
plt.title('Boxplot of lead_time')
plt.xlabel('lead_time')
plt.show()


# In[ ]:


#Outliers represent bookings with unusually long or short lead times.
"The boxplot of lead_time reveals several outliers with values exceeding 700 days."


# In[ ]:


The lead_time variable exhibits some outliers above 700 days, indicating a few guests book their stays more than 1.5 years in advance. 
While these outliers are relatively rare, they highlight the presence of long-term planners in the customer base. 
For most bookings, lead times are considerably shorter, suggesting that operational and marketing efforts should primarily
focus on typical booking windows. However, these extreme values could impact average lead time calculations and should be 
considered in deeper analyses."


# # 1. Univariate Analysis

# ### adr(Average Daily Rate)

# In[40]:


plt.figure(figsize=(8, 5))
sns.boxplot(y=df['adr'])
plt.title('ADR Distribution')
plt.xlabel('ADR')
plt.show()


# In[ ]:


#Most values fall between 50–150.
#The ADR distribution is normally distributed with a majority of bookings having ADR between 50 and 150


# In[41]:


# We use a histogram with KDE (Kernel Density Estimate) to visualize the distribution of the Average Daily Rate (ADR). It helps us understand the spread, skewness, and possible outliers in the pricing strategy of the hotel.
plt.figure(figsize=(8,5))
sns.histplot(df['adr'], kde=True)
plt.title("Distribution of ADR")
plt.show()


# ### lead_time Distribution

# In[56]:


plt.figure(figsize=(8, 5))
sns.histplot(df['lead_time'],kde=True, bins=40)
plt.title('Lead Time Distribution')
plt.xlabel('Lead Time (days)')
plt.ylabel('Count')
plt.show()


# In[ ]:


# Insights:-Most bookings are made within 0–100 days


# # 2.Bivariate analaysis

# In[68]:


#Insight: Understand cancellation patterns.
sns.countplot(data=df, x='is_canceled')
plt.title('Booking Cancellation Distribution')
plt.xticks([0,1], ['Not Canceled', 'Canceled'])
plt.xlabel('Booking Status')
plt.ylabel('Count')
plt.show()


# In[69]:


plt.figure(figsize=(10, 6))
sns.barplot(x='market_segment', y='adr',errorbar=None, data=df)
plt.title('ADR by Market Segment')
plt.xlabel('Market Segment')plt.ylabel('Average Daily Rate')
plt.xticks(rotation=45)
plt.show()


# In[63]:


# early booking cancelation status
sns.boxplot(data=df, x='is_canceled', y='lead_time')
plt.title('Lead Time by Cancellation Status')
plt.xticks([0,1], ['Not Canceled', 'Canceled'])
plt.show()


# #### Guest demographics and distribution by country

# In[64]:


#segment brings more business
sns.countplot(data=df, x='market_segment', order=df['market_segment'].value_counts().index)
plt.title('Bookings by Market Segment')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.show()


# In[65]:


top_countries = df['country'].value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_countries.index, y=top_countries.values)
plt.title('Top 10 Guest Countries')
plt.xlabel('Country')
plt.ylabel('Number of Guests')
plt.show()


# In[ ]:


#Market segment share and ADR (Average Daily Rate) comparison. Booking lead time distribution across customer types


# In[67]:


#. Booking lead time distribution across customer types
plt.figure(figsize=(8, 5))
sns.boxplot(x='customer_type', y='lead_time', data=df)
plt.title('Lead Time by Customer Type')
plt.xlabel('Customer Type')
plt.ylabel('Lead Time')
plt.show()


# In[79]:


room_counts = df['assigned_room_type'].value_counts()

# Plotting
plt.figure(figsize=(10, 6))
room_counts.plot(kind='bar', color='blue')
plt.title('Assigned Room Types Distribution')
plt.xlabel('Room Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:


high demand for A,D room types


# In[84]:


sns.countplot(data=df, x='meal', order=df['meal'].value_counts().index)
plt.title('Meal types')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.show()


# In[ ]:


from above graph BB(bed and breakfast) is most prefered type of meal by guest
full board are least preferred


# In[66]:


#Insight: Seasonality or peak/off-peak months.
order = ['January', 'February', 'March', 'April', 'May', 'June',
         'July', 'August', 'September', 'October', 'November', 'December']

sns.countplot(data=df, x='arrival_date_month', order=order)
plt.title('Bookings by Month')
plt.xticks(rotation=45)
plt.ylabel('Number of Bookings')
plt.show()


# In[85]:


sns.countplot(data=df, x='hotel', order=df['hotel'].value_counts().index)
plt.title('Bookings of hotels')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.show()


# In[92]:


sns.barplot(data=df, x='hotel', y=df['adr'],errorbar=None)
plt.title('Bookings hotels and average daily rate')
plt.xticks(rotation=45)
plt.ylabel('average daily rate')
plt.show()


# In[ ]:


If City Hotel shows a higher mean ADR than Resort Hotel:
It indicates City Hotel charges more on average—likely due to its urban location


# In[96]:


sns.barplot(data=df, x='hotel', y=df['Total_Night'],errorbar=None)
plt.title('Bookings hotels in Night')
plt.xticks(rotation=45)
plt.ylabel('average daily rate')
plt.show()


# In[ ]:


From above graph it is predicted average daily rate to stay in hotel for night in resort is greater than city hotel


# In[42]:


# Time-Series Analysis of Booking Trends (Monthly Count)
# Reason: Understanding seasonality helps optimize staffing and pricing.
df['arrival_month'] = pd.to_datetime(df['arrival_date_month'] + ' ' + df['arrival_date_year'].astype(str))
df['arrival_month'] = df['arrival_month'].dt.to_period('M').dt.to_timestamp()
monthly_bookings = df.groupby('arrival_month').size()
monthly_bookings.plot(figsize=(12,6), marker='o')
plt.title("Monthly Booking Trends")
plt.xlabel("Arrival Month")
plt.ylabel("Number of Bookings")


# ##### Time-series analysis of booking trends

# In[52]:


ts = df.groupby('reservation_status_date').size()
plt.figure(figsize=(12, 6))
ts.plot()
plt.title('Daily Booking Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Bookings')
plt.show()


# In[ ]:


#Observe whether bookings are increasing, decreasing, or stable over time.


# # Correlation matrix:

# In[70]:


corr_features = [
    'lead_time',
    'previous_cancellations',
    'previous_bookings_not_canceled',
    'booking_changes',
    'days_in_waiting_list',
    'adr',
    'required_car_parking_spaces'
]

# Calculate correlation matrix
corr_matrix = df[corr_features].corr()
#calculates the correlation coefficient between each pair: +1--> positive ,0--> no relation,-1--> negative

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='magma', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[ ]:


lead_time is slightly positively correlated with booking_changes and days_in_waiting------> 
correlated means more is the stay of customer more will be the lead time

adr shows a weak correlation with:
required_car_parking_spaces
previous_bookings_not_canceled

Adr and total people are highly correlated-more people more will be adr high adr high revenu
previous_cancellations and lead_time are positively correlated — customers who plan earlier may cancel more often.



# ### co-realtion 

# In[71]:


numeric_df = df.select_dtypes(include=['int64', 'float64'])
#Pearson Correlation
pearson_corr = numeric_df.corr(method='pearson')
#sepearman correlation
spearman_corr = numeric_df.corr(method='spearman')


# In[72]:


plt.figure(figsize=(12, 8))
sns.heatmap(pearson_corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Pearson Correlation Heatmap')
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='viridis')
plt.title('Spearman Correlation Heatmap')
plt.show()


# In[73]:


features = ['adr', 'lead_time', 'total_of_special_requests', 'booking_changes']
adr_corr = numeric_df[features].corr(method='pearson')
sns.heatmap(adr_corr, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Correlation of ADR with Selected Features')
plt.show()


# In[74]:


print("Correlation of ADR with lead_time:", adr_corr.loc['adr', 'lead_time'])
print("Correlation of ADR with special requests:", adr_corr.loc['adr', 'total_of_special_requests'])
print("Correlation of ADR with booking_changes:", adr_corr.loc['adr', 'booking_changes'])


# # Hypothesis testing

# In[ ]:


4.  Hypothesis Testing
Use statistical tests to validate business assumptions:
H0: There is no difference in ADR between bookings made through Online TA and Direct channels
H0: Room upgrades are independent of lead time
H0: Average stay duration does not differ between customer types


# In[ ]:


#1.
H0: There is no difference in ADR between bookings made through Online TA and Direct channels
df[online TA] abd df[Direct channels]
H0: There is no difference in the mean ADR between bookings made via Direct channel and TA/TO channel.
H1: There is a difference in the mean ADR between the two channels.


# In[75]:


online_ta_adr = df[df['distribution_channel'] == 'TA/TO']
direct_adr = df[df['distribution_channel'] == 'Direct']
#This is performing a two-sample Z-test on the ADR values from the two groups.
zscore,pvalue=ssw.ztest(direct_adr.adr,online_ta_adr.adr)
print(zscore,pvalue)


# In[ ]:


zscore: The test statistic, measuring how many standard deviations the difference in means is from zero.
pvalue: The probability of observing the data assuming the null hypothesis is true.


# In[ ]:


we reject Null hypothesis
If p-value < significance level (commonly 0.05), reject H0 → There is statistically significant evidence that ADR differs between Direct and TA/TO channels.
If p-value ≥ 0.05, fail to reject H0 → No sufficient evidence to say ADR differs between the two channels.
The zscore tells direction and magnitude of difference:
A large positive or negative z-score indicates a bigger difference between means.
The sign shows which group has higher mean (depending on order of subtraction in the test).


# In[ ]:





# In[ ]:


#2.
H0: Room upgrades are independent of lead time


# In[ ]:


Null Hypothesis (H₀):
Room upgrades are independent of lead time.
→ No significant difference in lead time between upgraded and non-upgraded bookings.

Alternative Hypothesis (H₁):
Room upgrades depend on lead time.
→ Guests who were reassigned rooms have different average lead times than those who were not.



# In[76]:


df['room_reassigned'] = (df['reserved_room_type'] != df['assigned_room_type']).astype(int)
lead_time_reassigned = df[df['room_reassigned'] == 1]
lead_time_not_reassigned = df[df['room_reassigned'] == 0]
ssw.ttest_ind(lead_time_reassigned.lead_time, lead_time_not_reassigned.lead_time, usevar='unequal')


# In[ ]:


t-statistic = -33.31  
p-value = 1.81e-236  
df (degrees of freedom) ≈ 18808
This is far less than 0.05, so we reject the null hypothesis.

Conclusion: There is a statistically significant difference in average lead times between upgraded and non-upgraded guests.



# In[ ]:





# In[ ]:


#3.


# In[ ]:


H0: Average stay duration is the same across customer types.
H1: Average stay duration differs across at least one customer type.


# In[ ]:


This is a one-way ANOVA test scenario because we are comparing means of a
numeric variable (stay duration) across multiple groups (customer types)


# In[77]:


df['customer_type'] = df['customer_type'].astype('category')


# In[78]:


model = ols('Total_Night ~ C(customer_type)', data=df).fit()
anova_table = sm.stats.anova_lm(model)
print(anova_table)


# In[ ]:


PR(>F) = 0.0 (p-value is extremely small):
This is the most important result.

Since p-value < 0.05, we reject the null hypothesis.
Conclusion: There is a statistically significant difference in average stay duration between at least one pair of customer types.


# In[ ]:


F-statistic = 886.52:
This is a very high F-value, indicating that the variation in stay duration between groups (customer types) is much larger than the variation within each group.

It confirms the groups are meaningfully different.



# In[ ]:


"The type of customer has a strong influence on the duration of their stay."

This means marketing strategies, pricing, and services can be tailored by customer


# # Key Buisness question

# In[ ]:


5.  Key Business Questions
get_ipython().run_line_magic('pinfo', 'most')
get_ipython().run_line_magic('pinfo', 'changes')
get_ipython().run_line_magic('pinfo', 'countries')
get_ipython().run_line_magic('pinfo', 'reassignment')
get_ipython().run_line_magic('pinfo', 'types')
●	What are the most common guest demographics (e.g., group size, nationality)?
●	Are there patterns in guest types (e.g., transient vs. corporate) that influence booking behavior? 
●	How does booking lead time vary across customer types and countries?
●	Are longer lead times associated with fewer booking changes or cancellations?
●	What is the typical duration of stay, and how does it vary by customer type or segment?
●	How often are guests upgraded or reassigned to a different room type?
●	Are guests who make special requests more likely to experience booking changes or longer stays?
●	Do certain market segments or distribution channels show higher booking consistency or revenue?
●	What factors are most strongly associated with higher ADR?
●	Are there customer types or segments consistently contributing to higher revenue?
●	Do bookings with more lead time or from specific countries yield higher ADR?
●	Are guests with higher ADR more likely to request special services or make booking modifications?


# ##### .What influences ADR the most?

# In[127]:


corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix[['adr']].sort_values(by='adr', ascending=False), annot=True)


# In[ ]:


From the heatmap, the strongest positive influencers of ADR are the number of special requests, lead time, and car parking spaces. Notably, 
cancellations are strongly negatively correlated with ADR. This suggests that high-paying customers tend to book early, request more services,
and are less likely to cancel.


# #### Do guests who book earlier tend to request more changes?

# In[134]:


corr_lead_changes = df['lead_time'].corr(df['booking_changes'])


# In[135]:


corr_lead_changes


# In[ ]:


There is a moderate positive correlation between lead_time and booking_changes.
So, the longer in advance people book, the more likely they are to make changes to their bookings.


# #### Are there pricing or booking differences across countries?

# In[ ]:


get_ipython().run_line_magic('pinfo', 'countries')
Whether guests from different countries are charged differently (ADR: Average Daily Rate)

Whether booking behavior (lead time, length of stay, cancellations) varies by country


# In[ ]:


Make a bar chart that shows the average ADR per country (top 10).
some countries might have much higher average rates.
(H₀): All countries pay the same average price
(H₁): Some countries pay different average prices


# In[138]:


adr_by_country = df.groupby('country')['adr'].mean().sort_values(ascending=False).head(10)


# In[139]:


adr_by_country


# In[132]:


adr_by_country = df.groupby('country')['adr'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=adr_by_country.index, y=adr_by_country.values)
plt.title("Top 10 Countries by Average Daily Rate")
plt.ylabel("Average Daily Rate (ADR)")
plt.xlabel("Country")
plt.xticks(rotation=45)
plt.show()


# In[ ]:


These top-paying countries may not be the highest in booking volume, but still bring more revenue per booking.


# #### Is there a pattern in room upgrades or reassignment?

# In[149]:


df['is_upgraded'] = df['reserved_room_type'] != df['assigned_room_type']
df['is_upgraded'].head()


# In[44]:


(df['reserved_room_type'] == df['assigned_room_type']).value_counts(normalize=True)


# In[ ]:





# ##### Are reserved room types consistently matched with assigned room types

# In[158]:


df['room_matched'] = df['reserved_room_type'] == df['assigned_room_type']
mismatched = df[df['room_matched'] == False]
print(mismatched.groupby(['reserved_room_type', 'assigned_room_type']).size().sort_values(ascending=False).head(10))


# In[159]:


match_counts = df['room_matched'].value_counts()
labels = ['Matched', 'Not Matched']
plt.pie(match_counts, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Room Assignment Match Rate')
plt.axis('equal')
plt.show()


# In[ ]:


The overall proportion of bookings where the reserved room type matches the assigned room type.
High Match % (e.g., 85–95%) → Hotel mostly honors reservations.

Low Match % (<70%) → Frequent room changes, suggesting:

Overbooking issues

Room type unavailability

Strategic upgrades for loyalty or dissatisfaction recovery


# In[ ]:





# #### What are the most common guest demographics (e.g., group size, nationality)?
# 

# In[161]:


sns.histplot(df['total_guests'], bins=range(1, 10), kde=False)
plt.title('Guest Group Size Distribution')
plt.xlabel('Total Guests per Booking')
plt.ylabel('Number of Bookings')
plt.show()


# In[ ]:


Most bookings are for 1 or 2 guests:

→ Targeted at solo travelers, business guests, or couples.

If larger groups (4–6) are common:

→ Suggests demand for family rooms or group offers.

Rare bookings for more than 6 → maybe conference or special event groups.


# In[162]:


top_nationalities = df['country'].value_counts().head(10)


# In[163]:


top_nationalities


# In[164]:


sns.barplot(x=top_nationalities.index, y=top_nationalities.values)
plt.title('Top 10 Guest Nationalities')
plt.ylabel('Number of Bookings')
plt.xlabel('Country')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


Top nationalities indicate your key guest source markets
If your top 3 are from the same country → strong domestic business.


# In[ ]:





# In[ ]:





# #### Are there patterns in guest types (e.g., transient vs. corporate) that influence booking behavior? 

# In[79]:


df['customer_type'].value_counts()


# In[80]:


lead_time_summary = df.groupby(['customer_type', 'country'])['lead_time'].mean().reset_index()


# In[81]:


lead_time_summary


# In[82]:


top_countries = df['country'].value_counts().head(10).index
lead_time_summary_top = lead_time_summary[lead_time_summary['country'].isin(top_countries)]


# In[ ]:


Hotels can see which customer types in which countries book earlier or later.

For example, Corporate guests from USA might book late, Group guests from UK might book early.


# In[83]:


plt.figure(figsize=(14, 7))
sns.barplot(data=lead_time_summary_top, x='country', y='lead_time', hue='customer_type')

plt.title('Average Booking Lead Time by Customer Type and Country')
plt.ylabel('Average Lead Time (days)')
plt.xlabel('Country')
plt.xticks(rotation=45)
plt.legend(title='Customer Type')
plt.tight_layout()
plt.show()


# ##### .Are longer lead times associated with fewer booking changes or cancellations?

# In[176]:


corr_lead_booking_changes = df['lead_time'].corr(df['booking_changes'])


# In[84]:


sns.boxplot(x='is_canceled', y='lead_time', data=df)
plt.title('Lead Time Distribution: Canceled vs. Non-Canceled Bookings')
plt.xlabel('Booking Canceled (0 = No, 1 = Yes)')
plt.ylabel('Lead Time (days)')
plt.show()


# In[ ]:


Correlation shows the strength and direction of association between lead time and booking changes. A negative correlation suggests longer lead times come with fewer changes.


# In[ ]:


Boxplot :-if canceled bookings tend to have systematically different lead times, indicating whether lead time can predict cancellations.


# In[ ]:


Use an independent t-test to check if the mean lead times are significantly different between canceled and non-canceled bookings.


# In[ ]:


Correlation shows the strength and direction of association between lead time and booking changes. A negative correlation suggests longer lead times come with fewer changes.


# ##### What is the typical duration of stay, and how does it vary by customer type or segment?

# In[185]:


mean_stay = df.groupby('customer_type')['Total_Night'].mean()
median_stay = df.groupby('customer_type')['Total_Night'].median()


# In[187]:


plt.figure(figsize=(10,6))
sns.boxplot(x='customer_type', y='Total_Night', data=df)
plt.title('Distribution of Stay Duration by Customer Type')
plt.xlabel('Customer Type')
plt.ylabel('Total Nights Stayed')
plt.show()


# In[ ]:


Which customer types tend to have longer or shorter stays on average. For example, corporate guests might have shorter stays than group bookings or transient tourists.


# #### Are there customer types or segments consistently contributing to higher revenue?

# In[88]:


df['total_revenue'] = df['adr'] * df['Total_Night']


# In[89]:


revenue_by_customer = df.groupby('customer_type')['total_revenue'].mean().sort_values(ascending=False)
revenue_by_segment = df.groupby('market_segment')['total_revenue'].mean().sort_values(ascending=False)


# In[90]:


plt.figure(figsize=(10,6))
sns.barplot(x=revenue_by_customer.index, y=revenue_by_customer.values)
plt.title("Average Revenue per Booking by Customer Type")
plt.ylabel("Avg Revenue (ADR × Nights)")
plt.xlabel("Customer Type")
plt.show()


# In[91]:


plt.figure(figsize=(10,6))
sns.barplot(x=revenue_by_segment.index, y=revenue_by_segment.values)
plt.title("Average Revenue per Booking by Market Segment")
plt.ylabel("Avg Revenue (ADR × Nights)")
plt.xlabel("Market Segment")
plt.xticks(rotation=45)
plt.show()


# In[ ]:


Drives data-informed pricing strategies
Helps with targeted marketing and loyalty programs
Improves revenue forecasting and customer segmentation


# #### Are guests with higher ADR more likely to request special services or make booking modifications?

# In[49]:


sns.scatterplot(data=df, x='adr', y='total_of_special_requests')
sns.scatterplot(data=df, x='adr', y='booking_changes')
plt.show()


# ##### Are guests who make booking changes more likely to request additional services or cancel?

# In[46]:


sns.boxplot(data=df, x='booking_changes', y='total_of_special_requests')
sns.boxplot(data=df, x='booking_changes', y='is_canceled')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




