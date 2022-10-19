#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import necessary  libraries 
import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt 
import plotly.express as px
import seaborn as sns
import pycountry
get_ipython().run_line_magic('matplotlib', 'inline')
#offline- mode
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
# Importing high-level chart objects
import plotly.graph_objs as go
# Import tools
from plotly import tools
import cufflinks as cf


# In[3]:


df = pd.read_csv('C://Users/Akara/Downloads/countries of the world.csv',decimal=',')


# In[4]:


df.head(10)


# In[5]:


df2 = df.iloc[:,2:]


# In[6]:


df2


# In[7]:


print('no of rows:{}'.format(df.shape[0]),'no of columns:{}'.format(df.shape[1]))


# In[8]:


df.isnull().sum()


# In[9]:


df.columns


# In[10]:


df.info()


# In[11]:


## creating an iso_alpha code of countries of easy use of visualizations
def do_fuzzy_search(country):
    try:
        result = pycountry.countries.search_fuzzy(country)
        return result[0].alpha_3
    except:
        return np.nan

df["country_code"] = df["Country"].apply(lambda country: do_fuzzy_search(country))


# In[12]:


# TO KNOW THE MISSING VALUE PERCENT
print("missing percent is:")
def missing_percent(df):
    nan_percent= 100*(df.isnull().sum()/len(df))
    nan_percent= nan_percent[nan_percent>0].sort_values()
    return nan_percent
missing_percent(df)


# In[13]:


cols = df[['Net migration', 'Infant mortality (per 1000 births)',
       'GDP ($ per capita)', 'Literacy (%)', 'Phones (per 1000)', 'Arable (%)',
       'Crops (%)', 'Other (%)', 'Climate', 'Birthrate', 'Deathrate',
       'Agriculture', 'Industry', 'Service']]
for i in cols:
    df[i].fillna(df[i].mean(axis=0), inplace=True)


# In[14]:


df.isnull().sum()


# In[15]:


df.rename(columns={'Literacy (%)': 'Literacy',
                  'Arable (%)':'Arable',
                  'Crops (%)':'Crops',
                  'Other (%)':'Other_factors',
                  'Pop. Density (per sq. mi.)':'Population_density',
                  'Coastline (coast/area ratio)':'Coastline',
                  'Area (sq. mi.)':'Area_land',
                  'Infant mortality (per 1000 births)':'Infant_mortality',
                  'GDP ($ per capita)':'GDP_PCP',
                  'Phones (per 1000)':'Phones'},inplace=True)


# In[16]:


df.round(4)


# In[17]:


df.describe()


# In[18]:


#df['Literacy']= df.apply(lambda x: x.Literacy * 100,axis=1)
#df['Crops']= df.apply(lambda x: x.Crops * 100,axis=1)
#df['Others_factors']= df.apply(lambda x: x.Other_factors * 100,axis=1)
#df['Arable']= df.apply(lambda x: x.Arable * 100,axis=1)


# # VISUALIZATION 
# 
# visualization
# on the net migration progress
# visualize on 1the  death rates
               2.birth rates
               3.infant morality (per 1000 births)
               4.Industry 
               5.death rates
               6.Pop. Density (per sq. mi.)
               7.phone(per 1000)
               8.Agriculture
               9.climate
               10.crops
# In[20]:


# CORRELATIONS OF THE DATASET
plt.figure(figsize=(30,20))
sns.pairplot(df2)
plt.show()


# In[21]:


# CORRELATION ALSO
plt.figure(figsize=(16,12))
sns.heatmap(data=df.iloc[:,2:].corr(),annot=True,fmt='.2f',cmap='coolwarm')
plt.show()


# In[22]:


top_deathrates=df.nlargest(10,['Net migration'] , keep='all')
#top_deathrates


# In[23]:


fig=px.bar(top_deathrates , y='Net migration',x='Country',title='TOP TEN NET MIGRATION COUNTRIES',text='Deathrate',
          color_discrete_sequence=px.colors.sequential.Magma_r)
fig.update_traces(texttemplate="%{text:.2s}" ,textposition='outside')


# In[24]:


top_population_coun=df.nlargest(20,'Population')
last_population_coun=df.nsmallest(20,'Population')
top_last_population_coun=pd.concat([top_population_coun,last_population_coun])
#top_population_coun


# In[25]:


fig = px.bar(top_population_coun ,y ='Population',x = 'Country', text='Population',title='THE TOP POPULATED COUNTRIES',
            color_discrete_sequence=px.colors.sequential.Darkmint,template='plotly_dark')
fig.update_traces(texttemplate="%{text:.2s}" ,textposition='outside')
fig.update_layout(uniformtext_minsize=8)
fig.update_layout(xaxis_tickangle=-45)
fig


# In[26]:


fig = px.bar(last_population_coun ,y ='Population',x = 'Country', text='Population',title='THE LAST POPULATED COUNTRIES')
fig.update_traces(texttemplate="%{text:.2s}" ,textposition='outside')
fig.update_traces(marker_color='rgb(267,006,225)', marker_line_color='rgb(200,408,127)',
                  marker_line_width=1.5, opacity=0.6)
fig.update_layout(uniformtext_minsize=8)
fig.update_layout(xaxis_tickangle=-45)
fig


# In[27]:


px.bar(df, x='Region',y='Coastline',title='ANALTSIS ON THE COASTLINE COUNTRIES ACCORDING TO REGION',
      hover_name='Country',color_discrete_sequence=px.colors.sequential.Jet,log_y=True,)


# In[28]:


fig = px.scatter_geo(df, locations='country_code',hover_name='Country',
                    size='Arable', projection='stereographic',title='THE ARABLE LAND WORLDWIDE COUNTRY VISUAL',
                    color_discrete_sequence=px.colors.sequential.gray,template='plotly',)
fig


# In[29]:


fig= px.choropleth(df,locations='country_code',color='Arable',color_continuous_scale=px.colors.sequential.Cividis_r,
                   scope='europe',title='further analysis on the europe arable land',hover_name='Country')
fig


# In[30]:


fig = px.sunburst(df, path=['Region', 'Country'], values='Population',
                  color='Area_land', hover_data=['country_code'],
                  color_continuous_scale='Oryel',
                  color_continuous_midpoint=np.average(df['Area_land'], weights=df['Population']),
                 title='Population and Area (square miles)(640arce==1area[square miles])')
fig.show()


# In[31]:


fig = px.choropleth(df,locations='country_code',color='Climate',
                   color_continuous_scale='oxy',range_color=(0,12),hover_name='Country',
                   title='GLOBAL CLIMATE(%) ANALYSIS',projection='mollweide')
fig


# In[32]:


#df3.rename({'df.Infant_mortality(per 1000 births)':'infant_mortality'},inplace=True)
trace0 = go.Scatter(x =df.Infant_mortality,
                  y =df.Birthrate,
                  mode = "markers",
                   name = "Birthrate(%)",
                  marker = dict(size = 12, color = "rgba(255, 70, 0, 0.9)"),xaxis='x1',yaxis='y2')

trace1 = go.Scatter(x =df.Infant_mortality,
                y =df.Deathrate,
                name = "Deathrate(%)",
                 mode = "markers",
                marker = dict(size = 12, color = "rgba(0, 190, 255, 0.9)"),xaxis='x2',yaxis='y2')

fig = tools.make_subplots(rows = 1,
                          cols = 2,
                         subplot_titles = ("birthrate", "deathrate"),)

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig.layout.update(title = "Correlation between infant_morality ,birthrate and Deathrate")

iplot(fig);


# In[33]:


#dir(px.colors.sequential)
#dir(px.colors.qualitative)


# In[34]:


#pie chart
#infant morality
fig = px.pie(df, values='Infant_mortality', names='Region',color_discrete_sequence=px.colors.sequential.BuGn,
             template='plotly_dark',
             title='piechart depicting(the number of deaths per 1,000 live births of children under the age of one year)in the region')
fig.update_traces(textposition='inside')
fig.update_layout(uniformtext_minsize=12, uniformtext_mode=False,)
fig.show()


# In[35]:


#donutchart
#deathrate
px.pie(df,values='Deathrate',names='Region',hole=0.5,color_discrete_sequence=px.colors.sequential.Burgyl,
       template='plotly_dark',title='A DONUT CHART DEATHRATE ANALYSIS')


# In[36]:


px.box(df['GDP_PCP'],hover_name=df['Country'] ,template='plotly_dark',title='A boxplot on the GDP',
       color_discrete_sequence=px.colors.sequential.Burgyl)


# In[37]:


high_GDP =df.nlargest(15,'GDP_PCP', keep='all')
low_GDP =df.nsmallest(15,'GDP_PCP')


# In[38]:


px.bar(high_GDP ,y='Country',x='GDP_PCP',color_discrete_sequence=px.colors.sequential.Rainbow,template='plotly',
      text_auto='GDP_PCP',orientation='h',title='A BAR CHART ON THE TOP TEN GDP COUNTRIES')


# In[39]:


px.bar(low_GDP ,y='Country',x='GDP_PCP',color_discrete_sequence=px.colors.sequential.Cividis,template='plotly',
      text_auto='GDP_PCP',orientation='h',title='A BAR CHART ON LAST GDP COUNTRIES')


# In[40]:


#industry
g=df['Industry'].max()
f=df['Industry'].min()
high_con_indu =df.nlargest(20,'Industry')
fig = px.scatter(df ,x='Country',y='Industry',size='Industry',color_discrete_sequence=px.colors.sequential.Viridis,
           log_y=True,range_y=[0.01,2],marginal_x='box',title='INDUSTRIES ANALYSIS ON COUNTRIES',width=1000,
          height=700)

fig.show()
fig.write_html('C://Users/Akara/Downloads/industries.html')


# In[41]:


#service
#df['Service'].max()
df['Service'].min()
px.area(df,y='Service',x='Country',hover_name='Country',color_discrete_sequence=px.colors.sequential.gray_r,
        markers=True,template='plotly_dark',title='A SERVICE AREA CHART BY COUNTRIES')


# In[42]:


#LITERACY
px.histogram(df, x='Country',y='Literacy',height=600,width=1000,marginal='violin',
             color_discrete_sequence=px.colors.sequential.Sunsetdark,template='plotly_dark',
            title='LITERACY ANALYSIS ON COUNTRIES',range_y=[20,100])


# In[43]:


#Agriculture
px.violin(df,y='Agriculture',box=True,points='all',color_discrete_sequence=px.colors.sequential.Plasma_r,
          template='plotly_dark',title='A VIOLIN ON AGRICULTURE BASED ON COUNTRIES',hover_name='Country')


# In[44]:


px.area(df,x='Country',y='Population_density',log_y=True,markers=True,
        color_discrete_sequence=px.colors.sequential.Aggrnyl,title='A AREA CHART ON THE POPULATION DENSITY')


# In[45]:


fig=px.choropleth(df,locations='country_code',color='Population_density',basemap_visible=True,
             color_continuous_scale=px.colors.sequential.solar,projection='robinson',scope='africa',
             range_color=[0,51],title='A AFRICA CHOROPLETH MAP ON THE POPULATION DENSITY')
fig.show()


# # MACHINE LEARNING ANALYSIS

# In[46]:


GDP=df.corr()['GDP_PCP']
GDP_CORR=pd.DataFrame(GDP)
cm1 = sns.light_palette('green',as_cmap=True)
GDP_CORR.style.background_gradient(cmap=cm1)


# In[47]:


#machine learnig dataframe copy
df3=pd.DataFrame(df)


# In[48]:


gdp_per_capita=df['GDP_PCP']
df3 =df3.drop(['Country', 'Region', 'Population','Area_land','Birthrate', 'Deathrate',
       'Agriculture', 'Industry', 'country_code','Coastline','Infant_mortality', 'Arable',
       'Crops', 'Other_factors','GDP_PCP','Climate'],axis=1)
df3['GDP_per_capita($)']=gdp_per_capita


# In[49]:


df3


# In[50]:


X  = df[['Phones']].to_numpy()
y = df[['GDP_PCP']].to_numpy()
# splitting into training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.3,random_state=0)


#train the model on training set
from sklearn.linear_model import LinearRegression
ML= LinearRegression()
ML.fit(X_train,y_train)

y_pred =ML.predict(X_test)
from sklearn.metrics import r2_score

r2_score(y_test,y_pred)*100
 


# In[ ]:





# In[51]:


import statsmodels.formula.api as smpi
zl=smpi.ols(formula='Phones~GDP_PCP',data=df).fit()
zl.summary()


# In[52]:


#ML.predict([[]])


# # machine learning predictions on infant  mortality

# In[53]:


#linear multiple regression
IFM=df.corr()['Infant_mortality']
IFM_CORR =pd.DataFrame(IFM)

cm = sns.light_palette("orange", as_cmap=True)

IFM_CORR.style.background_gradient(cmap=cm)


# In[54]:


#y = mx +c
# Generally: y[i] = alpha + (beta_1 * x_1[i]) + (beta_2 * x_2[i]) + (beta_3 * x_3[i]) + error
# Model:     y_hat[i] = alpha_hat + (beta_1_hat * x_1[i]) + (beta_2_hat * x_2[i]) + (beta_3_hat * x_3[i])
#alpha_hat = y_intercept = C
X = df[['Birthrate']].to_numpy()
y= df[['Infant_mortality']].to_numpy()
tl = LinearRegression()
tl.fit(X,y)


# In[55]:


y_pred=tl.predict(X)
plt.scatter(X,y)
plt.plot(X,y_pred ,color='r')


# In[56]:


print(tl.coef_)


# In[57]:


print(tl.intercept_)


# In[58]:


from sklearn.metrics import r2_score
r2_score(y,y_pred)


# In[59]:


# machine learning predictions on infant  mortality


# In[60]:


X=df[['Birthrate','Agriculture','Deathrate']].to_numpy()
y=df[['Infant_mortality']].to_numpy()


# In[61]:


X


# In[62]:


y


# In[63]:


df.head()


# In[64]:


# splitting into training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.3,random_state=0)


# In[65]:


#train the model on training set
from sklearn.linear_model import LinearRegression
ML= LinearRegression()
ML.fit(X_train,y_train)


# In[66]:


#predict the test set result
y_pred =ML.predict(X_test)
#print(y_predict)


# In[67]:


#ML.predict([[46.60,38.000000,20.34]])


# In[68]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,accuracy_score
r2_score(y_test,y_pred)*100


# In[69]:


plt.scatter(y_test,y_pred)


# In[70]:


print(ML.score(X_test,y_test)*100)
print(mean_squared_error(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))


# In[71]:


#ML.predict([[163]])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




