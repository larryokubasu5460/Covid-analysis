#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[219]:


df=pd.read_csv('country_wise_latest.csv')
df.head()


# In[220]:


df.shape


# In[221]:


df.info()


# In[222]:


df=df.rename(columns={'Country/Region':'Country'})


# In[223]:


df.head()


# In[224]:


df.dtypes


# ### Editing columns and augmenting them

# In[225]:


df.head()


# In[226]:


df['Deaths']=df['Deaths']+df['New deaths']


# In[227]:


df['Recovered']=df['Recovered']+df['New recovered']


# In[228]:


df['Confirmed']=df['Confirmed']+df['New cases']


# In[229]:


df=df.drop(['New cases','New deaths','New recovered'], axis=1)


# In[230]:


df.head()


# In[231]:


df2=df.groupby('WHO Region').sum()
df2.head()


# In[232]:


df2.isna().any()


# In[233]:


df2.head()


# ### Plot graph of by region

# In[234]:


df2
df4=df2.loc[:,'Confirmed':'Active'].head()


# In[235]:


df3=df.groupby('WHO Region')
df3.head()


# In[236]:


deaths=df2['Deaths']
recovered=df2['Recovered']
confirmed=df2['Confirmed']
active=df2['Active']


# In[237]:


region=[region for region, df in  df3]
N=6
fig, axes=plt.subplots(figsize=(12,8))
w=0.
ind=np.arange(N)
axes.bar(ind+0.00, deaths, width=0.25,color='r',align='center', label='Deaths')
axes.bar(ind+0.25, recovered, width=0.25,color='g',align='center', label='Recovered')
axes.bar(ind+0.50, confirmed, width=0.25,color='m', align='center', label='Confirmed')
axes.bar(ind+0.75, active, width=0.25,color='y',align='center', label='Active')

axes.set_xticklabels(region,size=8)
axes.set_xticks(ind + w / 2)
axes.set_title('Covid by regions')
axes.set_ylabel('Numbers')
axes.set_xlabel('WHO regions')
axes.legend()
axes.grid()
axes.autoscale_view()
fig.savefig('WHO covid regions.png')


# In[107]:


get_ipython().run_line_magic('pinfo', 'plt.bar')


# In[99]:





# In[141]:


df5=df.loc[:,'Country':'Active']
df5.head()


# In[152]:


df5=df.groupby('Country')[['Confirmed','Deaths','Recovered','Active']].sum().reset_index()


# In[153]:



df5.head()


# In[145]:


df5.info()


# ### Covid by countries

# In[ ]:



        


# In[ ]:





# In[146]:


h=df5['Country'].unique()


# In[162]:


N=1
ind=np.arange(N)
w=0.25

for idx in range(0,len(h)):
    C=df5[df5['Country']==h[idx]].reset_index()
    plt.bar(ind+0.00,C['Confirmed'],width=0.25, color='b', align='center', label='Confirmed')
    plt.bar(ind+0.25,C['Recovered'], width=w,color='g',align='center',label='Recovered')
    plt.bar(ind+.50,C['Deaths'], width=w,color='r',align='center',label='Deaths')
    plt.bar(ind+.75,C['Active'], width=w,color='y',align='center',label='Active')
    plt.title(h[idx])
    plt.ylabel=("Totals")
    plt.xlabel=("Parameters")
    plt.xticks(ind)
    plt.legend()
    plt.grid()
   
    plt.show()
  


# In[168]:


df[df["Deaths"]>10000]


# In[183]:


df[df['Recovered']==df['Recovered'].min()]


# In[238]:


data=pd.read_csv('day_wise.csv',index_col=0,parse_dates=True)
data.head()


# In[239]:


data['Confirmed']=data['New cases']+data['Confirmed']
data['Recovered']=data['New recovered']+data['Recovered']
data['Active']=data['New cases']+data['Active']
data['Deaths']=data['Deaths']+data['New deaths']


# In[240]:


data.head()


# In[241]:


data=data.drop(['Deaths / 100 Cases','Recovered / 100 Cases','Deaths / 100 Recovered'],axis=1)


# In[242]:


data.head()


# In[243]:


data.describe()


# In[244]:


data.info()


# In[245]:


data=data.reset_index()


# In[246]:


data.head()


# In[247]:


data.dtypes


# In[248]:


dat=data.set_index('Date')


# In[249]:


dat.head()


# In[250]:


df=data.groupby('Date')[['Confirmed','Deaths','Recovered','Active']].sum()


# In[251]:


df.head()


# # Plotting daywise

# In[252]:


fig, axes=plt.subplots(figsize=(12,9))

axes.plot(df.index,df['Confirmed'],color='y',label="Confirmed")
axes.plot(df.index, df['Deaths'],color='r',label='Deaths')
axes.plot(df.index,df['Recovered'],color='g',label='Recovered')
axes.plot(df.index, df['Active'],color='b',label='Active')
axes.set_title("Covid Daywise chart")
axes.set_xlabel("Timeline")
axes.set_ylabel("Numbers")
axes.legend()
axes.grid()


# ### Add a monthly column

# In[256]:


dat=dat.reset_index()


# In[257]:


dat.dtypes


# In[258]:


dat.head()


# In[272]:


dat["Month"]=dat['Date'].str[5:7].astype('int')
dat.head()


# In[268]:


# dat['Date']=dat['Date'].dt.strftime('%Y-%m-%d')


# In[273]:


dat.tail()


# In[274]:


df=dat.groupby('Month')[['Date','Confirmed','Deaths','Recovered','Active']].sum()


# In[276]:


df.tail()


# In[280]:


fig, axes=plt.subplots(figsize=(12,9))

axes.plot(df.index,df['Confirmed'],color='y',label='confirmed')
axes.plot(df.index,df['Deaths'],color='r',label='Deaths')
axes.plot(df.index, df['Recovered'],color='g',label='Recovered')
axes.plot(df.index, df['Active'],color='b',label='Active')
axes.set_title('Monthly Covid Chart')
axes.set_xlabel('Months')
axes.set_ylabel('Numbers of patients')
axes.legend()
axes.grid()
fig.savefig('Monthly covid.png')


# ## New group

# In[134]:


df=pd.read_csv('full_grouped.csv')
df.head()


# In[135]:


df=df.rename(columns={'Country/Region':'Country'})


# In[136]:


df.dtypes


# In[137]:


df['Month']=df['Date'].str[5:7].astype('int64')
df.head()


# In[138]:


df.dtypes


# In[139]:


df['Date']=pd.to_datetime(df['Date'])


# In[140]:


df.dtypes


# In[141]:


df.set_index('Date')


# In[142]:


df.head()


# In[ ]:





# In[143]:


df['Country']


# In[144]:


df1=df.groupby('Country')
df1.head()


# In[145]:



for name, Country in df1:
    fig, ax=plt.subplots(figsize=(10,10))
    Country
    ax.plot(Country['Date'],Country['Confirmed'],color='b',label='confirmed')
    ax.plot(Country['Date'],Country['Deaths'],color='r',label='deaths')
    ax.plot(Country['Date'],Country['Active'],color='y',label='active')
    ax.plot(Country['Date'],Country['Recovered'],color='g',label='recovered')
    ax.set_title(name)
    ax.grid()
    ax.set_xlabel('Date')
    ax.set_ylabel('Population')
    ax.legend()
    fig.savefig('./images/'+name)


# In[118]:





# In[146]:


df2=df.groupby(['Country','Date'])[['Confirmed','Deaths','Recovered','Active']].sum()


# In[147]:


df2.head()


# In[148]:


df2.index.names


# In[149]:


df_kenya=df2.loc['Kenya']


# In[150]:


df2.loc[['United Kingdom','Brazil','France','Germany','US','South Korea']]
df_usa=df2.loc['US']
df_sk=df2.loc['South Korea']
df_uk=df2.loc['United Kingdom']
df_brazil=df2.loc['Brazil']
df_france=df2.loc['France']
df_germ=df2.loc['Germany']


# In[151]:


countries=df['Country'].unique()


# In[152]:


df_kenya.head()


# In[153]:


fig, axes=plt.subplots(figsize=(12,12))
axes.plot(df_kenya.index,df_kenya['Confirmed'],label='confirmed')
axes.plot(df_kenya.index,df_kenya['Deaths'],color='r',label='deaths')
axes.plot(df_kenya.index,df_kenya['Recovered'],color='g',label='recovered')
axes.plot(df_kenya.index,df_kenya['Active'],label='Active')
axes.set_xlabel('Dates')
axes.set_ylabel('Population')
axes.set_title('Kenya chart')
axes.grid()
axes.legend()


# In[44]:


countries=df['Country'].unique()


# In[154]:



for i in range(0,len(countries)):
    fig,axes=plt.subplots(figsize=(10,10))
    C=df[df['Country']==countries[i]]
    axes.plot(C['Date'],C['Confirmed'],label='Confirmed')
    axes.plot(C['Date'],C['Deaths'],label='deaths')
    axes.plot(C['Date'],C['Active'],label='deaths')
    axes.plot(C['Date'],C['Recovered'],label='recovered')
    axes.set_title(countries[i])
    axes.grid()
    axes.legend()
    
    


# In[120]:


"""Today was a success 
Found two ways of plotting multiIndex dataframes
first group by the specifi id you want to use to plot and equate the dataframe to a pd
loop through the df in twos i.e for name(name of the id), Country(the id) in df(the pd you equated the groupby to)
set title to name and use country to access the columns
secondly then loop through the len of ID's and equate the IDs to the respective column names
C=df[df['Country']==countries[idx]]

"""


# In[123]:


countries


# In[ ]:


df.loc['Taiwan*']


# In[ ]:




