#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 18:52:51 2021

@author: mariawarter
"""

Run the normal model script first to get the synthetic reference series 
Check artifical_gr & artifical_shr median values for correctness 
Then run this script to get the CCS series 
When comparing synthetic and ccs check for weird values 
check the values of the KPE before making distributions for double values or weird ones 
#%%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from scipy import stats
from scipy import interpolate
import seaborn as sns
from scipy.stats import iqr

#Define water year and day of water year 

def assign_wy(df):
    if df.Date.month>=11:
        return(datetime(df.Date.year+1,1,1).year)
    else:
        return(datetime(df.Date.year,1,1).year)


def day_of_water_year(x):
    # Get the date of the previous October 1st
    water_year_start_date = datetime(x.year + x.month // 11 - 1, 11, 1)
    
    # Return the number of days since then
    return (x - water_year_start_date).days + 1#%%
#%%Historic SAVI - VALIDATION DATA

file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/SAVI/landsat_1986_2021/veg_grass_savi_cloud_na.csv"
#%%
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/SAVI/landsat_1986_2021/veg_shrub_savi_cloud_na.csv"

#%%
SAVI = pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0],sep=',')
SAVI['date'] = pd.to_datetime(SAVI['date'],dayfirst=True)

SAVI = SAVI.set_index(pd.DatetimeIndex(SAVI['date']))  
SAVI = SAVI.drop(['date'], axis=1)
SAVI=SAVI /10000
SAVI= SAVI.interpolate() #interpolating monthly values to daily
idx=pd.date_range('01-17-1986','12-19-2021',freq='D') #daily daterange
SAVI=SAVI.reindex(idx) #upsampling to daily values 
t= SAVI.loc[:'2011-11-11']
t=t.interpolate()
s=SAVI.loc['2013-03-26':]
s=s.interpolate()
p=SAVI.loc['2011-11-11':'2013-03-26']
SAVI=pd.concat([t,s],axis=0)
SAVI=SAVI.round(2)
SAVI.columns = SAVI.columns.astype(int)
SAVI['Date'] = pd.to_datetime(SAVI.index)
SAVI['WY'] = SAVI.apply(lambda x: assign_wy(x), axis=1)

SAVI['DOWY'] = SAVI.index.to_series().apply(day_of_water_year)

#%%
ag_gr=SAVI
ag_gr['median'] = ag_gr.iloc[:,0:30].median(axis=1)
ag_gr['iqr']=iqr(ag_gr.iloc[:,0:30],axis=1)
ag_gr = ag_gr.loc[(ag_gr.index < '2011-11-01') | (ag_gr.index > '2013-10-31')]
ag_gr = ag_gr.loc[(ag_gr.index < '1998-11-01') | (ag_gr.index > '2000-10-31')] #WY 2000 is incomplete - dont use 
ag_gr = ag_gr[ag_gr.DOWY != 366]
ag_gr=ag_gr[ag_gr['WY'].isin([1996,1998,2002,2006,2008,2010,2014,2016,2018])]

#%%
ag_shr=SAVI
ag_shr['median'] = ag_shr.iloc[:,0:30].median(axis=1)
ag_shr['iqr']=iqr(ag_shr.iloc[:,0:30],axis=1)
ag_shr = ag_shr.loc[(ag_shr.index < '2011-11-01') | (ag_shr.index > '2013-10-31')]
ag_shr = ag_shr.loc[(ag_shr.index < '1998-11-01') | (ag_shr.index > '2000-10-31')] #WY 2000 is incomplete - dont use 
ag_shr = ag_shr[ag_shr.DOWY != 366]
ag_shr=ag_shr[ag_shr['WY'].isin([1996,1998,2002,2006,2008,2010,2014,2016,2018])]
#%%
#Rain VALIDATION DATA
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/ag_rain_val.csv"
ag_rain= pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
ag_rain['Date'] = pd.to_datetime(ag_rain['Date'],dayfirst=True)
ag_rain['WY'] = ag_rain.apply(lambda x: assign_wy(x), axis=1)
ag_rain['DOWY'] = ag_rain.index.to_series().apply(day_of_water_year)
ag_rain = ag_rain.loc[(ag_rain.index < '1998-11-01') | (ag_rain.index > '2000-10-31')] #WY 2000 is incomplete - dont use 
ag_rain = ag_rain.loc[(ag_rain.index < '2011-11-01') | (ag_rain.index > '2013-10-31')]
ag_rain = ag_rain.loc[:'2018-10-31']
ag_rain = ag_rain[ag_rain.DOWY != 366]
ar=pd.DataFrame(ag_rain['P'])

#%%
#CCS of weaker monsoon 
months=[6,7,8,9]
rain1= ag_rain[['P','31D','62D','80D','94D']][ag_rain['Date'].map(lambda t: t.month in months)]

rain1 = rain1 * 0.75

months=[1,2,3,4]
rain2=ag_rain[['P','31D','62D','80D','94D']][ag_rain['Date'].map(lambda t: t.month in months)]
rain2 = rain2 * 0.75

months =[5,10,11,12]
rain3=ag_rain[['P','31D','62D','80D','94D']][ag_rain['Date'].map(lambda t: t.month in months)]

ag_rain=pd.concat([rain1,rain2,rain3],axis=0)
ag_rain=ag_rain.sort_index()
ag_rain['Date'] = ag_rain.index
ag_rain['WY'] = ag_rain.apply(lambda x: assign_wy(x), axis=1)
ag_rain['DOWY'] = ag_rain.index.to_series().apply(day_of_water_year)
ar1= pd.DataFrame(ag_rain['P'])
#%%

plt.plot(ar['P'])
plt.plot(ar1['P'])

#%%
#Temperature VALIDATION DATA
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/dT_SP/ESRL_temp_1986_2020.csv"
data_T = pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
data_T = data_T.set_index(pd.DatetimeIndex(data_T['date']))  
data_T.columns=['Date','maxT','minT']
data_T['Date'] = pd.to_datetime(data_T.index) #set P data to 1986-2019
data_T['mean']= (data_T['maxT'] + data_T['minT'] ) /2
data_T=data_T.interpolate()
data_T['31D']=data_T['maxT'].rolling(31).mean().round(2) 
data_T['62D']=data_T['maxT'].rolling(62).mean().round(2) 
data_T['94D']=data_T['maxT'].rolling(94).mean().round(2) 
data_T['80D']=data_T['maxT'].rolling(80).mean().round(2) 
#%%NORMAL
ag_temp=data_T.loc['1994-11-01':'2019-10-31']
ag_temp = ag_temp.loc[(ag_temp.index < '1998-11-01') | (ag_temp.index > '2000-10-31')]
ag_temp = ag_temp.loc[(ag_temp.index < '2011-11-01') | (ag_temp.index > '2013-10-31')]
ag_temp['WY'] = ag_temp.apply(lambda x: assign_wy(x), axis=1)
ag_temp['DOWY'] = ag_temp.index.to_series().apply(day_of_water_year)
ag_temp = ag_temp[ag_temp.DOWY != 366]
ag_temp=ag_temp[ag_temp['WY'].isin([1996,1998,2002,2006,2008,2010,2014,2016,2018])]
at=ag_temp
#%%
#CCS
ag_temp = data_T
ag_temp['Date1']= ag_temp['Date'] - timedelta(days=20)
ag_temp.index=ag_temp['Date1']

ag_temp=ag_temp.loc['1994-11-01':'2019-10-31']
ag_temp = ag_temp.loc[(ag_temp.index < '1998-11-01') | (ag_temp.index > '2000-10-31')]
ag_temp = ag_temp.loc[(ag_temp.index < '2011-11-01') | (ag_temp.index > '2013-10-31')]
ag_temp['Date'] = pd.to_datetime(ag_temp.index) #set P data to 1986-2019
ag_temp['WY'] = ag_temp.apply(lambda x: assign_wy(x), axis=1)
ag_temp['DOWY'] = ag_temp.index.to_series().apply(day_of_water_year)
ag_temp = ag_temp[ag_temp.DOWY != 366]
ag_temp=ag_temp[ag_temp['WY'].isin([1996,1998,2002,2006,2008,2010,2014,2016,2018])]
at1=ag_temp
#%%


dc=pd.pivot_table(at, index=at['DOWY'], columns=at['WY'], values='94D')
dc.mean(axis=1).plot()

dd=pd.pivot_table(ag_temp, index=ag_temp['DOWY'], columns=ag_temp['WY'], values='94D')
dd.mean(axis=1).plot()
#%%PRECIPITATION DIFF
 
rain_diff = np.diff(ag_rain['62D'])
rain_diff=pd.DataFrame(rain_diff)
rain_diff.columns=['slope']
rain_diff = rain_diff.set_index(ag_rain[:-1].index)
rain_diff['Date'] =rain_diff.index
rain_diff['WY'] = rain_diff.apply(lambda x: assign_wy(x), axis=1)
rain_diff['DOWY'] = rain_diff.index.to_series().apply(day_of_water_year)

#m_rain=pd.pivot_table(ag_rain, index=ag_rain['DOWY'], columns=ag_rain['WY'], values='P')
#m_rain=m_rain.dropna()
#%%
#START OF THE MONSOON - TIMING 
monsoon_months=[6,7,8,9]
som= pd.DataFrame(rain_diff.groupby(rain_diff['WY']).cumsum())
som['Date'] =som.index
som['WY'] = som.apply(lambda x: assign_wy(x), axis=1)
som['DOWY'] = som.index.to_series().apply(day_of_water_year)
som = som[som['Date'].map(lambda t: t.month in monsoon_months)]

som_doy= pd.DataFrame(som.groupby("WY")['slope'].idxmin()) #index of min value=start of monsoon season
#som_doy=pd.DataFrame(som_doy['slope'] + timedelta(days=20))
som_doy=som_doy.set_index(som_doy.slope)
som=pd.DataFrame(som['DOWY'][som.index.isin(som_doy.index)])

som=som.reset_index()
#%%TEMPERATURE diff

temp_diff = np.diff(ag_temp['94D'])
temp_diff=pd.DataFrame(temp_diff)
temp_diff.columns=['slope']
temp_diff = temp_diff.set_index(ag_temp[:-1].index)
temp_diff['Date'] =temp_diff.index
temp_diff['WY'] = temp_diff.apply(lambda x: assign_wy(x), axis=1)
temp_diff['DOWY'] = temp_diff.index.to_series().apply(day_of_water_year)

m_temp=pd.pivot_table(temp_diff, index=temp_diff['DOWY'], columns=temp_diff['WY'],values='slope')
m_temp.drop(m_temp.tail(1).index,inplace=True)
#c=m_temp.mean(axis=1)
#c.plot()
#%%
temp_max_doy = pd.DataFrame(temp_diff.groupby(temp_diff['WY']).cumsum())
temp_max_doy['Date'] = temp_max_doy.index
temp_max_doy['WY'] = temp_max_doy.apply(lambda x: assign_wy(x), axis=1)
temp_max_doy= pd.DataFrame(temp_max_doy.groupby("WY")['slope'].idxmax()) 
temp_max_doy = temp_max_doy.set_index(pd.DatetimeIndex(temp_max_doy.slope))

temp_min_doy=pd.DataFrame(temp_diff.groupby(temp_diff['WY']).cumsum())
temp_min_doy['Date'] = temp_min_doy.index
temp_min_doy['WY'] = temp_min_doy.apply(lambda x: assign_wy(x), axis=1)
temp_min_doy= pd.DataFrame(temp_min_doy.groupby("WY")['slope'].idxmin()) 
temp_min_doy = temp_min_doy.set_index(pd.DatetimeIndex(temp_min_doy.slope))

#%%
#MAX SAVI-VALUE
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/ci_max_gr1.csv"

ci_max_gr= pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
ci_max_gr.columns=['ci']
ci_max_gr['P'] = pd.Series(np.arange(0,220,1))
#ci_max_gr=np.asarray(ci_max_gr)
#ci_max_gr=ci_max_gr.reshape(-1)#ci_max_gr.columns=['CI']

file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/ci_max_shr1.csv"

ci_max_shr= pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
ci_max_shr.columns=['ci']
ci_max_shr['P'] = pd.Series(np.arange(0,220,1))
#ci_max_shr=np.asarray(ci_max_gr)
#ci_max_shr=ci_max_shr.reshape(-1)#ci_max_gr.columns=['CI']
#
rain = pd.DataFrame(ag_rain['31D'][ag_rain.index.isin(temp_max_doy.slope)])
rain=rain.round(0)
#
#max_savi_gr= pd.DataFrame((1.910*10**(-6))*rain**2 +0.0005*rain+ 0.1565) #THAT IS THE MEAN SAVI RESPONSE OF THE MODEL 
max_savi_gr= pd.DataFrame(0.001*rain+ 0.141) #THAT IS THE MEAN SAVI RESPONSE OF THE MODEL 
max_savi_gr.columns=['MAX-SAVI']

max_savi_shr= pd.DataFrame(0.001*rain + 0.131) #SHRUB MEAN modelled 
max_savi_shr.columns=['MAX-SAVI']
#%%GRASS
df_max_gr = pd.concat([max_savi_gr['MAX-SAVI'],rain['31D']],axis=1)
df_max_shr = pd.concat([max_savi_shr['MAX-SAVI'],rain['31D']],axis=1)
#%%
df_max_gr=df_max_gr.merge(ci_max_gr, left_on='31D', right_on='P')[['MAX-SAVI', 'ci','31D']]
df_max_shr=df_max_shr.merge(ci_max_shr, left_on='31D', right_on='P')[['MAX-SAVI', 'ci','31D']]
#
a=np.asarray(df_max_gr['MAX-SAVI'])
b=np.asarray(df_max_gr['ci'])
c=np.asarray(df_max_shr['MAX-SAVI'])
d=np.asarray(df_max_shr['ci'])

N=len(df_max_gr.index)
distr_max_gr=np.zeros([N, 100])
distr_max_shr=np.zeros([N, 100])

for t in range(N):
    
    distr_max_gr[t] = np.random.normal(a[t],b[t],100)
    distr_max_shr[t] = np.random.normal(c[t],d[t],100)

distr_max_gr=pd.DataFrame(distr_max_gr)
distr_max_gr=pd.concat([distr_max_gr,df_max_gr['ci']],axis=1)
distr_max_gr = distr_max_gr.set_index(max_savi_gr.index)
#distr_max_gr=distr_max_gr.iloc[:,0:100].clip(lower=0.18)

distr_max_shr=pd.DataFrame(distr_max_shr)
distr_max_shr=pd.concat([distr_max_shr,df_max_shr['ci']],axis=1)
distr_max_shr = distr_max_shr.set_index(max_savi_shr.index)
#distr_max_gr=distr_max_gr.iloc[:,0:100].clip(lower=0.18)

#%%
#END OF SEASON SAVI
#MAX SAVI-VALUE
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/ci_sos_gr.csv"

ci_sos_gr= pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
ci_sos_gr.columns=['ci']
ci_sos_gr['P'] = np.linspace(0, 220, 220).round(0)
ci_sos_gr=ci_sos_gr.drop_duplicates(subset=['P'])

file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/ci_sos_shr.csv"

ci_sos_shr= pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
ci_sos_shr.columns=['ci']
ci_sos_shr['P'] = np.linspace(0,220, 220).round(0)
ci_sos_shr=ci_sos_shr.drop_duplicates(subset=['P'])

#
rain = pd.DataFrame(ag_rain['94D'].loc[ag_rain['DOWY'] == 1]).round(0)
#sos_savi_gr = pd.DataFrame(((1.2688*10**(-6))*rain**2 + 0.0002*rain+0.09)).loc[ag_rain['DOWY'] == 1] #GRASS

sos_savi_gr = pd.DataFrame(0.0003*rain+0.076)#GRASS
sos_savi_gr.columns=['SOS-SAVI']
#eos_savi_gr['EOS-SAVI']=eos_savi_gr['EOS-SAVI'].clip(lower=0.1)


#sos_savi_shr= pd.DataFrame(((1.089*10**(-6))*rain**2 + 0.0002*rain+0.1048)).loc[ag_rain['DOWY'] == 1] #SHRUB
sos_savi_shr= pd.DataFrame(0.0002*rain+0.094) #SHRUB

sos_savi_shr.columns=['SOS-SAVI']
#eos_savi_shr['EOS-SAVI']=eos_savi_shr['EOS-SAVI'].clip(lower=0.1)
#%%

df_sos_gr = pd.concat([sos_savi_gr['SOS-SAVI'],rain['94D']],axis=1) #check duplicate values 2015
df_sos_shr = pd.concat([sos_savi_shr['SOS-SAVI'],rain['94D']],axis=1) #check duplicate values 
#%%
df_sos_gr=df_sos_gr.merge(ci_sos_gr, left_on='94D', right_on='P')[['SOS-SAVI', 'ci','94D']]
df_sos_shr=df_sos_shr.merge(ci_sos_shr, left_on='94D', right_on='P')[['SOS-SAVI', 'ci','94D']]
#
a=np.asarray(df_sos_gr['SOS-SAVI'])
b=np.asarray(df_sos_gr['ci'])
c=np.asarray(df_sos_shr['SOS-SAVI'])
d=np.asarray(df_sos_shr['ci'])

N=len(df_sos_gr.index)
distr_sos_gr=np.zeros([N, 100])
distr_sos_shr=np.zeros([N, 100])

for t in range(N): 
    
    distr_sos_gr[t] = np.random.normal(a[t],b[t],100)
    distr_sos_shr[t] = np.random.normal(c[t],d[t],100)


distr_sos_gr=pd.DataFrame(distr_sos_gr)
distr_sos_gr = distr_sos_gr.set_index(sos_savi_gr.index)
#distr_sos_gr=distr_sos_gr.iloc[:,0:100].clip(lower=0.1)

distr_sos_shr=pd.DataFrame(distr_sos_shr)
distr_sos_shr = distr_sos_shr.set_index(sos_savi_shr.index)
#distr_sos_shr=distr_sos_shr.iloc[:,0:100].clip(lower=0.1)

#%%
#START OF NEXT SEASON SAVI-GRASS

distr_eos_gr = pd.DataFrame(distr_sos_gr)
distr_eos_gr['Date'] = pd.DatetimeIndex(ag_rain['Date'].loc[ag_rain['DOWY'] == 365])
distr_eos_gr = distr_eos_gr.set_index(pd.DatetimeIndex(distr_eos_gr['Date']))
#SHRUB
distr_eos_shr = pd.DataFrame(distr_sos_shr)
distr_eos_shr['Date'] = pd.DatetimeIndex(ag_rain['Date'].loc[ag_rain['DOWY'] == 365])
distr_eos_shr = distr_eos_shr.set_index(pd.DatetimeIndex(distr_eos_shr['Date']))

#%% savi at START OF MONSOON

file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/ci_som_gr1.csv"

ci_som_gr= pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
ci_som_gr.columns=['ci']
ci_som_gr['P'] = np.linspace(0, 60, 100).round(0)
ci_som_gr=ci_som_gr.drop_duplicates(subset=['P'])

file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/ci_som_shr1.csv"
ci_som_shr= pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
ci_som_shr.columns=['ci']
ci_som_shr['P'] = np.linspace(0, 60, 100).round(0)
ci_som_shr=ci_som_shr.drop_duplicates(subset=['P'])

#
rain=pd.DataFrame(ag_rain['94D'][ag_rain.index.isin(som_doy.index)]).round(0)

#som_savi_gr = pd.DataFrame((6.933*10**(-5))*rain**2 + 0.0008*rain +0.0847)[ag_rain.index.isin(som_doy['slope'])] #SHRUB
som_savi_gr = pd.DataFrame(0.001*rain +0.0836) #SHRUB

som_savi_gr.columns=['SOM-SAVI']

#som_savi_shr = pd.DataFrame((2.584*10**(-5))*rain**2 + 0.0002*rain +0.0925)[ag_rain.index.isin(som_doy['slope'])] #SHRUB
som_savi_shr = pd.DataFrame(0.0007*rain +0.098)#SHRUB
som_savi_shr.columns=['SOM-SAVI']

#%%

df_som_gr = pd.concat([som_savi_gr['SOM-SAVI'],rain['94D']],axis=1) #change all the 3s 
df_som_shr = pd.concat([som_savi_shr['SOM-SAVI'],rain['94D']],axis=1) #change all the 3s in P 
#%%
df_som_shr=df_som_shr.merge(ci_som_shr, left_on='94D', right_on='P')[['SOM-SAVI', 'ci','94D']]
df_som_gr=df_som_gr.merge(ci_som_gr, left_on='94D', right_on='P')[['SOM-SAVI', 'ci','94D']]

a=np.asarray(df_som_gr['SOM-SAVI'])
b=np.asarray(df_som_gr['ci'])
c=np.asarray(df_som_shr['SOM-SAVI'])
d=np.asarray(df_som_shr['ci'])

N=len(df_som_shr.index)
distr_som_gr=np.zeros([N, 100])
distr_som_shr=np.zeros([N, 100])

for t in range(N): 
    
    distr_som_gr[t] = np.random.normal(a[t],b[t],100)
    distr_som_shr[t] = np.random.normal(c[t],d[t],100)


distr_som_gr=pd.DataFrame(distr_som_gr)
distr_som_gr = distr_som_gr.set_index(som_savi_gr.index)
#distr_som_gr=distr_som_gr.iloc[:,0:100].clip(lower=0.08)

distr_som_shr=pd.DataFrame(distr_som_shr)
distr_som_shr = distr_som_shr.set_index(som_savi_shr.index)
#distr_som_shr=distr_som_shr.iloc[:,0:100].clip(lower=0.08)

#%%

file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/ci_sob_gr1.csv"

ci_sob_gr= pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
ci_sob_gr.columns=['ci']
ci_sob_gr['P'] = np.linspace(0, 100, 100).round(0)
ci_sob_gr=ci_sob_gr.drop_duplicates(subset=['P'])

file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/ci_sob_shr1.csv"

ci_sob_shr= pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
ci_sob_shr.columns=['ci']
ci_sob_shr['P'] = np.linspace(0, 100, 220).round(0)
ci_sob_shr=ci_sob_shr.drop_duplicates(subset=['P'])
#
rain=pd.DataFrame(ag_rain['94D']).loc[ag_rain['DOWY'] == 173].round(0) #change date if needed for CCS

sob_gr = pd.DataFrame(0.001*rain +0.063 )#Grass

sob_gr.columns=['SOB-SAVI']


sob_shr = pd.DataFrame(0.0007*rain +0.07) #Shrub
sob_shr.columns=['SOB-SAVI']

#%%

df_sob_gr = pd.concat([sob_gr['SOB-SAVI'],rain['94D']],axis=1) #and change 13 to 14
df_sob_shr = pd.concat([sob_shr['SOB-SAVI'],rain['94D']],axis=1) #changed value 13 to 14
#%%
df_sob_gr=df_sob_gr.merge(ci_sob_gr, left_on='94D', right_on='P')[['SOB-SAVI', 'ci','94D']]
df_sob_shr=df_sob_shr.merge(ci_sob_shr, left_on='94D', right_on='P')[['SOB-SAVI', 'ci','94D']]

a=np.asarray(df_sob_gr['SOB-SAVI'])
b=np.asarray(df_sob_gr['ci'])
c=np.asarray(df_sob_shr['SOB-SAVI'])
d=np.asarray(df_sob_shr['ci'])

N=len(df_sob_gr.index)
distr_sob_gr=np.zeros([N, 100])
distr_sob_shr=np.zeros([N, 100])

for t in range(N): 
    
    distr_sob_gr[t] = np.random.normal(a[t],b[t],100)
    distr_sob_shr[t] = np.random.normal(c[t],d[t],100)


distr_sob_gr=pd.DataFrame(distr_sob_gr)
distr_sob_gr = distr_sob_gr.set_index(sob_gr.index)
distr_sob_shr=pd.DataFrame(distr_sob_shr)
distr_sob_shr = distr_sob_shr.set_index(sob_shr.index)
distr_sob_shr=distr_sob_shr.iloc[:,0:100].clip(lower=0.08)
distr_sob_shr=distr_sob_shr.iloc[:,0:100].clip(upper=0.13)

#distr_som_gr = distr_sob_gr.iloc[:,0:100]
#distr_som_gr=distr_som_gr.set_index(som_doy['slope'])
#%%
##distr_som_shr = distr_sob_shr.iloc[:,0:100]
#distr_som_shr=distr_som_shr.set_index(som_doy['slope'])

#to find the min of shrub and grass of the validated series
#%%MIN SAVI VALUE
min_savi_gr=pd.DataFrame(np.random.normal(0.08,0.04,size=[9,100])) #GRASS
#min_savi.columns=['MIN-SAVI']
min_savi_gr = min_savi_gr.set_index(pd.DatetimeIndex(temp_min_doy.index))
min_savi_gr['Date'] =min_savi_gr.index
min_savi_gr['WY'] = min_savi_gr.apply(lambda x: assign_wy(x), axis=1)
min_savi_gr['DOWY'] = min_savi_gr.index.to_series().apply(day_of_water_year)

#
min_savi_shr = pd.DataFrame(np.random.normal(0.09, 0.04, size=[9,100])) #SHRUB
#min_savi.columns=['MIN-SAVI']
min_savi_shr = min_savi_shr.set_index(pd.DatetimeIndex(temp_min_doy.index))
min_savi_shr['Date'] =min_savi_shr.index
min_savi_shr['WY'] = min_savi_shr.apply(lambda x: assign_wy(x), axis=1)
min_savi_shr['DOWY'] = min_savi_shr.index.to_series().apply(day_of_water_year)

#%%
artifical_gr_ccs=pd.concat([distr_max_gr.iloc[:,0:100], distr_eos_gr.iloc[:,0:100],
                            distr_sob_gr,distr_sos_gr.iloc[:,0:100],distr_som_gr,min_savi_gr.iloc[:,0:100]])
#
artifical_gr_ccs=artifical_gr_ccs.sort_index(ascending=True)
artifical_gr_ccs=artifical_gr_ccs.iloc[:,0:100].clip(lower=0.05)
artifical_gr_ccs['Date'] =artifical_gr_ccs.index
artifical_gr_ccs['WY'] = artifical_gr_ccs.apply(lambda x: assign_wy(x), axis=1)
artifical_gr_ccs['DOWY'] = artifical_gr_ccs.index.to_series().apply(day_of_water_year)
artifical_gr_ccs['median'] = artifical_gr_ccs.iloc[:,0:100].median(axis=1)
#artifical_gr=artifical_gr.loc[:'2019-10-31']
#artifical_gr = artifical_gr.loc[(artifical_gr.index < '2011-11-01') | (artifical_gr.index > '2013-11-01')]

artifical_shr_ccs=pd.concat([distr_max_shr.iloc[:,0:100],distr_eos_shr.iloc[:,0:100],
                             distr_sob_shr, distr_sos_shr.iloc[:,0:100], distr_som_shr,min_savi_shr.iloc[:,0:100]])
#
artifical_shr_ccs=artifical_shr_ccs.sort_index(ascending=True)
artifical_shr_ccs=artifical_shr_ccs.iloc[:,0:100].clip(lower=0.05)
artifical_shr_ccs['Date'] =artifical_shr_ccs.index
artifical_shr_ccs['WY'] = artifical_shr_ccs.apply(lambda x: assign_wy(x), axis=1)
artifical_shr_ccs['DOWY'] = artifical_shr_ccs.index.to_series().apply(day_of_water_year)
artifical_shr_ccs['median'] = artifical_shr_ccs.iloc[:,0:100].median(axis=1)

ag_art_gr_ccs = pd.pivot_table(artifical_gr_ccs, index=artifical_gr_ccs['DOWY'], columns=artifical_gr_ccs['WY'],values='median')
ag_art_shr_ccs = pd.pivot_table(artifical_shr_ccs, index=artifical_shr_ccs['DOWY'], columns=artifical_shr_ccs['WY'],values='median')

#%% Historical 
ag_shr=pd.pivot_table(ag_shr, index=ag_shr['DOWY'], columns=ag_shr['WY'],values='median')
ag_gr=pd.pivot_table(ag_gr, index=ag_gr['DOWY'], columns=ag_gr['WY'],values='median')

#%%
xx = np.linspace(1, 366, 366)
#%%
a=ag_art_gr_ccs
#%%
a=ag_art_shr_ccs
#%%
year=1996
x=np.asarray(a[year].dropna().index)
y=np.asarray(a[year].dropna())
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod0=pd.DataFrame(set_interp(xx)) #1996

#%%
year=1998
x=np.asarray(a[year].dropna().index)
y=np.asarray(a[year].dropna())
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod1=pd.DataFrame(set_interp(xx)) #1996
#%%
year=2002
x=np.asarray(a[year].dropna().index)
y=np.asarray(a[year].dropna())
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod2=pd.DataFrame(set_interp(xx)) #1996

#%%
year=2006
x=np.asarray(a[year].dropna().index)
y=np.asarray(a[year].dropna())
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod4=pd.DataFrame(set_interp(xx)) #1996

#%%
year=2008
x=np.asarray(a[year].dropna().index)
y=np.asarray(a[year].dropna())
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod5=pd.DataFrame(set_interp(xx)) #1996

#%%
year=2010
x=np.asarray(a[year].dropna().index)
y=np.asarray(a[year].dropna())
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod6=pd.DataFrame(set_interp(xx)) #1996

#%%
#year=2012
#x=np.asarray(a[year].dropna().index)
#y=np.asarray(a[year].dropna())
#set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
#mod7=pd.DataFrame(set_interp(xx)) #1996

#%%
year=2014
x=np.asarray(a[year].dropna().index)
y=np.asarray(a[year].dropna())
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod8=pd.DataFrame(set_interp(xx)) #1996

#%%
year=2016
x=np.asarray(a[year].dropna().index)
y=np.asarray(a[year].dropna())
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod9=pd.DataFrame(set_interp(xx)) #1996
#%%
year=2018
x=np.asarray(a[year].dropna().index)
y=np.asarray(a[year].dropna())
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod10=pd.DataFrame(set_interp(xx)) #1996

#%%
art_gr_ccs = pd.concat([mod0,mod1, mod2,mod4,mod5,mod6,mod8,mod9,mod10],axis=1)
art_gr_ccs.columns=['1996','1998','2002','2006','2008','2010','2014','2016','2018']

#%%
art_shr_ccs = pd.concat([mod0,mod1, mod2,mod4,mod5,mod6, mod8,mod9,mod10],axis=1)
art_shr_ccs.columns=['1996','1998','2002','2006','2008','2010','2014','2016','2018']

#%%

af_gr_ccs=art_gr_ccs.mean(axis=1)
af_gr_ccs_std=art_gr_ccs.std(axis=1)

af_shr_ccs=art_shr_ccs.mean(axis=1)
af_shr_ccs_std=art_shr_ccs.std(axis=1)
#%%
#CCS   syntehtic SAVI
savi_grass_ccs=art_gr_ccs
savi_grass_ccs.drop(savi_grass_ccs.tail(1).index,inplace=True)
savi_grass_ccs= savi_grass_ccs.melt()
savi_grass_ccs = savi_grass_ccs.set_index(pd.DatetimeIndex(ag_rain['Date']))  
savi_grass_ccs.columns=['Year','SAVI']

savi_shrub_ccs=art_shr_ccs
savi_shrub_ccs.drop(savi_shrub_ccs.tail(1).index,inplace=True)
savi_shrub_ccs= savi_shrub_ccs.melt()
savi_shrub_ccs = savi_shrub_ccs.set_index(pd.DatetimeIndex(ag_rain['Date']))  
savi_shrub_ccs.columns=['Year','SAVI']
#%% MEAN SAVI VALUES PLOT
plt.subplot(421)
plt.plot(np.asarray(ar['P']),color='blue',label='Historical')
plt.plot(np.asarray(ar1['P']),color='red',label='CCS')
plt.grid(alpha=0.2)
plt.ylabel('Precipitation (mm)')
plt.subplot(422)
plt.plot(dd.mean(axis=1),color='red',label='CCS')
plt.plot(dc.mean(axis=1),color='blue',label='Historical ')
plt.legend(frameon=False)
plt.grid(alpha=0.2)
plt.ylabel('T (°C)')
plt.subplot(423)
plt.plot(np.asarray(savi_grass['SAVI']),color='blue', label='Historical')
plt.plot(np.asarray(savi_grass_ccs['SAVI']),color='red',label='CCS')
plt.ylabel('SAVI')
plt.grid(alpha=0.2)
plt.legend(frameon=False)
plt.subplot(424)
plt.plot(np.asarray(savi_shrub['SAVI']),color='blue',label='Historical')
plt.plot(np.asarray(savi_shrub_ccs['SAVI']),color='red',label='CCS')
plt.grid(alpha=0.2)
plt.subplot(425)
plt.plot(af_gr,color='blue',label='Historical') #Modelled 
plt.fill_between(af_gr.index,af_gr-af_gr_std, af_gr+af_gr_std,  color='grey',alpha=0.2, edgecolor='none') #Modelled std
plt.plot(af_gr_ccs,color='red',label='CCS') #historic 
plt.fill_between(af_gr_ccs.index,af_gr_ccs-af_gr_ccs_std, af_gr_ccs+af_gr_ccs_std,  color='red',alpha=0.2, edgecolor='none') #historic std
plt.grid(alpha=0.2)
plt.ylim(0.04,0.27)
plt.ylabel('SAVI')
plt.legend(frameon=False)
plt.subplot(426)
plt.plot(af_shr,color='blue',label='Historical')
plt.fill_between(af_shr.index,af_shr-af_shr_std, af_shr+af_shr_std,  color='grey',alpha=0.2, edgecolor='none')
plt.plot(af_shr_ccs,color='red',label='CCS')
plt.fill_between(af_shr_ccs.index,af_shr_ccs-af_shr_ccs_std, af_shr_ccs+af_shr_ccs_std,  color='red',alpha=0.2, edgecolor='none') #historic std
plt.ylim(0.04,0.27)
plt.ylabel('SAVI')
plt.legend(frameon=False)
plt.grid(alpha=0.2)
plt.subplot(427)
sns.kdeplot(savi_grass['SAVI'],color='blue',label='Historical')
sns.kdeplot(savi_grass_ccs['SAVI'],color='red', label='CCS')
plt.legend(frameon=False)
plt.subplot(428)
sns.kdeplot(savi_shrub['SAVI'],color='blue',label='Historical')
sns.kdeplot(savi_shrub_ccs['SAVI'],color='red',label='CCS')
plt.legend(frameon=False)

#%%STATISTICS
months=[3,4,5,6,7,8,9]
x1= savi_grass[savi_grass.index.map(lambda t: t.month in months)]
x2=savi_grass_ccs[savi_grass_ccs.index.map(lambda t: t.month in months)]
x3= ag_rain['31D'][ag_rain.index.map(lambda t: t.month in months)]

print(stats.ks_2samp(x1['SAVI'],x2['SAVI'])) #p=0.21, s=0.19

stats.spearmanr(x2['SAVI'],x3)
plt.scatter(x2['SAVI'],x3)
#%%
months=[3,4,5,6,7,8,9]
x1= savi_shrub[savi_shrub.index.map(lambda t: t.month in months)]
x2=savi_shrub_ccs[savi_shrub_ccs.index.map(lambda t: t.month in months)]
x3= ag_rain['31D'][ag_rain.index.map(lambda t: t.month in months)]
stats.spearmanr(x2['SAVI'],x3)

print(stats.ks_2samp(x1['SAVI'],x2['SAVI'])) #p=0.21, s=0.19
sns.kdeplot(x1['SAVI'])
sns.kdeplot(x2['SAVI'])


#%%

x1= pd.DataFrame(savi_grass['SAVI'].groupby(savi_grass.Year).max()).round(2)
x2= pd.DataFrame(savi_grass_ccs['SAVI'].groupby(savi_grass_ccs.Year).max()).round(2)

tst=(x2['SAVI']*100) / x1['SAVI']
tst=100-tst
tst.mean()
stats.ks_2samp(x1['SAVI'],x2['SAVI'])
#%%
predictions = np.asarray(savi_shrub_ccs['SAVI'])
targets=np.asarray(savi_shrub['SAVI'])
NSE = (1-(np.sum((predictions-targets)**2)/np.sum((targets-np.mean(targets))**2)))
print(NSE)  






