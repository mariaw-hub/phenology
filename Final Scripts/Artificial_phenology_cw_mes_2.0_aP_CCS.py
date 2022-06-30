#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 12:41:47 2022

@author: mariawarter
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 11:29:49 2021

@author: mariawarter
"""
Run the original aaP model first to get reference synthetic series, then run this model to get CCS series
check df_max for weird values/doubles 
check artifical_ccs median and minimum if anything looks off. 

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from scipy import stats
from scipy import interpolate
from scipy.stats import iqr
import seaborn as sns
#%%
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
    return (x - water_year_start_date).days + 1

#%%
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/SAVI/landsat_1986_2021_edit/veg_cottonwood_savi_cloud_na.csv"
#%%
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/SAVI/landsat_1986_2021_edit/veg_mesquite_savi_cloud_na.csv"
#%%
SAVI = pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0],sep=',')
SAVI['date'] = pd.to_datetime(SAVI['date'],dayfirst=True)

SAVI = SAVI.set_index(pd.DatetimeIndex(SAVI['date']))  
SAVI = SAVI.drop(['date'], axis=1)
SAVI=SAVI /10000
#SAVI=SAVI.loc['1992-01-01':'2019-10-31']
SAVI= SAVI.interpolate() #interpolating monthly values to daily

SAVI=SAVI.sort_index()
#%%
idx=pd.date_range('01-17-1986','12-19-2021',freq='D') #daily daterange
SAVI=SAVI.reindex(idx) #upsampling to daily values 
t= SAVI.loc[:'2011-11-01']
t=t.interpolate()
s=SAVI.loc['2013-10-31':]
s=s.interpolate()
p=SAVI.loc['2011-11-01':'2013-03-26']

SAVI=pd.concat([t,s],axis=0)
SAVI['Date'] = pd.to_datetime(SAVI.index)

SAVI['WY'] = SAVI.apply(lambda x: assign_wy(x), axis=1)

SAVI['DOWY'] = SAVI.index.to_series().apply(day_of_water_year)
 
#%%
ax_cw=SAVI[['1','2','Date','DOWY','WY']].loc['1994-01-01':'2021-12-31']
ax_cw['median'] =ax_cw.iloc[:,0:2].median(axis=1)
ax_cw['iqr']=iqr(ax_cw.iloc[:,0:2],axis=1)
ax_cw = ax_cw.loc[(ax_cw.index < '2011-11-01') | (ax_cw.index > '2013-10-31')]
ax_cw = ax_cw[ax_cw.DOWY != 366]
ax_cw=ax_cw[ax_cw['WY'].isin([1995,1997,1999,2001,2003,2005,2007,2009,2011,2015,2017,2019])]
#%%
ax_mes =SAVI[['1','26','Date','DOWY','WY']].loc['1994-01-01':'2021-12-31']

ax_mes['median'] =ax_mes.iloc[:,0:2].median(axis=1)
ax_mes['iqr']=iqr(ax_mes.iloc[:,0:2],axis=1)
ax_mes = ax_mes.loc[(ax_mes.index < '2011-11-01') | (ax_mes.index > '2013-10-31')]
ax_mes = ax_mes[ax_mes.DOWY != 366]
ax_mes=ax_mes[ax_mes['WY'].isin([1995,1997,1999,2001,2003,2005,2007,2009,2011,2015,2017,2019])]
#%%
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/dP_SP/ESRL_CPC_Global_1986_2021.csv"

data_P_SP = pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
data_P_SP = data_P_SP.set_index(pd.DatetimeIndex(data_P_SP['date']))  
data_P_SP.columns=['Date','P']
data_P_SP['Date'] = pd.to_datetime(data_P_SP.index) #set P data to 1986-2019
#data_P_SP=data_P_SP.loc[(data_P_SP.index < '2012-01-01') | (data_P_SP.index > '2012-12-31')]
#data_P_SP = data_P_SP.loc[:'2019-12-31'] #data only starts in 1986

data_P_SP=data_P_SP.interpolate()
data_P_SP['WY'] = data_P_SP.apply(lambda x: assign_wy(x), axis=1)
data_P_SP['DOWY'] = data_P_SP.index.to_series().apply(day_of_water_year)
data_P_SP['DOY'] = data_P_SP['Date'].dt.dayofyear

ax_rain=data_P_SP.loc['1994-11-01':]
ax_rain=ax_rain[ax_rain['WY'].isin([1995,1997,1999,2001,2003,2005,2007,2009,2011,2015,2017,2019])]

#cwp=pd.pivot_table(ax_savi_cw, index=ax_savi_cw['DOWY'], columns=ax_savi_cw['WY'],values='median')
dir = r'/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/dPET_SP'
files = [i for i in glob.glob('*.txt')] #go into folder in nav tab

data=[pd.read_csv(file,sep=',',header=None, engine='python')
      for file in files]
idx=pd.date_range('01-01-1981','12-31-2021',freq='D') #daily daterange

data_PET_SP = pd.concat(data,ignore_index=True)
data_PET_SP.columns=['PET']
data_PET_SP =data_PET_SP.set_index(idx)
data_PET_SP['Date'] = data_PET_SP.index #set PET data to 1986-2019
#data_PET_SP = data_PET_SP.loc['1985':]
#data_PET_SP=data_PET_SP.loc[(data_PET_SP.index < '2012-01-01') | (data_PET_SP.index > '2012-12-31')]
data_PET_SP['WY'] = data_PET_SP.apply(lambda x: assign_wy(x), axis=1)
data_PET_SP['DOWY'] = data_PET_SP.index.to_series().apply(day_of_water_year)
ax_pet=data_PET_SP.loc['1994-11-01':]
ax_pet=ax_pet[ax_pet['WY'].isin([1995,1997,1999,2001,2003,2005,2007,2009,2011,2015,2017,2019])]

#%%
aP = pd.DataFrame(ax_rain['P'] - ax_pet['PET'])
aP.columns=['aP']
aP[aP < 0] = 0
aP['Date']=aP.index
#aP=aP.loc['1994':]
aP['WY'] = aP.apply(lambda x: assign_wy(x), axis=1)
aP['DOWY'] = aP.index.to_series().apply(day_of_water_year)
ax_aP1=aP[aP['WY'].isin([1995,1997,1999,2001,2003,2005,2007,2009,2011,2015,2017,2019])]
#%%
months=[5,6,7,8,9]
aP1= aP[['aP']][aP['Date'].map(lambda t: t.month in months)]

aP1=aP1*0.75

months=[1,2,3,4,10,11,12]
aP2= aP[['aP']][aP['Date'].map(lambda t: t.month in months)]


ax_aP2=pd.concat([aP1,aP2],axis=0)
ax_aP2=ax_aP2.sort_index()
ax_aP2['Date'] = ax_aP2.index
ax_aP2['WY'] = ax_aP2.apply(lambda x: assign_wy(x), axis=1)
ax_aP2['DOWY'] = ax_aP2.index.to_series().apply(day_of_water_year)


#%%

file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/ax_temp_all.csv"
ax_temp= pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
ax_temp['Date'] =ax_temp.index
ax_temp['WY'] = ax_temp.apply(lambda x: assign_wy(x), axis=1)
ax_temp['DOWY'] = ax_temp.index.to_series().apply(day_of_water_year)
#%%
ax_temp=ax_temp[ax_temp['WY'].isin([1995,1997,1999,2001,2003,2005,2007,2009,2011,2015,2017,2019])]
axt2=ax_temp
#%%

ax_temp['Date1']= ax_temp['Date'] - timedelta(days=20)
ax_temp.index=ax_temp['Date1']
ax_temp=ax_temp.loc['1994-11-01':'2020-10-31']
ax_temp['Date'] =ax_temp.index
ax_temp['WY'] = ax_temp.apply(lambda x: assign_wy(x), axis=1)
ax_temp['DOWY'] = ax_temp.index.to_series().apply(day_of_water_year)

ax_temp=ax_temp[ax_temp['WY'].isin([1995,1997,1999,2001,2003,2005,2007,2009,2011,2015,2017,2019])]
axt1=ax_temp
#%
dc=pd.pivot_table(axt2, index=axt2['DOWY'], columns=axt2['WY'], values='96D')
dc.mean(axis=1).plot()

dd=pd.pivot_table(axt1, index=axt1['DOWY'], columns=axt1['WY'], values='96D')
dd.mean(axis=1).plot()

#%%
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/ant_aP.csv"
ax_aP= pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
#ax_rain=ax_rain.loc[:'2017-10-31']
ax_aP=ax_aP[ax_aP['WY'].isin([1995,1997,1999,2001,2003,2005,2007,2009,2011,2015,2017,2019])]
ax_aP1=ax_aP
#%%

ax_aP = pd.DataFrame(ax_aP['aP'] *0.75 )
ax_aP['Date'] =ax_aP.index
ax_aP['WY'] = ax_aP.apply(lambda x: assign_wy(x), axis=1)
ax_aP['DOWY'] = ax_aP.index.to_series().apply(day_of_water_year)
#ax_aP2=ax_aP
#%%%

#%% DIFF T and GW for timing of min and max SAVI

temp_diff= np.diff(ax_temp['96D'])  
temp_diff=pd.DataFrame(temp_diff)
temp_diff.columns=['slope']

temp_diff=temp_diff.set_index(ax_temp[:-1].index)
temp_diff['Date'] =temp_diff.index
temp_diff['WY'] = temp_diff.apply(lambda x: assign_wy(x), axis=1)
temp_diff['DOWY'] = temp_diff.index.to_series().apply(day_of_water_year)

m_temp=pd.pivot_table(temp_diff, index=temp_diff['DOWY'], columns=temp_diff['WY'],values='slope')
m_temp=m_temp.dropna()
c=m_temp.mean(axis=1) #check for odd values 

#%%
#TIMING OF MAX SAVI
max_doy = pd.DataFrame(temp_diff.groupby(temp_diff['WY']).cumsum())
max_doy['Date'] = max_doy.index
max_doy['WY'] = max_doy.apply(lambda x: assign_wy(x), axis=1)
max_doy= pd.DataFrame(max_doy.groupby(max_doy['WY'])['slope'].idxmax()) 
max_doy = max_doy.set_index(pd.DatetimeIndex(max_doy.slope))
max_doy = max_doy.set_index(max_doy['slope'])
#max_doy=pd.DataFrame(max_doy['slope'] + timedelta(days=30))

# antecedent aP at the time of max doy (max doy from temperature)

max_aP = pd.DataFrame(ax_aP[ax_aP.index.isin(max_doy.slope)])
max_aP=max_aP.round(0)     
          
#%%CONFIDENCE INTERAVALS for aP 


file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/ci_maxsavi_cw_aP.csv"

ci_max_cw= pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
ci_max_cw.columns=['ci']
ci_max_cw['aP'] = pd.Series(np.linspace(0, 150, 150)).round(0)
ci_max_cw=ci_max_cw.drop_duplicates(subset=['aP'])


file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/ci_maxsavi_mes_aP.csv"

ci_max_mes= pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
ci_max_mes.columns=['ci']
ci_max_mes['aP'] = pd.Series(np.linspace(0,160,160)).round(0)
ci_max_mes=ci_max_mes.drop_duplicates(subset=['aP'])

#%%
#Calculate the max savi values 
max_savi_cw = pd.DataFrame(0.0003*max_aP['aP'] + 0.30) #Cottonwood - calcualte with relative difference not actual values
max_savi_cw.columns=['MAX-SAVI']
#max_savi_cw=max_savi_cw.round(1)

max_savi_mes = pd.DataFrame(0.0005*max_aP['aP'] + 0.31) #Mesquite 
max_savi_mes.columns=['MAX-SAVI']
#max_savi_mes=max_savi_mes.round(1)
#%%
#print((m_temp.mean(axis=1)).idxmax()) #change 14 days under CCS

#capture extended peak for cottonwood 
ms=c.idxmax() 

max_start =temp_diff['Date'].loc[temp_diff['DOWY'] == 230]


#%%
#stochastic Monte Carlo simulation to get distribution of SAVI
df_max_cw = pd.concat([max_savi_cw['MAX-SAVI'],max_aP['aP']],axis=1) #36 and 26 change 
df_max_mes = pd.concat([max_savi_mes['MAX-SAVI'],max_aP['aP']],axis=1)

#%%
df_max_cw=df_max_cw.merge(ci_max_cw, left_on='aP', right_on='aP')[['MAX-SAVI', 'ci','aP']]
df_max_mes=df_max_mes.merge(ci_max_mes, left_on='aP', right_on='aP')[['MAX-SAVI', 'ci','aP']]

a=np.asarray(df_max_cw['MAX-SAVI'])
b=np.asarray(df_max_cw['ci'])
c=np.asarray(df_max_mes['MAX-SAVI'])
d=np.asarray(df_max_mes['ci'])

N=len(max_aP.index)
distr_max_cw=np.zeros([N, 100])
distr_max_mes=np.zeros([N, 100])

for t in range(N): 
    
    distr_max_cw[t] = np.random.normal(a[t],b[t],100).round(2)
    distr_max_mes[t] = np.random.normal(c[t],d[t],100).round(2)


distr_max_cw=pd.DataFrame(distr_max_cw)
distr_max_cw= distr_max_cw.set_index(pd.DatetimeIndex(max_doy.slope))
#distr_max_cw=distr_max_cw.iloc[:,0:100].clip(lower=0.2)

distr_max_mes=pd.DataFrame(distr_max_mes)
distr_max_mes = distr_max_mes.set_index(max_doy.slope)
#distr_max_mes=distr_max_mes.iloc[:,0:100].clip(lower=0.15)
#
distr_max_cw_start= distr_max_cw
distr_max_cw_start=distr_max_cw_start.set_index(pd.DatetimeIndex(max_start.index))

#%% Timing of the start of green up linked to temperature 
#value from random distribution COTTONWOOD
temp_diff= np.diff(ax_temp['96D'])  
temp_diff=pd.DataFrame(temp_diff)
temp_diff.columns=['slope']

temp_diff=temp_diff.set_index(ax_temp[:-1].index)
temp_diff['Date'] =temp_diff.index
temp_diff['WY'] = temp_diff.apply(lambda x: assign_wy(x), axis=1)
temp_diff['DOWY'] = temp_diff.index.to_series().apply(day_of_water_year)

#
#Timing of the start of green up  for COTTONWOOD 
sog_doy = pd.DataFrame(temp_diff.groupby(temp_diff['WY']).cumsum())
sog_doy['Date'] = sog_doy.index
sog_doy['WY'] = sog_doy.apply(lambda x: assign_wy(x), axis=1)
sog_doy= pd.DataFrame(sog_doy.groupby(sog_doy['WY']) ['slope'].idxmin()) 
sog_doy = sog_doy.set_index(pd.DatetimeIndex(sog_doy.slope))
#%%
#Cottonwood minimum SAVI from distribution 
min_savi_cw = pd.DataFrame(np.random.normal(0.08,0.1,size=[12,100])) 
min_savi_cw = min_savi_cw.set_index(pd.DatetimeIndex(sog_doy.slope))
min_savi_cw['Date'] =min_savi_cw.index
min_savi_cw['WY'] = min_savi_cw.apply(lambda x: assign_wy(x), axis=1)
min_savi_cw['DOWY'] = min_savi_cw.index.to_series().apply(day_of_water_year)
#min_savi_cw=min_savi_cw.iloc[:,0:100].clip(lower=0.08)
#min_savi_mes=min_savi_mes.iloc[:,0:100].clip(upper=0.12)

#%%
#Mesquite
temp_diff= np.diff(ax_temp['64D'])  
temp_diff=pd.DataFrame(temp_diff)
temp_diff.columns=['slope']

temp_diff=temp_diff.set_index(ax_temp[:-1].index)
temp_diff['Date'] =temp_diff.index
temp_diff['WY'] = temp_diff.apply(lambda x: assign_wy(x), axis=1)
temp_diff['DOWY'] = temp_diff.index.to_series().apply(day_of_water_year)


#Timing 
sog_doy = pd.DataFrame(temp_diff.groupby(temp_diff['WY']).cumsum())
sog_doy['Date'] = sog_doy.index
sog_doy['WY'] = sog_doy.apply(lambda x: assign_wy(x), axis=1)
sog_doy= pd.DataFrame(sog_doy.groupby(sog_doy['WY'])['slope'].idxmin()) 
sog_doy = sog_doy.set_index(pd.DatetimeIndex(sog_doy.slope))
#%%
#Minimum SAVI Mesquite from distribution 
min_savi_mes = pd.DataFrame(np.random.normal(loc=0.05,scale=0.1,size=[12,100]))
min_savi_mes = min_savi_mes.set_index(pd.DatetimeIndex(sog_doy.slope))
min_savi_mes['Date'] =min_savi_mes.index
min_savi_mes['Date']= min_savi_mes['Date'] + timedelta(days=30)
min_savi_mes = min_savi_mes.set_index(pd.DatetimeIndex(min_savi_mes['Date']))
min_savi_mes['WY'] = min_savi_mes.apply(lambda x: assign_wy(x), axis=1)
min_savi_mes['DOWY'] = min_savi_mes.index.to_series().apply(day_of_water_year)
#min_savi_mes=min_savi_mes.iloc[:,0:100].clip(lower=0.01)
#min_savi_mes=min_savi_mes.iloc[:,0:100].clip(upper=0.12)

#%% End of season from distribution 
temp_eos=ax_temp.loc[ax_temp['DOWY'] == 365] 

eos_savi = pd.DataFrame(np.random.normal(loc=0.22, scale=0.09, size=[12,100])) #COTTONWOOD
distr_eos_cw = eos_savi.set_index(pd.DatetimeIndex(temp_eos['Date']))

eos_savi = pd.DataFrame(np.random.normal(loc=0.21, scale=0.09, size=[12,100])) #MESQUITE
distr_eos_mes = eos_savi.set_index(pd.DatetimeIndex(temp_eos['Date']))

#START OF SEASON from distribution 
temp_sos=ax_temp.loc[ax_temp['DOWY'] == 1] 

sos_savi = pd.DataFrame(np.random.normal(loc=0.22, scale=0.09, size=[12,100]))
distr_sos_cw = sos_savi.set_index(pd.DatetimeIndex(temp_sos['Date']))

sos_savi = pd.DataFrame(np.random.normal(loc=0.22, scale=0.09, size=[12,100]))
distr_sos_mes = sos_savi.set_index(pd.DatetimeIndex(temp_sos['Date']))

#%% Create dataframe with all events 

artifical_cw_ccs=pd.concat([distr_max_cw,distr_max_cw_start,distr_eos_cw,distr_sos_cw,min_savi_cw.iloc[:,0:100]]).round(2)
artifical_cw_ccs=artifical_cw_ccs.sort_index(ascending=True)
artifical_cw_ccs=pd.DataFrame(artifical_cw_ccs)
#artifical_cw=artifical_cw.iloc[:,0:100].clip(lower=0.1)
artifical_cw_ccs['Date'] =artifical_cw_ccs.index
artifical_cw_ccs['WY'] = artifical_cw_ccs.apply(lambda x: assign_wy(x), axis=1)
artifical_cw_ccs['DOWY'] = artifical_cw_ccs.index.to_series().apply(day_of_water_year)
artifical_cw_ccs['median'] = artifical_cw_ccs.iloc[:,0:100][artifical_cw_ccs.iloc[:,0:100] >= 0].median(axis=1)

#%% Create dataframe with all events 
artifical_mes_ccs=pd.concat([distr_max_mes,distr_eos_mes, distr_sos_mes,min_savi_mes.iloc[:,0:100]]).round(2)

artifical_mes_ccs=artifical_mes_ccs.sort_index(ascending=True)
artifical_mes_ccs=pd.DataFrame(artifical_mes_ccs)
#artifical_mes=artifical_mes.iloc[:,0:100].clip(lower=0.1)
artifical_mes_ccs['Date'] =artifical_mes_ccs.index
artifical_mes_ccs['WY'] = artifical_mes_ccs.apply(lambda x: assign_wy(x), axis=1)
artifical_mes_ccs['DOWY'] = artifical_mes_ccs.index.to_series().apply(day_of_water_year)
artifical_mes_ccs['median'] = artifical_mes_ccs.iloc[:,0:100][artifical_mes_ccs.iloc[:,0:100] >= 0].median(axis=1)

#%%
ag_art_cw_ccs = pd.pivot_table(artifical_cw_ccs, index=artifical_cw_ccs['DOWY'], columns=artifical_cw_ccs['WY'],values='median')
ag_art_cw_ccs.interpolate(method='spline',s=0, order=2, inplace=True)
#%%
ag_cw=ax_cw#.loc[:'2011-10-31']
ag_cw=pd.pivot_table(ag_cw, index=ag_cw['DOWY'], columns=ag_cw['WY'],values='median')
#ag_cw=ag_cw.dropna()
#ag_art_cw=ag_art_cw.clip(lower=0.12)

#%%
ag_art_mes_ccs = pd.pivot_table(artifical_mes_ccs, index=artifical_mes_ccs['DOWY'], columns=artifical_mes_ccs['WY'],values='median')
ag_art_mes_ccs.interpolate(method='spline', order=2, s=0, inplace=True)
#%%
ag_mes=ax_mes#.loc[:'2011-10-31']
ag_mes=pd.pivot_table(ag_mes, index=ag_mes['DOWY'], columns=ag_mes['WY'],values='median')
#ag_mes=ag_mes.dropna()
#ag_art_mes=ag_art_mes.clip(lower=0.115)

#%% Multivaraite spline interpolation between events 

#run all years first with a=cottonwood (2a) then reset a=mesquite (2b) 
#%% LOOP to interpolate all years. 
xx = np.linspace(1, 366, 366) #1
#%%
a=ag_art_cw_ccs #2a
#%%
a=ag_art_mes_ccs #2b
#%%
year=1995
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod2=pd.DataFrame(set_interp(xx)) #2003

#%%
year=1997
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod3=pd.DataFrame(set_interp(xx)) #2003

#%%
year=1999
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod4=pd.DataFrame(set_interp(xx)) #2003

#%%
year=2001
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod5=pd.DataFrame(set_interp(xx)) #2003
#%%
year=2003
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod6=pd.DataFrame(set_interp(xx)) #2003
#%%
year=2005
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod7=pd.DataFrame(set_interp(xx)) #2003
#%%
year=2007
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod8=pd.DataFrame(set_interp(xx)) #2003
#%%
year=2009
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod9=pd.DataFrame(set_interp(xx)) #2003
#%%
year=2011
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod10=pd.DataFrame(set_interp(xx)) #2003
#%%
#year=2013
#x=np.asarray(a[year].index)
#y=np.asarray(a[year])
#set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
#mod11=pd.DataFrame(set_interp(xx)) #2003
#%%
year=2015
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod12=pd.DataFrame(set_interp(xx)) #2003
#%%
year=2017
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod13=pd.DataFrame(set_interp(xx)) #2003
#%%
year=2019
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod14=pd.DataFrame(set_interp(xx)) #2003
#%%

art_cw_ccs = pd.concat([ mod2,mod3,mod4,mod5,mod6,mod7,mod8,mod9,mod10,mod12,mod13,mod14],axis=1)
art_cw_ccs.columns=['1995','1997','1999','2001','2003','2005','2007','2009','2011','2015','2017','2019']

#%%
art_mes_ccs = pd.concat([mod2,mod3,mod4,mod5,mod6,mod7,mod8,mod9,mod10,mod12,mod13,mod14],axis=1)
art_mes_ccs.columns=['1995','1997','1999','2001','2003','2005','2007','2009','2011','2015','2017','2019'] 

#%%
#Synthetic ensemble mean over multiple years 
af_cw_ccs=art_cw_ccs.mean(axis=1)
af_cw_ccs_std=art_cw_ccs.std(axis=1)

af_mes_ccs=art_mes_ccs.mean(axis=1)
af_mes_ccs_std=art_mes_ccs.std(axis=1)
#%%
#Historic ensemble mean 
his_cw=ag_cw.mean(axis=1)
his_cw_std=ag_cw.std(axis=1)


his_mes=ag_mes.mean(axis=1)
his_mes_std=ag_mes.std(axis=1)
#%%

savi_cw_ccs= art_cw_ccs
savi_cw_ccs.drop(savi_cw_ccs.tail(1).index,inplace=True)
savi_cw_ccs= savi_cw_ccs.melt()
savi_cw_ccs = savi_cw_ccs.set_index(pd.DatetimeIndex(ax_cw['Date']))  
savi_cw_ccs.columns=['Year','SAVI']

savi_mes_ccs= art_mes_ccs
savi_mes_ccs.drop(savi_mes_ccs.tail(1).index,inplace=True)
savi_mes_ccs= savi_mes_ccs.melt()
savi_mes_ccs = savi_mes_ccs.set_index(pd.DatetimeIndex(ax_mes['Date']))  
savi_mes_ccs.columns=['Year','SAVI']
#%%
#savi_mes.to_csv('savi_mes_artifical.csv',sep=',')
#savi_cw.to_csv('savi_cw_artifical.csv',sep=',')
#%%

plt.subplot(421)
plt.plot(np.asarray(ax_aP1['aP']),color='blue',label='Historic')
plt.plot(np.asarray(ax_aP2['aP']),color='red',label='CCS')
plt.grid(alpha=0.2)
plt.ylabel('aaP (mm)')
plt.subplot(422)
plt.plot(dd.mean(axis=1),color='red',label='CCS')
plt.plot(dc.mean(axis=1),color='blue',label='Historical ')
plt.legend(frameon=False)
plt.grid(alpha=0.2)
plt.ylabel('T (°C)')
plt.subplot(423)
plt.plot(np.asarray(savi_cw['SAVI']),color='blue', label='Historic')
plt.plot(np.asarray(savi_cw_ccs['SAVI']),color='red',label='CCS')
plt.ylim(0.04,0.41)
plt.ylabel('SAVI')
plt.grid(alpha=0.2)
plt.legend(frameon=False)
plt.subplot(424)
plt.plot(np.asarray(savi_mes['SAVI']),color='blue',label='Historic')
plt.plot(np.asarray(savi_mes_ccs['SAVI']),color='red',label='CCS')
plt.ylim(0.04,0.41)
plt.grid(alpha=0.2)
plt.subplot(425)
plt.plot(af_cw,color='blue',label='Historic') #Modelled 
plt.fill_between(af_cw.index,af_cw-af_cw_std, af_cw+af_cw_std,  color='grey',alpha=0.2, edgecolor='none') #Modelled std
plt.plot(af_cw_ccs,color='red',label='CCS') #historic 
plt.fill_between(af_cw_ccs.index,af_cw_ccs-af_cw_ccs_std, af_cw_ccs+af_cw_ccs_std,  color='red',alpha=0.2, edgecolor='none') #historic std
plt.grid(alpha=0.2)
plt.ylim(0.04,0.41)
plt.ylabel('SAVI')
plt.legend(frameon=False)
plt.subplot(426)
plt.plot(af_mes,color='blue',label='Historic')
plt.fill_between(af_mes.index,af_mes-af_mes_std, af_mes+af_mes_std,  color='grey',alpha=0.2, edgecolor='none')
plt.plot(af_mes_ccs,color='red',label='CCS')
plt.fill_between(af_mes_ccs.index,af_mes_ccs-af_mes_std, af_mes_ccs+af_mes_ccs_std,  color='red',alpha=0.2, edgecolor='none') #historic std
plt.ylim(0.04,0.41)
plt.ylabel('SAVI')
plt.legend(frameon=False)
plt.grid(alpha=0.2)
plt.subplot(427)
sns.kdeplot(savi_cw['SAVI'],color='blue',label='Historic')
sns.kdeplot(savi_cw_ccs['SAVI'],color='red', label='CCS')
plt.ylim(0,8)
plt.legend(frameon=False)
plt.subplot(428)
sns.kdeplot(savi_mes['SAVI'],color='blue',label='Historic')
sns.kdeplot(savi_mes_ccs['SAVI'],color='red',label='CCS')
plt.legend(frameon=False)
plt.ylim(0,8)


#%% STATISTICS BETWEEN HISTORIC AND MODELLED SAVI
months=[3,4,5,6,7,8,9]
x1= savi_cw[savi_cw.index.map(lambda t: t.month in months)]
x2=savi_cw_ccs[savi_cw_ccs.index.map(lambda t: t.month in months)]
x3= ax_aP2['aP'][ax_aP2.index.map(lambda t: t.month in months)]

print(stats.ks_2samp(x1['SAVI'],x2['SAVI'])) #p=0.21, s=0.19

stats.spearmanr(x2['SAVI'],x3)
plt.scatter(x2['SAVI'],x3)
#%%
months=[3,4,5,6,7,8,9]
x1= savi_mes[savi_mes.index.map(lambda t: t.month in months)]
x2=savi_mes_ccs[savi_mes_ccs.index.map(lambda t: t.month in months)]
x3= ax_aP2['aP'][ax_aP2.index.map(lambda t: t.month in months)]

print(stats.ks_2samp(x1['SAVI'],x2['SAVI'])) #p=0.21, s=0.19
stats.spearmanr(x2['SAVI'],x3)

#%%
plt.subplot(211)
sns.kdeplot(x1['SAVI'],color='blue')
sns.kdeplot(x2['SAVI'],color='red')
plt.subplot(212)
sns.kdeplot(x1['SAVI'],color='blue')
sns.kdeplot(x2['SAVI'],color='red')
