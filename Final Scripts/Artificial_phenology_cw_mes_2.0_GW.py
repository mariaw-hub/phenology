#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 12:17:12 2022

@author: mariawarter
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 11:29:49 2021

@author: mariawarter
"""


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
    return (x - water_year_start_date).days + 1#%%
#%% Historic SAVI

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
t= SAVI.loc[:'2011-10-05']
t=t.interpolate()
s=SAVI.loc['2013-03-26':]
s=s.interpolate()
p=SAVI.loc['2011-10-06':'2013-03-26']

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
ax_cw=ax_cw[ax_cw['WY'].isin([2002,2003,2006,2008,2010,2014,2015,2016,2017])]

#%%
ax_mes =SAVI[['1','26','Date','DOWY','WY']].loc['1994-01-01':'2021-12-31']

ax_mes['median'] =ax_mes.iloc[:,0:2].median(axis=1)
ax_mes['iqr']=iqr(ax_mes.iloc[:,0:2],axis=1)
ax_mes = ax_mes.loc[(ax_mes.index < '2011-11-01') | (ax_mes.index > '2013-10-31')]
ax_mes = ax_mes[ax_mes.DOWY != 366]
ax_mes=ax_mes[ax_mes['WY'].isin([2002,2003,2006,2008,2010,2014,2015,2016,2017])]

#%%
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/GW/SanPedro_GW_86_19.csv"

data_GW_SP = pd.read_csv(file,dayfirst=True,parse_dates=True,sep=';')
cols=[0,1,3]
data_GW_SP=data_GW_SP.drop(data_GW_SP.columns[cols],axis=1)
data_GW_SP['datetime'] = pd.to_datetime(data_GW_SP['datetime'],dayfirst=True)

data_GW_SP =data_GW_SP.set_index(pd.DatetimeIndex(data_GW_SP['datetime'])) 

data_GW_SP= data_GW_SP.groupby('station_ID',as_index=True).agg(lambda x: x.tolist())

ax_gw=pd.DataFrame(data_GW_SP['water_depth_m'].loc['GS_Sp17']) #suitable for mesquite, center of the samplign zone
b=pd.DataFrame(data_GW_SP['datetime'].loc['GS_Sp17'])
ax_gw=ax_gw.set_index(b[0])
ax_gw.columns=['DTG17']
ax_gw=ax_gw.sort_index()
ax_gw['Date'] =ax_gw.index
ax_gw['WY'] = ax_gw.apply(lambda x: assign_wy(x), axis=1)
ax_gw['DOWY'] = ax_gw.index.to_series().apply(day_of_water_year)
ax_gw['DTG'] =pd.DataFrame((ax_gw['DTG17']- ax_gw['DTG17'].min())/(ax_gw['DTG17'].max()-ax_gw['DTG17'].min()))
#ax_gw=ax_gw[ax_gw['WY'].isin([2002,2006,2008,2010,2014,2016])]
ax_gw = ax_gw.loc[(ax_gw.index < '2011-11-01') | (ax_gw.index > '2013-10-31')]
#ax_gw=ax_gw[ax_gw['WY'].isin([1995,1997,1999,2001,2003,2005,2007,2009,2011,2015,2017,2019])]

#%%

file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/ax_temp_all.csv"
ax_temp= pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
ax_temp['Date'] =ax_temp.index
ax_temp['WY'] = ax_temp.apply(lambda x: assign_wy(x), axis=1)
ax_temp['DOWY'] = ax_temp.index.to_series().apply(day_of_water_year)
ax_temp=ax_temp[ax_temp['WY'].isin([2002,2003,2006,2008,2010,2014,2015,2016,2017])]
#%% DIFF T and GW for timing of min and max SAVI

temp_diff= np.diff(ax_temp['96D'])  
temp_diff=pd.DataFrame(temp_diff)
temp_diff.columns=['slope']

temp_diff=temp_diff.set_index(ax_temp[:-1].index)
temp_diff['Date'] =temp_diff.index
temp_diff['WY'] = temp_diff.apply(lambda x: assign_wy(x), axis=1)
temp_diff['DOWY'] = temp_diff.index.to_series().apply(day_of_water_year)


m_temp=pd.pivot_table(temp_diff, index=temp_diff['DOWY'], columns=temp_diff['WY'],values='slope')
#m_temp=m_temp.dropna()
c=m_temp.mean(axis=1) #check for odd values 
#TIMING OF MAX SAVI
max_doy = pd.DataFrame(temp_diff.groupby(temp_diff['WY']).cumsum())
max_doy['Date'] = max_doy.index
max_doy['WY'] = max_doy.apply(lambda x: assign_wy(x), axis=1)
max_doy= pd.DataFrame(max_doy.groupby(max_doy['WY'])['slope'].idxmax()) 
max_doy = max_doy.set_index(pd.DatetimeIndex(max_doy.slope))
max_doy = max_doy.set_index(max_doy['slope'])
# antecedent aP at the time of max doy (max doy from temperature)

max_dtg = pd.DataFrame(ax_gw['DTG'][ax_gw.index.isin(max_doy.slope)]).round(2)
#max_dtg=max_dtg.round(0)     
ms=c.idxmax() 
max_start =ax_temp['Date'].loc[ax_temp['DOWY'] == 230]
          
#%%CONFIDENCE INTERAVALS for aP 

file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/ci_maxsavi_cw_gw.csv"

ci_max_cw= pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
ci_max_cw.columns=['ci']
ci_max_cw['DTG'] = pd.Series(np.linspace(0, 1, 100)).round(2)
#ci_max_cw=ci_max_cw.drop_duplicates(subset=['DTG'])


file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/ci_maxsavi_mes_gw.csv"

ci_max_mes= pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
ci_max_mes.columns=['ci']
ci_max_mes['DTG'] = pd.Series(np.linspace(0,1,100)).round(2)
#ci_max_mes=ci_max_mes.drop_duplicates(subset=['DTG'])

#%%
#Calculate the max savi values 
max_savi_cw = pd.DataFrame(-0.356*max_dtg['DTG']+0.607) #Cottonwood - calcualte with relative difference not actual values
max_savi_cw.columns=['MAX-SAVI']
#max_savi_cw=max_savi_cw.round(1)

max_savi_mes = pd.DataFrame(-0.268*max_dtg['DTG']+0.558) #Mesquite 
max_savi_mes.columns=['MAX-SAVI']
#max_savi_mes=max_savi_mes.round(1)

#%%
#stochastic Monte Carlo simulation to get distribution of SAVI
df_max_cw = pd.concat([max_savi_cw['MAX-SAVI'],max_dtg['DTG']],axis=1) #36 and 26 change 
df_max_mes = pd.concat([max_savi_mes['MAX-SAVI'],max_dtg['DTG']],axis=1)
#if it doesnt work, check that WTD values are all rounded to 1 decimal point 
#%%
df_max_cw=df_max_cw.merge(ci_max_cw, left_on='DTG', right_on='DTG')[['MAX-SAVI', 'ci','DTG']]
df_max_mes=df_max_mes.merge(ci_max_mes, left_on='DTG', right_on='DTG')[['MAX-SAVI', 'ci','DTG']]

a=np.asarray(df_max_cw['MAX-SAVI'])
b=np.asarray(df_max_cw['ci'])
c=np.asarray(df_max_mes['MAX-SAVI'])
d=np.asarray(df_max_mes['ci'])

N=len(max_dtg.index)
distr_max_cw=np.zeros([N, 100])
distr_max_mes=np.zeros([N, 100])

for t in range(N): 
    
    distr_max_cw[t] = np.random.normal(a[t],b[t],100).round(2)
    distr_max_mes[t] = np.random.normal(c[t],d[t],100).round(2)


distr_max_cw=pd.DataFrame(distr_max_cw)
distr_max_cw= distr_max_cw.set_index(pd.DatetimeIndex(max_dtg.index))
#distr_max_cw=distr_max_cw.iloc[:,0:100].clip(lower=0.2)

distr_max_mes=pd.DataFrame(distr_max_mes)
distr_max_mes = distr_max_mes.set_index(max_dtg.index)
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

#%%
#Timing of the start of green up  for COTTONWOOD 
sog_doy = pd.DataFrame(temp_diff.groupby(temp_diff['WY']).cumsum())
sog_doy['Date'] = sog_doy.index
sog_doy['WY'] = sog_doy.apply(lambda x: assign_wy(x), axis=1)
sog_doy= pd.DataFrame(sog_doy.groupby(sog_doy['WY']) ['slope'].idxmin()) 
sog_doy = sog_doy.set_index(pd.DatetimeIndex(sog_doy.slope))
#
#Cottonwood minimum SAVI from distribution 
min_savi_cw = pd.DataFrame(np.random.normal(0.11,0.09,size=[9,100])) 
min_savi_cw = min_savi_cw.set_index(pd.DatetimeIndex(sog_doy.slope))
min_savi_cw['Date'] =min_savi_cw.index
min_savi_cw['WY'] = min_savi_cw.apply(lambda x: assign_wy(x), axis=1)
min_savi_cw['DOWY'] = min_savi_cw.index.to_series().apply(day_of_water_year)
min_savi_cw=min_savi_cw.iloc[:,0:100].clip(lower=0.08)
#%%
#Mesquite
temp_diff= np.diff(ax_temp['80D'])  
temp_diff=pd.DataFrame(temp_diff)
temp_diff.columns=['slope']

temp_diff=temp_diff.set_index(ax_temp[:-1].index)
temp_diff['Date'] =temp_diff.index
temp_diff['WY'] = temp_diff.apply(lambda x: assign_wy(x), axis=1)
temp_diff['DOWY'] = temp_diff.index.to_series().apply(day_of_water_year)

#%%
#Timing 
sog_doy = pd.DataFrame(temp_diff.groupby(temp_diff['WY']).cumsum())
sog_doy['Date'] = sog_doy.index
sog_doy['WY'] = sog_doy.apply(lambda x: assign_wy(x), axis=1)
sog_doy= pd.DataFrame(sog_doy.groupby(sog_doy['WY'])['slope'].idxmin()) 
sog_doy = sog_doy.set_index(pd.DatetimeIndex(sog_doy.slope))
#
#Minimum SAVI Mesquite from distribution 
min_savi_mes = pd.DataFrame(np.random.normal(0.09,0.097,size=[9,100]))
min_savi_mes = min_savi_mes.set_index(pd.DatetimeIndex(sog_doy.slope))
min_savi_mes['Date'] =min_savi_mes.index
min_savi_mes['Date']= min_savi_mes['Date'] + timedelta(days=30)
min_savi_mes = min_savi_mes.set_index(pd.DatetimeIndex(min_savi_mes['Date']))
min_savi_mes['WY'] = min_savi_mes.apply(lambda x: assign_wy(x), axis=1)
min_savi_mes['DOWY'] = min_savi_mes.index.to_series().apply(day_of_water_year)
min_savi_mes=min_savi_mes.iloc[:,0:100].clip(lower=0.01)

#%% End of season from distribution 
temp_eos=ax_temp.loc[ax_temp['DOWY'] == 365] 

eos_savi = pd.DataFrame(np.random.normal(loc=0.22, scale=0.026, size=[9,100])) #COTTONWOOD
distr_eos_cw = eos_savi.set_index(pd.DatetimeIndex(temp_eos['Date']))

eos_savi = pd.DataFrame(np.random.normal(loc=0.22, scale=0.02, size=[9,100])) #MESQUITE
distr_eos_mes = eos_savi.set_index(pd.DatetimeIndex(temp_eos['Date']))

#START OF SEASON from distribution 
temp_sos=ax_temp.loc[ax_temp['DOWY'] == 1] 

sos_savi = pd.DataFrame(np.random.normal(loc=0.22, scale=0.03, size=[9,100]))
distr_sos_cw = sos_savi.set_index(pd.DatetimeIndex(temp_sos['Date']))

sos_savi = pd.DataFrame(np.random.normal(loc=0.22, scale=0.03, size=[9,100]))
distr_sos_mes = sos_savi.set_index(pd.DatetimeIndex(temp_sos['Date']))

#%% Create dataframe with all events 

artifical_cw=pd.concat([distr_max_cw,distr_eos_cw,distr_sos_cw,min_savi_cw.iloc[:,0:100]]).round(2)
artifical_cw=artifical_cw.sort_index(ascending=True)
artifical_cw=pd.DataFrame(artifical_cw)
#artifical_cw=artifical_cw.iloc[:,0:100].clip(lower=0.1)
artifical_cw['Date'] =artifical_cw.index
artifical_cw['WY'] = artifical_cw.apply(lambda x: assign_wy(x), axis=1)
artifical_cw['DOWY'] = artifical_cw.index.to_series().apply(day_of_water_year)
artifical_cw['median'] = artifical_cw.iloc[:,0:100][artifical_cw.iloc[:,0:100] >= 0].median(axis=1)
#%% Create dataframe with all events 
artifical_mes=pd.concat([distr_max_mes,distr_eos_mes, distr_sos_mes,min_savi_mes.iloc[:,0:100]]).round(2)

artifical_mes=artifical_mes.sort_index(ascending=True)
artifical_mes=pd.DataFrame(artifical_mes)
#artifical_mes=artifical_mes.iloc[:,0:100].clip(lower=0.1)
artifical_mes['Date'] =artifical_mes.index
artifical_mes['WY'] = artifical_mes.apply(lambda x: assign_wy(x), axis=1)
artifical_mes['DOWY'] = artifical_mes.index.to_series().apply(day_of_water_year)
artifical_mes['median'] = artifical_mes.iloc[:,0:100][artifical_mes.iloc[:,0:100] >= 0].median(axis=1)

#%%
ag_art_cw = pd.pivot_table(artifical_cw, index=artifical_cw['DOWY'], columns=artifical_cw['WY'],values='median')
ag_art_cw.interpolate(method='spline',s=0, order=2, inplace=True)


ag_art_mes = pd.pivot_table(artifical_mes, index=artifical_mes['DOWY'], columns=artifical_mes['WY'],values='median')
ag_art_mes.interpolate(method='spline', order=2, s=0, inplace=True)
#%%
ag_cw=ax_cw#.loc[:'2011-10-31']
ag_cw=pd.pivot_table(ag_cw, index=ag_cw['DOWY'], columns=ag_cw['WY'],values='median')
#ag_cw=ag_cw.dropna()
#ag_art_cw=ag_art_cw.clip(lower=0.12)

ag_mes=ax_mes#.loc[:'2011-10-31']
ag_mes=pd.pivot_table(ag_mes, index=ag_mes['DOWY'], columns=ag_mes['WY'],values='median')
#ag_mes=ag_mes.dropna()
#ag_art_mes=ag_art_mes.clip(lower=0.115)

#%% Multivaraite spline interpolation between events 

#run all years first with a=cottonwood (2a) then reset a=mesquite (2b) 
# LOOP to interpolate all years. 
xx = np.linspace(1, 366, 366) #1
#%%
a=ag_art_cw #2a
#%%
a=ag_art_mes #2b
#%%
year=2002
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod2=pd.DataFrame(set_interp(xx)) #2003
#%%
year=2003
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod3=pd.DataFrame(set_interp(xx)) #2003
#%%
year=2006
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod4=pd.DataFrame(set_interp(xx)) #2003
#%%
year=2008
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod5=pd.DataFrame(set_interp(xx)) #2003

#%%
year=2010
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod6=pd.DataFrame(set_interp(xx)) #2003
#%%
year=2014
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod8=pd.DataFrame(set_interp(xx)) #2003
#%%
year=2015
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod9=pd.DataFrame(set_interp(xx)) #2003
#%%
year=2016
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod10=pd.DataFrame(set_interp(xx)) #2003
#%%
year=2017
x=np.asarray(a[year].index)
y=np.asarray(a[year])
set_interp  =   interpolate.PchipInterpolator( x, y, extrapolate=True )
mod11=pd.DataFrame(set_interp(xx)) #2003

#%%

art_cw = pd.concat([mod2,mod3,mod4,mod5,mod6,mod8,mod9,mod10,mod11],axis=1)
art_cw.columns=['2002','2003','2006','2008','2010','2014','2015','2016','2017']

#%%
art_mes = pd.concat([mod2,mod3,mod4,mod5,mod6,mod8,mod9,mod10,mod11],axis=1)
art_mes.columns=['2002','2003','2006','2008','2010','2014','2015','2016','2017']

#%%
#Synthetic ensemble mean over multiple years 
af_cw=art_cw.mean(axis=1)
af_cw_std=art_cw.std(axis=1)

af_mes=art_mes.mean(axis=1)
af_mes_std=art_mes.std(axis=1)
#%%
#Historic ensemble mean 
#his_cw=(ag_cw.drop(ag_cw.columns[[7,8,9,10]],axis=1)).mean(axis=1)
#his_cw_std=(ag_cw.drop(ag_cw.columns[[7,8,9,10]],axis=1)).std(axis=1)

his_cw=ag_cw.mean(axis=1)
his_cw_std=ag_cw.std(axis=1)

#his_mes=(ag_mes.drop(ag_mes.columns[[0,4,7]],axis=1)).mean(axis=1)
#his_mes_std=(ag_mes.drop(ag_mes.columns[[0,4,7]],axis=1)).std(axis=1)

his_mes=ag_mes.mean(axis=1)
his_mes_std=ag_mes.std(axis=1)

#%%
#Timeseries of artifical SAVI 
#ax_temp.drop(ax_temp.tail(2).index,inplace=True)
savi_cw=art_cw
savi_cw.drop(savi_cw.tail(1).index,inplace=True)
savi_cw= savi_cw.melt()
savi_cw = savi_cw.set_index(pd.DatetimeIndex(ax_mes['Date']))  
savi_cw.columns=['Year','SAVI']

savi_mes=art_mes
savi_mes.drop(savi_mes.tail(1).index,inplace=True)
savi_mes= savi_mes.melt()
savi_mes = savi_mes.set_index(pd.DatetimeIndex(ax_mes['Date']))  
savi_mes.columns=['Year','SAVI']
#%%
#savi_cw.to_csv('artifical_savi_cw.csv',sep=',')
#savi_mes.to_csv('artifical_savi_mes.csv',sep=',')
#%%
#TS of modeled and observed SAVI 
cw=ag_cw.melt()
mes=ag_mes.melt()


plt.subplot(211)
plt.plot(np.asarray(savi_cw['SAVI']),label='synthetic')
plt.plot(np.asarray(cw['value']),label='histroic ')
plt.legend(frameon=False)
plt.subplot(212)
plt.plot(np.asarray(savi_mes['SAVI']),label='synthetic')
plt.plot(np.asarray(mes['value']),label='historic')
plt.legend(frameon=False)
#%% MEAN SAVI VALUES PLOT
plt.subplot(121)
plt.plot(af_cw,color='green')
plt.grid(alpha=0.2)
plt.fill_between(af_cw.index,af_cw-af_cw_std, af_cw+af_cw_std,  color='green',alpha=0.3, edgecolor='green')
plt.plot(his_cw,color='black')
plt.fill_between(his_cw.index,his_cw-his_cw_std, his_cw+his_cw_std,  color='grey',alpha=0.3, edgecolor='grey')
plt.ylim(0.04,0.45)
plt.ylabel('SAVI')
plt.subplot(122)
plt.plot(af_mes,color='orange')
plt.fill_between(af_mes.index,af_mes-af_mes_std, af_mes+af_mes_std,  color='orange',alpha=0.3, edgecolor='orange')
plt.plot(his_mes,color='black')
plt.fill_between(his_mes.index,his_mes-his_mes_std, his_mes+his_mes_std,  color='grey',alpha=0.3, edgecolor='grey')
plt.ylim(0.04,0.45)
plt.ylabel('SAVI')
plt.grid(alpha=0.2)

#%% STATISTICS BETWEEN HISTORIC AND MODELLED SAVI
#if the overall distribution of values is different or not 
tst1=pd.DataFrame(ax_cw['median'][ax_cw.index.isin(artifical_cw.index)])
tst2=artifical_cw.iloc[:,0:100].median(axis=1)

print(stats.ks_2samp(tst1['median'],tst2))  #p=0.08
#%%

tst3=pd.DataFrame(ax_mes['median'][ax_mes.index.isin(artifical_mes.index)])
tst4=artifical_mes.iloc[:,0:100].median(axis=1)

print(stats.ks_2samp(tst3['median'],tst4)) #p=0.05
#%%

stats.pearsonr(af_cw.dropna(),his_cw)
stats.pearsonr(af_mes.dropna(),his_mes)

#%%
plt.subplot(211)
plt.grid(alpha=0.2)
sns.kdeplot(af_cw,color='green',label='Modelled')
sns.kdeplot(his_cw,color='green',linestyle='--',label='Observed')
plt.legend(frameon=False)
plt.subplot(212)
sns.kdeplot(af_mes,color='orange')
sns.kdeplot(his_mes,color='orange',linestyle='--')
plt.grid(alpha=0.2)

#%%
x= ag_cw.min(axis=0) #median cw savi observed 
y=min_savi_cw.iloc[:,0:99].median(axis=1)

print(stats.kruskal(x,y)) #p=0.12 
#%%

x= ag_mes.min(axis=0)
y=min_savi_mes.iloc[:,0:99].median(axis=1)
print(stats.kruskal(x,y)) #p=0.56
#%%
x= ag_mes.max(axis=0)
y=df_max_mes['MAX-SAVI']
print(stats.kruskal(x,y)) #p=0.14
#########
x= ag_cw.max(axis=0)
y=df_max_cw['MAX-SAVI']
print(stats.kruskal(x,y)) #p=0.1


#%%%


