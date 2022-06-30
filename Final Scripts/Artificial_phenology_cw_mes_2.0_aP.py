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

#cwp=pd.pivot_table(ax_savi_cw, index=ax_savi_cw['DOWY'], columns=ax_savi_cw['WY'],values='median')
#%%

file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/ax_temp_all.csv"
ax_temp= pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
ax_temp['Date'] =ax_temp.index
ax_temp['WY'] = ax_temp.apply(lambda x: assign_wy(x), axis=1)
ax_temp['DOWY'] = ax_temp.index.to_series().apply(day_of_water_year)
ax_temp=ax_temp[ax_temp['WY'].isin([1995,1997,1999,2001,2003,2005,2007,2009,2011,2015,2017,2019])]

#%%
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/ant_aP.csv"
ax_aP= pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
#ax_rain=ax_rain.loc[:'2017-10-31']
ax_aP=ax_aP[ax_aP['WY'].isin([1995,1997,1999,2001,2003,2005,2007,2009,2011,2015,2017,2019])]
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

artifical_cw=pd.concat([distr_max_cw,distr_max_cw_start,distr_eos_cw,distr_sos_cw,min_savi_cw.iloc[:,0:100]]).round(2)
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
#%%
ag_cw=ax_cw#.loc[:'2011-10-31']
ag_cw=pd.pivot_table(ag_cw, index=ag_cw['DOWY'], columns=ag_cw['WY'],values='median')
#ag_cw=ag_cw.dropna()
#ag_art_cw=ag_art_cw.clip(lower=0.12)

#%%
ag_art_mes = pd.pivot_table(artifical_mes, index=artifical_mes['DOWY'], columns=artifical_mes['WY'],values='median')
ag_art_mes.interpolate(method='spline', order=2, s=0, inplace=True)
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
a=ag_art_cw #2a
#%%
a=ag_art_mes #2b
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

art_cw = pd.concat([ mod2,mod3,mod4,mod5,mod6,mod7,mod8,mod9,mod10,mod12,mod13,mod14],axis=1)
art_cw.columns=['1995','1997','1999','2001','2003','2005','2007','2009','2011','2015','2017','2019']

#%%
art_mes = pd.concat([mod2,mod3,mod4,mod5,mod6,mod7,mod8,mod9,mod10,mod12,mod13,mod14],axis=1)
art_mes.columns=['1995','1997','1999','2001','2003','2005','2007','2009','2011','2015','2017','2019'] 

#%%
#Synthetic ensemble mean over multiple years 
af_cw=art_cw.mean(axis=1)
af_cw_std=art_cw.std(axis=1)

af_mes=art_mes.mean(axis=1)
af_mes_std=art_mes.std(axis=1)
#%%
#Historic ensemble mean 
his_cw=ag_cw.mean(axis=1)
his_cw_std=ag_cw.std(axis=1)


his_mes=ag_mes.mean(axis=1)
his_mes_std=ag_mes.std(axis=1)
#%%

savi_cw= art_cw
savi_cw.drop(savi_cw.tail(1).index,inplace=True)
savi_cw= savi_cw.melt()
savi_cw = savi_cw.set_index(pd.DatetimeIndex(ax_temp['Date']))  
savi_cw.columns=['Year','SAVI']

savi_mes= art_mes
savi_mes.drop(savi_mes.tail(1).index,inplace=True)
savi_mes= savi_mes.melt()
savi_mes = savi_mes.set_index(pd.DatetimeIndex(ax_temp['Date']))  
savi_mes.columns=['Year','SAVI']
#%%
#savi_mes.to_csv('savi_mes_artifical.csv',sep=',')
#savi_cw.to_csv('savi_cw_artifical.csv',sep=',')
#%%
cw=ag_cw.melt()
mes=ag_mes.melt()

ab=np.asarray(savi_cw['SAVI'])
std_savi_cw=savi_cw['SAVI'].std()
ac=np.asarray(savi_mes['SAVI'])
std_savi_mes=savi_mes['SAVI'].std()
cd=np.asarray(ax_temp)
#%% TIMESERIES OF OBSERVED AND SYNTHETIC SAVI 
plt.subplot(211)
plt.plot(ab,color='red', label='synthetic')
plt.fill_between(cw.index,ab-std_savi_cw, ab+std_savi_cw,  color='red',alpha=0.1) #Modelled std
plt.plot(np.asarray(cw['value']),color='grey',label='observed')
plt.grid(alpha=0.2)
plt.ylabel('SAVI')
#plt.ylim(0,0.41)
plt.legend(frameon=False)
plt.subplot(212)
plt.plot(ac,color='red',label='synthetic')
plt.fill_between(mes.index,ac-std_savi_mes, ac+std_savi_mes,  color='red',alpha=0.1) #Modelled std
plt.plot(np.asarray(mes['value']),color='grey',label='observed')
plt.grid(alpha=0.2)
#plt.ylim(0,0.341)
plt.ylabel('SAVI')
#%%

plt.subplot(211)
sns.kdeplot(ab,color='green',label='Modelled')
sns.kdeplot(cw['value'],color='grey',linestyle='--',label='Observed')
plt.ylim(0,6)
plt.xlim(0,0.45)
plt.grid(alpha=0.2)
plt.legend(frameon=False)
plt.subplot(212)
sns.kdeplot(ac,color='orange')
sns.kdeplot(mes['value'],color='grey',linestyle='--')
plt.ylim(0,6)
plt.grid(alpha=0.2)
plt.xlim(0,0.45)
plt.xlabel('SAVI')

#%% MEAN SAVI VALUES PLOT
plt.subplot(121)
plt.plot(af_cw,color='green')
plt.grid(alpha=0.2)
plt.fill_between(af_cw.index,af_cw-af_cw_std, af_cw+af_cw_std,  color='green',alpha=0.3, edgecolor='green')
plt.plot(his_cw,color='black')
plt.fill_between(his_cw.index,his_cw-his_cw_std, his_cw+his_cw_std,  color='grey',alpha=0.3, edgecolor='grey')
plt.ylim(0.05,0.4)
plt.ylabel('SAVI')
plt.subplot(122)
plt.plot(af_mes,color='orange')
plt.fill_between(af_mes.index,af_mes-af_mes_std, af_mes+af_mes_std,  color='orange',alpha=0.3, edgecolor='orange')
plt.plot(his_mes,color='black')
plt.fill_between(his_mes.index,his_mes-his_mes_std, his_mes+his_mes_std,  color='grey',alpha=0.3, edgecolor='grey')
plt.ylim(0.05,0.4)
plt.ylabel('SAVI')
plt.grid(alpha=0.2)

#%% STATISTICS BETWEEN HISTORIC AND MODELLED SAVI
#if the overall distribution of values is different or not 
tst1=pd.DataFrame(ax_savi_cw['median'][ax_savi_cw.index.isin(artifical_cw.index)])
tst2=artifical_cw.iloc[:,0:100].median(axis=1)

print(stats.ks_2samp(tst1['median'],tst2)) 
#%%

tst3=pd.DataFrame(ax_savi_mes['median'][ax_savi_mes.index.isin(artifical_mes.index)])
tst4=artifical_mes.iloc[:,0:100].median(axis=1)

print(stats.ks_2samp(tst3['median'],tst4))
#%%

print(stats.pearsonr(af_cw.dropna(),his_cw))
stats.pearsonr(af_mes.dropna(),his_mes)

#%%
plt.subplot(211)
sns.kdeplot(af_cw,color='green',label='Modelled')
sns.kdeplot(his_cw,color='green',linestyle='--',label='Observed')
plt.ylim(0,6)
plt.xlim(0,0.45)
plt.grid(alpha=0.2)
plt.legend(frameon=False)
plt.subplot(212)
sns.kdeplot(af_mes,color='orange')
sns.kdeplot(his_mes,color='orange',linestyle='--')
plt.ylim(0,6)
plt.grid(alpha=0.2)
plt.xlim(0,0.45)
plt.xlabel('SAVI')

#%%
x= ag_cw.min(axis=0) #median cw savi observed 
y=min_savi_cw.iloc[:,0:99].median(axis=1)
y=pd.DataFrame(y[y.index.year.isin(x.index)])

print(stats.kruskal(x,y[0])) #p=0.294 
#%%

x= ag_mes.min(axis=0)
y=min_savi_mes.iloc[:,0:99].median(axis=1)
print(stats.kruskal(x,y)) #p=0.67
#%%
x= ag_mes.max(axis=0)
y=df_max_mes['MAX-SAVI']
print(stats.kruskal(x,y)) #p=0.58
#########
x= ag_cw.max(axis=0)
y=df_max_cw['MAX-SAVI']
print(stats.kruskal(x,y)) #p=0.5
      
#%%

