#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 11:19:22 2021

@author: mariawarter
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings
from scipy.stats import powerlaw
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import operator 
import glob
import matplotlib.dates as mdates
from scipy import signal
from sklearn import linear_model
import datetime as dt
from scipy.stats import iqr
from scipy.stats import pearsonr
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


#%%P and PET for the San Pedro

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

#%%
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
#%%
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/dT_SP/ESRL_temp_1986_2020.csv"

data_T_SP = pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
data_T_SP = data_T_SP.set_index(pd.DatetimeIndex(data_T_SP['date']))  
data_T_SP.columns=['Date','maxT','minT']
data_T_SP['Date'] = pd.to_datetime(data_T_SP.index) #set P data to 1986-2019
#data_T_SP=data_T_SP.loc[(data_T_SP.index < '2012-01-01') | (data_T_SP.index > '2012-12-31')]
#data_T_SP = data_T_SP.loc[:'2019-12-31'] #data only starts in 1986
data_T_SP['maxT'] =data_T_SP['maxT'].interpolate()
data_T_SP['minT'] =data_T_SP['minT'].interpolate()


data_T_SP['Date'] = pd.to_datetime(data_T_SP.index) #set P data to 1986-2019
data_T_SP['WY'] = data_T_SP.apply(lambda x: assign_wy(x), axis=1)
data_T_SP['DOWY'] = data_T_SP.index.to_series().apply(day_of_water_year)
data_T_SP['mean']= ((data_T_SP['maxT'] + data_T_SP['minT'] ) /2)
data_T_SP['32D']=data_T_SP['maxT'].rolling(32).mean().round(2) #1month
data_T_SP['64D']=data_T_SP['maxT'].rolling(64).mean().round(2) #1month
data_T_SP['80D']=data_T_SP['maxT'].rolling(80).mean().round(2) #1month
data_T_SP['96D']=data_T_SP['maxT'].rolling(96).mean().round(2) #1month

ax_temp=data_T_SP.loc['1992-11-01':'2020-10-31']
#ax_temp = ax_temp[ax_temp['WY'].isin([1996,1998,2000,2002,2006,2008,2010,2012,2014,2016,2018,2020])]

#%%Groundwater data for the San Pedro

file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/GW/SanPedro_GW_86_19.csv"

data_GW_SP = pd.read_csv(file,dayfirst=True,parse_dates=True,sep=';')
cols=[0,1,3]
data_GW_SP=data_GW_SP.drop(data_GW_SP.columns[cols],axis=1)
data_GW_SP['datetime'] = pd.to_datetime(data_GW_SP['datetime'],dayfirst=True)

data_GW_SP =data_GW_SP.set_index(pd.DatetimeIndex(data_GW_SP['datetime'])) 

data_GW_SP= data_GW_SP.groupby('station_ID',as_index=True).agg(lambda x: x.tolist())
#%%
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/GW/GW_GS_Sp19.csv"
data_GW_SP19 = pd.read_csv(file,dayfirst=True,parse_dates=True,sep=',')
cols=[0,1,2,3,5]
data_GW_SP19=data_GW_SP19.drop(data_GW_SP19.columns[cols],axis=1)
data_GW_SP19['datetime'] = pd.to_datetime(data_GW_SP19['datetime'],dayfirst=True)

data_GW_SP19 =data_GW_SP19.set_index(pd.DatetimeIndex(data_GW_SP19['datetime'])) 

#%%

file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/GW/USGS_3130811.csv"

data_gw10 = pd.read_csv(file,dayfirst=True,parse_dates=True,sep=';')
data_gw10['datetime'] = pd.to_datetime(data_gw10['datetime'],dayfirst=True)
data_gw10 =data_gw10.set_index(pd.DatetimeIndex(data_gw10['datetime'])) 

data_gw10['DTG_m'] = data_gw10['DTG_f'] /3.281

#%%
#file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/GW/SanPedro_SF_86_19.csv"

#data_SF_SP = pd.read_csv(file,dayfirst=True,parse_dates=True,sep=',')
#data_SF_SP['datetime'] = pd.to_datetime(data_SF_SP['datetime'],dayfirst=True)

#data_SF_SP =data_SF_SP.set_index(pd.DatetimeIndex(data_SF_SP['datetime'])) 

#data_SF_SP= data_SF_SP.groupby('station_ID.y',as_index=True).agg(lambda x: x.tolist())

#%%Import SAVI
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/SAVI/landsat_1986_2021_edit/veg_cottonwood_savi_cloud_na.csv"

#file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/SAVI/landsat_veg_cottonwood_south_savi_cloudless.csv"
#%%
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/SAVI/landsat_1986_2021_edit/veg_mesquite_savi_cloud_na.csv"
#%%
#file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/SAVI/mesquite_dry_savi_cloud_na.csv"

#%%
#file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/SAVI/cottonwood_dry_savi_cloud_na.csv"

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

#%%
SAVI['Date'] = pd.to_datetime(SAVI.index)
#SAVI['DOY'] = SAVI['Date'].dt.dayofyear
#SAVI =SAVI.loc['2013-11-01':'2018-10-31']
#SAVI.drop(SAVI.tail(1).index,inplace=True)
def assign_wy(SAVI):
    if SAVI.Date.month>=11:
        return(datetime(SAVI.Date.year+1,1,1).year)
    else:
        return(datetime(SAVI.Date.year,1,1).year)

SAVI['WY'] = SAVI.apply(lambda x: assign_wy(x), axis=1)

x=SAVI['Date']
def day_of_water_year(x):
    # Get the date of the previous October 1st
    water_year_start_date = datetime(x.year + x.month // 11 - 1, 11, 1)
    
    # Return the number of days since then
    return (x - water_year_start_date).days + 1#%%
SAVI['DOWY'] = SAVI.index.to_series().apply(day_of_water_year)
 

#%%
SAVI_cottonwood=SAVI
SAVI_cottonwood = SAVI_cottonwood.loc['1994-11-01':'2021-12-31']
#SAVI_cottonwood=SAVI_cottonwood.loc[(SAVI_cottonwood.index < '2008-01-01') | (SAVI_cottonwood.index > '2008-12-31')]
#%%
SAVI_mesquite=SAVI.loc['1994-11-01':'2021-12-31']
#SAVI_mesquite=SAVI_mesquite.loc[(SAVI_mesquite.index < '2008-01-01') | (SAVI_mesquite.index > '2008-12-31')]
#%%

SAVI_cottonwood['median'] = SAVI_cottonwood.iloc[:,0:30].median(axis=1)
ag_cw = pd.pivot_table(SAVI_cottonwood, index=SAVI_cottonwood['DOWY'], columns=SAVI_cottonwood['WY'],values='median')


SAVI_mesquite['median'] = SAVI_mesquite.iloc[:,0:30].median(axis=1)
ag_mes = pd.pivot_table(SAVI_mesquite, index=SAVI_mesquite['DOWY'], columns=SAVI_mesquite['WY'],values='median')

#%%
#To get valuidation dataset per WY with full daily values run separately to expand DF
#ax_cw_val=SAVI_cottonwood
#ax_cw_val=ax_cw_val[ax_cw_val['WY'].isin([1996,1998,1999,2000,2002,2004,2006,2008,2010,2014,2016,2018,2019,2020])]
#ax_cw_val=ax_cw_val[['1','2']] #only wet pixels 
#ax_cw_val.to_csv('ax_cw_val_wet_cy.csv',sep=',')

#%%
#ax_mes_val=SAVI_mesquite
#ax_mes_val=ax_mes_val[ax_mes_val['WY'].isin([1996,1998,1999,2000,2002,2004,2006,2008,2010,2014,2016,2018,2019,2020])]
#ax_mes_val=ax_mes_val[['1','26']] # wet pixels 
#ax_mes_val.to_csv('ax_mes_val_wet_cy.csv',sep=',')
#%%

plt.subplot(211)
plt.grid(alpha=0.2)
plt.ylim(0.07,0.35)
plt.plot(ax_gr.mean(axis=1))
plt.subplot(212)
plt.plot(ag_mes.mean(axis=1))
plt.plot(ag_cw.mean(axis=1))
plt.ylim(0.07,0.35)
plt.grid(alpha=0.2)

#%%

#ax_mes_all=SAVI_mesquite
#ax_cw_all = SAVI_cottonwood 

#ax_mes_all.to_csv('ax_mes_all.csv',sep=',')
#ax_cw_all.to_csv('ax_cw_all.csv',sep=',')

a=SAVI_cottonwood['1']
a=a.loc['2014':'2020']
b=SAVI_mesquite['1']
b=b.loc['2014':'2020']
c=SAVI_grassland.iloc[:,0:30].median(axis=1)
c=c.loc['2014':'2020']
#d=SAVI_shrub.iloc[:,0:30].median(axis=1)
#d=d.loc['2014':'2020']

savi=pd.concat([c,b,a],axis=1).round(2)
savi.columns=['gr','mes','cw']

savi.to_csv('savi_all.csv',sep=',')
#%%
#get max and min every year for every veg type.

ab= pd.DataFrame(a.groupby(a.index.year).max()).round(2)
#ab.index=SAVI_cottonwood.groupby(SAVI_cottonwood.index.year)['1'].idxmax()
#ab=ab.loc['2014':'2020']

bb= pd.DataFrame(b.groupby(b.index.year).max()).round(2)
#bb.index=SAVI_mesquite.groupby(SAVI_mesquite.index.year)['1'].idxmax()
#bb=bb.loc['2014':'2020']

cb= pd.DataFrame(c.groupby(c.index.year).max()).round(2)
#cb.index=SAVI_grassland.groupby(SAVI_grassland.index.year)['median'].idxmax()
#cb=cb.loc['2014':'2020']

savi_max=pd.concat([cb,bb,ab],axis=1)
savi_max.columns=['gr','mes','cw']
savi_max.to_csv('savi_max.csv',sep=',')
#%%
ab= pd.DataFrame(a.groupby(a.index.year).min()).round(2)
#ab.index=SAVI_cottonwood.groupby(SAVI_cottonwood.index.year)['1'].idxmax()
#ab=ab.loc['2014':'2020']

bb= pd.DataFrame(b.groupby(b.index.year).min()).round(2)
#bb.index=SAVI_mesquite.groupby(SAVI_mesquite.index.year)['1'].idxmax()
#bb=bb.loc['2014':'2020']

cb= pd.DataFrame(c.groupby(c.index.year).min()).round(2)
#cb.index=SAVI_grassland.groupby(SAVI_grassland.index.year)['median'].idxmax()
#cb=cb.loc['2014':'2020']

savi_min=pd.concat([cb,bb,ab],axis=1)
savi_min.columns=['gr','mes','cw']
savi_min.to_csv('savi_min.csv',sep=',')

#%%
#INTERMEDIATE REACH
#ax=SAVI_cottonwood[['4','30','6','10','13','12','20','15','18','16','26']] #intermediate pixels
#ax_cw_inter= pd.DataFrame(ax.median(axis=1))
#ax_cw_inter.columns=['median']
#ax_cw_inter_iqr = pd.DataFrame(iqr(ax,axis=1))
#ax_cw_inter_iqr.index = ax_cw_inter.index
#ax_cw_inter_iqr.columns=['iqr']
#ax_cw_inter['Date'] = pd.to_datetime(ax_cw_inter.index)
#ax_cw_inter['DOWY'] =ax_cw_inter.index.to_series().apply(day_of_water_year)
#ax_cw_inter['WY'] =ax_cw_inter.apply(lambda x: assign_wy(x), axis=1)
#%%
#WET REACH
ax=SAVI_cottonwood[['1','2']] # wet pixels 

ax_cw_wet= pd.DataFrame(ax.median(axis=1))
ax_cw_wet.columns=['median']
ax_cw_wet_iqr = pd.DataFrame(iqr(ax,axis=1))
ax_cw_wet_iqr.index = ax_cw_wet.index
ax_cw_wet_iqr.columns=['iqr']
ax_cw_wet['Date'] = pd.to_datetime(ax_cw_wet.index)
#%%
#DRY REACH

#ax_cw_dry= pd.DataFrame(SAVI_cottonwood.median(axis=1))
#ax_cw_dry.columns=['median']
#ax_cw_dry_iqr = pd.DataFrame(iqr(SAVI_cottonwood,axis=1))
#ax_cw_dry_iqr.index = ax_cw_dry.index
#ax_cw_dry_iqr.columns=['iqr']
#ax_cw_dry['Date'] = pd.to_datetime(ax_cw_dry.index)


#%%
#INTERMEDIATE REACH 
#ax=SAVI_mesquite

#ax_mes_inter=ax.drop(columns=['1','26']) #intermediate are pixels all but those tw0
#ax_mes_inter_iqr=pd.DataFrame(iqr(ax_mes_inter.iloc[:,0:28],axis=1))
#ax_mes_inter=pd.DataFrame(ax_mes_inter.median(axis=1))
#ax_mes_inter.columns=['median']

#ax_mes_inter['Date'] = pd.to_datetime(ax_mes_inter.index)

#ax_mes_inter_iqr.index=ax_mes_inter.index
#ax_mes_inter_iqr.columns=['iqr']
#%%
#WET REACH 
ax_mes_wet=pd.DataFrame(SAVI_mesquite[['1','26']])
ax_mes_wet_iqr=pd.DataFrame(iqr(ax_mes_wet,axis=1))
ax_mes_wet=pd.DataFrame(ax_mes_wet.median(axis=1))
ax_mes_wet.columns=['median']
ax_mes_wet['Date'] = pd.to_datetime(ax_mes_wet.index)
ax_mes_wet_iqr.columns=['iqr']
ax_mes_wet_iqr.index=ax_mes_wet.index
#%%

#ax_mes_dry=pd.DataFrame(SAVI_mesquite_dry.median(axis=1))
#ax_mes_dry.columns=['median']
#ax_mes_dry_iqr=pd.DataFrame(iqr(SAVI_mesquite_dry,axis=1))
#ax_mes_dry_iqr.columns=['iqr']
#ax_mes_dry_iqr.index=ax_mes_dry.index


#%%
mc=ax_cw_wet
mci=ax_cw_wet_iqr
#%%
#mc= ax_cw_inter
#mci=ax_cw_inter_iqr

#%%
#mc=ax_cw_dry
#mci=ax_cw_dry_iqr
#%% MAX SAVI 
cw_max= pd.DataFrame(mc.groupby(mc.index.year)['median'].max())
cw_max_id= pd.DataFrame(mc.groupby(mc.index.year)['median'].idxmax())

#%%
cw_max_iqr = pd.DataFrame(mci[mci.index.isin(cw_max_id['median'])])
cw_max_iqr.columns=['iqr']
cw_max_iqr.index=cw_max.index


#%%
mm=ax_mes_wet.loc['1995':]
mmi=ax_mes_wet_iqr.loc['1995':]
#%%
#mm=ax_mes_inter
#mmi=ax_mes_inter_iqr
#%%
#mm=ax_mes_dry
#mmi=ax_mes_dry_iqr

#%%
mes_max = pd.DataFrame(mm.groupby(mm.index.year)['median'].max())
mes_max_id = pd.DataFrame(mm.groupby(mm.index.year)['median'].idxmax())
#%%
mes_max_iqr = pd.DataFrame(mmi[mmi.index.isin(mes_max_id['median'])])
mes_max_iqr.columns=['iqr']
mes_max_iqr.index=mes_max.index
#%% Trend of max savi 

#tr=df['']

#coefficients, residuals, _, _, _ = np.polyfit(range(len(tr.index)),tr,1,full=True)
#mse = residuals[0]/(len(tr.index))
#nrmse = np.sqrt(mse)/(tr.max() - tr.min())
#print('Slope ' + str(coefficients[0]))
#print('NRMSE: ' + str(nrmse))

#plt.plot(tr)
#plt.plot(tr.index, [coefficients[0]*x + coefficients[1] for x in range(len(tr))])

#%%
#Precipitaiton sort per WY
data_P_SP['WY'] = data_P_SP.index.year
data_P_SP['DOY'] = data_P_SP['Date'].dt.dayofyear
ax_rain=data_P_SP.loc['1994-11-01':]
ax_pet=data_PET_SP.loc['1994-11-01':]

#%% #%%Groundwater and tree sampling points nearby
a=pd.DataFrame(data_GW_SP['water_depth_m'].loc['GS_Sp17']) #suitable for mesquite, center of the samplign zone
b=pd.DataFrame(data_GW_SP['datetime'].loc['GS_Sp17'])
a=a.set_index(b[0])
a.columns=['DTG17']
a=a.sort_index()
a['Date'] =a.index
#%%#%% DTG17 CALIBRATION WELL IN THE WET REACH

ax_gw17 = a
#ax_gw17['WY'] = ax_gw17.apply(lambda x: assign_wy(x), axis=1)
ax_gw17['DOY'] = ax_gw17['Date'].dt.dayofyear
ax_gw17['WY'] = ax_gw17.index.year
#%%%
ax_gw17=ax_gw17.loc[(ax_gw17.index < '2004') | (ax_gw17.index > '2006')]
ax_gw17=ax_gw17.loc[(ax_gw17.index < '2007') | (ax_gw17.index > '2008')]
ax_gw17=ax_gw17.loc[(ax_gw17.index < '2018') | (ax_gw17.index > '2019')]

#%% add extra years from second DF 
data_GW_SP19.columns=['Date','DTG19']
data_GW_SP19=data_GW_SP19.sort_index()

ax_gw19 = pd.concat([a['DTG19'],data_GW_SP19['DTG19']],axis=0)
ax_gw19=ax_gw19.sort_index()
ax_gw19=pd.DataFrame(ax_gw19)
ax_gw19['Date'] = ax_gw19.index
ax_gw19=ax_gw19.drop_duplicates('Date', 'last')

#%% DTG10 - VALIDATION WELL - WET 
ax_gw10= pd.DataFrame(data_gw10['DTG_m'])
ax_gw10.columns=['DTG10']
ax_gw10=ax_gw10.sort_index()
#idx=pd.date_range('06-22-2001','02-01-2022',freq='D')
#ax_gw10=ax_gw10.reindex(idx)

ax_gw10['Date'] =ax_gw10.index
ax_gw10['DOY'] = ax_gw10['Date'].dt.dayofyear

#%% INTERMDEIATE WELL 
ax_gw23= pd.DataFrame(a['DTG23'])
ax_gw23=ax_gw23.sort_index()
ax_gw23['Date'] =ax_gw23.index
#%%
ax_gw27= pd.DataFrame(a['DTG27'])
ax_gw27=ax_gw27.sort_index()
ax_gw27['Date'] =ax_gw27.index

#%% aP P-PET

aP = pd.DataFrame(ax_rain['P'] - ax_pet['PET'])
aP.columns=['aP']
aP[aP < 0] = 0
aP['Date']=aP.index
#aP=aP.loc['1994':]
aP['WY'] = aP.apply(lambda x: assign_wy(x), axis=1)
aP['DOWY'] = aP.index.to_series().apply(day_of_water_year)

#%%
months=[5,6,7,8,9]
ant_aP= aP[aP['Date'].map(lambda t: t.month in months)]
ant_aP = pd.DataFrame(ant_aP['aP'].groupby(ant_aP.index.year).cumsum())
ant_aP['Date']=ant_aP.index
ant_aP['WY'] = ant_aP.apply(lambda x: assign_wy(x), axis=1)
ant_aP['DOWY'] = ant_aP.index.to_series().apply(day_of_water_year)

#ant_aP.to_csv('ant_aP.csv',sep=',')


#%% CALIBRATION DATA 
ant_aP_cal= ant_aP[ant_aP.index.year.isin([1995,1997,199,2001,2003,2005,2007,2009,2011,2013,2015,2017,2019,2021])]
ant_aP=ant_aP_cal
#ant_aP=ant_aP.loc[:'2015']
ant_aP_val= ant_aP[ant_aP.index.year.isin([1996,1998,1999,2000,2002,2004,2006,2008,2010,2014,2016,2018,2019,2020])]
#%%
aP_cw=pd.DataFrame(ant_aP[ant_aP.index.isin(cw_max_id['median'])])
cw=pd.DataFrame(cw_max[cw_max.index.isin(aP_cw.index.year)])

stat, p = pearsonr(aP_cw['aP'], cw['median'])
print('stat=%.3f, p=%.4f' % (stat, p))
if p > 0.1:
	print('Probably independent')
else:
	print('Probably dependent')

plt.scatter(aP_cw['aP'],cw['median'],color='green')

#%%
months=[5,6,7,8,9]
ant_aP= aP[aP['Date'].map(lambda t: t.month in months)]
ant_aP = pd.DataFrame(ant_aP['aP'].groupby(ant_aP.index.year).cumsum())
ant_aP['Date']=ant_aP.index
ant_aP['WY'] = ant_aP.apply(lambda x: assign_wy(x), axis=1)
ant_aP['DOWY'] = ant_aP.index.to_series().apply(day_of_water_year)
ant_aP=ant_aP_cal
#ant_aP=ant_aP.loc[:'2017']

aP_mes=pd.DataFrame(ant_aP[ant_aP.index.isin(mes_max_id['median'])])
#aP_mes=aP_mes.dropna()
mes=pd.DataFrame(mes_max[mes_max.index.isin(aP_mes.index.year)])

stat, p = pearsonr(aP_mes['aP'], mes['median'])
print('stat=%.3f, p=%.4f' % (stat, p))
if p > 0.1:
	print('Probably independent')
else:
	print('Probably dependent')
    
plt.scatter(aP_mes['aP'],mes['median'],color='orange')



#%% VALIDATION DATA SHOWN ON TS PLOTS 
aP_cw_val=pd.DataFrame(ant_aP_val[ant_aP_val.index.isin(cw_max_id['median'])])
cw_val=pd.DataFrame(cw_max[cw_max.index.isin(aP_cw_val.index.year)])


aP_mes_val=pd.DataFrame(ant_aP_val[ant_aP_val.index.isin(mes_max_id['median'])])
#aP_mes=aP_mes.dropna()
mes_val=pd.DataFrame(mes_max[mes_max.index.isin(aP_mes_val.index.year)])
#%%
plt.scatter(aP_mes_val['aP'],mes_val['median'],color='orange')
plt.scatter(aP_cw_val['aP'],cw_val['median'],color='green')

#%% Normalize DTG 10 to check - validation well 

gw19_norm = pd.DataFrame((ax_gw19['DTG19']- ax_gw19['DTG19'].min())/(ax_gw19['DTG19'].max()-ax_gw19['DTG19'].min()))
#gw10_norm['16D'] = pd.DataFrame((ax_gw10['16D']- ax_gw10['16D'].min())/(ax_gw10['16D'].max()-ax_gw10['16D'].min()))

gw19_norm=gw19_norm.dropna()
#gw10_norm=gw10_norm.loc[:'2014']

gw_max_mes19= pd.DataFrame(gw19_norm[gw19_norm.index.isin(mes_max_id['median'])])
gw_max_mes19=gw_max_mes19.dropna() #2007
mes_max19=pd.DataFrame(mes_max[mes_max.index.isin(gw_max_mes19.index.year)])


gw_max_cw19= pd.DataFrame(gw19_norm[gw19_norm.index.isin(cw_max_id['median'])])
gw_max_cw19=gw_max_cw19.dropna() #2007
cw_max19=pd.DataFrame(cw_max[cw_max.index.isin(gw_max_cw19.index.year)])
#%%
plt.scatter(gw_max_cw19['DTG19'], cw_max19['median'],color='green')   
plt.scatter(gw_max_mes19['DTG19'], mes_max19['median'],color='orange')   

#%% validation data set of DTG10 

#gw10_norm.to_csv('ax_gw_val_dtg10.csv',sep=',') #save all data (incl > 2014)

#%%
stat, p = pearsonr(gw_max_mes19['DTG19'], mes_max19['median'])
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.1:
	print('Probably independent')
else:
	print('Probably dependent')
    
stat, p = pearsonr(gw_max_cw19['DTG19'], cw_max19['median'])
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.1:
	print('Probably independent')
else:
	print('Probably dependent')


#%% Normalize dtg of well 17 
gw17_norm =pd.DataFrame((ax_gw17['DTG17']- ax_gw17['DTG17'].min())/(ax_gw17['DTG17'].max()-ax_gw17['DTG17'].min()))

gw17_norm = gw17_norm[gw17_norm.index.year.isin([1995,1997,2001,2003,2005,2007,2009,2011,2013,2015,2017,2021])]

#gw17_norm=gw17_norm.dropna()
#gw17_norm=gw17_norm.loc[:'2014']

gw_max_cw17= pd.DataFrame(gw17_norm[gw17_norm.index.isin(cw_max_id['median'])])
#gw_max_cw17.index=gw_max_cw17.index.year
cw_max17=pd.DataFrame(cw_max[cw_max.index.isin(gw_max_cw17.index.year)])


gw_max_mes17= pd.DataFrame(gw17_norm[gw17_norm.index.isin(mes_max_id['median'])])
#gw_max_mes17.index=gw_max_mes17.index.year
mes_max17=pd.DataFrame(mes_max[mes_max.index.isin(gw_max_mes17.index.year)])
plt.scatter(gw_max_cw17['DTG17'], cw_max17['median'],color='green')    
plt.scatter(gw_max_mes17['DTG17'], mes_max17['median'],color='orange')

#%%
gw17_norm =pd.DataFrame((ax_gw17['DTG17']- ax_gw17['DTG17'].min())/(ax_gw17['DTG17'].max()-ax_gw17['DTG17'].min()))

gw17_norm_val = gw17_norm[gw17_norm.index.year.isin([2002,2004,2006,2008,2010,2014,2016,2018])]

#gw17_norm=gw17_norm.dropna()
#gw17_norm=gw17_norm.loc[:'2014']

gw_max_cw17_val= pd.DataFrame(gw17_norm_val[gw17_norm_val.index.isin(cw_max_id['median'])])
#gw_max_cw17.index=gw_max_cw17.index.year
cw_max17_val=pd.DataFrame(cw_max[cw_max.index.isin(gw_max_cw17_val.index.year)])


gw_max_mes17_val= pd.DataFrame(gw17_norm_val[gw17_norm_val.index.isin(mes_max_id['median'])])
#gw_max_mes17.index=gw_max_mes17.index.year
mes_max17_val=pd.DataFrame(mes_max[mes_max.index.isin(gw_max_mes17_val.index.year)])



#%%
plt.scatter(gw_max_cw17_val['DTG17'], cw_max17_val['median'],color='green')    
plt.scatter(gw_max_mes17_val['DTG17'], mes_max17_val['median'],color='orange')
#%%
stat, p = pearsonr(gw_max_cw17['DTG17'], cw_max17['median'])
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.1:
	print('Probably independent')
else:
	print('Probably dependent')

stat, p = pearsonr(gw_max_mes17['DTG17'], mes_max17['median'])
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.1:
	print('Probably independent')
else:
	print('Probably dependent')
#%% INTERMEDIATE REACH
gw23_norm =pd.DataFrame((ax_gw23['DTG23']- ax_gw23['DTG23'].min())/(ax_gw23['DTG23'].max()-ax_gw23['DTG23'].min()))

gw23_norm=gw23_norm.dropna()
#gw17_norm=gw17_norm.loc[:'2014']

gw_max_cw23= pd.DataFrame(gw23_norm[gw23_norm.index.isin(cw_max_id['median'])])
#gw_max_cw17.index=gw_max_cw17.index.year
cw_max23=pd.DataFrame(cw_max[cw_max.index.isin(gw_max_cw23.index.year)])

gw_max_mes23= pd.DataFrame(gw23_norm[gw23_norm.index.isin(mes_max_id['median'])])
#gw_max_mes17.index=gw_max_mes17.index.year
mes_max23=pd.DataFrame(mes_max[mes_max.index.isin(gw_max_mes23.index.year)])

stat, p = pearsonr(gw_max_cw23['DTG23'], cw_max23['median'])
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.1:
	print('Probably independent')
else:
	print('Probably dependent')

stat, p = pearsonr(gw_max_mes23['DTG23'], mes_max23['median'])
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.1:
	print('Probably independent')
else:
	print('Probably dependent')
#%%
plt.scatter(gw_max_mes23['DTG23'], mes_max23['median'],color='orange')
plt.scatter(gw_max_cw23['DTG23'], cw_max23['median'],color='green')
#%%
gw27_norm =pd.DataFrame((ax_gw27['DTG27']- ax_gw27['DTG27'].min())/(ax_gw27['DTG27'].max()-ax_gw27['DTG27'].min()))

gw27_norm=gw27_norm.dropna()
#gw17_norm=gw17_norm.loc[:'2014']

gw_max_cw27= pd.DataFrame(gw27_norm[gw27_norm.index.isin(cw_max_id['median'])])
#gw_max_cw17.index=gw_max_cw17.index.year
cw_max27=pd.DataFrame(cw_max[cw_max.index.isin(gw_max_cw27.index.year)])
gw_max_mes27= pd.DataFrame(gw27_norm[gw27_norm.index.isin(mes_max_id['median'])])
#gw_max_mes17.index=gw_max_mes17.index.year
mes_max27=pd.DataFrame(mes_max[mes_max.index.isin(gw_max_mes27.index.year)])


stat, p = pearsonr(gw_max_cw27['DTG27'], cw_max27['median'])
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.1:
	print('Probably independent')
else:
	print('Probably dependent')

stat, p = pearsonr(gw_max_mes27['DTG27'], mes_max27['median'])
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.1:
	print('Probably independent')
else:
	print('Probably dependent')

plt.scatter(gw_max_cw27['DTG27'], cw_max27['median'],color='green')
plt.scatter(gw_max_mes27['DTG27'], mes_max27['median'],color='orange')



#%% Regression between aP and DTG 

xa= aP_mes['aP'] #needs all values of all years
ya=gw_max_mes23['DTG23']
xa=pd.DataFrame(xa[xa.index.isin(ya.index)])
ya=pd.DataFrame(ya[ya.index.isin(xa.index)])


xb=aP_cw['aP']
#xb.index=xb.index.year
yb=gw_max_cw23['DTG23']
xb=pd.DataFrame(xb[xb.index.isin(yb.index)])
yb=pd.DataFrame(yb[yb.index.isin(xb.index)])



Xc=pd.concat([xa,xb],axis=0)
Yc=pd.concat([ya,yb],axis=0)

plt.scatter(Xc,Yc)
#%%
stat, p = pearsonr(Xc['aP'], Yc['DTG23'])
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.1:
	print('Probably independent')
else:
	print('Probably dependent')
#%%
x=Xc['aP'].values.flatten()
y=Yc.values.flatten()
new_x = np.linspace(0,160) #adapt depending on relationship
x_line = np.linspace(0,160, 100) #adapt depending on relationship
coeffs = np.polyfit(x,y,1) #1D Regression for Water table depth 
a,b = coeffs
y_model = np.polyval([a,b], x)
y_line = np.polyval([a,b], x_line)

poly = np.poly1d(coeffs)
new_y = poly(new_x)
n=len(Xc)

yerr=np.ones(n)*0.07 #gw17.std()

x_mean=np.mean(x)
y_mean=np.mean(y)
n=x.size
m=3
dof=n-m
v=stats.t.ppf(0.95,dof)

residual = y-y_model
std_error = (np.sum(residual**2) / dof)**.5

numerator = np.sum((x - x_mean)*(y - y_mean))
denominator = ( np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2) )**.5
correlation_coef = numerator / denominator
r2 = correlation_coef**2

# mean squared error
#MSE = 1/n * np.sum( (y - y_model)**2 )

#ci = t * std_error * np.sqrt((1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2)))
pi = v * std_error * np.sqrt((1 + 1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2)))

plt.errorbar(x, y,yerr=yerr,fmt='o',linestyle='none',color='black')    
plt.plot(x,y,'.', new_x,new_y,markersize=8.0,color='black',label="y={0:.4f}*x+{1:.4f}".format(a,b)) #1D
plt.fill_between(x_line, y_line + pi, y_line - pi, color = 'grey', alpha=0.2,label = '95% prediction interval',edgecolor='none')
plt.legend(frameon=False)
    
    #%%MESQUITE
x=gw_max_mes17['DTG17'].values.flatten()
y=mes_max17.values.flatten()
new_x = np.linspace(0,1) #adapt depending on relationship
x_line = np.linspace(0,1, 100) #adapt depending on relationship
coeffs = np.polyfit(x,y,1) #1D Regression for Water table depth 
a,b = coeffs
y_model = np.polyval([a,b], x)
y_line = np.polyval([a,b], x_line)

poly = np.poly1d(coeffs)
new_y = poly(new_x)

#Statistics
x_mean=np.mean(x)
y_mean=np.mean(y)
n=x.size
m=3
dof=n-m
v=stats.t.ppf(0.95,dof)

residual = y-y_model
std_error = (np.sum(residual**2) / dof)**.5

numerator = np.sum((x - x_mean)*(y - y_mean))
denominator = ( np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2) )**.5
correlation_coef = numerator / denominator
r2 = correlation_coef**2


#ci = t * std_error * np.sqrt((1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2)))
pi = v * std_error * np.sqrt((1 + 1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2)))
print(coeffs)
#%%
#pi=pd.DataFrame(pi)
#pi.to_csv('ci_maxsavi_mes_gw.csv',sep=',')

#%% COTTONWOOD
s=gw_max_cw17['DTG17'].values.flatten()
t=cw_max17.values.flatten()
new_s = np.linspace(0,1) #adapt depending on relationship
s_line = np.linspace(0,1, 100) #adapt depending on relationship
coeffs = np.polyfit(s,t,1) #1D Regression for Water table depth 
c,d = coeffs
t_model = np.polyval([c,d], s)
t_line = np.polyval([c,d], s_line)

poly = np.poly1d(coeffs)
new_t = poly(new_s)

#Statistics
s_mean=np.mean(s)
t_mean=np.mean(t)
n=x.size
m=3
dof=n-m
v=stats.t.ppf(0.95,dof)

residual = t-t_model
std_error = (np.sum(residual**2) / dof)**.5

numerator = np.sum((s - s_mean)*(t - t_mean))
denominator = ( np.sum((s - s_mean)**2) * np.sum((t - t_mean)**2) )**.5
correlation_coef = numerator / denominator
r2 = correlation_coef**2

# mean squared error
#MSE = 1/n * np.sum( (y - y_model)**2 )

#ci = t * std_error * np.sqrt((1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2)))
pic = v * std_error * np.sqrt((1 + 1/n + (s_line - s_mean)**2 / np.sum((s - s_mean)**2)))
print(coeffs)
#%%
#pic=pd.DataFrame(pic)
#pic.to_csv('ci_maxsavi_cw_gw.csv',sep=',')


#%%
#, ANTECEDENT aP, max mes
o=aP_mes['aP'].values.flatten()
p=mes.values.flatten()

new_o = np.linspace(0,160) #adapt depending on relationship
o_line = np.linspace(0,160, 160) #adapt depending on relationship
coeffs = np.polyfit(o,p,1) #1D Regression for Water table depth 
e,f = coeffs
p_model = np.polyval([e,f], o)
p_line = np.polyval([e,f], o_line)

poly = np.poly1d(coeffs)
new_p = poly(new_o)

#Statistics
o_mean=np.mean(o)
p_mean=np.mean(p)
n=x.size
m=3
dof=n-m
v=stats.t.ppf(0.95,dof)

residual = p-p_model
std_error = (np.sum(residual**2) / dof)**.5

numerator = np.sum((o - o_mean)*(p - p_mean))
denominator = ( np.sum((o - o_mean)**2) * np.sum((p - p_mean)**2) )**.5
correlation_coef = numerator / denominator
r2 = correlation_coef**2

# mean squared error
#MSE = 1/n * np.sum( (y - y_model)**2 )

#ci = t * std_error * np.sqrt((1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2)))
pig = v * std_error * np.sqrt((1 + 1/n + (o_line - o_mean)**2 / np.sum((o - o_mean)**2)))
print(coeffs)
#%%
#pig=pd.DataFrame(pig)
#pig.to_csv('ci_maxsavi_mes_aP.csv',sep=',')
#%% COTTONWOOD max savi and antecedent ap
g=aP_cw['aP'].values.flatten()
h=cw.values.flatten()
new_g = np.linspace(0,150) #adapt depending on relationship
g_line = np.linspace(0,150, 150) #adapt depending on relationship
coeffs = np.polyfit(g,h,1) #1D Regression for Water table depth 
r,q = coeffs
h_model = np.polyval([r,q], g)
h_line = np.polyval([r,q], g_line)

poly = np.poly1d(coeffs)
new_h = poly(new_g)

#Statistics
g_mean=np.mean(g)
h_mean=np.mean(h)
n=x.size
m=3
dof=n-m
v=stats.t.ppf(0.95,dof)

residual = t-t_model
std_error = (np.sum(residual**2) / dof)**.5

numerator = np.sum((g - g_mean)*(h - h_mean))
denominator = ( np.sum((g - g_mean)**2) * np.sum((h - h_mean)**2) )**.5
correlation_coef = numerator / denominator
r2 = correlation_coef**2

# mean squared error
#MSE = 1/n * np.sum( (y - y_model)**2 )

#ci = t * std_error * np.sqrt((1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2)))
pin = v * std_error * np.sqrt((1 + 1/n + (g_line - g_mean)**2 / np.sum((g - g_mean)**2)))
print(coeffs)

#%%
#pin=pd.DataFrame(pin)
#pin.to_csv('ci_maxsavi_cw_aP.csv',sep=',')

#%%
#Max mesquite slice
a1=mes_max#.loc[:'2017']
a2=mes_max_iqr#.loc[:'2017']
aP_mes.index=mes.index
#%%Max cottonwood slice
b1=cw_max#.loc[:'2017']
b2=cw_max_iqr#.loc[:'2017']
aP_cw.index=cw.index
#%% Error around max mesquite for aP -max savi plot
pp=pd.DataFrame(mes_max_iqr[mes_max_iqr.index.isin(aP_mes.index)])
pp=np.asarray(pp['iqr'])
ppp=pd.DataFrame(mes_max_iqr[mes_max_iqr.index.isin(aP_mes_val.index.year)])
ppp=np.asarray(ppp['iqr'])
#%% Error around max cottonwood for aP-max savi plot
qq=pd.DataFrame(cw_max_iqr[cw_max_iqr.index.isin(aP_cw.index)])
qq=np.asarray(qq['iqr'])
qqq=pd.DataFrame(cw_max_iqr[cw_max_iqr.index.isin(aP_cw_val.index.year)])
qqq=np.asarray(qqq['iqr'])
#%% Error around DTG - change which dtg 
n=6 #mesquite
dd=np.ones(n)*0.17 #gw23.std() # gw17_norm.std =0.07
gwc=gw_max_cw17 #or gw_max_cw17 for wet reach 
n=6 #cottonwood
d=np.ones(n)*0.17 #gw17.std() 0.07 for DTG17
gwm=gw_max_mes17 #or anything else 
#%% Error around max savi - dtg plot 
mm=pd.DataFrame(mes_max_iqr[mes_max_iqr.index.isin(gwm.index.year)])
mm=np.asarray(mm['iqr'])
cc=pd.DataFrame(cw_max_iqr[cw_max_iqr.index.isin(gwc.index.year)])
cc=np.asarray(cc['iqr'])
#%%
plt.subplot(422)
plt.bar(aP_mes.index,aP_mes['aP'],color='blue',align='center',alpha =0.6)
plt.bar(aP_mes_val.index.year,aP_mes_val['aP'],color='grey',align='center',alpha =0.6)
plt.grid(alpha=0.2)
plt.ylabel('antecedent aP (mm)',fontsize='small')
plt.twinx()
plt.plot(a1,marker='.',color='orange')
plt.fill_between(a2.index,a1['median']-a2['iqr'], a1['median']+a2['iqr'],  color='grey',edgecolor='none',alpha=0.2)
plt.ylim(0.2,0.5)
plt.ylabel('max SAVI',fontsize='small')


plt.subplot(421)
plt.bar(aP_cw.index,aP_cw['aP'],color='blue',align='center',alpha=0.6)
plt.bar(aP_cw_val.index.year,aP_cw_val['aP'],color='grey',align='center',alpha=0.6)
plt.ylabel('antecedent aP (mm)',fontsize='small')
plt.grid(alpha=0.2)
plt.twinx()
plt.plot(b1,marker='.',color='green')
plt.fill_between(b2.index,b1['median']-b2['iqr'], b1['median']+b2['iqr'],  color='grey',edgecolor='none',alpha=0.2)
plt.ylim(0.2,0.5)
plt.ylabel('max SAVI',fontsize='small')

plt.subplot(424)
plt.grid(alpha=0.3)

plt.errorbar(o, p,yerr=pp,fmt='o',linestyle='none',color='blue')    
plt.plot(o,p,'.', new_o,new_p,markersize=5.0,color='blue')#,label="y={0:.4f}*x+{1:.4f}".format(e,f)) #1D
plt.fill_between(o_line, p_line + pig, p_line - pig, color = 'grey', alpha=0.2,label = '95% prediction interval',edgecolor='none')
plt.errorbar(aP_mes_val['aP'], mes_val['median'],yerr=ppp,fmt='v',linestyle='none',color='grey')    
plt.ylabel('max SAVI',fontsize='small')
plt.ylim(0.2,0.5)
plt.xlim(-7,165)
plt.xlabel('antecedent aP (mm)',fontsize='small')
plt.xticks(fontsize='small')


plt.subplot(423)
plt.grid(alpha=0.3)

plt.errorbar(g, h,yerr=qq,fmt='o',linestyle='none',color='blue',label='Pearsonr=-0.54,p=0.01 ')    
plt.plot(g,h,'.', new_g,new_h,markersize=5.0,color='blue')#,label="y={0:.4f}*x+{1:.4f}".format(e,f)) #1D
plt.fill_between(g_line, h_line + pin, h_line - pin, color = 'grey', alpha=0.2,label = '95% prediction interval',edgecolor='none')
plt.errorbar(aP_cw_val['aP'], cw_val['median'],yerr=qqq,fmt='v',linestyle='none',color='grey')    
plt.ylabel('max SAVI',fontsize='small')
plt.ylim(0.2,0.5)
plt.xlim(-7,165)
plt.xlabel('antecedent aP (mm)',fontsize='small')
plt.xticks(fontsize='small')

plt.subplot(426)
plt.plot(a1['median'],marker='.',color='black')
plt.fill_between(a1.index,a1['median']-a2['iqr'], a1['median']+a2['iqr'],  color='grey',edgecolor='none',alpha=0.2)
plt.ylabel('max SAVI',fontsize='small')
plt.ylim(0.2,0.5)
plt.grid(alpha=0.3)
plt.twinx()

plt.errorbar(gwm.index.year,np.asarray(gwm),yerr=d,color='blue',linestyle='none',marker='d')
plt.ylim(1,0)
plt.ylabel('DTG (m)',fontsize='small')
plt.xticks(fontsize='small')

plt.subplot(425)
plt.plot(b1['median'],marker='.',color='black')
plt.fill_between(b1.index,b1['median']-b2['iqr'], b1['median']+b2['iqr'],  color='grey',edgecolor='none',alpha=0.2)
plt.ylim(0.2,0.5)
plt.ylabel('max SAVI',fontsize='small')
#plt.ylim(0.01,0.24)
plt.grid(alpha=0.3)
plt.twinx()
plt.xticks(fontsize='small')

plt.errorbar(gwc.index.year,np.asarray(gwc),yerr=dd,color='blue',linestyle='none',marker='d')
plt.errorbar(gw17_norm_val.index.year,gw17_norm_val['DTG17'],yerr=1,color='grey')
plt.ylim(1,0)
plt.ylabel('DTG (m)',fontsize='small')

plt.subplot(428)
#mmm=pd.DataFrame(mes_max_iqr[mes_max_iqr.index.isin(gw_max_mes10.index.year)])
#mmm=np.asarray(mmm['iqr'])

plt.errorbar(x, y,yerr=mm,fmt='o',linestyle='none',color='black',label='Pearsonr=-0.56,p=0.07')
plt.plot(x,y,'.', new_x,new_y,markersize=5.0,color='black')#,label="y={0:.4f}*x+{1:.4f}".format(a,b)) #1D
#plt.errorbar(gw_max_mes10['DTG10'].values.flatten(),mes_max10.values.flatten(),yerr=mmm,linestyle='none',marker='v',color='grey')
plt.fill_between(x_line, y_line + pic, y_line - pic, color = 'grey', alpha=0.2,label = '95% prediction interval',edgecolor='none')
plt.ylim(0.2,0.55)
plt.xlim(0.6,1)
#plt.legend(fontsize='small',frameon=False)
plt.ylabel('max SAVI',fontsize='small')
plt.xlabel('DTG (m)',fontsize='small')
plt.grid(alpha=0.3)
plt.xticks(fontsize='small')

plt.subplot(427)
#ccc=pd.DataFrame(cw_max_iqr[cw_max_iqr.index.isin(gw_max_cw10.index.year)])
#ccc=np.asarray(ccc['iqr'])

plt.errorbar(s, t,yerr=cc,fmt='o',linestyle='none',color='black',label='Pearsonr=-0.67,p=0.01 ')    
plt.plot(s,t,'.', new_s,new_t,markersize=5.0,color='black')#,label="y={0:.4f}*x+{1:.4f}".format(c,d)) #1D
#plt.errorbar(gw_max_cw10['DTG10'].values.flatten(),cw_max10.values.flatten(),linestyle='none',yerr=ccc,marker='v',color='grey')
plt.grid(alpha=0.2)
plt.fill_between(s_line, t_line + pic, t_line - pic, color = 'grey', alpha=0.2,label = '95% prediction interval',edgecolor='none')
#plt.legend(fontsize='small',frameon=False)
plt.ylabel('max SAVI',fontsize='small')
plt.xlabel('DTG (m)',fontsize='small')
plt.ylim(0.2,0.55)
plt.xlim(0.6,1)
plt.xticks(fontsize='small')

#%%% Multiple regression using aP and DTG to determine SAVI

plt.subplot(221)
plt.plot(a1['median'],marker='.',color='black')
plt.fill_between(a1.index,a1['median']-a2['iqr'], a1['median']+a2['iqr'],  color='grey',edgecolor='none',alpha=0.2)
plt.ylabel('POS',fontsize='small')
plt.ylim(0.2,0.5)
plt.grid(alpha=0.3)
plt.twinx()

plt.errorbar(gw_max_mes17.index.year, gw_max_mes17['DTG17'],yerr=mm,fmt='o',linestyle='none',color='blue',label='Pearsonr=-0.56,p=0.07')
plt.errorbar(gw_max_mes17_val.index.year,gw_max_mes17_val['DTG17'],yerr=mm1,fmt='d',linestyle='none',color='grey')
plt.ylim(1,0)
plt.ylabel('DTG (m)',fontsize='small')
plt.xticks(fontsize='small')

plt.subplot(222)
plt.plot(b1['median'],marker='.',color='black')
plt.fill_between(b1.index,b1['median']-b2['iqr'], b1['median']+b2['iqr'],  color='grey',edgecolor='none',alpha=0.2)
plt.grid(alpha=0.3)
plt.twinx()
plt.errorbar(gw_max_cw17.index.year, gw_max_cw17['DTG17'],yerr=mm,fmt='o',linestyle='none',color='blue',label='Pearsonr=-0.56,p=0.07')
plt.errorbar(gw_max_cw17_val.index.year,gw_max_cw17_val['DTG17'],yerr=mm1,fmt='d',linestyle='none',color='grey')
#plt.ylim(0.2,0.5)
plt.ylabel('POS',fontsize='small')
#plt.ylim(0.01,0.24)
plt.xticks(fontsize='small')
plt.ylim(1,0)
plt.ylabel('DTG (m)',fontsize='small')

plt.subplot(211)
mm=pd.DataFrame(mes_max_iqr[mes_max_iqr.index.isin(gw_max_mes17.index.year)])
mm=np.asarray(mm['iqr'])
mm1=pd.DataFrame(mes_max_iqr[mes_max_iqr.index.isin(gw_max_mes17_val.index.year)])
mm1=np.asarray(mm1['iqr'])

plt.errorbar(x, y,yerr=mm,fmt='o',linestyle='none',color='blue',label='Pearsonr=-0.56,p=0.07')
plt.plot(x,y,'.', new_x,new_y,markersize=5.0,color='black')#,label="y={0:.4f}*x+{1:.4f}".format(a,b)) #1D
#plt.errorbar(gw_max_mes10['DTG10'].values.flatten(),mes_max10.values.flatten(),yerr=mmm,linestyle='none',marker='v',color='grey')
plt.fill_between(x_line, y_line + pic, y_line - pic, color = 'grey', alpha=0.2,label = '95% prediction interval',edgecolor='none')
plt.errorbar(gw_max_mes17_val['DTG17'],mes_max17_val['median'],yerr=mm1,fmt='d',linestyle='none',color='grey')

plt.ylim(0.2,0.55)
plt.xlim(0.6,1)
#plt.legend(fontsize='small',frameon=False)
plt.ylabel('POS',fontsize='small')
plt.xlabel('DTG (m)',fontsize='small')
plt.grid(alpha=0.3)
plt.xticks(fontsize='small')

plt.subplot(212)
cc=pd.DataFrame(cw_max_iqr[cw_max_iqr.index.isin(gw_max_cw17.index.year)])
cc=np.asarray(cc['iqr'])
cc1=pd.DataFrame(cw_max_iqr[cw_max_iqr.index.isin(gw_max_cw17_val.index.year)])
cc1=np.asarray(cc1['iqr'])

plt.errorbar(s, t,yerr=cc,fmt='o',linestyle='none',color='blue',label='Pearsonr=-0.67,p=0.01 ')    
plt.plot(s,t,'.', new_s,new_t,markersize=5.0,color='black')#,label="y={0:.4f}*x+{1:.4f}".format(c,d)) #1D
plt.errorbar(gw_max_cw17_val['DTG17'],cw_max17_val['median'],yerr=cc1,fmt='d',linestyle='none',color='grey')
plt.grid(alpha=0.2)
plt.fill_between(s_line, t_line + pic, t_line - pic, color = 'grey', alpha=0.2,label = '95% prediction interval',edgecolor='none')
#plt.legend(fontsize='small',frameon=False)
plt.ylabel('POS',fontsize='small')
plt.xlabel('DTG (m)',fontsize='small')
plt.ylim(0.2,0.55)
plt.xlim(0.6,1)
plt.xticks(fontsize='small')




