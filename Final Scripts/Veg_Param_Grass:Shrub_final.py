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
from scipy.signal import find_peaks
from scipy.stats import iqr
from scipy.stats import pearsonr
from datetime import datetime,timedelta

#%%
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


#%% Data Input - Walnut Gulch/input_1D

#Evapotranspiration for a central location in the Walnut Gulch - from dPET dataset 
idx=pd.date_range('01-01-1981','12-31-2019',freq='D') #daily daterange


dir = r'/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/dPET_WG' #go into dir in file 
files = [i for i in glob.glob('*.txt')]

data=[pd.read_csv(file,sep=',',header=None, engine='python')
      for file in files]

data_PET_WG = pd.concat(data,ignore_index=True)
data_PET_WG.columns=['PET']
data_PET_WG =data_PET_WG.set_index(idx)
data_PET_WG['Date']=data_PET_WG.index
data_PET_WG=data_PET_WG.loc[(data_PET_WG.index < '2012-01-01') | (data_PET_WG.index > '2012-12-31')]

#%%
#Walnut GUlch Rain gauges for grass
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/dP_WG/precip_shrub_1.csv"
data_P_shrub1 = pd.read_csv(file,sep=';')
data_P_shrub1['Date']= pd.to_datetime(data_P_shrub1[["Year", "Month", "Day"]])
data_P_shrub1 = data_P_shrub1.set_index(pd.DatetimeIndex(data_P_shrub1['Date']))
idx=pd.date_range('01-01-1981','09-23-1999',freq='D')
data_P_shrub1=data_P_shrub1.reindex(idx).replace(np.nan, 0)
data_P_shrub1['Date']=data_P_shrub1.index
data_P_shrub1.columns = data_P_shrub1.columns.str.replace("[_]", " ")


file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/dP_WG/precip_shrub_2.csv"
data_P_shrub2 = pd.read_csv(file,sep=';')
data_P_shrub2['Date']= pd.to_datetime(data_P_shrub2[["Year", "Month", "Day"]])
data_P_shrub2 = data_P_shrub2.set_index(pd.DatetimeIndex(data_P_shrub2['Date']))
idx=pd.date_range('01-01-2000','12-31-2021',freq='D')
data_P_shrub2=data_P_shrub2.reindex(idx).replace(np.nan, 0)
data_P_shrub2['Date']=data_P_shrub2.index
data_P_shrub2=data_P_shrub2.drop('Gage 92',1)

data_P_shrub = pd.concat([data_P_shrub1,data_P_shrub2],join='inner',axis=0)

data_P_shrub=data_P_shrub.drop(columns=['Year','Month','Day'])
#data_P_shrub=data_P_shrub.loc[(data_P_shrub.index < '2012-01-01') | (data_P_shrub.index > '2012-12-31')]

#%%
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/dP_WG/precip_grass_1.csv"
data_P_grass1 = pd.read_csv(file,sep=';')
data_P_grass1['Date']= pd.to_datetime(data_P_grass1[["Year", "Month", "Day"]])
data_P_grass1 = data_P_grass1.set_index(pd.DatetimeIndex(data_P_grass1['Date']))
idx=pd.date_range('01-01-1981','10-16-1999',freq='D')
data_P_grass1=data_P_grass1.reindex(idx).replace(np.nan, 0)
data_P_grass1['Date']=data_P_grass1.index
data_P_grass1.columns = data_P_grass1.columns.str.replace("[_]", " ")


file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/dP_WG/precip_grass_2.csv"
data_P_grass2 = pd.read_csv(file,sep=';')
data_P_grass2['Date']= pd.to_datetime(data_P_grass2[["Year", "Month", "Day"]])
data_P_grass2 = data_P_grass2.set_index(pd.DatetimeIndex(data_P_grass2['Date']))
idx=pd.date_range('01-01-2000','01-31-2021',freq='D')
data_P_grass2=data_P_grass2.reindex(idx).replace(np.nan, 0)
data_P_grass2['Date']=data_P_grass2.index
data_P_grass2=data_P_grass2.drop('Gage 88',1)

data_P_grass = pd.concat([data_P_grass1,data_P_grass2],join='inner',axis=0)

data_P_grass=data_P_grass.drop(columns=['Year','Month','Day'])

data_P_grass=data_P_grass.loc[(data_P_grass.index < '2012-01-01') | (data_P_grass.index > '2012-12-31')]
#%%
data_PET_WG['WY'] = data_PET_WG.apply(lambda x: assign_wy(x), axis=1)
data_PET_WG['DOWY'] = data_PET_WG.index.to_series().apply(day_of_water_year)
data_PET_WG['16D']=data_PET_WG['PET'].rolling(16).sum().round(2) #2weeks
data_PET_WG['31D']=data_PET_WG['PET'].rolling(31).sum().round(2) #1month
data_PET_WG['62D']=data_PET_WG['PET'].rolling(62).sum().round(2) #2months
data_PET_WG['94D']=data_PET_WG['PET'].rolling(94).sum().round(2) #3months
data_PET_WG['126D']=data_PET_WG['PET'].rolling(126).sum().round(2) #3months
data_PET_WG['145D']=data_PET_WG['PET'].rolling(145).sum().round(2) #3months

data_PET_WG=data_PET_WG.round(2)
ag_pet=data_PET_WG.loc['1994-01-01':'2019-10-31']
#ag_pet = ag_pet.loc[(ag_pet.index < '2011-11-01') | (ag_pet.index > '2013-10-31')]
#ag_pet=pd.DataFrame(ag_pet[ag_pet.index.isin(ag_gr.index)])

#%%
rain = pd.DataFrame(data_P_shrub.iloc[:,0:15].mean(axis=1))
#rain=data_P_SP
rain.columns=['P']
rain=rain.interpolate()
rain['Date'] =rain.index
rain['WY'] = rain.apply(lambda x: assign_wy(x), axis=1)
rain['DOWY'] = rain.index.to_series().apply(day_of_water_year)
rain['16D']=rain['P'].rolling(16).sum().round(2) #1month
rain['31D']=rain['P'].rolling(31).sum().round(2) #1month
rain['62D']=rain['P'].rolling(62).sum().round(2) #1month
rain['80D']=rain['P'].rolling(80).sum().round(2) #3months
rain['94D']=rain['P'].rolling(94).sum().round(2) #3months

ag_rain=rain.loc['1994-01-01':]
#%%
file = r"/Users/mariawarter/Box Sync/PhD/PhD/Riparian Model/Data/dT_SP/ESRL_temp_1986_2020.csv"

data_T = pd.read_csv(file,dayfirst=True,parse_dates=True, index_col=[0])
data_T = data_T.set_index(pd.DatetimeIndex(data_T['date']))  
data_T.columns=['Date','maxT','minT']
data_T['Date'] = pd.to_datetime(data_T.index) #set P data to 1986-2019
#data_T=data_T.loc[(data_T.index < '2012-01-01') | (data_T.index > '2012-12-31')]
#data_T = data_T.loc[:'2019-12-31'] #data only starts in 1986

data_T['WY'] = data_T.apply(lambda x: assign_wy(x), axis=1)
data_T['DOWY'] = data_T.index.to_series().apply(day_of_water_year)
data_T['mean']= (data_T['maxT'] + data_T['minT'] ) /2
data_T['31D']=data_T['maxT'].rolling(31).mean().round(2) #1month
data_T['62D']=data_T['maxT'].rolling(62).mean().round(2) #1month
data_T['94D']=data_T['maxT'].rolling(94).mean().round(2) #1month
data_T['80D']=data_T['maxT'].rolling(80).mean().round(2) #1month

ag_temp=data_T.loc['1994-01-01':'2019-10-31']
#ag_temp = ag_temp.loc[(ag_temp.index < '2011-11-01') | (ag_temp.index > '2013-10-31')]
#ag_temp=pd.DataFrame(ag_temp[ag_temp.index.isin(ag_gr.index)])
#ag_temp_Y = pd.DataFrame(ag_temp['maxT'].groupby(ag_temp['WY']).idxmax())
#ax_temp = pd.pivot_table(ag_temp, index=ag_temp['DOWY'], columns=ag_temp['WY'],values='62D')

#%%SAVI for all veg types 
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

#%%
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

#%%
SAVI['Date'] = pd.to_datetime(SAVI.index)

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
SAVI_grassland=SAVI

#%%
SAVI_shrub=SAVI

#%%SHRUB/GRASS PET AND P

ag_gr=SAVI_grassland.loc['1994-11':]
ag_gr['median'] = ag_gr.iloc[:,0:30].median(axis=1)
ag_gr['iqr']=iqr(ag_gr.iloc[:,0:30],axis=1)

ax_gr = pd.pivot_table(ag_gr, index=ag_gr['DOWY'], columns=ag_gr['WY'],values='median')

#%%
ag_shr =SAVI_shrub.loc['1994-11':]

ag_shr['median'] = ag_shr.median(axis=1)
ag_shr['iqr']=iqr(ag_shr.iloc[:,0:30],axis=1)

#ag_gr.to_csv('ag_grass_WG.csv',sep=',')
#ag_shr.to_csv('ag_shrub_WG.csv',sep=',')
ax_shr = pd.pivot_table(ag_shr, index=ag_shr['DOWY'], columns=ag_shr['WY'],values='median')

 #%% Split of dfs into CAL and VAL data sets - select every second year for CAL 

ag_rain_cal = ag_rain[ag_rain.index.year.isin([1995,1997,1999,2001,2003,2005,2007,2009,2011,2013,2015,2017,2019,2020,2021])]
ag_gr_cal=ag_gr[ag_gr.index.year.isin([1995,1997,1999,2001,2003,2005,2007,2009,2011,2013,2015,2017,2019,2020,2021])]
ag_shr_cal=ag_shr[ag_shr.index.year.isin([1995,1997,1999,2001,2003,2005,2007,2009,2011,2013,2015,2017,2019,2020,2021])]
#g_temp_cal = ag_temp[ag_temp['WY'].isin([1995,1997,2001,2003,2005,2007,2009,2011,2013,2015,2017,2019,2020,2021])]
#ag_pet_cal = ag_pet[ag_pet['WY'].isin([1995,1997,1999,2001,2003,2005,2007,2009,2011,2013,2015,2017,2019])]
#%%
ag_rain_cal.to_csv('ag_rain_cal.csv',sep=',')
ag_gr_cal.to_csv('ag_gr_cal.csv',sep=',')
ag_shr_cal.to_csv('ag_shr_cal.csv',sep=',')
ag_temp_cal.to_csv('ag_temp_cal.csv',sep=',')

#%%
ag_rain_val=ag_rain[ag_rain['WY'].isin([1996,1998,2000,2002,2004,2006,2008,2010,2012, 2014,2016,2018,2020])]
ag_gr_val=ag_gr[ag_gr.index.year.isin([1996,1998,2000,2000,2002,2004,2006,2008,2010,2012,2014,2016,2018,2020])]
ag_shr_val=ag_shr[ag_shr.index.year.isin([1996,1998,2000,2002,2004,2006,2008,2010,2012,2014,2016,2018,2020])]
#ag_temp_val = ag_temp[ag_temp.index.year.isin([1996,1998,2000,2002,2004,2006,2008,2010,2012,2014,2016,2018,2020])]
#ag_pet_val = ag_pet[ag_pet.index.year.isin([1996,1998,2000,2002,2004, 2006,2008,2010,2012,2014,2016,2018,2020])]

#%%
ag_rain_val.to_csv('ag_rain_val.csv',sep=',')
ag_gr_val.to_csv('ag_gr_val_cy.csv',sep=',')
ag_shr_val.to_csv('ag_shr_val_cy.csv',sep=',')
ag_temp_val.to_csv('ag_temp_val_cy.csv',sep=',')
ag_pet_val.to_csv('ag_pet_val_cy.csv',sep=',')

#%%
#If i want to run all the values not just CAL

#ag_rain = ag_rain_cal
#ag_gr = ag_gr_cal
#ag_shr = ag_shr_cal
#ag_temp_cal=ag_temp
#%% Find all max values from all sampling points and corresponding rain
# need to add each point separately to the df, frist run test 1 and add to big df, then keep addng points
#In this dataframe all the max values each year from all 30 sampling points 

#Find max values for all points SRUBS
max_shr = pd.DataFrame(ag_shr['median'].groupby(ag_shr.index.year).max())
max_shr_id=pd.DataFrame(ag_shr['median'].groupby(ag_shr.index.year).idxmax())

max_gr = pd.DataFrame(ag_gr['median'].groupby(ag_gr.index.year).max())
max_gr_id=pd.DataFrame(ag_gr['median'].groupby(ag_gr.index.year).idxmax())

max_gr_iqr = pd.DataFrame(ag_gr['iqr'][ag_gr.index.isin(max_gr_id['median'])])
max_shr_iqr = pd.DataFrame(ag_shr['iqr'][ag_shr.index.isin(max_shr_id['median'])])

#%%# Antecedent rain on date of max SAVI-SHRUBS
#CALIBRATION

rain_max_gr_cal= pd.DataFrame(ag_rain_cal['31D'][ag_rain_cal.index.isin(max_gr_id['median'])])

rain_max_shr_cal= pd.DataFrame(ag_rain_cal['31D'][ag_rain_cal.index.isin(max_shr_id['median'])])
#%%
max_gr_cal= pd.DataFrame(max_gr[max_gr.index.isin(rain_max_gr_cal.index.year)])
max_shr_cal= pd.DataFrame(max_shr[max_shr.index.isin(rain_max_shr_cal.index.year)])

#%%
stat, p = pearsonr(rain_max_gr_cal['31D'], max_gr_cal['median'])
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.1:
	print('Probably independent')
else:
	print('Probably dependent')
    
stat, p = pearsonr(rain_max_shr_cal['31D'], max_shr_cal['median'])
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.1:
	print('Probably independent')
else:
	print('Probably dependent')

0.86#%%

rain_max_gr_val= pd.DataFrame(ag_rain_val['31D'][ag_rain_val.index.isin(max_gr_id['median'])])
rain_max_shr_val= pd.DataFrame(ag_rain_val['31D'][ag_rain_val.index.isin(max_shr_id['median'])])
#%%
max_gr_val=pd.DataFrame(max_gr['median'][max_gr.index.isin(rain_max_gr_val.index.year)])
max_shr_val=pd.DataFrame(max_gr['median'][max_gr.index.isin(rain_max_shr_val.index.year)])


#%%D
#Timeseries of Max SAVI and Antecedent Precipitation  with median values 
a=np.asarray(max_gr['median'])
b=np.asarray(max_gr_iqr['iqr'])
c=np.asarray(max_shr['median'])
d=np.asarray(max_shr_iqr['iqr'])
rain_std=ag_rain['31D'].std()
rain_std1=ag_rain['31D'].std()


plt.subplot(221)
plt.plot(max_gr.index, a,marker='.',color='green') 
plt.fill_between(max_gr.index,a-b, a+b,  color='grey',alpha=0.2,edgecolor='none')
plt.grid(alpha=0.3)
plt.ylabel('POS_2')
plt.ylim(0.05,0.45)
plt.subplot(222)
plt.plot(max_shr.index, c,marker='.',color='orange') 
plt.fill_between(max_shr.index,c-d, c+d,  color='grey',alpha=0.2,edgecolor='none')
plt.ylabel('POS_2')
plt.ylim(0.05,0.45)
plt.grid(alpha=0.3)

plt.subplot(223)
plt.bar(rain_max_gr_cal.index.year,rain_max_gr_cal['31D'],color='blue')
plt.bar(rain_max_gr_val.index.year,rain_max_gr_val['31D'],color='grey')
plt.ylabel('antecedent Precipitation (mm)',fontsize='small')
plt.grid(alpha=0.3)
plt.ylim(-10,220)
plt.subplot(224)
plt.bar(rain_max_shr_cal.index.year,rain_max_shr_cal['31D'],color='blue')
plt.bar(rain_max_shr_val.index.year,rain_max_shr_val['31D'],color='grey')
plt.grid(alpha=0.3)
plt.ylabel('antecedent Precipitation (mm)',fontsize='small')
plt.ylim(-10,220)
#%%
x=rain_max_gr_cal['31D'].values.flatten()
y=max_gr_cal['median'].values.flatten()

new_x= np.linspace(x.min(),x.max()) #for max SAVI
x_line = np.linspace(0, 220, 220) #MAX
#x_line=np.sort(x)

#%%

x=rain_max_shr_cal['31D'].values.flatten()
y=max_shr_cal['median'].values.flatten() 
new_x= np.linspace(0,220) #for max SAVI
x_line = np.linspace(0, 220, 220) #MAX
#%%
coeffs = np.polyfit(x,y,1)
a,b = coeffs
y_model = np.polyval([a,b], x)
y_line = np.polyval([a,b], x_line)
poly = np.poly1d(coeffs)

new_y = poly(new_x)
#Statistics
x_mean=np.mean(x)
y_mean=np.mean(y)
n=x.size                #number of samples
m=3                   #number of parameters (3 parameters abc)
dof=n-m   
t=stats.t.ppf(0.975,dof)  #alpha = 0.05, student statistic of interval confidence 

residual = y-y_model
std_error = np.sqrt(np.sum(residual**2) / dof)

pi = t * std_error * np.sqrt(1 + 1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))
print(coeffs)

#%%
err1=pd.DataFrame(max_gr_iqr['iqr'][max_gr_iqr.index.year.isin(max_gr_cal.index)])
err2=pd.DataFrame(max_gr_iqr['iqr'][max_gr_iqr.index.year.isin(max_gr_val.index)])

err3=pd.DataFrame(max_shr_iqr['iqr'][max_shr_iqr.index.year.isin(max_shr_cal.index)])
err4=pd.DataFrame(max_shr_iqr['iqr'][max_shr_iqr.index.year.isin(max_shr_val.index)])
#%%
plt.subplot(121)
plt.errorbar(rain_max_gr_cal['31D'],max_gr_cal['median'],yerr=err1['iqr'],linestyle='none',marker='o',color='blue')
plt.errorbar(rain_max_gr_val['31D'],max_gr_val['median'],yerr=err2['iqr'],linestyle='none',marker='v',color='grey')
plt.plot(x,y,'.', new_x,new_y,color='blue', label="y={0:e}*x+{1:.4f}".format(a,b))
plt.fill_between(x_line, y_line + pi, y_line - pi, color = 'grey', alpha=0.2,label = '95% prediction interval',edgecolor='none')
plt.legend(frameon=False)
plt.xlabel('Antecedent P (mm)')
plt.ylabel('SOM SAVI')
plt.ylabel('POS_2')
plt.subplot(122)
plt.errorbar(rain_max_shr_cal['31D'],max_shr_cal['median'],yerr=err3['iqr'],linestyle='none',marker='o',color='blue')
plt.errorbar(rain_max_shr_val['31D'],max_shr_val['median'],yerr=err4['iqr'],linestyle='none',marker='v',color='grey')
plt.plot(x,y,'.', new_x,new_y,color='blue', label="y={0:e}*x+{1:.4f}".format(a,b))
plt.fill_between(x_line, y_line + pi, y_line - pi, color = 'grey', alpha=0.2,label = '95% prediction interval',edgecolor='none')
plt.legend(frameon=False)
plt.xlabel('Antecedent P (mm)')
plt.ylabel('SOM SAVI')
plt.ylabel('POS_2')



#%%

m_rain= np.diff(ag_rain['P'])  #used max T to calculate 64D average 
m_rain=pd.DataFrame(m_rain)
m_rain.columns=['slope']

m_rain=m_rain.set_index(ag_rain[:-1].index)
m_rain['Date'] =m_rain.index
m_rain['WY'] = m_rain.apply(lambda x: assign_wy(x), axis=1)
m_rain['DOWY'] =m_rain.index.to_series().apply(day_of_water_year)
#%%
m_rain=pd.pivot_table(m_rain, index=m_rain['DOWY'], columns=m_rain['WY'],values='slope')
spring_P=m_rain.mean(axis=1)

#%%

plt.subplot(211)
plt.plot(cum_P.cumsum(), color='black')
plt.ylabel('Precipitation (mm)')
plt.grid(alpha=0.2)
plt.subplot(212)
plt.plot(spring_P)
plt.grid(alpha=0.2)
plt.axhline(0,color='grey')
plt.axvline(173)
plt.ylabel('Precipitation (mm)')

#%%START OF MONSOON SEASON SAVI
#start of the monsoon season - determine DOWY
ag_rain_cal = ag_rain[ag_rain['WY'].isin([1995,1997,2001,2003,2005,2007,2009,2011,2013,2015,2017,2021])]
ag_gr_cal=ag_gr[ag_gr['WY'].isin([1995,1997,2001,2003,2005,2007,2009,2011,2013,2015,2017,2021])]
ag_shr_cal=ag_shr[ag_shr['WY'].isin([1995,1997,2001,2003,2005,2007,2009,2011,2013,2015,2017,2021])]
#%%
ag_rain_val=ag_rain[ag_rain['WY'].isin([1996,1998,1999,2000,2002,2004,2006,2008,2010,2014,2016,2018,2019,2020])]
ag_gr_val=ag_gr[ag_gr['WY'].isin([1996,1998,2000,1999,2000,2002,2004,2006,2008,2010,2012,2014,2016,2018,2019,2020])]
ag_shr_val=ag_shr[ag_shr['WY'].isin([1996,1998,1999,2000,2000,2002,2004,2006,2008,2010,2012,2014,2016,2018,2019,2020])]
#%%
monsoon_months=[6,7,8,9]

start_monsoon= pd.DataFrame(m_rain['slope'].groupby(m_rain['WY']).cumsum()) #m_rain must NOT be pivoted, check furhter up
start_monsoon['Date'] =start_monsoon.index
start_monsoon['WY'] = start_monsoon.apply(lambda x: assign_wy(x), axis=1)
start_monsoon['DOWY'] = start_monsoon.index.to_series().apply(day_of_water_year)
start_monsoon= start_monsoon[start_monsoon['Date'].map(lambda t: t.month in monsoon_months)]

#identify date of start of monsoon - where elbow is 
date_mon= pd.DataFrame(start_monsoon.groupby("WY")["slope"].idxmin()) #index of min value=start of monsoon season
date_mon=pd.DataFrame(date_mon['slope'] + timedelta(days=30))

date_mon=date_mon.set_index(date_mon.slope)

#%%ANTECEDENT RAIN BEFORE MONSOON (incl. spring rain)
rain_mon_cal = pd.DataFrame(ag_rain_cal['80D'][ag_rain_cal.index.isin(date_mon.index)])

som_shr_cal = ag_shr_cal['median'][ag_shr_cal.index.isin(date_mon.index)]
som_gr_cal = ag_gr_cal['median'][ag_gr_cal.index.isin(date_mon.index)]

#%%
som_shr_val = ag_shr_val['median'][ag_shr_val.index.isin(date_mon.index)]
som_gr_val = ag_gr_val['median'][ag_gr_val.index.isin(date_mon.index)]
rain_mon_val=pd.DataFrame(ag_rain_val['80D'][ag_rain_val.index.isin(date_mon.index)])

#%%
stat, p = pearsonr(rain_mon_cal['80D'], som_shr_cal)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.1:
	print('Probably independent')
else:
	print('Probably dependent')
    
stat, p = pearsonr(rain_mon_cal['80D'], som_gr_cal)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.1:
	print('Probably independent')
else:
	print('Probably dependent')
    
plt.scatter(rain_mon_cal['80D'], som_gr_cal,color='green')
plt.scatter(rain_mon_cal['80D'], som_shr_cal,color='orange')

#%%
x=rain_mon_cal.values.flatten()
y=som_gr_cal.values.flatten()
col='green'
new_x = np.linspace(0, 90) #SOM
x_line = np.linspace(0, 90, 100)
#%%
x=rain_mon_cal.values.flatten()
y=som_shr_cal.values.flatten()
col='orange'
new_x = np.linspace(0, 90) #SOM
x_line = np.linspace(0, 90, 100) #EOS
#%%
coeffs = np.polyfit(x,y,1)
a,b = coeffs
y_model = np.polyval([a,b], x)
y_line = np.polyval([a,b], x_line)
poly = np.poly1d(coeffs)

new_y = poly(new_x)
#Statistics
x_mean=np.mean(x)
y_mean=np.mean(y)
n=x.size                #number of samples
m=3                   #number of parameters (3 parameters abc)
dof=n-m   
t=stats.t.ppf(0.975,dof)  #alpha = 0.05, student statistic of interval confidence 

residual = y-y_model
std_error = np.sqrt(np.sum(residual**2) / dof)

pi = t * std_error * np.sqrt(1 + 1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))
print(coeffs)

#%%
err1=pd.DataFrame(ag_gr['iqr'][ag_gr.index.isin(rain_mon_cal.index)])
err2=pd.DataFrame(ag_gr['iqr'][ag_gr.index.isin(rain_mon_val.index)])

err3=pd.DataFrame(ag_shr['iqr'][ag_shr.index.isin(rain_mon_cal.index)])
err4=pd.DataFrame(ag_shr['iqr'][ag_shr.index.isin(rain_mon_val.index)])
#%%
plt.subplot(121)
plt.errorbar(rain_mon_cal['80D'],som_gr_cal,yerr=err1['iqr'],linestyle='none',marker='o',color='black')
plt.errorbar(rain_mon_val['80D'],som_gr_val,yerr=err2['iqr'],linestyle='none',marker='v',color='grey')
plt.plot(x,y,'.', new_x,new_y,color='black', label="y={0:e}*x+{1:.4f}".format(a,b))
plt.fill_between(x_line, y_line + pi, y_line - pi, color = 'grey', alpha=0.2,label = '95% prediction interval',edgecolor='none')
plt.legend(frameon=False)
plt.xlabel('Antecedent P (mm)')
plt.ylabel('SOM SAVI')
plt.subplot(122)
plt.errorbar(rain_mon_cal['80D'],som_shr_cal,yerr=err3['iqr'],linestyle='none',marker='o',color='black')
plt.errorbar(rain_mon_val['80D'],som_shr_val,yerr=err4['iqr'],linestyle='none',marker='v',color='grey')
plt.plot(x,y,'.', new_x,new_y,color='black', label="y={0:e}*x+{1:.4f}".format(a,b))
plt.fill_between(x_line, y_line + pi, y_line - pi, color = 'grey', alpha=0.2,label = '95% prediction interval',edgecolor='none')
plt.legend(frameon=False)
plt.xlabel('Antecedent P (mm)')
plt.ylabel('SOM SAVI')

#%%SPRING PEAK

spring_months=[4]

spring_gr_cal= ag_gr_cal['median'][ag_gr_cal['Date'].map(lambda t: t.month in spring_months)]
spring_shr_cal= ag_shr_cal['median'][ag_shr_cal['Date'].map(lambda t: t.month in spring_months)]
#%%
spring_rain_gr_cal = pd.DataFrame(ag_rain_cal['94D'][ag_rain_cal.index.isin(spring_gr_cal.index)])
spring_rain_shr_cal = pd.DataFrame(ag_rain_cal['94D'][ag_rain_cal.index.isin(spring_shr_cal.index)])
#%%

spring_gr_val= ag_gr_val['median'][ag_gr_val['Date'].map(lambda t: t.month in spring_months)]
spring_shr_val= ag_shr_val['median'][ag_shr_val['Date'].map(lambda t: t.month in spring_months)]

spring_rain_gr_val = pd.DataFrame(ag_rain_val['94D'][ag_rain_val.index.isin(spring_gr_val.index)])
spring_rain_shr_val = pd.DataFrame(ag_rain_val['94D'][ag_rain_val.index.isin(spring_shr_val.index)])

#%%#%%
err1=pd.DataFrame(ag_gr['iqr'][ag_gr.index.isin(spring_gr_cal.index)])
err2=pd.DataFrame(ag_gr['iqr'][ag_gr.index.isin(spring_gr_val.index)])

err3=pd.DataFrame(ag_shr['iqr'][ag_shr.index.isin(spring_shr_cal.index)])
err4=pd.DataFrame(ag_shr['iqr'][ag_shr.index.isin(spring_shr_val.index)])


#%%
stat, p = pearsonr(spring_rain_gr_cal['94D'], spring_gr_cal)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.1:
	print('Probably independent')
else:
	print('Probably dependent')
    
stat, p = pearsonr(spring_rain_shr_cal['94D'], spring_shr_cal)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.1:
	print('Probably independent')
else:
	print('Probably dependent')

plt.scatter(spring_rain_gr_cal['94D'], spring_gr_cal,color='green')
plt.scatter(spring_rain_shr_cal['94D'], spring_shr_cal,color='orange')

#%%
x=spring_rain_gr_cal.values.flatten()
y=spring_gr_cal.values.flatten()
col='green'
new_x = np.linspace(0,95) #SOM
x_line = np.linspace(0, 95, 100)
#%%
x=spring_rain_shr_cal.values.flatten()
y=spring_shr_cal.values.flatten()
col='orange'
new_x = np.linspace(0, 95) #SOM
x_line = np.linspace(0, 95, 100) #EOS
#%%
coeffs = np.polyfit(x,y,1)
a,b = coeffs
y_model = np.polyval([a,b], x)
y_line = np.polyval([a,b], x_line)
poly = np.poly1d(coeffs)

new_y = poly(new_x)
#Statistics
x_mean=np.mean(x)
y_mean=np.mean(y)
n=x.size                #number of samples
m=3                   #number of parameters (3 parameters abc)
dof=n-m   
t=stats.t.ppf(0.975,dof)  #alpha = 0.05, student statistic of interval confidence 

residual = y-y_model
std_error = np.sqrt(np.sum(residual**2) / dof)

pi = t * std_error * np.sqrt(1 + 1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))
print(coeffs)

#%%
plt.subplot(121)
plt.errorbar(spring_rain_gr_cal['94D'],spring_gr_cal,yerr=err1['iqr'],linestyle='none',marker='o',color='black')
plt.errorbar(spring_rain_gr_val['94D'],spring_gr_val,yerr=err2['iqr'],linestyle='none',marker='v',color='grey')
plt.plot(x,y,'.', new_x,new_y,color='black', label="y={0:e}*x+{1:.4f}".format(a,b))
plt.fill_between(x_line, y_line + pi, y_line - pi, color = 'grey', alpha=0.2,label = '95% prediction interval',edgecolor='none')
plt.legend(frameon=False)
plt.xlabel('Antecedent P (mm)')
plt.ylabel('Spring SAVI')
plt.subplot(122)
plt.errorbar(spring_rain_shr_cal['94D'],spring_shr_cal,yerr=err3['iqr'],linestyle='none',marker='o',color='black')
plt.errorbar(spring_rain_shr_val['94D'],spring_shr_val,yerr=err4['iqr'],linestyle='none',marker='v',color='grey')
plt.plot(x,y,'.', new_x,new_y,color='black', label="y={0:e}*x+{1:.4f}".format(a,b))
plt.fill_between(x_line, y_line + pi, y_line - pi, color = 'grey', alpha=0.2,label = '95% prediction interval',edgecolor='none')
plt.legend(frameon=False)
plt.xlabel('Antecedent P (mm)')
plt.ylabel('Spring SAVI')

#%% START/END OF THE SEASON 

sos_months=[1]

sos_gr_cal= ag_gr_cal['median'][ag_gr_cal['Date'].map(lambda t: t.month in sos_months)]
sos_shr_cal= ag_shr_cal['median'][ag_shr_cal['Date'].map(lambda t: t.month in sos_months)]

sos_rain_gr_cal = pd.DataFrame(ag_rain_cal['94D'][ag_rain_cal.index.isin(sos_gr_cal.index)])
sos_rain_shr_cal = pd.DataFrame(ag_rain_cal['94D'][ag_rain_cal.index.isin(sos_shr_cal.index)])
#%%

stat, p = pearsonr(sos_rain_gr_cal['94D'], sos_gr_cal)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.1:
	print('Probably independent')
else:
	print('Probably dependent')
    
stat, p = pearsonr(sos_rain_shr_cal['94D'], sos_shr_cal)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.1:
	print('Probably independent')
else:
	print('Probably dependent')
    
plt.scatter(sos_rain_gr_cal, sos_gr_cal,color='green')
plt.scatter(sos_rain_shr_cal, sos_shr_cal,color='orange')
#%%

sos_gr_val= ag_gr_val['median'][ag_gr_val['Date'].map(lambda t: t.month in sos_months)]
sos_shr_val= ag_shr_val['median'][ag_shr_val['Date'].map(lambda t: t.month in sos_months)]


sos_rain_gr_val = pd.DataFrame(ag_rain_val['94D'][ag_rain_val.index.isin(sos_gr_val.index)])
sos_rain_shr_val = pd.DataFrame(ag_rain_val['94D'][ag_rain_val.index.isin(sos_shr_val.index)])

sos_gr_val = pd.DataFrame(sos_gr_val[sos_gr_val.index.isin(sos_rain_gr_val.index)])
sos_shr_val = pd.DataFrame(sos_shr_val[sos_shr_val.index.isin(sos_rain_shr_val.index)])
#%%
x=sos_rain_gr_cal['94D'].values.flatten()
y=sos_gr_cal.values.flatten()
col='green'
new_x = np.linspace(0,220) #SOM
x_line = np.linspace(0, 220, 220)
#%%
x=sos_rain_shr_cal['94D'].values.flatten()
y=sos_shr_cal.values.flatten()
col='orange'
new_x = np.linspace(0, 220) #SOM
x_line = np.linspace(0, 220, 220) #EOS
#%%
coeffs = np.polyfit(x,y,1)
a,b = coeffs
y_model = np.polyval([a,b], x)
y_line = np.polyval([a,b], x_line)
poly = np.poly1d(coeffs)

new_y = poly(new_x)
#Statistics
x_mean=np.mean(x)
y_mean=np.mean(y)
n=x.size                #number of samples
m=3                   #number of parameters (3 parameters abc)
dof=n-m   
t=stats.t.ppf(0.975,dof)  #alpha = 0.05, student statistic of interval confidence 

residual = y-y_model
std_error = np.sqrt(np.sum(residual**2) / dof)

pi = t * std_error * np.sqrt(1 + 1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))
print(coeffs)

#%%
pi=pd.DataFrame(pi)
pi.to_csv('ci_sos_shr.csv',sep=',')

#%%
err1=pd.DataFrame(ag_gr['iqr'][ag_gr.index.isin(sos_gr_cal.index)])
err2=pd.DataFrame(ag_gr['iqr'][ag_gr.index.isin(sos_gr_val.index)])

err3=pd.DataFrame(ag_shr['iqr'][ag_shr.index.isin(sos_shr_cal.index)])
err4=pd.DataFrame(ag_shr['iqr'][ag_shr.index.isin(sos_shr_val.index)])
#%%
plt.subplot(121)
plt.errorbar(sos_rain_gr_cal['94D'],sos_gr_cal,yerr=err1['iqr'],linestyle='none',marker='o',color='black')
plt.errorbar(sos_rain_gr_val['94D'],sos_gr_val['median'],yerr=err2['iqr'],linestyle='none',marker='v',color='grey')
plt.plot(x,y,'.', new_x,new_y,color='black', label="y={0:e}*x+{1:.4f}".format(a,b))
plt.fill_between(x_line, y_line + pi, y_line - pi, color = 'grey', alpha=0.2,label = '95% prediction interval',edgecolor='none')
plt.legend(frameon=False)
plt.xlabel('Antecedent P (mm)')
plt.ylabel('End of Season SAVI')
plt.subplot(122)
plt.errorbar(sos_rain_shr_cal['94D'],sos_shr_cal,yerr=err3['iqr'],linestyle='none',marker='o',color='black')
plt.errorbar(sos_rain_shr_val['94D'],sos_shr_val['median'],yerr=err4['iqr'],linestyle='none',marker='v',color='grey')
plt.plot(x,y,'.', new_x,new_y,color='black', label="y={0:e}*x+{1:.4f}".format(a,b))
plt.fill_between(x_line, y_line + pi, y_line - pi, color = 'grey', alpha=0.2,label = '95% prediction interval',edgecolor='none')
plt.legend(frameon=False)
plt.xlabel('Antecedent P (mm)')
plt.ylabel('End of Season SAVI')

#%% DIFFERENCED SERIES - P,T, SAVI
#%%
m_temp= np.diff(ag_temp['80D'])  #used max T to calculate 64D average 
m_temp=pd.DataFrame(m_temp)
m_temp.columns=['slope']

m_temp=m_temp.set_index(ag_temp[:-1].index)
m_temp['Date'] =m_temp.index
m_temp['WY'] = m_temp.apply(lambda x: assign_wy(x), axis=1)
m_temp['DOWY'] = m_temp.index.to_series().apply(day_of_water_year)
#%%
m_temp=pd.pivot_table(m_temp, index=m_temp['DOWY'], columns=m_temp['WY'],values='slope')

b=m_temp.mean(axis=1) #ensemble mean 
b_std = m_temp.std(axis=1) #ensemble std
#%%
m= np.diff(ag_pet_cal['94D'])  #used max T to calculate 64D average 
m=pd.DataFrame(m)
m.columns=['slope']

m=m.set_index(ag_pet_cal[:-1].index)
m['Date'] =m.index
m['WY'] = m.apply(lambda x: assign_wy(x), axis=1)
m['DOWY'] = m.index.to_series().apply(day_of_water_year)

m_pet=pd.pivot_table(m, index=m['DOWY'], columns=m['WY'],values='slope')
m_pet=m_pet.dropna()
b1=m_pet.mean(axis=1)

b1_std = m_pet.std(axis=1)
#%%
#Slope for SAVI 
me=ag_shr
#%%
me=ag_gr
#%%
m= pd.DataFrame(np.diff(me['median'])) #series of all differences
m.columns=['slope']

m=m.set_index(me[:-1].index)
m['Date'] =m.index
m['WY'] = m.apply(lambda x: assign_wy(x), axis=1)
m['DOWY'] = m.index.to_series().apply(day_of_water_year)
#%%
m_savi_shr=pd.pivot_table(m, index=m['DOWY'], columns=m['WY'],values='slope')
d_shr=m_savi_shr.mean(axis=1)
d_shr_std=m_savi_shr.std(axis=1)
#%%

m_savi_gr=pd.pivot_table(m, index=m['DOWY'], columns=m['WY'],values='slope')
d_gr=m_savi_gr.mean(axis=1)
d_gr_std=m_savi_gr.std(axis=1)

#%%#%%plotting diff of SAVI and P for GRass and shrub

#always check d_std for odd value at the end
c=ax_rain.mean(axis=1)
c_std=ax_rain.std(axis=1)
#%%
x=b.index

plt.subplot(121)
plt.plot(b,label='T',color='red')
plt.fill_between(x,b-b_std, b+b_std,  color='red',alpha=0.1, edgecolor='red')
#plt.plot(c,label='P',color='blue')
#plt.fill_between(x,c-c_std, c+c_std,  color='blue',alpha=0.1, edgecolor='blue')
plt.ylabel('diffT')
plt.xlabel('Antecedent Precipiatation (mm)')
#plt.ylim(-20,20)
plt.legend(frameon=False, loc='upper left')
plt.twinx()
plt.plot(d_shr,color='green',label='diffSAVI',linewidth=1.5)
plt.fill_between(d_shr.index,d_shr-d_shr_std, d_shr+d_shr_std,  color='green',alpha=0.3, edgecolor='green')
plt.ylim(-0.005,0.005)
plt.legend(frameon=False,loc='upper right')
plt.ylabel('diffSAVI')
plt.xlabel('DOWY')
plt.axhline(0,color='black')

plt.subplot(122)
#plt.plot(c,label='P',color='blue')
#plt.fill_between(x,c-c_std, c+c_std,  color='blue',alpha=0.1, edgecolor='blue')
#plt.ylabel('P')
#plt.ylim(-20,20)
plt.plot(b,label='T',color='red')
plt.xlabel('Antecedent Precipiatation (mm)')
plt.fill_between(x,b-b_std, b+b_std,  color='red',alpha=0.1, edgecolor='red')
plt.ylabel('diffT')
plt.legend(frameon=False, loc='upper left')
plt.twinx()
plt.plot(d_gr,color='g',label='diffSAVI',linewidth=1.5)
plt.fill_between(d_gr.index,d_gr-d_gr_std, d_gr+d_gr_std,  color='green',alpha=0.3, edgecolor='green')
plt.ylim(-0.005,0.005)
plt.legend(frameon=False,loc='upper right')
plt.ylabel('diffSAVI')
plt.xlabel('DOWY')
plt.axhline(0,color='black')


#%%
#%%
x=rain_sos.iloc[:,0:30].values.flatten()
#y=savi_eos_gr.drop([savi_eos_gr.index[1]])
y=sos_gr.iloc[:,0:30].values.flatten() #apted in 2000 and 2006
col='green'
new_x= np.linspace(0,180) 
x_line = np.linspace(0, 180, 200) 

xo=rain_sos_o.values.flatten()
yo=sos_gr_o.values.flatten()
#%%
x=rain_sos.iloc[:,0:30].values.flatten()
#y=savi_eos_shr.drop([savi_eos_shr.index[1]])
y=sos_shr.iloc[:,0:30].values.flatten() #adapted 2000 and 2006
col='orange'
new_x= np.linspace(0,180) 
x_line = np.linspace(0, 180, 200) 
xo=rain_sos_o.values.flatten()
yo=sos_shr_o.values.flatten()

#%%
x=sob_rain_gr.iloc[:,0:30].values.flatten()
y=sob_gr.iloc[:,0:30].values.flatten()
col='green'
new_x = np.linspace(0, 110) 
x_line = np.linspace(0, 110, 100) 

xo=sob_rain_gr_o.values.flatten()
yo=sob_gr_o.values.flatten()
#%%
x=sob_rain_shr.iloc[:,0:30].values.flatten()
y=sob_shr.iloc[:,0:30].values.flatten()
col='orange'
new_x = np.linspace(0, 110) 
x_line = np.linspace(0, 110, 100) 

xo=sob_rain_shr_o.values.flatten()
yo=sob_shr_o.values.flatten()
#%%

x=rain_max_gr.iloc[:,0:30].values.flatten()
y=max_doy_gr.iloc[:,0:30].values.flatten()

x=x[np.logical_not(np.isnan(y))]
y=y[np.logical_not(np.isnan(y))]
new_x = np.linspace(0, 220) 
x_line = np.linspace(0, 220, 100) 

#%% LINEAR
coeffs = np.polyfit(x,y,1)
a,b = coeffs
y_model = np.polyval([a,b], x)
y_line = np.polyval([a,b], x_line)


#%%
coeffs = np.polyfit(x,y,2)

poly = np.poly1d(coeffs)
a,b,c=coeffs
y_model = np.polyval([a,b,c], x)
y_line = np.polyval([a,b,c], x_line)

#%%
poly = np.poly1d(coeffs)

new_y = poly(new_x)

#Statistics
x_mean=np.mean(x)
y_mean=np.mean(y)
n=x.size                #number of samples
m=3                   #number of parameters (3 parameters abc)
dof=n-m   
t=stats.t.ppf(0.975,dof)  #alpha = 0.05, student statistic of interval confidence 

residual = y-y_model
std_error = np.sqrt(np.sum(residual**2) / dof)

pi = t * std_error * np.sqrt(1 + 1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))
#%%
# to plot the adjusted model
plt.subplot(121)
plt.plot(rain_max_gr['31D'],)


plt.plot(x,y,'.', new_x,new_y,color=col, label="y={0:e}*x+{1:.4f}".format(a,b))

plt.plot(x_line, y_line, color = col)
plt.fill_between(x_line, y_line + pi, y_line - pi, color = col, alpha=0.2,label = '95% prediction interval')
plt.legend(frameon=False)
plt.xlabel('Antecedent P (mm)')
#plt.ylabel('SOM SAVI')
#plt.ylabel(' SAVI at the start of browning')
#plt.ylabel('Max SAVI')
print(coeffs)
#%%
pi=pd.DataFrame(pi)
pi.to_csv('ci_sob_shr1.csv',sep=',')

#%% Seasonal plots of P and T and GR and Shr

t1=ag_gr
t1=t1.loc[(t1.index < '2011-11-01') | (t1.index > '2013-10-31')]
t1=pd.pivot_table(t1, index=t1['DOWY'], columns=t1['WY'],values='mean')
t1=t1.mean(axis=1)

t2=ag_shr
t2=t2.loc[(t2.index < '2011-11-01') | (t2.index > '2013-10-31')]
t2=pd.pivot_table(t2, index=t2['DOWY'], columns=t2['WY'],values='mean')
t2=t2.mean(axis=1)

t4=ag_temp
t4=t4.loc[(t4.index < '2011-11-01') | (t4.index > '2013-10-31')]
t4= pd.pivot_table(t4, index=t4['DOWY'], columns=t4['WY'],values='94D')
t4=t4.mean(axis=1)

t3 = ag_rain
t3=t3.loc[(t3.index < '2011-11-01') | (t3.index > '2013-10-31')]
t3=pd.pivot_table(t3, index=t3['DOWY'], columns=t3['WY'],values='31D')
t3=t3.mean(axis=1)

#%%
plt.subplot(211)
plt.plot(t1,color='green',label='SAVI')  #CW
plt.plot(t2,color='orange',label='SAVI')  #MEs
plt.ylim(0.07,0.21)
plt.twinx()
plt.plot(t3,color='blue', linestyle='--',label='Precipitation (mm)')
plt.ylim(-3,115)
plt.twinx()
plt.plot(t4,color='red', label='T')
plt.ylim(15,35)
plt.legend(frameon=False)

plt.subplot(212)
plt.ylim(0.07,0.21)
plt.twinx()
plt.plot(t3,color='blue', linestyle='--',label='Precipitation (mm)')
plt.ylim(-3,115)
plt.twinx()
plt.plot(t4,color='red', label='T')
plt.ylim(15,35)
plt.legend(frameon=False)

#%%
cw1=d_gr.loc[0:182]
cw1std=d_gr_std.loc[0:182]
cw2=d_gr.loc[182:365]
cw2std=d_gr_std.loc[182:365]
tmp1=b.loc[0:182]
tmp1std=b_std.loc[0:182]
tmp2=b.loc[182:364]
tmp2std=b_std.loc[182:364]

mes1=d_shr.loc[0:182]
mes1std=d_shr_std.loc[0:182]
mes2=d_shr.loc[182:365]
mes2std=d_shr_std.loc[182:365]

#%%
plt.subplot(141)
plt.plot(tmp1,color='red')
#plt.ylim(-6.5,6.5)
plt.fill_between(tmp1.index,tmp1-tmp1std, tmp1+tmp1std,  color='red',alpha=0.1, edgecolor='red')
plt.twinx()
plt.plot(cw1,color='green')
plt.fill_between(cw1.index,cw1-cw1std, cw1+cw1std,  color='green',alpha=0.1, edgecolor='green')
plt.ylim(-0.0045,0.0045)
plt.axhline(0)

plt.subplot(142)
plt.plot(tmp2,color='red')
plt.fill_between(tmp2.index,tmp2-tmp2std, tmp2+tmp2std,  color='red',alpha=0.1, edgecolor='red')
#plt.ylim(-6.5,6.5)
plt.twinx()
plt.plot(cw2,color='green')
plt.fill_between(cw2.index,cw2-cw2std, cw2+cw2std,  color='green',alpha=0.1, edgecolor='green')
plt.ylim(-0.0045,0.0045)
plt.axhline(0)

plt.subplot(143)
plt.plot(tmp1,color='red')
plt.fill_between(tmp1.index,tmp1-tmp1std, tmp1+tmp1std,  color='red',alpha=0.1, edgecolor='red')
#plt.ylim(-6.5,6.5)
plt.twinx()
plt.plot(mes1,color='green')
plt.fill_between(mes1.index,mes1-mes1std, mes1+mes1std,  color='green',alpha=0.1, edgecolor='green')
plt.ylim(-0.0045,0.0045)
plt.axhline(0)

plt.subplot(144)
plt.plot(tmp2,color='red')
plt.fill_between(tmp2.index,tmp2-tmp2std, tmp2+tmp2std,  color='red',alpha=0.1, edgecolor='red')
#plt.ylim(-6.5,6.5)
plt.twinx()
plt.plot(mes2,color='green')
plt.fill_between(mes2.index,mes2-mes2std, mes2+mes2std,  color='green',alpha=0.1, edgecolor='green')
plt.ylim(-0.0045,0.0045)
plt.axhline(0)

#%%












