"""
Plot the mean temperature for each data set to compare

Reference  : Deser et al. [2020, JCLI]
Author    : Zachary M. Labe
Date      : 14 December 2020
"""

### Import packages
import matplotlib.pyplot as plt
import numpy as np
import calc_Utilities as UT
import calc_dataFunctions as df

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directorydataLLS = '/Users/zlabe/Data/LENS/SINGLE/'
directorydataLLL = '/Users/zlabe/Data/LENS/monthly'
directoryfigure =  '/Users/zlabe/Desktop/SINGLE_v2.0/'
datasetsingleq = ['XGHG','XAER','LENS','20CRv3','ERA5']
datasetsingle = ['XGHG','XAER','lens','20CRv3','ERA5']
seasons = ['annual','JFM','AMJ','JAS','OND']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
years = np.arange(1920,2080+1,1)
yearsobs = np.arange(1920,2015+1,1)
yearsobs2 = np.arange(1979,2019+1,1)
directoriesall = [directorydataLLS,directorydataLLS,directorydataLLL]

variq = 'T2M'
monthlychoice = seasons[0]
reg_name = 'Globe'

def read_primary_dataset(variq,dataset,lat_bounds,lon_bounds,monthlychoice):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons

def read_obs_dataset(variq,dataset_obs,lat_bounds,lon_bounds,monthlychoice):
    data_obs,lats_obs,lons_obs = df.readFiles(variq,dataset_obs,monthlychoice)
    data_obs,lats_obs,lons_obs = df.getRegion(data_obs,lats_obs,lons_obs,
                                            lat_bounds,lon_bounds)
    if dataset_obs == '20CRv3':
        if monthlychoice == 'DJF':
            year20cr = np.arange(1837,2015+1)
        else:
            year20cr = np.arange(1836,2015+1)
        yearqq = np.where((year20cr >= yearsobs.min()) & (year20cr <= yearsobs.max()))[0]
        data_obs = data_obs[yearqq,:,:]
    
    print('our OBS dataset: ',dataset_obs,' is shaped',data_obs.shape)
    return data_obs,lats_obs,lons_obs

### Read in data
lat_bounds,lon_bounds = UT.regions(reg_name)
ghg,lat1,lon1 = read_primary_dataset(variq,datasetsingle[0],lat_bounds,lon_bounds,
                            monthlychoice)
aer,lat1,lon1 = read_primary_dataset(variq,datasetsingle[1],lat_bounds,lon_bounds,
                            monthlychoice)
lens,lat1,lon1 = read_primary_dataset(variq,datasetsingle[2],lat_bounds,lon_bounds,
                            monthlychoice)
obs,lat1,lon1 = read_obs_dataset(variq,datasetsingle[3],lat_bounds,lon_bounds,
                            monthlychoice)
obs2,lat1,lon1 = read_obs_dataset(variq,datasetsingle[4],lat_bounds,lon_bounds,
                            monthlychoice)

### Calculate global average
lon2,lat2 = np.meshgrid(lon1,lat1)
globe_ghg = UT.calc_weightedAve(ghg,lat2)
globe_aer = UT.calc_weightedAve(aer,lat2)
globe_lens = UT.calc_weightedAve(lens,lat2)
globe_obs = UT.calc_weightedAve(obs,lat2)
globe_obs2 = UT.calc_weightedAve(obs2,lat2)

### Calculate ensemble means
meanghg = np.nanmean(globe_ghg,axis=0)
meanaer = np.nanmean(globe_aer,axis=0)
meanlens = np.nanmean(globe_lens,axis=0)

###############################################################################
###############################################################################
###############################################################################
### Create time series
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

fig = plt.figure()
ax = plt.subplot(111)

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')  
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_ticks([])

adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')

plt.plot(years,meanghg,'-',
            color='steelblue',linewidth=2.5,clip_on=True,
            label=r'\textbf{AER+}')

plt.plot(years,meanaer,'-',
            color='darkgoldenrod',linewidth=2.5,clip_on=True,
            label=r'\textbf{GHG+}')

plt.plot(years,meanlens,'-',
            color='crimson',linewidth=2.5,clip_on=True,
            label=r'\textbf{ALL}')

plt.plot(yearsobs,globe_obs,color='k',linewidth=2.5,
          dashes=(1,0.3),linestyle='--',label=r'\textbf{20CRv3}',zorder=11)
plt.plot(yearsobs2,globe_obs2,color='dimgrey',linewidth=1.5,
         linestyle='-',label=r'\textbf{ERA5}',zorder=11)

plt.ylabel(r'\textbf{T2M [$^{\circ}$C]}',fontsize=10,color='k')
plt.yticks(np.arange(-10,101,0.5),map(str,np.round(np.arange(-10,101,0.5),2)),size=6)
plt.xticks(np.arange(1920,2080+1,20),map(str,np.arange(1920,2080+1,20)),size=6)
plt.xlim([1920,2080])   
plt.ylim([12,16])

plt.title(r'\textbf{GLOBAL MEAN AIR TEMPERATURE}',fontsize=17,color='dimgrey')

leg = plt.legend(shadow=False,fontsize=9.5,loc='upper left',
              bbox_to_anchor=(-0.01,1.05),fancybox=True,ncol=1,frameon=False,
              handlelength=1,handletextpad=0.5)

plt.savefig(directoryfigure + 'TimeSeries_T2M_MeanState_XGHG-XAER-LENS-20CRv3.png',
            dpi=300)