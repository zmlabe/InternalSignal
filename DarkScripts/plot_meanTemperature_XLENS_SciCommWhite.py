"""
Plot the mean temperature anomalies for each data set to compare

Reference  : Deser et al. [2020, JCLI]
Author    : Zachary M. Labe
Date      : 22 October 2020
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
directoryfigure =  '/Users/zlabe/Documents/Projects/InternalSignal/DarkFigures/'
datasetsingleq = ['XGHG','XAER','LENS','20CRv3']
datasetsingle = ['XGHG','XAER','lens','20CRv3']
seasons = ['annual','JFM','AMJ','JAS','OND']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
years = np.arange(1920,2080+1,1)
yearsobs = np.arange(1920,2015+1,1)
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

### Calculate anomalies with 1951-1980 baseline
yearq = np.where((years >= 1951) & (years <= 1980))[0]
mean_ghg = np.nanmean(ghg[:,yearq,:,:],axis=1)
mean_aer = np.nanmean(aer[:,yearq,:,:],axis=1)
mean_lens = np.nanmean(lens[:,yearq,:,:],axis=1)
mean_obs = np.nanmean(obs[yearq,:,:],axis=0)

anom_ghg = np.empty((ghg.shape))
anom_aer = np.empty((aer.shape))
anom_lens = np.empty((lens.shape))
for i in range(ghg.shape[1]):
    anom_ghg[:,i,:,:] = ghg[:,i,:,:] - mean_ghg
    anom_aer[:,i,:,:] = aer[:,i,:,:] - mean_aer
    anom_lens[:,i,:,:] = lens[:,i,:,:] - mean_lens
anom_obs = obs - mean_obs

### Calculate global average
lon2,lat2 = np.meshgrid(lon1,lat1)
globe_ghg = UT.calc_weightedAve(anom_ghg,lat2)
globe_aer = UT.calc_weightedAve(anom_aer,lat2)
globe_lens = UT.calc_weightedAve(anom_lens,lat2)
globe_obs = UT.calc_weightedAve(anom_obs,lat2)

### Calculate ensemble means
ensanom_ghg = np.nanmean(globe_ghg,axis=0)
ensanom_aer = np.nanmean(globe_aer,axis=0)
ensanom_lens = np.nanmean(globe_lens,axis=0)

###############################################################################
###############################################################################
###############################################################################
### Create time series
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
# plt.rc('savefig',facecolor='black')
# plt.rc('axes',edgecolor='darkgrey')
# plt.rc('xtick',color='darkgrey')
# plt.rc('ytick',color='darkgrey')
# plt.rc('axes',labelcolor='darkgrey')
# plt.rc('axes',facecolor='black')

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

max2tdghg = ensanom_ghg + (np.nanstd(globe_ghg,axis=0)*2)
min2tdghg = ensanom_ghg - (np.nanstd(globe_ghg,axis=0)*2)

max2tdaer = ensanom_aer + (np.nanstd(globe_aer,axis=0)*2)
min2tdaer = ensanom_aer - (np.nanstd(globe_aer,axis=0)*2)

max2tdlens = ensanom_lens + (np.nanstd(globe_lens,axis=0)*2)
min2tdlens = ensanom_lens - (np.nanstd(globe_lens,axis=0)*2)

ax.fill_between(years, min2tdghg, max2tdghg, facecolor='deepskyblue',
                alpha=0.3,zorder=2,clip_on=False)
plt.plot(years,ensanom_ghg,'-',
            color='deepskyblue',linewidth=2,clip_on=False,
            label=r'\textbf{AER+}')

ax.fill_between(years, min2tdaer, max2tdaer, facecolor='forestgreen',
                alpha=0.3,zorder=2,clip_on=False)
plt.plot(years,ensanom_aer,'-',
            color='forestgreen',linewidth=2,clip_on=False,
            label=r'\textbf{GHG+}')

ax.fill_between(years, min2tdlens, max2tdlens, facecolor='crimson',
                alpha=0.3,zorder=2,clip_on=False)
plt.plot(years,ensanom_lens,'-',
            color='crimson',linewidth=2,clip_on=False,
            label=r'\textbf{ALL}')

# plt.plot(yearsobs,globe_obs,color='w',linewidth=2,
#           dashes=(1,0.3),linestyle='--',label=r'\textbf{20CRv3}',zorder=11)

plt.ylabel(r'\textbf{Temperature Anomaly [$^{\circ}$C]}',fontsize=10,color='dimgrey')
plt.yticks(np.arange(-10,101,0.5),map(str,np.round(np.arange(-10,101,0.5),2)),size=6)
plt.xticks(np.arange(1920,2080+1,20),map(str,np.arange(1920,2080+1,20)),size=6)
plt.xlim([1920,2080])   
plt.ylim([-0.5,4])

leg = plt.legend(shadow=False,fontsize=20,loc='upper left',
              bbox_to_anchor=(-0.01,1),fancybox=True,ncol=1,frameon=False,
              handlelength=1,handletextpad=0.5)
for text in leg.get_texts():
    text.set_color('k')

plt.savefig(directoryfigure + 'TimeSeries_T2M_XGHG-XAER-LENS-20CRv3_SciCommWhite.png',
            dpi=300)