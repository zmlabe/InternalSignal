"""
Plot signal-to-noise ratios for XLENS simulations

Method 4 = mean temperature change / mean std of temperature in 1920-1959

Reference  : Deser et al. [2020, JCLI] & Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 28 October 2020
"""

### Import packages
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import cmocean as cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import itertools

##############################################################################
##############################################################################
##############################################################################
## Data preliminaries 
directorydataLLS = '/Users/zlabe/Data/LENS/SINGLE/'
directorydataLLL = '/Users/zlabe/Data/LENS/monthly'
directoryfigure =  '/Users/zlabe/Desktop/SINGLE_v2.0/Composites/T2M/'
datasetsingleq = np.repeat(['AER+ALL','GHG+ALL','TOTAL'],3)
datasetsingle = ['XGHG','XAER','lens']
timeq = ['1960-1999','2000-2039','2040-2079']
seasons = ['annual','JFM','AMJ','JAS','OND']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
years = np.arange(1920,2079+1,1)
directoriesall = [directorydataLLS,directorydataLLS,directorydataLLL]

variq = 'T2M'
monthlychoice = seasons[0]
reg_name = 'Globe'

def read_primary_dataset(variq,dataset,lat_bounds,lon_bounds,monthlychoice):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons

### Read in data
lat_bounds,lon_bounds = UT.regions(reg_name)
ghg,lat1,lon1 = read_primary_dataset(variq,datasetsingle[0],lat_bounds,lon_bounds,
                            monthlychoice)
aer,lat1,lon1 = read_primary_dataset(variq,datasetsingle[1],lat_bounds,lon_bounds,
                            monthlychoice)
lens,lat1,lon1 = read_primary_dataset(variq,datasetsingle[2],lat_bounds,lon_bounds,
                            monthlychoice)

### Calculate to 2079
ghgn = ghg[:,:-1,:,:]
aern = aer[:,:-1,:,:]
lensn = lens[:,:-1,:,:]

### Calculate early means
ghg_20 = np.nanmean(ghg[:,:40,:,:],axis=1)
aer_20 = np.nanmean(aer[:,:40,:,:],axis=1)
lens_20 = np.nanmean(lens[:,:40,:,:],axis=1)
ghg_20std = np.nanstd(ghg[:,:40,:,:],axis=1)
aer_20std = np.nanstd(aer[:,:40,:,:],axis=1)
lens_20std = np.nanstd(lens[:,:40,:,:],axis=1)

ghg_20mean = np.nanmean(ghg_20,axis=0)
aer_20mean = np.nanmean(aer_20,axis=0)
lens_20mean = np.nanmean(lens_20,axis=0)
ghg_20meanstd = np.nanmean(ghg_20std,axis=0)
aer_20meanstd = np.nanmean(aer_20std,axis=0)
lens_20meanstd = np.nanmean(lens_20std,axis=0)

### Calculate ensemble mean
meanghg = np.nanmean(ghgn,axis=0)
meanaer = np.nanmean(aern,axis=0)
meanlens = np.nanmean(lensn,axis=0)

diffens_ghg = np.empty((ghgn.shape[0],3,lat1.shape[0],lon1.shape[0]))
diffens_aer = np.empty((aern.shape[0],3,lat1.shape[0],lon1.shape[0]))
diffens_lens = np.empty((lensn.shape[0],3,lat1.shape[0],lon1.shape[0]))
stdens_ghg = np.empty((ghgn.shape[0],3,lat1.shape[0],lon1.shape[0]))
stdens_aer = np.empty((aern.shape[0],3,lat1.shape[0],lon1.shape[0]))
stdens_lens = np.empty((lensn.shape[0],3,lat1.shape[0],lon1.shape[0]))
### Calculate change in temperature from 1920-1959 and sigma per period
for count,i in enumerate(range(40,len(years),40)):
    diffens_ghg[:,count,:,:] = np.nanmean(ghgn[:,i:i+40,:,:],axis=1) - ghg_20
    diffens_aer[:,count,:,:] = np.nanmean(aern[:,i:i+40,:,:],axis=1) - aer_20
    diffens_lens[:,count,:,:] = np.nanmean(lensn[:,i:i+40,:,:],axis=1) - lens_20
    
    stdens_ghg[:,count,:,:] = np.nanstd(ghgn[:,i:i+40,:,:],axis=1)
    stdens_aer[:,count,:,:] = np.nanstd(aern[:,i:i+40,:,:],axis=1)
    stdens_lens[:,count,:,:] = np.nanstd(lensn[:,i:i+40,:,:],axis=1)
    
### Calculate change statistics
meanchange_ghg = np.nanmean(diffens_ghg,axis=0)
meanchange_aer = np.nanmean(diffens_aer,axis=0)
meanchange_lens = np.nanmean(diffens_lens,axis=0)
meanstd_ghg = np.nanmean(stdens_ghg,axis=0)
meanstd_aer = np.nanmean(stdens_aer,axis=0)
meanstd_lens = np.nanmean(stdens_lens,axis=0)

maxchange_ghg = np.nanmax(diffens_ghg,axis=0)
maxchange_aer = np.nanmax(diffens_aer,axis=0)
maxchange_lens = np.nanmax(diffens_lens,axis=0)
maxstd_ghg = np.nanmax(stdens_ghg,axis=0)
maxstd_aer = np.nanmax(stdens_aer,axis=0)
maxstd_lens = np.nanmax(stdens_lens,axis=0)

minchange_ghg = np.nanmin(diffens_ghg,axis=0)
minchange_aer = np.nanmin(diffens_aer,axis=0)
minchange_lens = np.nanmin(diffens_lens,axis=0)
minstd_ghg = np.nanmin(stdens_ghg,axis=0)
minstd_aer = np.nanmin(stdens_aer,axis=0)
minstd_lens = np.nanmin(stdens_lens,axis=0)

### Calculate ensemble spread in change
spread_ghg = maxchange_ghg - minchange_ghg
spread_aer = maxchange_aer - minchange_aer
spread_lens = maxchange_lens - minchange_lens
spreadstd_ghg = maxstd_ghg - minstd_ghg
spreadstd_aer = maxstd_aer - minstd_aer
spreadstd_lens = maxstd_lens - minstd_lens

### Calculate signal-to-noise ratios
snr_ghg = meanchange_ghg/ghg_20meanstd
snr_aer = meanchange_aer/aer_20meanstd
snr_lens = meanchange_lens/lens_20meanstd

runs = list(itertools.chain.from_iterable([snr_ghg,snr_aer,snr_lens]))

###########################################################################
###########################################################################
###########################################################################
### Plot variable data for trends
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

### Set limits for contours and colorbars
limit = np.arange(-5,5.1,0.1)
barlim = np.arange(-5,6,5)
cmap = cmocean.cm.balance
label = r'\textbf{T2M [signal-to-noise]}'
    
fig = plt.figure(figsize=(5,3))
for r in range(len(runs)):
    var = runs[r]
    
    ax1 = plt.subplot(3,3,r+1)
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    circle = m.drawmapboundary(fill_color='k')
    circle.set_clip_on(False) 
    m.drawcoastlines(color='dimgrey',linewidth=0.35)
    
    var, lons_cyclic = addcyclic(var, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='white',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    cs = m.contourf(x,y,var,limit,extend='both')
            
    cs.set_cmap(cmap) 
    if any([r==0,r==3,r==6]):
        ax1.annotate(r'\textbf{%s}' % datasetsingleq[r],xy=(0,0),xytext=(-0.1,0.5),
                      textcoords='axes fraction',color='k',fontsize=9,
                      rotation=90,ha='center',va='center')
    if any([r==0,r==1,r==2]):
        ax1.annotate(r'\textbf{%s}' % timeq[r],xy=(0,0),xytext=(0.5,1.22),
                      textcoords='axes fraction',color='dimgrey',fontsize=9,
                      rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.87,0.97),
                  textcoords='axes fraction',color='k',fontsize=6,
                  rotation=330,ha='center',va='center')

###########################################################################
cbar_ax = fig.add_axes([0.32,0.095,0.4,0.03])                
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)

cbar.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  

cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(top=0.85,wspace=0.01,hspace=0,bottom=0.14)

plt.savefig(directoryfigure + 'SNRPeriods_T2M_XGHG-XAER-LENS_Method4.png',dpi=300)