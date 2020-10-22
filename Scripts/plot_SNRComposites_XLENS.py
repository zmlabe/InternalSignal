"""
Plot signal-to-noise ratios for XLENS simulations

Reference  : Deser et al. [2020, JCLI]
Author    : Zachary M. Labe
Date      : 22 October 2020
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

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directorydataLLS = '/Users/zlabe/Data/LENS/SINGLE/'
directorydataLLL = '/Users/zlabe/Data/LENS/monthly'
directoryfigure =  '/Users/zlabe/Desktop/SINGLE_v2.0/Composites/T2M/'
datasetsingleq = np.repeat(['XGHG','XAER','LENS'],4)
datasetsingle = ['XGHG','XAER','lens']
timeq = ['1920-1959','1960-1999','2000-2039','2040-2079']
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

### Calculate ensemble mean
meanghg = np.nanmean(ghg,axis=0)[:-1,:,:] #to 2079
meanaer = np.nanmean(aer,axis=0)[:-1,:,:] #to 2079
meanlens = np.nanmean(lens,axis=0)[:-1,:,:] #to 2079

### Calculate linear trends
def calcTrend(data):
    if data.ndim == 3:
        slopes = np.empty((data.shape[1],data.shape[2]))
        x = np.arange(data.shape[0])
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                mask = np.isfinite(data[:,i,j])
                y = data[:,i,j]
                
                if np.sum(mask) == y.shape[0]:
                    xx = x
                    yy = y
                else:
                    xx = x[mask]
                    yy = y[mask]      
                if np.isfinite(np.nanmean(yy)):
                    slopes[i,j],intercepts, \
                    r_value,p_value,std_err = sts.linregress(xx,yy)
                else:
                    slopes[i,j] = np.nan
    elif data.ndim == 4:
        slopes = np.empty((data.shape[0],data.shape[2],data.shape[3]))
        x = np.arange(data.shape[1])
        for ens in range(data.shape[0]):
            for i in range(data.shape[2]):
                for j in range(data.shape[3]):
                    mask = np.isfinite(data[ens,:,i,j])
                    y = data[ens,:,i,j]
                    
                    if np.sum(mask) == y.shape[0]:
                        xx = x
                        yy = y
                    else:
                        xx = x[mask]
                        yy = y[mask]      
                    if np.isfinite(np.nanmean(yy)):
                        slopes[ens,i,j],intercepts, \
                        r_value,p_value,std_err = sts.linregress(xx,yy)
                    else:
                        slopes[ens,i,j] = np.nan
    
    dectrend = slopes * 10.   
    print('Completed: Finished calculating trends!')      
    return dectrend

### Process trends
trend_ghg = np.empty((len(years)//40,meanghg.shape[1],meanghg.shape[2]))
trend_aer = np.empty((len(years)//40,meanaer.shape[1],meanaer.shape[2]))
trend_lens = np.empty((len(years)//40,meanlens.shape[1],meanlens.shape[2]))
for count,i in enumerate(range(0,len(years),40)):
    trend_ghg[count,:,:] = calcTrend(meanghg[i:i+40,:,:])
    trend_aer[count,:,:] = calcTrend(meanaer[i:i+40,:,:])
    trend_lens[count,:,:] = calcTrend(meanlens[i:i+40,:,:])
  
trendens_ghg = np.empty((ghg.shape[0],len(years)//40,meanghg.shape[1],meanghg.shape[2]))
trendens_aer = np.empty((aer.shape[0],len(years)//40,meanaer.shape[1],meanaer.shape[2]))
trendens_lens = np.empty((lens.shape[0],len(years)//40,meanlens.shape[1],meanlens.shape[2]))
for count,i in enumerate(range(0,len(years),40)):
    trendens_ghg[:,count,:,:] = calcTrend(ghg[:,i:i+40,:,:])
    trendens_aer[:,count,:,:] = calcTrend(aer[:,i:i+40,:,:])
    trendens_lens[:,count,:,:] = calcTrend(lens[:,i:i+40,:,:])

std_ghg = np.nanstd(trendens_ghg[:,:,:,:],axis=0)
std_aer = np.nanstd(trendens_aer[:,:,:,:],axis=0)
std_lens = np.nanstd(trendens_lens[:,:,:,:],axis=0)

### Calculate signal-to-noise ratios
snr_ghg = abs(trend_ghg)/std_ghg
snr_aer = abs(trend_aer)/std_aer
snr_lens = abs(trend_lens)/std_lens

runs = list(itertools.chain.from_iterable([snr_ghg,snr_aer,snr_lens]))

###########################################################################
###########################################################################
###########################################################################
### Plot variable data for trends
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

### Set limits for contours and colorbars
limit = np.arange(0,4.1,0.1)
barlim = np.arange(0,4.1,0.5)
cmap = plt.cm.CMRmap
label = r'\textbf{T2M [signal-to-noise]}'
    
fig = plt.figure(figsize=(5,3))
for r in range(len(runs)):
    var = runs[r]
    
    ax1 = plt.subplot(3,4,r+1)
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
    
    cs = m.contourf(x,y,var,limit,extend='max')
            
    cs.set_cmap(cmap) 
    if any([r==0,r==4,r==8]):
        ax1.annotate(r'\textbf{%s}' % datasetsingleq[r],xy=(0,0),xytext=(-0.1,0.5),
                      textcoords='axes fraction',color='k',fontsize=13,
                      rotation=90,ha='center',va='center')
    if any([r==0,r==1,r==2,r==3]):
        ax1.annotate(r'\textbf{%s}' % timeq[r],xy=(0,0),xytext=(0.5,1.22),
                      textcoords='axes fraction',color='dimgrey',fontsize=9,
                      rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.87,0.97),
                  textcoords='axes fraction',color='k',fontsize=6,
                  rotation=330,ha='center',va='center')

###########################################################################
cbar_ax = fig.add_axes([0.32,0.095,0.4,0.03])                
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)

cbar.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  

cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(top=0.85,wspace=0.01,hspace=0,bottom=0.14)

plt.savefig(directoryfigure + 'SNRPeriods_T2M_XGHG-XAER-LENS.png',dpi=300)