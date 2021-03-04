"""
Trends for different climate model runs

Reference  : Deser et al. [2020, JCLI]
Author    : Zachary M. Labe
Date      : 29 December 2020
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
directoryfigure =  '/Users/zlabe/Documents/Projects/InternalSignal/DarkFigures/'
datasetsingleq = ['AER+','GHG+','ALL']
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
    
    dectrend = slopes * 10.   
    print('Completed: Finished calculating trends!')      
    return dectrend

### Procress trends
trend_ghg = np.empty((len(years)//40,meanghg.shape[1],meanghg.shape[2]))
trend_aer = np.empty((len(years)//40,meanaer.shape[1],meanaer.shape[2]))
trend_lens = np.empty((len(years)//40,meanlens.shape[1],meanlens.shape[2]))
for count,i in enumerate(range(0,len(years),40)):
    trend_ghg[count,:,:,] = calcTrend(meanghg[i:i+40,:,:])
    trend_aer[count,:,:,] = calcTrend(meanaer[i:i+40,:,:])
    trend_lens[count,:,:,] = calcTrend(meanlens[i:i+40,:,:])
    
pvals_ghg = np.empty((len(years)//40,meanghg.shape[1],meanghg.shape[2]))
pvals_aer = np.empty((len(years)//40,meanaer.shape[1],meanaer.shape[2]))
pvals_lens = np.empty((len(years)//40,meanlens.shape[1],meanlens.shape[2]))
for count,y in enumerate(range(0,len(years),40)):
    for i in range(lat1.shape[0]):
        for j in range(lon1.shape[0]):
            trend,h,pvals_ghg[count,i,j],z = UT.mk_test(meanghg[y:y+40,i,j],0.05)
            trend,h,pvals_aer[count,i,j],z = UT.mk_test(meanaer[y:y+40,i,j],0.05)
            trend,h,pvals_lens[count,i,j],z = UT.mk_test(meanlens[y:y+40,i,j],0.05)
            
pvals_ghg[np.where(pvals_ghg == 1.)] = 0.
pvals_ghg[np.where(np.isnan(pvals_ghg))] = 1.
pvals_ghg[np.where(pvals_ghg == 0.)] = np.nan

pvals_aer[np.where(pvals_aer == 1.)] = 0.
pvals_aer[np.where(np.isnan(pvals_aer))] = 1.
pvals_aer[np.where(pvals_aer == 0.)] = np.nan

pvals_lens[np.where(pvals_lens == 1.)] = 0.
pvals_lens[np.where(np.isnan(pvals_lens))] = 1.
pvals_lens[np.where(pvals_lens == 0.)] = np.nan

ghgc = np.nanmean(trend_ghg[1:2,:,:],axis=0)
aerc = np.nanmean(trend_aer[1:2,:,:],axis=0)
lensc = np.nanmean(trend_lens[1:2,:,:],axis=0)
  
runs = [ghgc,aerc,lensc]  
    
#######################################################################
#######################################################################
#######################################################################
### Plot trends
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
plt.rc('savefig',facecolor='black')
plt.rc('axes',edgecolor='darkgrey')
plt.rc('xtick',color='darkgrey')
plt.rc('ytick',color='darkgrey')
plt.rc('axes',labelcolor='darkgrey')
plt.rc('axes',facecolor='black')

limit = np.arange(-0.4,0.401,0.001)
barlim = np.round(np.arange(-0.4,0.41,0.2),2)
cmap = cmocean.cm.balance
label = r'\textbf{$^{\circ}$C per decade}'

fig = plt.figure(figsize=(10,2.5))
for i in range(len(runs)):
    ax1 = plt.subplot(1,3,i+1)
            
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    circle = m.drawmapboundary(fill_color='dimgrey')
    circle.set_clip_on(False) 
    m.drawcoastlines(color='dimgrey',linewidth=0.35)
    
    ### Take lrp mean over all years
    dataall = runs[i]
    
    var, lons_cyclic = addcyclic(dataall, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
    
    ### Make the plot continuous
    cs = m.contourf(x,y,var,limit,
                    extend='both')                     
    cs.set_cmap(cmap)
    
    ax1.annotate(r'\textbf{%s}' % (datasetsingleq[i]),xy=(0,0),xytext=(0.88,0.92),
                      textcoords='axes fraction',color='w',fontsize=16,
                      rotation=333,ha='center',va='center')
    # ax1.annotate(r'\textbf{[%s]}' % letters[i],xy=(0,0),xytext=(0.085,0.93),
    #                       textcoords='axes fraction',color='darkgrey',fontsize=8,
    #                       rotation=0,ha='center',va='center')
    
cbar_ax = fig.add_axes([0.293,0.13,0.4,0.03])             
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)

cbar.set_label(label,fontsize=11,color='w',labelpad=1.4)  
cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=6)
cbar.outline.set_edgecolor('darkgrey')

plt.tight_layout()
plt.subplots_adjust(bottom=0.17)
plt.savefig(directoryfigure + 'ModelTrends_PAPER_DARK_SciComm.png',dpi=600)