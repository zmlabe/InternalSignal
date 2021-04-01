"""
Compares LRP maps for only testing data

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 29 March 2021
"""

### Import packages
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import cmocean
import palettable.cubehelix as cm
import scipy.stats as sts
import calc_Utilities as UT

### Set parameters
variables = [r'T2M']
datasets = [r'AER+',r'GHG+',r'ALL',r'AER+',r'GHG+',r'ALL']
seasons = [r'annual']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
SAMPLEQ = 100

### Set directories
directorydata = '/Users/zlabe/Documents/Research/InternalSignal/Data/FINAL/'
directorydata2 = '/Users/zlabe/Documents/Research/InternalSignal/Data/FINAL/R1/LRP/'
directoryfigure = '/Users/zlabe/Desktop/PAPER/R1/'

### Read in LRP maps (XGHG)
data = Dataset(directorydata + 'LRP_Maps_XGHG_AllSeasons_PAPER.nc')
lat1 = data.variables['lat'][:]
lon1 = data.variables['lon'][:]
lrp_ghg = data.variables['LRP'][:]
data.close()
ghg = np.nanmean(lrp_ghg[:,:,:,:],axis=1).squeeze()

### Read in LRP maps (XAER)
data = Dataset(directorydata + 'LRP_Maps_XAER_AllSeasons_PAPER.nc')
lat1 = data.variables['lat'][:]
lon1 = data.variables['lon'][:]
lrp_aer = data.variables['LRP'][:]
data.close()
aer = np.nanmean(lrp_aer[:,:,:,:],axis=1).squeeze()

### Read in LRP maps (LENS)
data = Dataset(directorydata + 'LRP_Maps_lens_AllSeasons_PAPER.nc')
lat1 = data.variables['lat'][:]
lon1 = data.variables['lon'][:]
lrp_lens = data.variables['LRP'][:]
data.close()
lens = np.nanmean(lrp_lens[:,:,:,:],axis=1).squeeze()

##############################################################################
##############################################################################
##############################################################################

### Read in LRP maps (XGHG)
datates = Dataset(directorydata2 + 'LRP_Maps_XGHG_AllSeasons_PAPER-Testing.nc')
lat1tes = datates.variables['lat'][:]
lon1tes = datates.variables['lon'][:]
lrp_ghgtes = datates.variables['LRP'][:]
datates.close()
ghgtes = np.nanmean(lrp_ghgtes[:,:,:,:],axis=1).squeeze()

### Read in LRP maps (XAER)
datates = Dataset(directorydata2 + 'LRP_Maps_XAER_AllSeasons_PAPER-Testing.nc')
lat1tes = datates.variables['lat'][:]
lon1tes = datates.variables['lon'][:]
lrp_aertes = datates.variables['LRP'][:]
datates.close()
aertes = np.nanmean(lrp_aertes[:,:,:,:],axis=1).squeeze()

### Read in LRP maps (LENS)
datates = Dataset(directorydata2 + 'LRP_Maps_lens_AllSeasons_PAPER-Testing.nc')
lat1tes = datates.variables['lat'][:]
lon1tes = datates.variables['lon'][:]
lrp_lenstes = datates.variables['LRP'][:]
datates.close()
lenstes = np.nanmean(lrp_lenstes[:,:,:,:],axis=1).squeeze()

##############################################################################
##############################################################################
##############################################################################
## Assess data
data = [ghg,aer,lens,ghgtes,aertes,lenstes]

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of LRP means
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

fig = plt.figure(figsize=(8,4))
for i in range(len(datasets)):
    ax1 = plt.subplot(2,3,i+1)
            
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    circle = m.drawmapboundary(fill_color='dimgrey')
    circle.set_clip_on(False) 
    m.drawcoastlines(color='darkgrey',linewidth=0.35)
    
    ### Colorbar limits
    barlim = np.round(np.arange(0,0.6,0.1),2)
    
    ### Take lrp mean over all years
    if i < 3:
        lrpmean = data[i][0]
    else:
        lrpmean = data[i]
    
    var, lons_cyclic = addcyclic(lrpmean, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
    
    ### Make the plot continuous
    cs = m.contourf(x,y,var,np.arange(0,0.5001,0.005),
                    extend='max')                
    cmap = cm.classic_16.mpl_colormap          
    cs.set_cmap(cmap)
    
    ax1.annotate(r'\textbf{%s}' % (datasets[i]),xy=(0,0),xytext=(0.88,0.92),
                      textcoords='axes fraction',color='k',fontsize=14,
                      rotation=334,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[i],xy=(0,0),xytext=(0.085,0.93),
                          textcoords='axes fraction',color='dimgrey',fontsize=8,
                          rotation=0,ha='center',va='center')
    
    if i == 0:
        ax1.annotate(r'\textbf{TRAINING+TESTING}',xy=(0,0),xytext=(-0.3,0.55),
                          textcoords='axes fraction',color='k',fontsize=6,
                          rotation=0,ha='center',va='center')
        ax1.annotate(r'\textbf{COMPOSITE}',xy=(0,0),xytext=(-0.3,0.46),
                          textcoords='axes fraction',color='k',fontsize=6,
                          rotation=0,ha='center',va='center')
    if i == 3:
        ax1.annotate(r'\textbf{ONLY TESTING}',xy=(0,0),xytext=(-0.3,0.55),
                          textcoords='axes fraction',color='k',fontsize=6,
                          rotation=0,ha='center',va='center')
        ax1.annotate(r'\textbf{COMPOSITE}',xy=(0,0),xytext=(-0.3,0.46),
                          textcoords='axes fraction',color='k',fontsize=6,
                          rotation=0,ha='center',va='center')
    
cbar_ax = fig.add_axes([0.3,0.13,0.4,0.03])             
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)

cbar.set_label(r'\textbf{RELEVANCE}',fontsize=11,color='dimgrey',labelpad=1.4)  
cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=6)
cbar.outline.set_edgecolor('dimgrey')

plt.subplots_adjust(hspace=-0.2)
plt.savefig(directoryfigure + 'TestingMaps_LRPmean_3XLENS_PAPER.png',dpi=600)