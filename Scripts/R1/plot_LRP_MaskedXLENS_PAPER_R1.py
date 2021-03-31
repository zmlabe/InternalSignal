"""
Plots LRP map that is masked out using a statistical test

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 7 October 2020
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
datasets = [r'AER+',r'GHG+',r'ALL']
seasons = [r'annual']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
SAMPLEQ = 100
SAMPLEQ2 = 500
typer = 'SHUFFLE'
years = np.arange(1920,2080+1,1)
yearq = np.where((years >= 1960) & (years <= 2039))[0]

### Set directories
directorydata = '/Users/zlabe/Documents/Research/InternalSignal/Data/FINAL/'
directoryfigure = '/Users/zlabe/Desktop/PAPER/'

### Read in LRP maps for X(LENS)
data = Dataset(directorydata + 'LRP_YearlyMaps_%s_20ens_%s_%s.nc' % (SAMPLEQ,variables[0],seasons[0]))
lat1 = data.variables['lat'][:]
lon1 = data.variables['lon'][:]
lrp = data.variables['LRP'][:,:,:,:]
data.close()

lrpghgq = lrp[0]
lrpaerq = lrp[1]
lrplensq = lrp[2]
lrpghg = np.nanmean(lrpghgq[:,yearq,:,:],axis=1)
lrpaer = np.nanmean(lrpaerq[:,yearq,:,:],axis=1)
lrplens = np.nanmean(lrplensq[:,yearq,:,:],axis=1)

if typer == 'SHUFFLE':
    ### Read in LRP maps for shuffle data
    data = Dataset(directorydata + 'LRP_Maps_%s_20ens_SHUFFLE-TIMENS.nc' % (SAMPLEQ2))
    lat1 = data.variables['lat'][:]
    lon1 = data.variables['lon'][:]
    lrprandom = data.variables['LRP'][:].squeeze()
    data.close()
else:
    print(ValueError('WRONG TYPE OF RANDOMIZED DATA!'))

###############################################################################
###############################################################################
###############################################################################
### Calculate statistics over the 500 random samples
### Mean
mean_ghg = np.nanmean(lrpghg[:,:,:],axis=0)
mean_aer = np.nanmean(lrpaer[:,:,:],axis=0)
mean_lens = np.nanmean(lrplens[:,:,:],axis=0)
mean_random = np.nanmean(lrprandom[:,:,:],axis=0)
max_random = np.nanmax(lrprandom[:,:,:])
min_random = np.nanmax(lrprandom[:,:,:])
mean = [mean_ghg, mean_aer, mean_lens]

###############################################################################
###############################################################################
###############################################################################
### Calculate 95th percentile of all points
thresh = np.percentile(lrprandom,95)
lrpmask_ghg = mean_ghg 
lrpmask_ghg[lrpmask_ghg<=thresh] = np.nan
lrpmask_aer = mean_aer 
lrpmask_aer[lrpmask_aer<=thresh] = np.nan
lrpmask_lens = mean_lens 
lrpmask_lens[lrpmask_lens<=thresh] = np.nan
maskdata = [lrpmask_ghg,lrpmask_aer,lrpmask_lens]

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of LRP means
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

fig = plt.figure(figsize=(10,2.5))
for i in range(len(datasets)):
    ax1 = plt.subplot(1,3,i+1)
            
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    circle = m.drawmapboundary(fill_color='dimgrey')
    circle.set_clip_on(False) 
    m.drawcoastlines(color='darkgrey',linewidth=0.35)
    
    ### Colorbar limits
    barlim = np.round(np.arange(0,0.6,0.1),2)
    
    ### Take lrp mean over all years
    lrpmean = maskdata[i]
    
    var, lons_cyclic = addcyclic(lrpmean, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
    
    ### Make the plot continuous
    cs = m.contourf(x,y,var,np.arange(0,0.5001,0.005),
                    extend='max')                
    cmap = cm.classic_16.mpl_colormap          
    cs.set_cmap(cmap)
    
    ax1.annotate(r'\textbf{%s}' % (datasets[i]),xy=(0,0),xytext=(0.865,0.91),
                      textcoords='axes fraction',color='k',fontsize=14,
                      rotation=331,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[i],xy=(0,0),xytext=(0.085,0.93),
                          textcoords='axes fraction',color='dimgrey',fontsize=8,
                          rotation=0,ha='center',va='center')
    
cbar_ax = fig.add_axes([0.293,0.13,0.4,0.03])             
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)

cbar.set_label(r'\textbf{RELEVANCE}',fontsize=11,color='dimgrey',labelpad=1.4)  
cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=6)
cbar.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(bottom=0.17)
plt.savefig(directoryfigure + 'LRPmean_3XLENS_Masked.png',dpi=600)