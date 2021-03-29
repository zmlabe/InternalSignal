"""
Compares LRP maps optimized models

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
directorydata = '/Users/zlabe/Documents/Research/InternalSignal/Data/FINAL/R1/Prediction/Trials/'
directoryfigure = '/Users/zlabe/Desktop/PAPER/R1/'

# ### Read in LRP maps for X(LENS) for L2=0.01 at 500 epochs
# data = Dataset(directorydata + 'LRP_YearlyMaps_%s_20ens_%s_%s_R1-Trials_L2-%s_epochs-%s.nc' % (SAMPLEQ,variables[0],seasons[0],0.01,500))
# lat1 = data.variables['lat'][:]
# lon1 = data.variables['lon'][:]
# lrp = data.variables['LRP'][:,:,:,:]
# data.close()
# lrpghg = np.nanmean(lrp[0,:,:,:],axis=1)
# lrpaer = np.nanmean(lrp[1,:,:,:],axis=1)
# lrplens = np.nanmean(lrp[2,:,:,:],axis=1)

# ### Read in LRP maps for X(LENS) for L2=0.001 at 1500 epochs
# dataother = Dataset(directorydata + 'LRP_YearlyMaps_%s_20ens_%s_%s_R1-Trials_L2-%s_epochs-%s.nc' % (SAMPLEQ,variables[0],seasons[0],0.001,1500))
# lat1other = dataother.variables['lat'][:]
# lon1other = dataother.variables['lon'][:]
# lrpother = dataother.variables['LRP'][:,:,:,:]
# dataother.close()

# lrpghgother = np.nanmean(lrpother[0,:,:,:],axis=1)
# lrpaerother = np.nanmean(lrpother[1,:,:,:],axis=1)
# lrplensother = np.nanmean(lrpother[2,:,:,:],axis=1)

# ##############################################################################
# ##############################################################################
# ##############################################################################
# ## Assess data
# data = [lrpghg,lrpaer,lrplens,
#         lrpghgother,lrpaerother,lrplensother]

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
    lrpmean = np.nanmean(data[i],axis=0)
    
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
        ax1.annotate(r'\textbf{L$_{2}$=0.01}',xy=(0,0),xytext=(-0.4,0.55),
                          textcoords='axes fraction',color='k',fontsize=8,
                          rotation=0,ha='left',va='center')
        ax1.annotate(r'\textbf{Epochs=500}',xy=(0,0),xytext=(-0.4,0.42),
                          textcoords='axes fraction',color='k',fontsize=8,
                          rotation=0,ha='left',va='center')
    if i == 3:
        ax1.annotate(r'\textbf{L$_{2}$=0.001}',xy=(0,0),xytext=(-0.4,0.55),
                          textcoords='axes fraction',color='k',fontsize=8,
                          rotation=0,ha='left',va='center')
        ax1.annotate(r'\textbf{Epochs=1500}',xy=(0,0),xytext=(-0.4,0.42),
                          textcoords='axes fraction',color='k',fontsize=8,
                          rotation=0,ha='left',va='center')
    
cbar_ax = fig.add_axes([0.3,0.13,0.4,0.03])             
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)

cbar.set_label(r'\textbf{RELEVANCE}',fontsize=11,color='dimgrey',labelpad=1.4)  
cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=6)
cbar.outline.set_edgecolor('dimgrey')

plt.subplots_adjust(hspace=-0.2)
plt.savefig(directoryfigure + 'L2_LRPmean_3XLENS_PAPER.png',dpi=600)