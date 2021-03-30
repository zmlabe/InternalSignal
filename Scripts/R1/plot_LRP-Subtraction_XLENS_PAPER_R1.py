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
import palettable.scientific.diverging as dddd
import scipy.stats as sts
import calc_Utilities as UT

### Set parameters
variables = [r'T2M']
datasets = [r'AER+ minus ALL',r'GHG+ minus ALL']
seasons = [r'annual']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
SAMPLEQ = 100

### Set directories
directorydata = '/Users/zlabe/Documents/Research/InternalSignal/Data/FINAL/R1/Prediction/Trials/'
directoryfigure = '/Users/zlabe/Desktop/PAPER/R1/'

### Read in LRP maps for X(LENS) for L2=0.01 at 500 epochs
data = Dataset(directorydata + 'LRP_YearlyMaps_%s_20ens_%s_%s_R1-Trials_L2-%s_epochs-%s.nc' % (SAMPLEQ,variables[0],seasons[0],0.01,500))
lat1 = data.variables['lat'][:]
lon1 = data.variables['lon'][:]
lrp = data.variables['LRP'][:,:,:,:]
data.close()

### Calculate ensemble means
lrpghg = np.nanmean(lrp[0,:,:,:],axis=1)
lrpaer = np.nanmean(lrp[1,:,:,:],axis=1)
lrplens = np.nanmean(lrp[2,:,:,:],axis=1)

### Calculate differences
ghganom = np.nanmean(lrpghg - lrplens,axis=0)
aeranom = np.nanmean(lrpaer - lrplens,axis=0)

### Assess data
data = [ghganom,aeranom]

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of LRP means
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

fig = plt.figure(figsize=(9,3))
for i in range(len(data)):
    ax1 = plt.subplot(1,2,i+1)
            
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    circle = m.drawmapboundary(fill_color='dimgrey')
    circle.set_clip_on(False) 
    m.drawcoastlines(color='w',linewidth=0.35)
    
    ### Colorbar limits
    barlim = np.round(np.arange(-0.2,0.3,0.1),2)
    
    ### Take lrp mean over all years
    lrpmean = data[i]
    
    var, lons_cyclic = addcyclic(lrpmean, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
    
    ### Make the plot continuous
    cs = m.contourf(x,y,var,np.arange(-0.2,0.201,0.001),
                    extend='both')                
    cmap = dddd.Berlin_20.mpl_colormap         
    cs.set_cmap(cmap)
    
    ax1.annotate(r'\textbf{%s}' % (datasets[i]),xy=(0,0),xytext=(0.5,1.07),
                      textcoords='axes fraction',color='k',fontsize=14,
                      rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[i],xy=(0,0),xytext=(0.085,0.93),
                          textcoords='axes fraction',color='dimgrey',fontsize=8,
                          rotation=0,ha='center',va='center')
    
    # if i == 0:
    #     ax1.annotate(r'\textbf{L$_{2}$=0.01}',xy=(0,0),xytext=(-0.4,0.55),
    #                       textcoords='axes fraction',color='k',fontsize=8,
    #                       rotation=0,ha='left',va='center')
    
cbar_ax = fig.add_axes([0.32,0.09,0.4,0.03])          
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)

cbar.set_label(r'\textbf{RELEVANCE DIFFERENCE}',fontsize=9,color='dimgrey',labelpad=1.4)  
cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(top=0.85,wspace=0.01,hspace=0,bottom=0.14)
plt.savefig(directoryfigure + 'LRP-Subtraction_PAPER_R1.png',dpi=600)