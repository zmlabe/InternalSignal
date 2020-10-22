"""
Plots LRP maps of uncertainty for RANDOM data

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 21 October 2020
"""

### Import packages
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import cmocean
import palettable.cubehelix as cm
import scipy.stats as sts

### Set parameters
variables = [r'T2M']
datasets = [r'RANDOM']
seasons = [r'annual']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
SAMPLEQ = 100

### Set directories
directorydata = '/Users/zlabe/Documents/Research/InternalSignal/Data/'
directoryfigure = '/Users/zlabe/Desktop/SINGLE_v2.0/Histograms/LRP/%s/' % variables[0]

### Read in LRP maps
data = Dataset(directorydata + 'LRP_Maps_%s_20ens_%s_%s_RANDOM.nc' % (SAMPLEQ,variables[0],seasons[0]))
lat1 = data.variables['lat'][:]
lon1 = data.variables['lon'][:]
lrp = data.variables['LRP'][:].squeeze()
data.close()

###############################################################################
###############################################################################
###############################################################################
### Calculate statistics over the 100 random samples
### Standard Deviation 
std = np.nanstd(lrp[:,:,:],axis=0)

### Mean
mean = np.nanmean(lrp[:,:,:],axis=0)

### Percentile
percentile = np.percentile(lrp[:,:,:],90,axis=0)

percn = np.empty((SAMPLEQ,lat1.shape[0]*lon1.shape[0]))
for i in range(SAMPLEQ):  
    x = np.ravel(lrp[i,:,:])
    percn[i,:] = (sts.rankdata(x)-1)/len(x)
perc = np.reshape(percn,(SAMPLEQ,lat1.shape[0],lon1.shape[0]))
percmean = np.nanmean(perc,axis=0)*100. # percent

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of LRP means
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

### Prepare plotting parameters
dataq = [mean,std,percmean]
labelq = ['RELEVANCE','STD. DEV. [RELEVANCE]','PERCENTILE [\%]']
limitsq = [np.arange(0,0.2001,0.001),np.arange(0,0.051,0.001),np.arange(0,101,1)]
barlimq = [np.round(np.arange(0,0.201,0.05),2),np.round(np.arange(0,0.051,0.01),2),np.arange(0,101,25)]
datasetsq = [r'RANDOM']*3
colorbarendq = ['max','max','neither']
cmapq = [cm.classic_16.mpl_colormap,cm.cubehelix2_16.mpl_colormap,cmocean.cm.dense_r]

fig = plt.figure(figsize=(10,2.5))
for i in range(len(dataq)):
    ax1 = plt.subplot(1,3,i+1)
            
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    circle = m.drawmapboundary(fill_color='k')
    circle.set_clip_on(False) 
    m.drawcoastlines(color='darkgrey',linewidth=0.35)
    
    ### Colorbar limits
    barlim = barlimq[i]
    
    ### Take lrp mean over all years
    lrpstats = dataq[i]
    
    var, lons_cyclic = addcyclic(lrpstats, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
    
    ### Make the plot continuous
    cs = m.contourf(x,y,var,limitsq[i],
                    extend=colorbarendq[i])                        
    cs.set_cmap(cmapq[i])
    
    ax1.annotate(r'\textbf{%s}' % (datasetsq[i]),xy=(0,0),xytext=(0.865,0.91),
                      textcoords='axes fraction',color='k',fontsize=14,
                      rotation=335,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[i],xy=(0,0),xytext=(0.085,0.93),
                          textcoords='axes fraction',color='dimgrey',fontsize=8,
                          rotation=0,ha='center',va='center')
    
    cbar = m.colorbar(cs,drawedges=False,location='bottom',extendfrac=0.07,
                      extend=colorbarendq[i])                  
    cbar.set_label(r'\textbf{%s}' % labelq[i],fontsize=11,color='dimgrey',labelpad=1.4)  
    cbar.set_ticks(barlim)
    cbar.set_ticklabels(list(map(str,barlim)))
    cbar.ax.tick_params(axis='x', size=.01,labelsize=6)
    cbar.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(bottom=0.17)
plt.savefig(directoryfigure + 'LRPstats_RANDOMDATA_%s_%s_%s.png' % (variables[0],
                                                        seasons[0],
                                                        SAMPLEQ),
                                                        dpi=300)