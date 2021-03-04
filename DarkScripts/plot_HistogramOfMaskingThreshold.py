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
import calc_Utilities as UT

### Set parameters
variables = [r'T2M']
datasets = [r'RANDOM']
seasons = [r'annual']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
SAMPLEQ = 100
SAMPLEQ2 = 500

### Set directories
directorydata = '/Users/zlabe/Documents/Research/InternalSignal/Data/'
directoryfigure = '/Users/zlabe/Desktop/SINGLE_v2.0/Histograms/LRP/%s/' % variables[0]

### Read in LRP maps
data = Dataset(directorydata + 'LRP_Maps_%s_20ens_%s_%s_SHUFFLE-TIMENS.nc' % (SAMPLEQ,variables[0],seasons[0]))
lat1 = data.variables['lat'][:]
lon1 = data.variables['lon'][:]
lrp = data.variables['LRP'][:].squeeze()
data.close()

###############################################################################
###############################################################################
###############################################################################
### Calculate statistics over the 100 random samples

### Mean
mean = np.nanmean(lrp[:,:,:],axis=0)
lon2,lat2 = np.meshgrid(lon1,lat1)
meanall = lrp.ravel()
thresh = np.percentile(meanall,95)

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of LRP means
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Create plot for histograms of slope
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

fig = plt.figure()        
ax = plt.subplot(111)
adjust_spines(ax, ['left','bottom'])            
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none') 
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2) 
ax.tick_params('both',length=5.5,width=2,which='major',color='dimgrey')  
ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)

plt.axvline(x=thresh,color=cm.classic_16.mpl_colormap(0.57),linewidth=2,linestyle='--',dashes=(1,0.3),
            zorder=10,label=r'\textbf{THRESHOLD TO MASK DATA - 95th percentile}')

weights_random = np.ones_like(meanall)/len(meanall)
n_random, bins_random, patches_random = plt.hist(meanall,bins=np.arange(0.0,0.152,0.002)-0.001,
                                        density=False,alpha=1,
                                        weights=weights_random,zorder=3)

for i in range(len(patches_random)):
    patches_random[i].set_facecolor(cm.classic_16.mpl_colormap(0.35))
    patches_random[i].set_edgecolor('white')
    patches_random[i].set_linewidth(0.5)
    
leg = plt.legend(shadow=False,fontsize=7,loc='upper center',
    bbox_to_anchor=(0.28,1),fancybox=True,ncol=1,frameon=False,
    handlelength=3,handletextpad=1)

plt.ylabel(r'\textbf{PROPORTION[%s]}' % SAMPLEQ,fontsize=10,color='k')
plt.xlabel(r'\textbf{DISTRIBUTION OF LRP FOR SHUFFLED DATA [RELEVANCE]}',fontsize=10,color='k')
plt.yticks(np.arange(0,1.1,0.05),map(str,np.round(np.arange(0,1.1,0.05),2)),size=6)
plt.xticks(np.arange(0,1.1,0.01),map(str,np.round(np.arange(0,1.1,0.01),2)),size=6)
plt.xlim([0,0.15])   
plt.ylim([0,0.1])

plt.savefig(directoryfigure + 'LRPstats_ThresholdHistONLY_TIMENS_%s_%s_%s.png' % (variables[0],
                                                        seasons[0],
                                                        SAMPLEQ),
                                                        dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Prepare plotting parameters
dataq = [mean]
labelq = ['LRP RELEVANCE']
limitsq = [np.arange(0,0.5001,0.005)]
barlimq = [np.round(np.arange(0,0.6,0.1),2)]
datasetsq = [r'CESM-LENS']
colorbarendq = ['max']
cmapq = [cm.classic_16.mpl_colormap]

### Read in LRP maps for X(LENS)
data = Dataset(directorydata + 'LRP_Maps_%s_%s_%s.nc' % (variables[0],seasons[0],SAMPLEQ2))
lat1 = data.variables['lat'][:]
lon1 = data.variables['lon'][:]
lrpc = data.variables['LRP'][:]
data.close()

lrplens = lrpc[2,:,:,:]
# lrpmask_lens = np.nanmean(lrplens[:,:,:],axis=0) # Mean
lrpmask_lens = lrplens[5,:,:] # Pick example
lrpmask_lens[lrpmask_lens<=thresh] = np.nan
maskdata = [lrpmask_lens]

###############################################################################
###############################################################################
###############################################################################
fig = plt.figure()
ax1 = plt.subplot(111)
        
m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
circle = m.drawmapboundary(fill_color='dimgrey')
circle.set_clip_on(False) 
m.drawcoastlines(color='darkgrey',linewidth=0.35)

### Colorbar limits
barlim = barlimq[0]

### Take lrp mean over all years
lrpstats = maskdata[0]

var, lons_cyclic = addcyclic(lrpstats, lon1)
var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
x, y = m(lon2d, lat2d)

### Make the plot continuous
cs = m.contourf(x,y,var,limitsq[0],
                extend=colorbarendq[0])                        
cs.set_cmap(cmapq[0])

ax1.annotate(r'\textbf{%s}' % (datasetsq[0]),xy=(0,0),xytext=(0.865,0.91),
                  textcoords='axes fraction',color='k',fontsize=21,
                  rotation=335,ha='center',va='center')

cbar = m.colorbar(cs,drawedges=False,location='bottom',extendfrac=0.07,
                  extend=colorbarendq[0],pad=0.2)                  
cbar.set_label(r'\textbf{%s}' % labelq[0],fontsize=11,color='dimgrey',labelpad=1.4)  
cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=6,labelcolor='dimgrey')
cbar.outline.set_edgecolor('dimgrey')

plt.tight_layout()
# plt.subplots_adjust(bottom=0.17)
plt.savefig(directoryfigure + 'LRPstats_ThresholdHistMap_TIMENS_%s_%s_%s.png' % (variables[0],
                                                        seasons[0],
                                                        SAMPLEQ),
                                                        dpi=300)