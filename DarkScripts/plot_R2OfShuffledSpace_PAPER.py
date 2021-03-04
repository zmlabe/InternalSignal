"""
Plots normalized histograms of R^2 of observations using shuffled ensemble
and years for paper

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 11 November 2020
"""

### Import packages
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import cmocean
import palettable.cubehelix as cm

### Set parameters
variables = [r'T2M']
seasons = [r'annual']
SAMPLEQ = 500
SAMPLEQ2 = 100

### Set directories
directorydata2 = '/Users/zlabe/Documents/Research/InternalSignal/Data/FINAL/'
directoryfigure =  '/Users/zlabe/Documents/Projects/InternalSignal/DarkFigures/'

### Read in slope data
filename_slope = 'Slopes_20CRv3-SHUFFLE-TIMENS_%s_RANDOMSEED_20ens.txt' % SAMPLEQ
slopes = np.genfromtxt(directorydata2 + filename_slope,unpack=True)

### Read in R2 data
filename_R2 = 'R2_20CRv3-SHUFFLE-TIMENS_%s_RANDOMSEED_20ens.txt' % SAMPLEQ
r2 = np.genfromtxt(directorydata2 + filename_R2,unpack=True)

### Read in other R2 data
filename_R2all = 'R2_20CRv3-Obs_XGHG-XAER-LENS_%s_RANDOMSEED_Medians_20ens.txt' % SAMPLEQ2
r2_allmodel = np.genfromtxt(directorydata2 + filename_R2all,unpack=True)
ghg_r2 = r2_allmodel[0]
aer_r2 = r2_allmodel[1]
lens_r2 = r2_allmodel[2]

###############################################################################
###############################################################################
###############################################################################
### Create plot for histograms of slopes
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
plt.rc('savefig',facecolor='black')
plt.rc('axes',edgecolor='darkgrey')
plt.rc('xtick',color='darkgrey')
plt.rc('ytick',color='darkgrey')
plt.rc('axes',labelcolor='darkgrey')
plt.rc('axes',facecolor='black')

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
ax.spines['bottom'].set_color('darkgrey')
ax.spines['left'].set_color('darkgrey')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2) 
ax.tick_params('both',length=5.5,width=2,which='major',color='darkgrey')  

### Plot histograms
plt.axvline(x=ghg_r2,color='deepskyblue',linewidth=2,linestyle='--',dashes=(1,0.3),
            zorder=10,label=r'\textbf{AER+}')
plt.axvline(x=aer_r2,color='gold',linewidth=2,linestyle='--',dashes=(1,0.3),
            zorder=10,label=r'\textbf{GHG+}')
plt.axvline(x=lens_r2,color='crimson',linewidth=2,linestyle='--',dashes=(1,0.3),
            zorder=10,label=r'\textbf{ALL}')


weights = np.ones_like(r2)/len(r2)
n, bins, patches = plt.hist(r2,bins=np.arange(0,1.01,0.025),
                                        density=False,alpha=1,
                                        label=r'\textbf{SHUFFLE}',
                                        weights=weights,zorder=3,color='w')
for i in range(len(patches)):
    patches[i].set_facecolor('w')
    patches[i].set_edgecolor('k')
    patches[i].set_linewidth(1)
    
leg = plt.legend(shadow=False,fontsize=7,loc='upper center',
    bbox_to_anchor=(0.5,1.1),fancybox=True,ncol=4,frameon=False,
    handlelength=3,handletextpad=1)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.text(0.763,0.3136,r'\textbf{SHUFFLE}',fontsize=7,color='w',zorder=12)
plt.ylabel(r'\textbf{PROPORTION}',fontsize=10,color='w')
plt.xlabel(r'\textbf{R$^{2}$ OF OBSERVATIONS}',fontsize=10,color='w')
plt.yticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=6)
plt.xticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=6)
plt.xlim([0,1])   
plt.ylim([0,0.3])

###############################################################################
###############################################################################
###############################################################################
### Read in LRP maps for shuffle data
data = Dataset(directorydata2 + 'LRP_Maps_%s_20ens_SHUFFLE-TIMENS.nc' % (SAMPLEQ))
lat1 = data.variables['lat'][:]
lon1 = data.variables['lon'][:]
lrprandom = data.variables['LRP'][:].squeeze()
data.close()

### Average across all 500
# mean = np.nanmean(lrprandom,axis=0)
mean = lrprandom[0] # example

labelq = ['LRP-RELEVANCE']
limitsq = [np.arange(0,0.5001,0.005)]
barlimq = [np.round(np.arange(0,0.6,0.1),2)]
datasetsq = [r'SHUFFLE']
colorbarendq = ['max']
cmapq = [cm.classic_16.mpl_colormap]

ax1 = plt.axes([.24,.54,.4,.25])
        
m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
circle = m.drawmapboundary(fill_color='dimgrey')
circle.set_clip_on(False) 
m.drawcoastlines(color='w',linewidth=0.35)

### Colorbar limits
barlim = barlimq[0]

var, lons_cyclic = addcyclic(mean, lon1)
var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
x, y = m(lon2d, lat2d)

### Make the plot continuous
cs = m.contourf(x,y,var,limitsq[0],
                extend=colorbarendq[0])                        
cs.set_cmap(cmapq[0])

ax1.annotate(r'\textbf{%s}' % (datasetsq[0]),xy=(0,0),xytext=(0.865,0.92),
                  textcoords='axes fraction',color='w',fontsize=9,
                  rotation=332,ha='center',va='center')

cbar = m.colorbar(cs,drawedges=False,location='bottom',extendfrac=0.07,
                  extend=colorbarendq[0],pad=0.1)                  
cbar.set_label(r'\textbf{%s}' % labelq[0],fontsize=8,color='w',labelpad=1.4)  
cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=6,labelcolor='darkgrey')
cbar.outline.set_edgecolor('darkgrey')
    
plt.savefig(directoryfigure + 'HistogramR2OfShuffledEns_PAPER_DARK.png',dpi=600)