"""
Plots normalized histograms of R^2 of observations using shuffled ensemble
and years for paper

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 30 March 2021
version   : R1
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
directoryfigure = '/Users/zlabe/Desktop/PAPER/R1/'

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

### Plot histograms
plt.axvline(x=ghg_r2,color='steelblue',linewidth=2,linestyle='--',dashes=(1,0.3),
            zorder=10,label=r'\textbf{AER+}')
plt.axvline(x=aer_r2,color='darkgoldenrod',linewidth=2,linestyle='--',dashes=(1,0.3),
            zorder=10,label=r'\textbf{GHG+}')
plt.axvline(x=lens_r2,color='crimson',linewidth=2,linestyle='--',dashes=(1,0.3),
            zorder=10,label=r'\textbf{ALL}')


weights = np.ones_like(r2)/len(r2)
n, bins, patches = plt.hist(r2,bins=np.arange(0,1.01,0.025),
                                        density=False,alpha=0.5,
                                        label=r'\textbf{SHUFFLE}',
                                        weights=weights,zorder=3)
for i in range(len(patches)):
    patches[i].set_facecolor('k')
    patches[i].set_edgecolor('white')
    patches[i].set_linewidth(1)
    
leg = plt.legend(shadow=False,fontsize=7,loc='upper center',
    bbox_to_anchor=(0.5,1.1),fancybox=True,ncol=4,frameon=False,
    handlelength=3,handletextpad=1)

plt.ylabel(r'\textbf{PROPORTION}',fontsize=10,color='k')
plt.xlabel(r'\textbf{R$^{2}$ OF OBSERVATIONS}',fontsize=10,color='k')
plt.yticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=6)
plt.xticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=6)
plt.xlim([0,1])   
plt.ylim([0,0.3])
    
plt.savefig(directoryfigure + 'HistogramR2OfShuffledEns_PAPER_R1.png',dpi=600)