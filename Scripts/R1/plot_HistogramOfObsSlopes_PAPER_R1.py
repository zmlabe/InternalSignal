"""
Plots normalized histograms of slope of observations for the paper

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 30 March 2021
Version   : R1
"""

### Import packages
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import scipy.stats as stats

### Set parameters
variables = [r'T2M']
datasets = [r'XGHG',r'XAER',r'lens']
seasons = [r'annual']
SAMPLEQ = 100

### Set directories
directorydata = '/Users/zlabe/Documents/Research/InternalSignal/Data/FINAL/'
directoryfigure = '/Users/zlabe/Desktop/PAPER/R1/'

### Read in slope data
filename_slope = 'Slopes_20CRv3-Obs_XGHG-XAER-LENS_%s_RANDOMSEED_20ens.txt' % SAMPLEQ
slopes = np.genfromtxt(directorydata + filename_slope,unpack=True)
ghg_slopes = slopes[:,0]
aer_slopes = slopes[:,1]
lens_slopes = slopes[:,2]

### Read in R2 data
filename_R2= 'R2_20CRv3-Obs_XGHG-XAER-LENS_%s_RANDOMSEED_20ens.txt' % SAMPLEQ
slopes = np.genfromtxt(directorydata + filename_R2,unpack=True)
ghg_r2 = slopes[:,0]
aer_r2 = slopes[:,1]
lens_r2 = slopes[:,2]

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
plt.axvline(x=1,color='k',linewidth=2,linestyle='--',dashes=(1,0.3),
            zorder=10)

weights_ghg = np.ones_like(ghg_slopes)/len(ghg_slopes)
n_ghg, bins_ghg, patches_ghg = plt.hist(ghg_slopes,bins=np.arange(-1.2,2.1,0.1)-0.05,
                                        density=False,alpha=0.5,
                                        label=r'\textbf{AER+}',
                                        weights=weights_ghg,zorder=3)
for i in range(len(patches_ghg)):
    patches_ghg[i].set_facecolor('steelblue')
    patches_ghg[i].set_edgecolor('white')
    patches_ghg[i].set_linewidth(0.5)
 
weights_aer = np.ones_like(aer_slopes)/len(aer_slopes)
n_aer, bins_aer, patches_aer = plt.hist(aer_slopes,bins=np.arange(-1.2,2.1,0.1)-0.05,
                                        density=False,alpha=0.5,
                                        label=r'\textbf{GHG+}',
                                        weights=weights_aer,zorder=4)
for i in range(len(patches_aer)):
    patches_aer[i].set_facecolor('darkgoldenrod')
    patches_aer[i].set_edgecolor('white')
    patches_aer[i].set_linewidth(0.5)
    
weights_lens = np.ones_like(lens_slopes)/len(lens_slopes)
n_lens, bins_lens, patches_lens = plt.hist(lens_slopes,bins=np.arange(-1.2,2.1,0.1)-0.05,
                                        density=False,alpha=0.5,
                                        label=r'\textbf{ALL}',
                                        weights=weights_lens,zorder=5)
for i in range(len(patches_lens)):
    patches_lens[i].set_facecolor('crimson')
    patches_lens[i].set_edgecolor('white')
    patches_lens[i].set_linewidth(0.5)
    
leg = plt.legend(shadow=False,fontsize=14,loc='upper center',
        bbox_to_anchor=(0.15,1),fancybox=True,ncol=1,frameon=False,
        handlelength=2,handletextpad=1)

plt.ylabel(r'\textbf{PROPORTION}',fontsize=10,color='k')
plt.xlabel(r'\textbf{SLOPE OF OBSERVATIONS}',fontsize=10,color='k')
plt.yticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=6)
plt.xticks(np.arange(-1.2,10.1,0.2),map(str,np.round(np.arange(-1.2,10.1,0.2),2)),size=6)
plt.xlim([-1.2,2])   
plt.ylim([0,0.6])
    
plt.savefig(directoryfigure + 'HistogramOfObsSlopes_PAPER_R1.png',dpi=600)