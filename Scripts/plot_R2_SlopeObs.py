"""
Plots histograms of R^2 of observations

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 7 October 2020
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
SAMPLEQ = 500

### Set directories
directorydata = '/Users/zlabe/Documents/Research/InternalSignal/Data/'
directoryfigure = '/Users/zlabe/Desktop/SINGLE_v1.2-HISTOGRAM/%s/' % variables[0]

### Read in slope data
filename_slope = 'Slopes_20CRv3-Obs_XGHG-XAER-LENS_%s_RANDOMSEED.txt' % SAMPLEQ
slopes = np.genfromtxt(directorydata + filename_slope,unpack=True)
ghg_slopes = slopes[:,0]
aer_slopes = slopes[:,1]
lens_slopes = slopes[:,2]

### Read in R2 data
filename_R2= 'R2_20CRv3-Obs_XGHG-XAER-LENS_%s_RANDOMSEED.txt' % SAMPLEQ
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
ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)

### Plot histograms
weights_ghg = np.ones_like(ghg_r2)/len(ghg_r2)
n_ghg, bins_ghg, patches_ghg = plt.hist(ghg_r2,bins=np.arange(0,1.01,0.025),
                                        density=False,alpha=0.5,
                                        label=r'\textbf{XGHG}',
                                        weights=weights_ghg,zorder=3)
for i in range(len(patches_ghg)):
    patches_ghg[i].set_facecolor('steelblue')
    patches_ghg[i].set_edgecolor('white')
    patches_ghg[i].set_linewidth(0.5)
 
weights_aer = np.ones_like(aer_r2)/len(aer_r2)
n_aer, bins_aer, patches_aer = plt.hist(aer_r2,bins=np.arange(0,1.01,0.025),
                                        density=False,alpha=0.5,
                                        label=r'\textbf{XAER}',
                                        weights=weights_aer,zorder=4)
for i in range(len(patches_aer)):
    patches_aer[i].set_facecolor('goldenrod')
    patches_aer[i].set_edgecolor('white')
    patches_aer[i].set_linewidth(0.5)
    
weights_lens = np.ones_like(lens_r2)/len(lens_r2)
n_lens, bins_lens, patches_lens = plt.hist(lens_r2,bins=np.arange(0,1.01,0.025),
                                        density=False,alpha=0.5,
                                        label=r'\textbf{LENS}',
                                        weights=weights_lens,zorder=5)
for i in range(len(patches_lens)):
    patches_lens[i].set_facecolor('forestgreen')
    patches_lens[i].set_edgecolor('white')
    patches_lens[i].set_linewidth(0.5)
    
leg = plt.legend(shadow=False,fontsize=7,loc='upper center',
        bbox_to_anchor=(0.5,1.1),fancybox=True,ncol=3,frameon=False,
        handlelength=3,handletextpad=1)

plt.ylabel(r'\textbf{PROPORTION[%s]}' % SAMPLEQ,fontsize=10,color='k')
plt.xlabel(r'\textbf{R$^{2}$} [ANNUAL -- T2M -- 20CRv3 -- (1920-2015)]',fontsize=10,color='k')
plt.yticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=6)
plt.xticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=6)
plt.xlim([0,1])   
plt.ylim([0,0.7])
    
plt.savefig(directoryfigure + 'Histogram_r2_XGHG-XAER-LENS_T2M_%s.png' % SAMPLEQ,
            dpi=300)