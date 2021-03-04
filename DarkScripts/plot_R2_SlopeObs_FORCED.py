"""
Plots histograms of R^2 of observations from ensembles with the forced mean
removed

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 27 October 2020
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
directorydata = '/Users/zlabe/Documents/Research/InternalSignal/Data/'
directoryfigure = '/Users/zlabe/Desktop/SINGLE_v2.0/Histograms/rm_ensemble_mean/%s/' % variables[0]

### Read in slope data
filename_slope = 'Slopes_1920-2080-FORCEDens_XGHG-XAER-LENS_%s_RANDOMSEED_20ens.txt' % SAMPLEQ
slopes = np.genfromtxt(directorydata + filename_slope,unpack=True)
ghg_slopes = slopes[:,0]
aer_slopes = slopes[:,1]
lens_slopes = slopes[:,2]

### Read in R2 data
filename_R2= 'R2_1920-2080-FORCEDens_XGHG-XAER-LENS_%s_RANDOMSEED_20ens.txt' % SAMPLEQ
slopes = np.genfromtxt(directorydata + filename_R2,unpack=True)
ghg_r2 = slopes[:,0]
aer_r2 = slopes[:,1]
lens_r2 = slopes[:,2]

### Read in other R2 data
filename_R2all = 'R2_20CRv3-Obs_XGHG-XAER-LENS_%s_RANDOMSEED_Medians.txt' % SAMPLEQ
r2_allmodel = np.genfromtxt(directorydata + filename_R2all,unpack=True)
ghg_r2q = r2_allmodel[0]
aer_r2q = r2_allmodel[1]
lens_r2q = r2_allmodel[2]

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
plt.axvline(x=ghg_r2q,color='steelblue',linewidth=2,linestyle='--',dashes=(1,0.3),
            zorder=10)
plt.axvline(x=aer_r2q,color='goldenrod',linewidth=2,linestyle='--',dashes=(1,0.3),
            zorder=10)
plt.axvline(x=lens_r2q,color='forestgreen',linewidth=2,linestyle='--',dashes=(1,0.3),
            zorder=10)

weights_ghg = np.ones_like(ghg_r2)/len(ghg_r2)
n_ghg, bins_ghg, patches_ghg = plt.hist(ghg_r2,bins=np.arange(0,1.01,0.025),
                                        density=False,alpha=0.5,
                                        label=r'\textbf{XGHG: RM-ENSmean}',
                                        weights=weights_ghg,zorder=3)
for i in range(len(patches_ghg)):
    patches_ghg[i].set_facecolor('steelblue')
    patches_ghg[i].set_edgecolor('white')
    patches_ghg[i].set_linewidth(0.5)
 
weights_aer = np.ones_like(aer_r2)/len(aer_r2)
n_aer, bins_aer, patches_aer = plt.hist(aer_r2,bins=np.arange(0,1.01,0.025),
                                        density=False,alpha=0.5,
                                        label=r'\textbf{XAER: RM-ENSmean}',
                                        weights=weights_aer,zorder=4)
for i in range(len(patches_aer)):
    patches_aer[i].set_facecolor('goldenrod')
    patches_aer[i].set_edgecolor('white')
    patches_aer[i].set_linewidth(0.5)
    
weights_lens = np.ones_like(lens_r2)/len(lens_r2)
n_lens, bins_lens, patches_lens = plt.hist(lens_r2,bins=np.arange(0,1.01,0.025),
                                        density=False,alpha=0.5,
                                        label=r'\textbf{LENS: RM-ENSmean}',
                                        weights=weights_lens,zorder=5)
for i in range(len(patches_lens)):
    patches_lens[i].set_facecolor('forestgreen')
    patches_lens[i].set_edgecolor('white')
    patches_lens[i].set_linewidth(0.5)
    
leg = plt.legend(shadow=False,fontsize=7,loc='upper center',
        bbox_to_anchor=(0.5,1.1),fancybox=True,ncol=3,frameon=False,
        handlelength=3,handletextpad=1)

plt.ylabel(r'\textbf{PROPORTION[%s]}' % SAMPLEQ,fontsize=10,color='k')
plt.xlabel(r'\textbf{R$^{2}$} [ANNUAL -- T2M -- ENSEMBLES -- 1920-2080]',fontsize=10,color='k')
plt.yticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=6)
plt.xticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=6)
plt.xlim([0,1])   
plt.ylim([0,0.7])
    
plt.savefig(directoryfigure + 'Histogram_r2_XGHG-XAER-LENS_T2M_%s_FORCED.png' % SAMPLEQ,
            dpi=300)