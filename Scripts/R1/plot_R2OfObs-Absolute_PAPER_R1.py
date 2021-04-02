"""
Plot largest R^2 for both GHG+ and ALL in one figure histogram

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 2 April 2021
Version   : 1
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
directorydata = '/Users/zlabe/Documents/Research/InternalSignal/Data/FINAL/R1/Prediction/Trials/'
directoryfigure = '/Users/zlabe/Desktop/PAPER/R1/ANNtrials/'

### Select ANNs to analyze
l2 = [0.01,0.01,0.1,0.001,0.001,0.001]
epochs = [500,100,500,100,500,1500]
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]

def readData(directorydata,SAMPLEQ,l2,epochs):
    ### Read in R2 data
    filename_R2 = 'R2_20CRv3-Obs_XGHG-XAER-LENS_%s_RANDOMSEED_20ens_R1-Trials_L2-%s_epochs-%s.txt' % (SAMPLEQ,l2,epochs)
    slopes = np.genfromtxt(directorydata + filename_R2,unpack=True)
    ghg_r2q = slopes[:,0]
    aer_r2q = slopes[:,1]
    lens_r2q = slopes[:,2]
    
    median_ghg = np.median(ghg_r2q)
    median_aer = np.median(aer_r2q)
    median_lens = np.median(lens_r2q)
    mediansall = np.array([median_ghg,median_aer,median_lens])

    return ghg_r2q,aer_r2q,lens_r2q,mediansall
    
### Read in data
r_ghgm = np.empty((len(l2),SAMPLEQ))
r_aerm = np.empty((len(l2),SAMPLEQ))
r_lensm = np.empty((len(l2),SAMPLEQ))
mediansm = np.empty((len(l2),len(datasets)))
for i in range(len(l2)):
    r_ghgm[i,:],r_aerm[i,:],r_lensm[i,:],mediansm[i,:] = readData(directorydata,SAMPLEQ,l2[i],epochs[i])
    
### See where maximum is
maxr = np.argmax(mediansm,axis=0)
maxaer = maxr[1]
maxlen = maxr[2]

### Histogram of maximum
maxr_aer = r_aerm[maxaer]
maxr_len = r_lensm[maxlen]

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
weights_aer = np.ones_like(maxr_aer)/len(maxr_aer)
n_aer, bins_aer, patches_aer = plt.hist(maxr_aer,bins=np.arange(0,1.1,0.02)-0.01,
                                        density=False,alpha=0.5,
                                        label=r'\textbf{[GHG+]: epoch=%s, L$_{2}$=%s}' % (epochs[maxaer],l2[maxaer]),
                                        weights=weights_aer,zorder=4)
for i in range(len(patches_aer)):
    patches_aer[i].set_facecolor('darkgoldenrod')
    patches_aer[i].set_edgecolor('white')
    patches_aer[i].set_linewidth(0.5)
    
weights_lens = np.ones_like(maxr_len)/len(maxr_len)
n_lens, bins_lens, patches_lens = plt.hist(maxr_len,bins=np.arange(0,1.1,0.02)-0.01,
                                        density=False,alpha=0.5,
                                        label=r'\textbf{[ALL]: epoch=%s, L$_{2}$=%s}' % (epochs[maxlen],l2[maxlen]),
                                        weights=weights_lens,zorder=5)
for i in range(len(patches_lens)):
    patches_lens[i].set_facecolor('crimson')
    patches_lens[i].set_edgecolor('white')
    patches_lens[i].set_linewidth(0.5)
    
leg = plt.legend(shadow=False,fontsize=14,loc='upper center',
        bbox_to_anchor=(0.35,1),fancybox=True,ncol=1,frameon=False,
        handlelength=2,handletextpad=1)

plt.ylabel(r'\textbf{PROPORTION}',fontsize=10,color='k')
plt.xlabel(r'\textbf{R$^{2}$ OF OBSERVATIONS}',fontsize=10,color='k')
plt.xticks(np.arange(0,5,0.05),map(str,np.round(np.arange(0,5,0.05),2)),size=5)
plt.yticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=5)
plt.xlim([0,1.])   
plt.ylim([0,0.6])
    
plt.savefig(directoryfigure + 'HistogramR2OfObs-Absolute_PAPER.png',dpi=600)