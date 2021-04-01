"""
Plots correlation (Rval) of slope of observations for paper showing possible 
ANNs with different combinations of epochs and L2

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 1 April 2021
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
    filename_R2 = 'Rval_20CRv3-Obs_XGHG-XAER-LENS_%s_RANDOMSEED_20ens_R1-Trials_L2-%s_epochs-%s.txt' % (SAMPLEQ,l2,epochs)
    slopes = np.genfromtxt(directorydata + filename_R2,unpack=True)
    ghg_r2q = slopes[:,0]
    aer_r2q = slopes[:,1]
    lens_r2q = slopes[:,2]
    
    median_ghg = np.median(ghg_r2q)
    median_aer = np.median(aer_r2q)
    median_lens = np.median(lens_r2q)
    mediansall = np.array([median_ghg,median_aer,median_lens])
    np.savetxt(directorydata + 'Rval_20CRv3-Obs_XGHG-XAER-LENS_%s_RANDOMSEED_Medians_20ens_R1-Trials_L2-%s_epochs-%s.txt' % (SAMPLEQ,l2,epochs),
               mediansall)

    return ghg_r2q,aer_r2q,lens_r2q,mediansall
    
### Read in data
r2_ghgm = np.empty((len(l2),SAMPLEQ))
r2_aerm = np.empty((len(l2),SAMPLEQ))
r2_lensm = np.empty((len(l2),SAMPLEQ))
mediansm = np.empty((len(l2),len(datasets)))
for i in range(len(l2)):
    r2_ghgm[i,:],r2_aerm[i,:],r2_lensm[i,:],mediansm[i,:] = readData(directorydata,SAMPLEQ,l2[i],epochs[i])

###############################################################################
###############################################################################
###############################################################################
### Create plot for histograms of slopes
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 


### LENS medians
lenmed = mediansm[:,-1]

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 2.5))
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

for s in range(len(l2)):
    ax = plt.subplot(2,3,s+1)
    adjust_spines(ax, ['left','bottom'])            
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none') 
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2) 
    ax.tick_params('both',length=3,width=2,which='major',color='dimgrey')  
    # ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)
    
    ### Plot data
    ghg_r2 = r2_ghgm[s,:]
    aer_r2 = r2_aerm[s,:]
    lens_r2 = r2_lensm[s,:]
    
    ### Plot histograms
    weights_ghg = np.ones_like(ghg_r2)/len(ghg_r2)
    n_ghg, bins_ghg, patches_ghg = plt.hist(ghg_r2,bins=np.arange(-1.2,2.1,0.1)-0.05,
                                            density=False,alpha=0.5,
                                            label=r'\textbf{AER+}',
                                            weights=weights_ghg,zorder=3,clip_on=False)
    for i in range(len(patches_ghg)):
        patches_ghg[i].set_facecolor('steelblue')
        patches_ghg[i].set_edgecolor('white')
        patches_ghg[i].set_linewidth(0.17)
     
    weights_aer = np.ones_like(aer_r2)/len(aer_r2)
    n_aer, bins_aer, patches_aer = plt.hist(aer_r2,bins=np.arange(-1.2,2.1,0.1)-0.05,
                                            density=False,alpha=0.5,
                                            label=r'\textbf{GHG+}',
                                            weights=weights_aer,zorder=4,clip_on=False)
    for i in range(len(patches_aer)):
        patches_aer[i].set_facecolor('darkgoldenrod')
        patches_aer[i].set_edgecolor('white')
        patches_aer[i].set_linewidth(0.17)
        
    weights_lens = np.ones_like(lens_r2)/len(lens_r2)
    n_lens, bins_lens, patches_lens = plt.hist(lens_r2,bins=np.arange(-1.2,2.1,0.1)-0.05,
                                            density=False,alpha=0.5,
                                            label=r'\textbf{ALL}',
                                            weights=weights_lens,zorder=5,clip_on=False)
    for i in range(len(patches_lens)):
        patches_lens[i].set_facecolor('crimson')
        patches_lens[i].set_edgecolor('white')
        patches_lens[i].set_linewidth(0.17)     
    
    ### Create legend
    if s == 1:
        leg = plt.legend(shadow=False,fontsize=11,loc='upper center',
                bbox_to_anchor=(0.5,1.305),fancybox=True,ncol=3,frameon=False,
                handlelength=3,handletextpad=1,handleheight=0.08)
        
    ### Create x/y labels
    if any([s==0]):
        plt.text(-2.2,-0.3,r'\textbf{PROPORTION}',fontsize=10,color='k',
                 rotation=90) 
    if s ==4:
        plt.xlabel(r'\textbf{R OF OBSERVATIONS}',fontsize=10,color='k')
    if s == 0:
        plt.text(-1.2,0.63,r'\textbf{[ANN-Paper]}',color='k',fontsize=6,
                 ha='left')
    
    plt.xticks(np.arange(-5,5,0.5),map(str,np.round(np.arange(-5,5,0.5),2)),size=6)
    plt.yticks(np.arange(0,1.1,0.2),map(str,np.round(np.arange(0,1.1,0.2),2)),size=6)
    
    if s == 0:
        ax.axes.xaxis.set_ticklabels([])
    elif any([s==1,s==2]):
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
    elif any([s==4,s==5]):
        ax.axes.yaxis.set_ticklabels([])
    
    ### Add information    
    plt.text(-1.2,0.69,r'\textbf{L$_{\bf{2}}$ = %s}' % l2[s],color='dimgrey',fontsize=8)
    plt.text(-1.2,0.75,r'\textbf{Epochs = %s}' % epochs[s],color='dimgrey',fontsize=8)
    plt.text(1,0.8,r'\textbf{[%s]}' % letters[s],color='dimgrey',fontsize=8,
                          rotation=0,ha='center',va='center')
        
    plt.xlim([-1.2,1.])   
    plt.ylim([0,0.8])
    
plt.savefig(directoryfigure + 'HistogramRvalOfObs-AllANNs_PAPER.png',dpi=600)