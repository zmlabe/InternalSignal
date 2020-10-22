"""
Plots normalized histograms of R^2 of observations using randomized data ANN

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 21 October 2020
"""

### Import packages
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import scipy.stats as stats

### Set parameters
variables = [r'T2M']
seasons = [r'annual']
SAMPLEQ = 100

### Set directories
directorydata = '/Users/zlabe/Documents/Research/InternalSignal/Data/'
directoryfigure = '/Users/zlabe/Desktop/SINGLE_v2.0/Histograms/%s/' % variables[0]

### Read in slope data
filename_slope = 'Slopes_20CRv3-RANDOM_%s_RANDOMSEED_20ens.txt' % SAMPLEQ
slopes = np.genfromtxt(directorydata + filename_slope,unpack=True)

### Read in R2 data
filename_R2= 'R2_20CRv3-RANDOM_%s_RANDOMSEED_20ens.txt' % SAMPLEQ
r2 = np.genfromtxt(directorydata + filename_R2,unpack=True)

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
weights = np.ones_like(r2)/len(r2)
n, bins, patches = plt.hist(r2,bins=np.arange(0,1.01,0.025),
                                        density=False,alpha=1,
                                        label=r'\textbf{XGHG}',
                                        weights=weights,zorder=3)
for i in range(len(patches)):
    patches[i].set_facecolor('darkgoldenrod')
    patches[i].set_edgecolor('white')
    patches[i].set_linewidth(1)

plt.ylabel(r'\textbf{PROPORTION[%s]}' % SAMPLEQ,fontsize=10,color='k')
plt.xlabel(r'\textbf{R$^{2}$} [ANNUAL -- T2M -- 20CRv3 -- (1920-2015)]',fontsize=10,color='k')
plt.yticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=6)
plt.xticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=6)
plt.xlim([0,1])   
plt.ylim([0,0.3])
    
plt.savefig(directoryfigure + 'Histogram_r2_RANDOM_T2M_%s.png' % SAMPLEQ,
            dpi=300)