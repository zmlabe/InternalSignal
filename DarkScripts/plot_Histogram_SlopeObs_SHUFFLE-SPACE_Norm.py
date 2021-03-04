"""
Plots normalized histograms of slope of observations using shuffled data of
lat and lon ANN

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 5 November 2020
"""

### Import packages
import numpy as np
import matplotlib.pyplot as plt

### Set parameters
variables = [r'T2M']
seasons = [r'annual']
SAMPLEQ = 500

### Set directories
directorydata = '/Users/zlabe/Documents/Research/InternalSignal/Data/'
directoryfigure = '/Users/zlabe/Desktop/SINGLE_v2.0/Histograms/%s/' % variables[0]

### Read in slope data
filename_slope = 'Slopes_20CRv3-SHUFFLE-SPACE_%s_RANDOMSEED_20ens.txt' % SAMPLEQ
slopes = np.genfromtxt(directorydata + filename_slope,unpack=True)

### Read in R2 data
filename_R2= 'R2_20CRv3-SHUFFLE-SPACE_%s_RANDOMSEED_20ens.txt' % SAMPLEQ
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
plt.axvline(x=1,color='k',linewidth=2,linestyle='--',dashes=(1,0.3),
            zorder=10)
    
weights_lens = np.ones_like(slopes)/len(slopes)
n, bins, patches = plt.hist(slopes,bins=np.arange(-1,2.21,0.1)-0.05,
                                        density=False,alpha=1,
                                        label=r'\textbf{SHUFFLE-SPACE DATA ANN}',
                                        weights=weights_lens,zorder=5)
for i in range(len(patches)):
    patches[i].set_facecolor('olivedrab')
    patches[i].set_edgecolor('white')
    patches[i].set_linewidth(1)
    
leg = plt.legend(shadow=False,fontsize=7,loc='upper center',
        bbox_to_anchor=(0.2,1),fancybox=True,ncol=1,frameon=False,
        handlelength=3,handletextpad=1)

plt.ylabel(r'\textbf{PROPORTION[%s]}' % SAMPLEQ,fontsize=10,color='k')
plt.xlabel(r'\textbf{SLOPES} [ANNUAL -- T2M -- 20CRv3 -- (1920-2015)]',fontsize=10,color='k')
plt.yticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=6)
plt.xticks(np.arange(-1.2,10.1,0.2),map(str,np.round(np.arange(-1.2,10.1,0.2),2)),size=6)
plt.xlim([-1.0,2.2])   
plt.ylim([0,0.3])
    
plt.savefig(directoryfigure + 'Histogram_Slopes_SHUFFLE-SPACE_T2M_%s_Norm.png' % SAMPLEQ,
            dpi=300)