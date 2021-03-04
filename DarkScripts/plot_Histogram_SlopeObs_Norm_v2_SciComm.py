"""
Plots normalized histograms of slope of observations version 2

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 28 October 2020
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
directoryfigure = '/Users/zlabe/Desktop/SINGLE_v2.0/SciComm/'

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
plt.axvline(x=1,color='w',linewidth=2,linestyle='--',dashes=(1,0.3),
            zorder=10)

weights_ghg = np.ones_like(ghg_slopes)/len(ghg_slopes)
n_ghg, bins_ghg, patches_ghg = plt.hist(ghg_slopes,bins=np.arange(-1,2.1,0.1)-0.05,
                                        density=False,alpha=0.6,
                                        label=r'\textbf{AER+ALL}',
                                        weights=weights_ghg,zorder=3)
for i in range(len(patches_ghg)):
    patches_ghg[i].set_facecolor('steelblue')
    patches_ghg[i].set_edgecolor('k')
    patches_ghg[i].set_linewidth(0.5)
 
weights_aer = np.ones_like(aer_slopes)/len(aer_slopes)
n_aer, bins_aer, patches_aer = plt.hist(aer_slopes,bins=np.arange(-1,2.1,0.1)-0.05,
                                        density=False,alpha=0.6,
                                        label=r'\textbf{GHG+ALL}',
                                        weights=weights_aer,zorder=4)
for i in range(len(patches_aer)):
    patches_aer[i].set_facecolor('goldenrod')
    patches_aer[i].set_edgecolor('k')
    patches_aer[i].set_linewidth(0.5)
    
weights_lens = np.ones_like(lens_slopes)/len(lens_slopes)
n_lens, bins_lens, patches_lens = plt.hist(lens_slopes,bins=np.arange(-1,2.1,0.1)-0.05,
                                        density=False,alpha=0.6,
                                        label=r'\textbf{TOTAL}',
                                        weights=weights_lens,zorder=5)
for i in range(len(patches_lens)):
    patches_lens[i].set_facecolor('forestgreen')
    patches_lens[i].set_edgecolor('k')
    patches_lens[i].set_linewidth(0.5)
    
leg = plt.legend(shadow=False,fontsize=10,loc='upper center',
        bbox_to_anchor=(0.13,1),fancybox=True,ncol=1,frameon=False,
        handlelength=3,handletextpad=1)
for text in leg.get_texts():
    text.set_color('w')

plt.ylabel(r'\textbf{PROPORTION[%s]}' % SAMPLEQ,fontsize=10,color='darkgrey')
plt.xlabel(r'\textbf{SLOPES} [ANNUAL -- T2M -- 20CRv3 -- (1920-2015)]',fontsize=10,color='darkgrey')
plt.yticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=6)
plt.xticks(np.arange(-1,10.1,0.2),map(str,np.round(np.arange(-1,10.1,0.2),2)),size=6)
plt.xlim([-1,2])   
plt.ylim([0,0.6])

###############################################################################
###############################################################################
###############################################################################
### Add subplots

### Add subplot
ax1 = plt.axes([.3,.38,.10,.15])
         
ax1.spines['top'].set_color('none')
ax1.spines['right'].set_color('none') 
ax1.spines['bottom'].set_color('darkgrey')
ax1.spines['left'].set_color('darkgrey')
ax1.spines['bottom'].set_linewidth(1)
ax1.spines['left'].set_linewidth(1) 
ax1.tick_params('both',length=3,width=1,which='major',color='darkgrey')  

line = np.arange(1920,2080+1,1)
lineobs = np.arange(1920,2015+1,1)
time = np.arange(lineobs.shape[0])
line_ghg = np.median(ghg_slopes)*time + ((1920+2080)/2)
plt.plot(line,line,color='w',linewidth=1,clip_on=True,
         linestyle='--',dashes=(1,0.3))
plt.plot(lineobs,line_ghg,color='steelblue',linewidth=2,clip_on=True,zorder=4)

plt.xlabel(r'\textbf{ACTUAL YEAR}',fontsize=4,color='w',labelpad=-6)
plt.ylabel(r'\textbf{PREDICTED YEAR}',fontsize=4,color='w',labelpad=-11.5)
plt.xticks(np.arange(1920,2101,160),map(str,np.arange(1920,2101,160)),size=4)
plt.yticks(np.arange(1920,2101,160),map(str,np.arange(1920,2101,160)),size=4)
plt.xlim([1920,2080])   
plt.ylim([1920,2080])
ax1.tick_params(axis='x',which='major',pad=1)
ax1.tick_params(axis='y',which='major',pad=0.4)

###############################################################################
### Add subplot
ax2 = plt.axes([.48,.7,.10,.15])
         
ax2.spines['top'].set_color('none')
ax2.spines['right'].set_color('none') 
ax2.spines['bottom'].set_color('darkgrey')
ax2.spines['left'].set_color('darkgrey')
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['left'].set_linewidth(1) 
ax2.tick_params('both',length=3,width=1,which='major',color='k')  

line = np.arange(1920,2080+1,1)
time = np.arange(lineobs.shape[0])
line_aer = np.median(aer_slopes)*time + 1920
plt.plot(line,line,color='w',linewidth=1,clip_on=True,
         linestyle='--',dashes=(1,0.3))
plt.plot(lineobs,line_aer,color='goldenrod',linewidth=2,clip_on=True,zorder=4)

plt.xlabel(r'\textbf{ACTUAL YEAR}',fontsize=4,color='w',labelpad=-6)
plt.ylabel(r'\textbf{PREDICTED YEAR}',fontsize=4,color='w',labelpad=-11.5)
plt.xticks(np.arange(1920,2101,160),map(str,np.arange(1920,2101,160)),size=4)
plt.yticks(np.arange(1920,2101,160),map(str,np.arange(1920,2101,160)),size=4)
plt.xlim([1920,2080])   
plt.ylim([1920,2080])
ax1.tick_params(axis='x',which='major',pad=1)
ax1.tick_params(axis='y',which='major',pad=0.4)

###############################################################################
### Add subplot
ax2 = plt.axes([.74,.57,.10,.15])
         
ax2.spines['top'].set_color('none')
ax2.spines['right'].set_color('none') 
ax2.spines['bottom'].set_color('darkgrey')
ax2.spines['left'].set_color('darkgrey')
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['left'].set_linewidth(1) 
ax2.tick_params('both',length=3,width=1,which='major',color='k')  

line = np.arange(1920,2080+1,1)
time = np.arange(lineobs.shape[0])
line_lens = np.median(lens_slopes)*time + 1920
plt.plot(line,line,color='w',linewidth=1,clip_on=True,
         linestyle='--',dashes=(1,0.3))
plt.plot(lineobs,line_lens,color='forestgreen',linewidth=2,clip_on=True,zorder=4)

plt.xlabel(r'\textbf{ACTUAL YEAR}',fontsize=4,color='w',labelpad=-6)
plt.ylabel(r'\textbf{PREDICTED YEAR}',fontsize=4,color='w',labelpad=-11.5)
plt.xticks(np.arange(1920,2101,160),map(str,np.arange(1920,2101,160)),size=4)
plt.yticks(np.arange(1920,2101,160),map(str,np.arange(1920,2101,160)),size=4)
plt.xlim([1920,2080])   
plt.ylim([1920,2080])
ax1.tick_params(axis='x',which='major',pad=1)
ax1.tick_params(axis='y',which='major',pad=0.4)
    
plt.savefig(directoryfigure + 'Histogram_Slopes_XGHG-XAER-LENS_T2M_%s_Norm_SciComm.png' % SAMPLEQ,
            dpi=300)