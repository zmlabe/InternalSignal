"""
Plots histograms of LRP for specific regions

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 7 October 2020
"""

### Import packages
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import cmocean
import palettable.cubehelix as cm
import scipy.stats as sts
import calc_Utilities as UT

### Set parameters
variables = [r'T2M']
datasets = [r'XGHG',r'XAER',r'LENS']
seasons = [r'annual']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
SAMPLEQ = 500

### Set directories
directorydata = '/Users/zlabe/Documents/Research/InternalSignal/Data/'
directoryfigure = '/Users/zlabe/Desktop/SINGLE_v1.2-HISTOGRAM/LRP/%s/' % variables[0]

### Read in LRP maps
data = Dataset(directorydata + 'LRP_Maps_%s_%s_%s.nc' % (variables[0],seasons[0],SAMPLEQ))
lats1 = data.variables['lat'][:]
lons1 = data.variables['lon'][:]
lrp = data.variables['LRP'][:]
data.close()

lrpghg = lrp[0,:,:,:]
lrpaer = lrp[1,:,:,:]
lrplens = lrp[2,:,:,:]

###############################################################################
###############################################################################
###############################################################################
### Calculate statistics over the 500 random samples
### Standard Deviation 
std_ghg = np.nanstd(lrpghg[:,:,:],axis=0)
std_aer = np.nanstd(lrpaer[:,:,:],axis=0)
std_lens = np.nanstd(lrplens[:,:,:],axis=0)
std = [std_ghg,std_aer,std_lens]

### Mean
mean_ghg = np.nanmean(lrpghg[:,:,:],axis=0)
mean_aer = np.nanmean(lrpaer[:,:,:],axis=0)
mean_lens = np.nanmean(lrplens[:,:,:],axis=0)
mean = [mean_ghg, mean_aer, mean_lens]

### Percentile
percentile_ghg = np.percentile(lrpghg[:,:,:],90,axis=0)
percentile_aer = np.percentile(lrpaer[:,:,:],90,axis=0)
percentile_lens = np.percentile(lrplens[:,:,:],90,axis=0)
percentile = [percentile_ghg, percentile_aer, percentile_lens]

ghg_percn = np.empty((SAMPLEQ,lats1.shape[0]*lons1.shape[0]))
for i in range(SAMPLEQ):  
    x = np.ravel(lrpghg[i,:,:])
    ghg_percn[i,:] = (sts.rankdata(x)-1)/len(x)
ghg_perc = np.reshape(ghg_percn,(SAMPLEQ,lats1.shape[0],lons1.shape[0]))
ghg_percmean = np.nanmean(ghg_perc,axis=0)*100. # percent

aer_percn = np.empty((SAMPLEQ,lats1.shape[0]*lons1.shape[0]))
for i in range(SAMPLEQ):  
    x = np.ravel(lrpaer[i,:,:])
    aer_percn[i,:] = (sts.rankdata(x)-1)/len(x)
aer_perc = np.reshape(aer_percn,(SAMPLEQ,lats1.shape[0],lons1.shape[0]))
aer_percmean = np.nanmean(aer_perc,axis=0)*100. # percent

lens_percn = np.empty((SAMPLEQ,lats1.shape[0]*lons1.shape[0]))
for i in range(SAMPLEQ):  
    x = np.ravel(lrplens[i,:,:])
    lens_percn[i,:] = (sts.rankdata(x)-1)/len(x)
lens_perc = np.reshape(lens_percn,(SAMPLEQ,lats1.shape[0],lons1.shape[0]))
lens_percmean = np.nanmean(lens_perc,axis=0)*100. # percent

perc = [ghg_percmean,aer_percmean,lens_percmean]

###############################################################################
###############################################################################
###############################################################################
### Calculate regions

def calculateRegions(time_lrp,lats1,lons1):
    ### Southeast Asia
    lataq = np.where((lats1 >= 10) & (lats1 <= 35))[0]
    lata1 = lats1[lataq]
    lonaq = np.where((lons1 >= 90) & (lons1 <= 115))[0]
    lona1 = lons1[lonaq]
    time_lrpA1 = time_lrp[:,lataq,:]
    time_lrpA = time_lrpA1[:,:,lonaq]
    lona2,lata2 = np.meshgrid(lona1,lata1)
    mean_lrpA = UT.calc_weightedAve(time_lrpA,lata2)
    
    
    ### India
    latiq = np.where((lats1 >= 10) & (lats1 <= 35))[0]
    lati1 = lats1[latiq]
    loniq = np.where((lons1 >= 60) & (lons1 <= 90))[0]
    loni1 = lons1[loniq]
    time_lrpI1 = time_lrp[:,latiq,:]
    time_lrpI = time_lrpI1[:,:,loniq]
    loni2,lati2 = np.meshgrid(loni1,lati1)
    mean_lrpI = UT.calc_weightedAve(time_lrpI,lati2)
    
    ### North Atlantic Warming Hole
    latwq = np.where((lats1 >= 40) & (lats1 <= 60))[0]
    latw1 = lats1[latwq]
    lonwq = np.where((lons1 >= 290) & (lons1 <= 345))[0]
    lonw1 = lons1[lonwq]
    time_lrpW1 = time_lrp[:,latwq,:]
    time_lrpW = time_lrpW1[:,:,lonwq]
    lonw2,latw2 = np.meshgrid(lonw1,latw1)
    mean_lrpW = UT.calc_weightedAve(time_lrpW,latw2)
    
    ### Sahara
    latdq = np.where((lats1 >= -5) & (lats1 <= 15))[0]
    latd1 = lats1[latdq]
    londq = np.where((lons1 >= 0) & (lons1 <= 50))[0]
    lond1 = lons1[londq]
    time_lrpD1 = time_lrp[:,latdq,:]
    time_lrpD = time_lrpD1[:,:,londq]
    lond2,latd2 = np.meshgrid(lond1,latd1)
    mean_lrpD = UT.calc_weightedAve(time_lrpD,latd2)
    
    ### Southern Ocean
    latsoq = np.where((lats1 >= -65) & (lats1 <= -45))[0]
    latso1 = lats1[latsoq]
    lonsoq = np.where((lons1 >= 0) & (lons1 <= 360))[0]
    lonso1 = lons1[lonsoq]
    time_lrpSO1 = time_lrp[:,latsoq,:]
    time_lrpSO = time_lrpSO1[:,:,lonsoq]
    lonso2,latso2 = np.meshgrid(lonso1,latso1) 
    mean_lrpSO = UT.calc_weightedAve(time_lrpSO,latso2)
    
    ### Add together
    regions = [mean_lrpA,mean_lrpI,mean_lrpW,mean_lrpD,mean_lrpSO]
    
    return regions

regions_ghg = calculateRegions(lrpghg,lats1,lons1)
regions_aer = calculateRegions(lrpaer,lats1,lons1)
regions_lens = calculateRegions(lrplens,lats1,lons1)
regionnames = ['Southeast Asia','India','North Atlantic Warming Hole',
               'Sahara','Southern Ocean']

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
        
fig = plt.figure(figsize=(9,3))
for ii in range(len(regions_ghg)):
    
    ax = plt.subplot(1,5,ii+1)
    
    if ii == 0:
        adjust_spines(ax, ['left','bottom'])            
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none') 
        ax.spines['bottom'].set_color('dimgrey')
        ax.spines['left'].set_color('dimgrey')
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2) 
        ax.tick_params('both',length=5.5,width=2,which='major',color='dimgrey')  
        ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)
    else:
        adjust_spines(ax, ['left','bottom'])            
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none') 
        ax.spines['bottom'].set_color('dimgrey')
        ax.spines['left'].set_color('dimgrey')
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2) 
        ax.tick_params('both',length=5.5,width=2,which='major',color='dimgrey')  
        ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)
        ax.set_yticklabels([]) 
    
    ### Plot histograms    
    weights_ghg = np.ones_like(regions_ghg[ii])/len(regions_ghg[ii])
    n_ghg, bins_ghg, patches_ghg = plt.hist(regions_ghg[ii],bins=np.arange(0,1.05,0.02)-0.01,
                                            density=False,alpha=0.5,
                                            label=r'\textbf{XGHG}',
                                            weights=weights_ghg,zorder=3)
    for i in range(len(patches_ghg)):
        patches_ghg[i].set_facecolor('steelblue')
        patches_ghg[i].set_edgecolor('white')
        patches_ghg[i].set_linewidth(0.5)
     
    weights_aer = np.ones_like(regions_aer[ii])/len(regions_aer[ii])
    n_aer, bins_aer, patches_aer = plt.hist(regions_aer[ii],bins=np.arange(0,1.05,0.02)-0.01,
                                            density=False,alpha=0.5,
                                            label=r'\textbf{XAER}',
                                            weights=weights_aer,zorder=4)
    for i in range(len(patches_aer)):
        patches_aer[i].set_facecolor('goldenrod')
        patches_aer[i].set_edgecolor('white')
        patches_aer[i].set_linewidth(0.5)
        
    weights_lens = np.ones_like(regions_lens[ii])/len(regions_lens[ii])
    n_lens, bins_lens, patches_lens = plt.hist(regions_lens[ii],bins=np.arange(0,1.05,0.02)-0.01,
                                            density=False,alpha=0.5,
                                            label=r'\textbf{LENS}',
                                            weights=weights_lens,zorder=5)
    for i in range(len(patches_lens)):
        patches_lens[i].set_facecolor('crimson')
        patches_lens[i].set_edgecolor('white')
        patches_lens[i].set_linewidth(0.5)
    
    if ii == 2:    
        leg = plt.legend(shadow=False,fontsize=8,loc='upper center',
                bbox_to_anchor=(0.40,1.18),fancybox=True,ncol=3,frameon=False,
                handlelength=4,handletextpad=1)
    
    if ii == 0:
        plt.ylabel(r'\textbf{PROPORTION[%s]}' % SAMPLEQ,fontsize=10,color='k')
    plt.xlabel(r'\textbf{%s} [LRP-Relevance]' % regionnames[ii],fontsize=5,color='k')
    plt.yticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=5)
    plt.xticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=5)
    plt.xlim([0,0.4])   
    plt.ylim([0,1])
    
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.savefig(directoryfigure + 'LRP_Regions_Histograms_XGHG-XAER-LENS_T2M_%s_Norm.png' % SAMPLEQ,
            dpi=300)