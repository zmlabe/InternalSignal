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
SAMPLEQ = 100
years = np.arange(1920,2080+1,1)
timeq = ['1920-1959','1960-1999','2000-2039','2040-2079']

### Set directories
directorydata = '/Users/zlabe/Documents/Research/InternalSignal/Data/FINAL/'
directoryfigure =  '/Users/zlabe/Documents/Projects/InternalSignal/DarkFigures/'

### Read in LRP maps
data = Dataset(directorydata + 'LRP_YearlyMaps_%s_20ens_%s_%s.nc' % (SAMPLEQ,variables[0],seasons[0]))
lats1 = data.variables['lat'][:]
lons1 = data.variables['lon'][:]
lrp = data.variables['LRP'][:]
data.close()

### Select model runs
lrp1 = lrp[0,:,:,:,:]
lrp2 = lrp[1,:,:,:,:]
lrp3 = lrp[2,:,:,:,:]

### Select years to average over
yearq = np.where((years >= 1960) & (years <= 2039))[0]
lrpghg = np.nanmean(lrp1[:,yearq,:,:],axis=1)
lrpaer = np.nanmean(lrp2[:,yearq,:,:],axis=1)
lrplens = np.nanmean(lrp3[:,yearq,:,:],axis=1)

###############################################################################
###############################################################################
###############################################################################
### Calculate statistics over the 100 random samples
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
    lataq = np.where((lats1 >= 10) & (lats1 <= 40))[0]
    lata1 = lats1[lataq]
    lonaq = np.where((lons1 >= 105) & (lons1 <= 120))[0]
    lona1 = lons1[lonaq]
    time_lrpA1 = time_lrp[:,lataq,:]
    time_lrpA = time_lrpA1[:,:,lonaq]
    lona2,lata2 = np.meshgrid(lona1,lata1)
    mean_lrpA = UT.calc_weightedAve(time_lrpA,lata2)
    
    
    ### India
    latiq = np.where((lats1 >= 15) & (lats1 <= 40))[0]
    lati1 = lats1[latiq]
    loniq = np.where((lons1 >= 70) & (lons1 <= 105))[0]
    loni1 = lons1[loniq]
    time_lrpI1 = time_lrp[:,latiq,:]
    time_lrpI = time_lrpI1[:,:,loniq]
    loni2,lati2 = np.meshgrid(loni1,lati1)
    mean_lrpI = UT.calc_weightedAve(time_lrpI,lati2)
    
    ### North Atlantic Warming Hole
    latwq = np.where((lats1 >= 50) & (lats1 <= 60))[0]
    latw1 = lats1[latwq]
    lonwq = np.where((lons1 >= 315) & (lons1 <= 340))[0]
    lonw1 = lons1[lonwq]
    time_lrpW1 = time_lrp[:,latwq,:]
    time_lrpW = time_lrpW1[:,:,lonwq]
    lonw2,latw2 = np.meshgrid(lonw1,latw1)
    mean_lrpW = UT.calc_weightedAve(time_lrpW,latw2)
    
    ### Sahara
    latdq = np.where((lats1 >= 0) & (lats1 <= 15))[0]
    latd1 = lats1[latdq]
    londq1 = np.where((lons1 >= 0) & (lons1 <= 45))[0]
    londq2 = np.where((lons1 >= 350) & (lons1 <= 360))[0]
    londq = np.append(londq1 ,londq2)
    lond1 = lons1[londq]
    time_lrpD1 = time_lrp[:,latdq,:]
    time_lrpD = time_lrpD1[:,:,londq]
    lond2,latd2 = np.meshgrid(lond1,latd1)
    mean_lrpD = UT.calc_weightedAve(time_lrpD,latd2)
    
    ### Southern Ocean Section
    latsoq = np.where((lats1 >= -66) & (lats1 <= -40))[0]
    latso1 = lats1[latsoq]
    lonsoq = np.where((lons1 >= 5) & (lons1 <= 70))[0]
    # lonsoq2 = np.where((lons1 >= 330) & (lons1 <= 360))[0]
    # lonsoq = np.append(lonsoq1 ,lonsoq2)
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
regionnames = ['Southeast Asia','India','North Atlantic',
                'Central Africa','Southern Ocean']

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
        
fig = plt.figure(figsize=(9,3))
for ii in range(len(regions_ghg)):
    
    ax = plt.subplot(1,5,ii+1)
    
    if ii == 0:
        adjust_spines(ax, ['left','bottom'])            
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none') 
        ax.spines['bottom'].set_color('darkgrey')
        ax.spines['left'].set_color('darkgrey')
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2) 
        ax.tick_params('both',length=5.5,width=2,which='major',color='darkgrey')  
        ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.35)
    else:
        adjust_spines(ax, ['left','bottom'])            
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none') 
        ax.spines['bottom'].set_color('darkgrey')
        ax.spines['left'].set_color('darkgrey')
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2) 
        ax.tick_params('both',length=5.5,width=2,which='major',color='darkgrey')  
        ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.35)
        ax.set_yticklabels([]) 
    
    ### Plot histograms    
    weights_ghg = np.ones_like(regions_ghg[ii])/len(regions_ghg[ii])
    n_ghg, bins_ghg, patches_ghg = plt.hist(regions_ghg[ii],bins=np.arange(0,1.05,0.02)-0.01,
                                            density=False,alpha=0.7,
                                            label=r'\textbf{AER+}',
                                            weights=weights_ghg,zorder=3)
    for i in range(len(patches_ghg)):
        patches_ghg[i].set_facecolor('deepskyblue')
        patches_ghg[i].set_edgecolor('k')
        patches_ghg[i].set_linewidth(0.5)
     
    weights_aer = np.ones_like(regions_aer[ii])/len(regions_aer[ii])
    n_aer, bins_aer, patches_aer = plt.hist(regions_aer[ii],bins=np.arange(0,1.05,0.02)-0.01,
                                            density=False,alpha=0.7,
                                            label=r'\textbf{GHG+}',
                                            weights=weights_aer,zorder=4)
    for i in range(len(patches_aer)):
        patches_aer[i].set_facecolor('gold')
        patches_aer[i].set_edgecolor('k')
        patches_aer[i].set_linewidth(0.5)
        
    weights_lens = np.ones_like(regions_lens[ii])/len(regions_lens[ii])
    n_lens, bins_lens, patches_lens = plt.hist(regions_lens[ii],bins=np.arange(0,1.05,0.02)-0.01,
                                            density=False,alpha=0.7,
                                            label=r'\textbf{ALL}',
                                            weights=weights_lens,zorder=5)
    for i in range(len(patches_lens)):
        patches_lens[i].set_facecolor('crimson')
        patches_lens[i].set_edgecolor('k')
        patches_lens[i].set_linewidth(0.5)
    
    if ii == 2:    
        leg = plt.legend(shadow=False,fontsize=15,loc='upper center',
                bbox_to_anchor=(0.40,1.35),fancybox=True,ncol=3,frameon=False,
                handlelength=2,handletextpad=1)
        ccc = ['deepskyblue','gold','crimson']
        # for text in leg.get_texts():
        for counter, text in enumerate(leg.get_texts()):
            text.set_color(ccc[counter])
    
    if ii == 0:
        plt.ylabel(r'\textbf{PROPORTION}',fontsize=10,color='w')
    plt.xlabel(r'\textbf{%s}' % regionnames[ii],fontsize=12,color='w')
    plt.yticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=6)
    plt.xticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=6)
    plt.xlim([0,0.6])   
    plt.ylim([0,0.9])
    
    # ax.annotate(r'\textbf{[%s]}' % letters[ii],xy=(0,0),xytext=(0.95,0.945),
    #                  textcoords='axes fraction',color='k',fontsize=8,
    #                  rotation=0,ha='center',va='center')
    
plt.tight_layout()
plt.subplots_adjust(wspace=0.3,top=0.8)
plt.savefig(directoryfigure + 'HistogramsOfLRPRegions_PAPER_DARK.png',dpi=300)