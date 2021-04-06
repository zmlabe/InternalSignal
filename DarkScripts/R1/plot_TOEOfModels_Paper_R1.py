"""
Plot signal-to-noise ratios for XLENS simulations for paper

[1] Method: 10-yr running mean exceeds 1920-1949 baseline by 2sigma

Reference  : Deser et al. [2020, JCLI] and Lehner et al. [2017, JCLI]
Author    : Zachary M. Labe
Date      : 4 March 2021
"""

### Import packages
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import palettable.scientific.sequential as scm
import cmocean as cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import pandas as pd

##############################################################################
##############################################################################
##############################################################################
## Data preliminaries 
directorydataLLS = '/Users/zlabe/Data/LENS/SINGLE/'
directorydataLLL = '/Users/zlabe/Data/LENS/monthly'
directoryfigure =  '/Users/zlabe/Documents/Projects/InternalSignal/DarkFigures/'
directoriesall = [directorydataLLS,directorydataLLS,directorydataLLL]
##############################################################################
##############################################################################
##############################################################################
datasetsingleq = ['AER+','GHG+','ALL']
datasetsingle = ['XGHG','XAER','lens']
##############################################################################
##############################################################################
##############################################################################
timeq = ['1920-1959','1960-1999','2000-2039','2040-2079']
seasons = ['annual','JFM','AMJ','JAS','OND']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
years = np.arange(1920,2079+1,1)
##############################################################################
##############################################################################
##############################################################################
variq = 'T2M'
monthlychoice = seasons[0]
reg_name = 'Globe'
##############################################################################
##############################################################################
##############################################################################
def read_primary_dataset(variq,dataset,lat_bounds,lon_bounds,monthlychoice):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons

### Read in data
lat_bounds,lon_bounds = UT.regions(reg_name)
ghg,lat1,lon1 = read_primary_dataset(variq,datasetsingle[0],lat_bounds,lon_bounds,
                            monthlychoice)
aer,lat1,lon1 = read_primary_dataset(variq,datasetsingle[1],lat_bounds,lon_bounds,
                            monthlychoice)
lens,lat1,lon1 = read_primary_dataset(variq,datasetsingle[2],lat_bounds,lon_bounds,
                            monthlychoice)

### Calculate ensemble mean
meanghg = np.nanmean(ghg,axis=0)[:-1,:,:] #to 2079
meanaer = np.nanmean(aer,axis=0)[:-1,:,:] #to 2079
meanlens = np.nanmean(lens,axis=0)[:-1,:,:] #to 2079

### Functions for calculating moving averages
def moving_average(data,window):
    """ 
    Calculating rolling mean over set window
    """
    ### Import functions
    import numpy as np
    
    movemean = np.convolve(data,np.ones(window),'valid') / window
    return movemean

def rollingMean(data,w,mp):
    """ 
    Calculating rolling mean over set window
    """
    ### Import functions
    import numpy as np
    import pandas as pd
    
    datadf = pd.Series(data)
    movemean = datadf.rolling(window=w,min_periods=mp).mean().to_numpy()
    return movemean

### 10-year running mean
window = 10 
min_periods = 10
smooth_ghg = np.empty((ghg.shape[0],ghg.shape[1]-1,ghg.shape[2],ghg.shape[3]))
smooth_aer = np.empty((aer.shape[0],aer.shape[1]-1,aer.shape[2],aer.shape[3]))
smooth_lens = np.empty((lens.shape[0],lens.shape[1]-1,lens.shape[2],lens.shape[3]))
for ens in range(ghg.shape[0]):
    for i in range(ghg.shape[2]):
        for j in range(ghg.shape[3]):
            smooth_ghg[ens,:,i,j] = rollingMean(ghg[ens,:-1,i,j],window,min_periods)
            smooth_aer[ens,:,i,j] = rollingMean(aer[ens,:-1,i,j],window,min_periods)
            smooth_lens[ens,:,i,j] = rollingMean(lens[ens,:-1,i,j],window,min_periods)
    print('Completed: Ensemble #%s running mean!' % (ens+1))
    
### Slice baseline of 1920-1949
minyr = 1920
maxyr = 1949
yearq = np.where((years >= minyr) & ((years <= maxyr)))[0]
ghgbase = smooth_ghg[:,yearq,:,:]
aerbase = smooth_aer[:,yearq,:,:]
lensbase = smooth_lens[:,yearq,:,:]

### 2 Sigma of 1920-1949
ghg2 = np.nanstd(ghgbase[:,:,:,:],axis=1) * 2.
aer2 = np.nanstd(aerbase[:,:,:,:],axis=1) * 2.
lens2 = np.nanstd(lensbase[:,:,:,:],axis=1) * 2.

### Limit of baseline
ghgbasemean = np.nanmean(ghgbase[:,:,:,:],axis=1)
aerbasemean = np.nanmean(aerbase[:,:,:,:],axis=1)
lensbasemean = np.nanmean(lensbase[:,:,:,:],axis=1)

ghglimit = ghgbasemean + ghg2
aerlimit = aerbasemean + aer2
lenslimit = lensbasemean + lens2

### Calculate ToE
def calcToE(database,datalimit,years):
    """ 
    Calculate ToE from Lehner et al. 2017
    """
    
    toe= np.empty((database.shape[0],database.shape[2],database.shape[3]))
    toe[:,:,:] = np.nan
    for ens in range(database.shape[0]):
        for i in range(database.shape[2]):
            for j in range(database.shape[3]):
                limit = datalimit[ens,i,j]
                for yr in range(database.shape[1]):
                    smooth = database[ens,yr,i,j]
                    if smooth > limit:
                        if np.nanmax(database[ens,yr:,i,j]) > limit:
                            if np.isnan(toe[ens,i,j]):
                                toe[ens,i,j] = years[yr]
                        
        print('Completed: Ensemble #%s ToE!' % (ens+1))
    
    return toe

toe_ghg = calcToE(smooth_ghg,ghglimit,years)
toe_aer = calcToE(smooth_aer,aerlimit,years)
toe_lens = calcToE(smooth_lens,lenslimit,years)

### Calculate ensemble mean ToE
mtoe_ghg = np.nanmean(toe_ghg,axis=0)
mtoe_aer = np.nanmean(toe_aer,axis=0)
mtoe_lens = np.nanmean(toe_lens,axis=0)
alltoe = [mtoe_ghg,mtoe_aer,mtoe_lens]

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of ToE
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
plt.rc('savefig',facecolor='black')
plt.rc('axes',edgecolor='darkgrey')
plt.rc('xtick',color='darkgrey')
plt.rc('ytick',color='darkgrey')
plt.rc('axes',labelcolor='darkgrey')
plt.rc('axes',facecolor='black')

fig = plt.figure(figsize=(10,2.5))
for i in range(len(alltoe)):
    ax1 = plt.subplot(1,3,i+1)
            
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    circle = m.drawmapboundary(fill_color='dimgrey')
    circle.set_clip_on(False) 
    m.drawcoastlines(color='w',linewidth=0.35)
    
    ### Colorbar limits
    barlim = np.round(np.arange(1920,2040+1,10),2)
    barlim2 = np.round(np.arange(1925,2060+1,10),2)
    barlim3 = [r'1920s',r'1930s',r'1940s',r'1950s',r'1960s',r'1970s',r'1980s',r'1990s',
               r'2000s',r'2010s',r'2020s',r'2030s',r'2040s']
    
    ### Take toe mean over all years
    toemodel = alltoe[i]
    
    var, lons_cyclic = addcyclic(toemodel , lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
    
    ### Make the plot continuous
    cs = m.contourf(x,y,var,np.arange(1920,2040+1,10),
                    extend='max')                
    # cmap = cm.cubehelix2_16_r.mpl_colormap    
    cmap = scm.Batlow_17.mpl_colormap    
    cs.set_cmap(cmap)
    
    ax1.annotate(r'\textbf{%s}' % (datasetsingleq[i]),xy=(0,0),xytext=(0.865,0.93),
                      textcoords='axes fraction',color='w',fontsize=19,
                      rotation=334,ha='center',va='center')
    # ax1.annotate(r'\textbf{[%s]}' % letters[i],xy=(0,0),xytext=(0.085,0.93),
    #                       textcoords='axes fraction',color='dimgrey',fontsize=8,
    #                       rotation=0,ha='center',va='center')
    
cbar_ax = fig.add_axes([0.293,0.145,0.4,0.03])             
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=True)

cbar.set_label(r'\textbf{TIMING OF EMERGENCE [Years]}',fontsize=11,color='w',labelpad=5)  
cbar.set_ticks(barlim2)
cbar.set_ticklabels(barlim3)
cbar.ax.tick_params(axis='x', size=.01,labelsize=5,labelcolor='w')
cbar.outline.set_edgecolor('w')
cbar.outline.set_linewidth(1)
cbar.dividers.set_color('w')
cbar.dividers.set_linewidth(1)

plt.tight_layout()
plt.subplots_adjust(bottom=0.17)
plt.savefig(directoryfigure + 'TOEPeriods_T2M_DARK.png',dpi=600)