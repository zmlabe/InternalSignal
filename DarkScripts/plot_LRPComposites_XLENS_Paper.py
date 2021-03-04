"""
Plot composite periods for LRP between experiments

Reference  : Deser et al. [2020, JCLI]
Author    : Zachary M. Labe
Date      : 11 November 2020
"""

### Import packages
import math
import time
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import scipy.stats as sts
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import calc_Utilities as UT
import calc_dataFunctions as df
import itertools

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directorydata = '/Users/zlabe/Documents/Research/InternalSignal/Data/FINAL/'
directoryfigure =  '/Users/zlabe/Documents/Projects/InternalSignal/DarkFigures/'
datasetsingleq = np.repeat(['AER+','GHG+','ALL'],4)
datasetsingle = ['XGHG','XAER','lens']
timeq = ['1920-1959','1960-1999','2000-2039','2040-2079']
seasons = ['annual','JFM','AMJ','JAS','OND']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
years = np.arange(1920,2079+1,1)
variables = ['T2M']
monthlychoice = seasons[0]
reg_name = 'Globe'
SAMPLEQ = 100

### Read in LRP maps (XGHG)
data = Dataset(directorydata + 'LRP_Maps_XGHG_AllSeasons_PAPER.nc')
lat1 = data.variables['lat'][:]
lon1 = data.variables['lon'][:]
lrp_ghg = data.variables['LRP'][:]
data.close()
lrpghg = lrp_ghg[0]

### Read in LRP maps (XAER)
data = Dataset(directorydata + 'LRP_Maps_XAER_AllSeasons_PAPER.nc')
lat1 = data.variables['lat'][:]
lon1 = data.variables['lon'][:]
lrp_aer = data.variables['LRP'][:]
data.close()
lrpaer = lrp_aer[0]

### Read in LRP maps (LENS)
data = Dataset(directorydata + 'LRP_Maps_lens_AllSeasons_PAPER.nc')
lat1 = data.variables['lat'][:]
lon1 = data.variables['lon'][:]
lrp_lens = data.variables['LRP'][:]
data.close()
lrplens = lrp_lens[0]

### Procress time
comp_ghg = np.empty((len(years)//40,lrpghg.shape[1],lrpghg.shape[2]))
comp_aer = np.empty((len(years)//40,lrpaer.shape[1],lrpaer.shape[2]))
comp_lens = np.empty((len(years)//40,lrplens.shape[1],lrplens.shape[2]))
for count,i in enumerate(range(0,len(years),40)):
    comp_ghg[count,:,:,] = np.nanmean(lrpghg[i:i+40,:,:],axis=0)
    comp_aer[count,:,:,] = np.nanmean(lrpaer[i:i+40,:,:],axis=0)
    comp_lens[count,:,:,] = np.nanmean(lrplens[i:i+40,:,:],axis=0)

thresh = np.genfromtxt(directorydata + 'Threshold_MaskingLRP_95.txt',
                       unpack=True)
comp_ghgm = comp_ghg.copy()
comp_aerm = comp_aer.copy()
comp_lensm = comp_lens.copy()
comp_ghg[comp_ghg<=thresh] = np.nan
comp_aer[comp_aer<=thresh] = np.nan
comp_lens[comp_lens<=thresh] = np.nan

runs = list(itertools.chain.from_iterable([comp_ghg,comp_aer,comp_lens]))
runsm = list(itertools.chain.from_iterable([comp_ghgm,comp_aerm,comp_lensm]))
    
###########################################################################
###########################################################################
###########################################################################
### Plot variable data for trends
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
plt.rc('savefig',facecolor='black')
plt.rc('axes',edgecolor='darkgrey')
plt.rc('xtick',color='darkgrey')
plt.rc('ytick',color='darkgrey')
plt.rc('axes',labelcolor='darkgrey')
plt.rc('axes',facecolor='black')

### Set limits for contours and colorbars
limit = np.arange(0,0.5001,0.005)
barlim = np.round(np.arange(0,0.6,0.1),2)
cmap = cm.classic_16.mpl_colormap 
label = r'\textbf{RELEVANCE}'
    
fig = plt.figure(figsize=(5,3))
for r in range(len(runs)):
    var = runs[r]
    
    ax1 = plt.subplot(3,4,r+1)
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    circle = m.drawmapboundary(fill_color='dimgrey')
    circle.set_clip_on(False) 
    m.drawcoastlines(color='darkgrey',linewidth=0.27)
    
    var, lons_cyclic = addcyclic(var, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    cs = m.contourf(x,y,var,limit,extend='max')
            
    cs.set_cmap(cmap) 
    if any([r==0,r==4,r==8]):
        ax1.annotate(r'\textbf{%s}' % datasetsingleq[r],xy=(0,0),xytext=(-0.1,0.5),
                      textcoords='axes fraction',color='w',fontsize=9,
                      rotation=90,ha='center',va='center')
    if any([r==0,r==1,r==2,r==3]):
        ax1.annotate(r'\textbf{%s}' % timeq[r],xy=(0,0),xytext=(0.5,1.22),
                      textcoords='axes fraction',color='darkgrey',fontsize=9,
                      rotation=0,ha='center',va='center')
    # ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.87,0.97),
    #               textcoords='axes fraction',color='k',fontsize=6,
    #               rotation=330,ha='center',va='center')

###########################################################################
cbar_ax = fig.add_axes([0.32,0.095,0.4,0.03])                
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)

cbar.set_label(label,fontsize=9,color='w',labelpad=1.4)  

cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar.outline.set_edgecolor('darkgrey')

plt.tight_layout()
plt.subplots_adjust(top=0.85,wspace=0.01,hspace=0,bottom=0.14)

plt.savefig(directoryfigure + 'LRPPeriods_T2M_PAPER_DARK.png',dpi=600)

###########################################################################
###########################################################################
###########################################################################

fig = plt.figure(figsize=(5,3))
for r in range(len(runs)):
    var = runsm[r]
    
    ax1 = plt.subplot(3,4,r+1)
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    circle = m.drawmapboundary(fill_color='dimgrey')
    circle.set_clip_on(False) 
    m.drawcoastlines(color='darkgrey',linewidth=0.27)
    
    var, lons_cyclic = addcyclic(var, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    cs = m.contourf(x,y,var,limit,extend='max')
            
    cs.set_cmap(cmap) 
    if any([r==0,r==4,r==8]):
        ax1.annotate(r'\textbf{%s}' % datasetsingleq[r],xy=(0,0),xytext=(-0.1,0.5),
                      textcoords='axes fraction',color='w',fontsize=9,
                      rotation=90,ha='center',va='center')
    if any([r==0,r==1,r==2,r==3]):
        ax1.annotate(r'\textbf{%s}' % timeq[r],xy=(0,0),xytext=(0.5,1.22),
                      textcoords='axes fraction',color='darkgrey',fontsize=9,
                      rotation=0,ha='center',va='center')
    # ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.87,0.97),
    #               textcoords='axes fraction',color='k',fontsize=6,
    #               rotation=330,ha='center',va='center')

###########################################################################
cbar_ax = fig.add_axes([0.32,0.095,0.4,0.03])                
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)

cbar.set_label(label,fontsize=9,color='w',labelpad=1.4)  

cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar.outline.set_edgecolor('darkgrey')

plt.tight_layout()
plt.subplots_adjust(top=0.85,wspace=0.01,hspace=0,bottom=0.14)

plt.savefig(directoryfigure + 'LRPPeriods_T2M_NOMASK_DARK.png',dpi=600)

