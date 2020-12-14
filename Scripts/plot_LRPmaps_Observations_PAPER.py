"""
Plot LRP maps of observations

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 14 December 2020
"""

### Import packages
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm

### Set parameters
variables = [r'T2M']
datasets = [r'[OBS]AER+',r'[OBS]AER+',r'[OBS]GHG+',r'[OBS]GHG+',
            r'[OBS]ALL',r'[OBS]ALL']
seasons = [r'annual']
timeq = ['1920-1959','1960-1999']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
years = np.arange(1920,2015+1,)

### Set directories
directorydata = '/Users/zlabe/Documents/Research/InternalSignal/Data/FINAL/'
directoryfigure = '/Users/zlabe/Desktop/PAPER/'

### Read in LRP maps for X(LENS)
dataghg = Dataset(directorydata + 'LRP_ObservationMaps_XGHG_annual_PAPER.nc')
lat1 = dataghg.variables['lat'][:]
lon1 = dataghg.variables['lon'][:]
lrpghg = dataghg.variables['LRP'][:]*1000
dataghg.close()

dataaer = Dataset(directorydata + 'LRP_ObservationMaps_XAER_annual_PAPER.nc')
lat1 = dataaer.variables['lat'][:]
lon1 = dataaer.variables['lon'][:]
lrpaer = dataaer.variables['LRP'][:]*1000
dataaer.close()

datalens = Dataset(directorydata + 'LRP_ObservationMaps_lens_annual_PAPER.nc')
lat1 = datalens.variables['lat'][:]
lon1 = datalens.variables['lon'][:]
lrplens = datalens.variables['LRP'][:]*1000
datalens.close()

### Read in masking threshold
thresh = np.genfromtxt(directorydata + 'Threshold_MaskingLRP_95.txt',
                       unpack=True)

### Gridded lat x lon
lon2,lat2 = np.meshgrid(lon1,lat1)

### Calculate period means
ghgmean = np.nanmean(lrpghg,axis=0)
aermean = np.nanmean(lrpaer,axis=0)
lensmean = np.nanmean(lrplens,axis=0)

### Period 1
ghg1 = np.nanmean(lrpghg[:40,:,:],axis=0)
aer1 = np.nanmean(lrpaer[:40,:,:],axis=0)
lens1 = np.nanmean(lrplens[:40,:,:],axis=0)

### Period 2 
ghg2 = np.nanmean(lrpghg[40:80,:,:],axis=0)
aer2 = np.nanmean(lrpaer[40:80,:,:],axis=0)
lens2 = np.nanmean(lrplens[40:80,:,:],axis=0)

### Mask data
ghgmean[ghgmean<=thresh] = np.nan
aermean[aermean<=thresh] = np.nan
lensmean[lensmean<=thresh] = np.nan

ghg1[ghg1<=thresh] = np.nan
aer1[aer1<=thresh] = np.nan
lens1[lens1<=thresh] = np.nan

ghg2[ghg2<=thresh] = np.nan
aer2[aer2<=thresh] = np.nan
lens2[lens2<=thresh] = np.nan

### Create data arrays
totalmean = [ghgmean,aermean,lensmean]
periodmean = [ghg1,ghg2,aer1,aer2,lens1,lens2]

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of LRP means
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

limit = np.arange(0,0.5001,0.005)
barlim = np.round(np.arange(0,0.6,0.1),2)
cmap = cm.classic_16.mpl_colormap 
label = r'\textbf{RELEVANCE}'

fig = plt.figure()
for r in range(len(periodmean)):
    var = periodmean[r]
    
    ax1 = plt.subplot(3,2,r+1)
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
    if any([r==0,r==2,r==4]):
        ax1.annotate(r'\textbf{%s}' % datasets[r],xy=(0,0),xytext=(-0.07,0.5),
                      textcoords='axes fraction',color='k',fontsize=10,
                      rotation=90,ha='center',va='center')
    if any([r==0,r==1]):
        ax1.annotate(r'\textbf{%s}' % timeq[r],xy=(0,0),xytext=(0.5,1.08),
                      textcoords='axes fraction',color='dimgrey',fontsize=11,
                      rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                  textcoords='axes fraction',color='k',fontsize=6,
                  rotation=330,ha='center',va='center')
    
###############################################################################
cbar_ax = fig.add_axes([0.32,0.09,0.4,0.03])                
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)

cbar.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  

cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(top=0.85,wspace=-0.3,hspace=0.02,bottom=0.14)

plt.savefig(directoryfigure + 'OBSPeriods_LRP_PAPER.png',dpi=600)

##############################################################################
##############################################################################
##############################################################################
fig = plt.figure(figsize=(10,2.5))
for i in range(len(totalmean)):
    ax1 = plt.subplot(1,3,i+1)
            
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    circle = m.drawmapboundary(fill_color='dimgrey')
    circle.set_clip_on(False) 
    m.drawcoastlines(color='darkgrey',linewidth=0.35)
    
    ### Colorbar limits
    barlim = np.round(np.arange(0,0.6,0.1),2)
    
    ### Take lrp mean over all years
    lrpmean = totalmean[i]
    
    var, lons_cyclic = addcyclic(lrpmean, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
    
    ### Make the plot continuous
    cs = m.contourf(x,y,var,np.arange(0,0.5001,0.005),
                    extend='max')                
    cmap = cm.classic_16.mpl_colormap          
    cs.set_cmap(cmap)
    
    ax1.annotate(r'\textbf{%s}' % (datasets[i]),xy=(0,0),xytext=(0.865,0.91),
                      textcoords='axes fraction',color='k',fontsize=11,
                      rotation=332.5,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[i],xy=(0,0),xytext=(0.085,0.93),
                          textcoords='axes fraction',color='dimgrey',fontsize=8,
                          rotation=0,ha='center',va='center')
    
cbar_ax = fig.add_axes([0.293,0.13,0.4,0.03])             
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)

cbar.set_label(r'\textbf{RELEVANCE}',fontsize=11,color='dimgrey',labelpad=1.4)  
cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=6)
cbar.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(bottom=0.17)
plt.savefig(directoryfigure + 'OBS_LRP_PAPER.png',dpi=600)