"""
Plots LRP maps of uncertainty 

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
lat1 = data.variables['lat'][:]
lon1 = data.variables['lon'][:]
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

ghg_percn = np.empty((SAMPLEQ,lat1.shape[0]*lon1.shape[0]))
for i in range(SAMPLEQ):  
    x = np.ravel(lrpghg[i,:,:])
    ghg_percn[i,:] = (sts.rankdata(x)-1)/len(x)
ghg_perc = np.reshape(ghg_percn,(SAMPLEQ,lat1.shape[0],lon1.shape[0]))
ghg_percmean = np.nanmean(ghg_perc,axis=0)*100. # percent

aer_percn = np.empty((SAMPLEQ,lat1.shape[0]*lon1.shape[0]))
for i in range(SAMPLEQ):  
    x = np.ravel(lrpaer[i,:,:])
    aer_percn[i,:] = (sts.rankdata(x)-1)/len(x)
aer_perc = np.reshape(aer_percn,(SAMPLEQ,lat1.shape[0],lon1.shape[0]))
aer_percmean = np.nanmean(aer_perc,axis=0)*100. # percent

lens_percn = np.empty((SAMPLEQ,lat1.shape[0]*lon1.shape[0]))
for i in range(SAMPLEQ):  
    x = np.ravel(lrplens[i,:,:])
    lens_percn[i,:] = (sts.rankdata(x)-1)/len(x)
lens_perc = np.reshape(lens_percn,(SAMPLEQ,lat1.shape[0],lon1.shape[0]))
lens_percmean = np.nanmean(lens_perc,axis=0)*100. # percent

perc = [ghg_percmean,aer_percmean,lens_percmean]

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of LRP means
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

fig = plt.figure(figsize=(10,2.5))
for i in range(len(datasets)):
    ax1 = plt.subplot(1,3,i+1)
            
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    circle = m.drawmapboundary(fill_color='k')
    circle.set_clip_on(False) 
    m.drawcoastlines(color='darkgrey',linewidth=0.35)
    
    ### Colorbar limits
    barlim = np.round(np.arange(0,0.6,0.1),2)
    
    ### Take lrp mean over all years
    lrpmean = mean[i]
    
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
                     textcoords='axes fraction',color='k',fontsize=14,
                     rotation=335,ha='center',va='center')
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
plt.savefig(directoryfigure + 'LRPmean_%s_%s_%s.png' % (variables[0],
                                                        seasons[0],
                                                        SAMPLEQ),
                                                        dpi=300)

# #######################################################################
# #######################################################################
# #######################################################################
# ### Plot subplot of LRP means with statistical masking
# plt.rc('text',usetex=True)
# plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

# fig = plt.figure(figsize=(10,2.5))
# for i in range(len(datasets)):
#     ax1 = plt.subplot(1,3,i+1)
            
#     m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
#     circle = m.drawmapboundary(fill_color='k')
#     circle.set_clip_on(False) 
#     m.drawcoastlines(color='darkgrey',linewidth=0.35)
    
#     ### Colorbar limits
#     barlim = np.round(np.arange(0,0.6,0.1),2)
    
#     ### Take lrp mean over all years
#     lrpmean = mean[i]
    
#     var, lons_cyclic = addcyclic(lrpmean, lon1)
#     var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
#     lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
#     x, y = m(lon2d, lat2d)
    
#     ### Make the plot continuous
#     cs = m.contourf(x,y,var,np.arange(0,0.5001,0.005),
#                     extend='max')                
#     cmap = cm.classic_16.mpl_colormap          
#     cs.set_cmap(cmap)
    
    # ax1.annotate(r'\textbf{%s}' % (datasets[i]),xy=(0,0),xytext=(0.865,0.91),
    #                   textcoords='axes fraction',color='k',fontsize=14,
    #                   rotation=335,ha='center',va='center')
    # ax1.annotate(r'\textbf{[%s]}' % letters[i],xy=(0,0),xytext=(0.085,0.93),
    #                       textcoords='axes fraction',color='dimgrey',fontsize=8,
    #                       rotation=0,ha='center',va='center')
    
# cbar_ax = fig.add_axes([0.293,0.13,0.4,0.03])             
# cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
#                     extend='max',extendfrac=0.07,drawedges=False)

# cbar.set_label(r'\textbf{RELEVANCE}',fontsize=11,color='dimgrey',labelpad=1.4)  
# cbar.set_ticks(barlim)
# cbar.set_ticklabels(list(map(str,barlim)))
# cbar.ax.tick_params(axis='x', size=.01,labelsize=6)
# cbar.outline.set_edgecolor('dimgrey')

# plt.tight_layout()
# plt.subplots_adjust(bottom=0.17)
# plt.savefig(directoryfigure + 'LRPmean_%s_%s_%s_Mask.png' % (variables[0],
#                                                         seasons[0],
#                                                         SAMPLEQ),
#                                                         dpi=300)

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of LRP STD
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

fig = plt.figure(figsize=(10,2.5))
for i in range(len(datasets)):
    ax1 = plt.subplot(1,3,i+1)
            
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    circle = m.drawmapboundary(fill_color='k')
    circle.set_clip_on(False) 
    m.drawcoastlines(color='darkgrey',linewidth=0.35)
    
    ### Colorbar limits
    barlim = np.round(np.arange(0,0.1001,0.025),3)
    
    ### Take lrp mean over all years
    lrpstd = std[i]
    
    var, lons_cyclic = addcyclic(lrpstd, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
    
    ### Make the plot continuous
    cs = m.contourf(x,y,var,np.arange(0,0.1001,0.001),
                    extend='max')                
    cmap = cm.cubehelix2_16.mpl_colormap          
    cs.set_cmap(cmap)
    
    ax1.annotate(r'\textbf{%s}' % (datasets[i]),xy=(0,0),xytext=(0.865,0.91),
                     textcoords='axes fraction',color='k',fontsize=14,
                     rotation=335,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[i],xy=(0,0),xytext=(0.085,0.93),
                         textcoords='axes fraction',color='dimgrey',fontsize=8,
                         rotation=0,ha='center',va='center')
    
cbar_ax = fig.add_axes([0.293,0.13,0.4,0.03])             
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)

cbar.set_label(r'\textbf{STD. DEV. [RELEVANCE]}',fontsize=11,
               color='dimgrey',labelpad=1.4)  
cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=6)
cbar.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(bottom=0.17)
plt.savefig(directoryfigure + 'LRPstd_%s_%s_%s.png' % (variables[0],
                                                        seasons[0],
                                                        SAMPLEQ),
                                                        dpi=300)

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of percentile means
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

fig = plt.figure(figsize=(10,2.5))
for i in range(len(datasets)):
    ax1 = plt.subplot(1,3,i+1)
            
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    circle = m.drawmapboundary(fill_color='k')
    circle.set_clip_on(False) 
    m.drawcoastlines(color='darkgrey',linewidth=0.35)
    
    ### Colorbar limits
    barlim = np.round(np.arange(0,101,20),2)
    
    ### Take percentile mean over all samples
    percmean = perc[i]
    
    var, lons_cyclic = addcyclic(percmean, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
    
    ### Make the plot continuous
    cs = m.contourf(x,y,var,np.arange(0,100.01,1),
                    extend='max')                
    cmap = cmocean.cm.dense_r
    cs.set_cmap(cmap)
    
    ax1.annotate(r'\textbf{%s}' % (datasets[i]),xy=(0,0),xytext=(0.865,0.91),
                     textcoords='axes fraction',color='k',fontsize=14,
                     rotation=335,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[i],xy=(0,0),xytext=(0.085,0.93),
                         textcoords='axes fraction',color='dimgrey',fontsize=8,
                         rotation=0,ha='center',va='center')
    
cbar_ax = fig.add_axes([0.293,0.13,0.4,0.03])             
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)

cbar.set_label(r'\textbf{PERCENTILE [\%]}',fontsize=11,color='dimgrey',labelpad=1.4)  
cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=6)
cbar.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(bottom=0.17)
plt.savefig(directoryfigure + 'Percmean_%s_%s_%s.png' % (variables[0],
                                                        seasons[0],
                                                        SAMPLEQ),
                                                        dpi=300)