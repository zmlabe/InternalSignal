"""
Plot trends of observations for selected periods

Reference  : Deser et al. [2020, JCLI]
Author    : Zachary M. Labe
Date      : 19 October 2020
"""

### Import packages
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import cmocean as cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import itertools

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directorydataLLS = '/Users/zlabe/Data/LENS/SINGLE/'
directorydataLLL = '/Users/zlabe/Data/LENS/monthly'
directoryfigure =  '/Users/zlabe/Desktop/PAPER/'
datasetsingleq = ['20CRv3']
datasetsingle = ['20CRv3']
timeq = ['1920-1959','1960-1999','2000-2015']
seasons = ['annual','JFM','AMJ','JAS','OND']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
years = np.arange(1920,2015+1,1)
directoriesall = [directorydataLLS,directorydataLLS,directorydataLLL]

variq = 'T2M'
monthlychoice = seasons[0]
reg_name = 'Globe'

def read_primary_dataset(variq,dataset,lat_bounds,lon_bounds,monthlychoice):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons

### Read in data
lat_bounds,lon_bounds = UT.regions(reg_name)
obs,lat1,lon1 = read_primary_dataset(variq,datasetsingle[0],lat_bounds,lon_bounds,
                            monthlychoice)

### Select periods
obsnew = obs[-years.shape[0]:,:,:]
obs1 = obsnew[0:40,:,:]
obs2 = obsnew[40:80,:,:]
obs3 = obsnew[80:,:,:]

### Calculate linear trends
def calcTrend(data):
    slopes = np.empty((data.shape[1],data.shape[2]))
    x = np.arange(data.shape[0])
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            mask = np.isfinite(data[:,i,j])
            y = data[:,i,j]
            
            if np.sum(mask) == y.shape[0]:
                xx = x
                yy = y
            else:
                xx = x[mask]
                yy = y[mask]      
            if np.isfinite(np.nanmean(yy)):
                slopes[i,j],intercepts, \
                r_value,p_value,std_err = sts.linregress(xx,yy)
            else:
                slopes[i,j] = np.nan
    
    dectrend = slopes * 10.   
    print('Completed: Finished calculating trends!')      
    return dectrend

### Procress trends
trends1 = calcTrend(obs1)
trends2 = calcTrend(obs2)
trends3 = calcTrend(obs3)
    
pvals1 = np.empty((lat1.shape[0],lon1.shape[0]))
pvals2 = np.empty((lat1.shape[0],lon1.shape[0]))
pvals3 = np.empty((lat1.shape[0],lon1.shape[0]))
for i in range(lat1.shape[0]):
    for j in range(lon1.shape[0]):
        trend,h,pvals1[i,j],z = UT.mk_test(obs1[:,i,j],0.05)
        trend,h,pvals2[i,j],z = UT.mk_test(obs2[:,i,j],0.05)
        trend,h,pvals3[i,j],z = UT.mk_test(obs3[:,i,j],0.05)
            
pvals1[np.where(pvals1 == 1.)] = 0.
pvals1[np.where(np.isnan(pvals1))] = 1.
pvals1[np.where(pvals1 == 0.)] = np.nan

pvals2[np.where(pvals2 == 1.)] = 0.
pvals2[np.where(np.isnan(pvals2))] = 1.
pvals2[np.where(pvals2 == 0.)] = np.nan

pvals3[np.where(pvals3 == 1.)] = 0.
pvals3[np.where(np.isnan(pvals3))] = 1.
pvals3[np.where(pvals3 == 0.)] = np.nan
    
runs = [trends1,trends2]
pvals = [pvals1,pvals2]
    
###########################################################################
###########################################################################
###########################################################################
### Plot variable data for trends
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

### Set limits for contours and colorbars
limit = np.arange(-0.75,0.751,0.025)
barlim = np.arange(-0.75,0.76,0.25)
cmap = cmocean.cm.balance
label = r'\textbf{$^{\circ}$C decade$^{-1}$}'
    
fig = plt.figure(figsize=(9,3))
for r in range(len(runs)):
    var = runs[r]
    pvar = pvals[r]
    
    ax1 = plt.subplot(1,2,r+1)
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    circle = m.drawmapboundary(fill_color='k')
    circle.set_clip_on(False) 
    m.drawcoastlines(color='k',linewidth=0.35)
    
    var, lons_cyclic = addcyclic(var, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    pvar,lons_cyclic = addcyclic(pvar, lon1)
    pvar,lons_cyclic = shiftgrid(180.,pvar,lons_cyclic,start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='white',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    cs = m.contourf(x,y,var,limit,extend='both')
    cs1 = m.contourf(x,y,pvar,colors='None',hatches=['///////////'])
            
    cs.set_cmap(cmap) 
    if any([r==0]):
        ax1.annotate(r'\textbf{%s}' % datasetsingleq[r],xy=(0,0),xytext=(-0.07,0.5),
                      textcoords='axes fraction',color='k',fontsize=15,
                      rotation=90,ha='center',va='center')
    if any([r==0,r==1,r==2]):
        ax1.annotate(r'\textbf{%s}' % timeq[r],xy=(0,0),xytext=(0.5,1.08),
                      textcoords='axes fraction',color='dimgrey',fontsize=11,
                      rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                  textcoords='axes fraction',color='k',fontsize=6,
                  rotation=330,ha='center',va='center')

###########################################################################
cbar_ax = fig.add_axes([0.32,0.09,0.4,0.03])                
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)

cbar.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  

cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(top=0.85,wspace=0.01,hspace=0,bottom=0.14)

plt.savefig(directoryfigure + 'OBSTrendPeriods_T2M_PAPER.png',dpi=600)

