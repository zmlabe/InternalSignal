"""
Plots LRP time series for selected regions v1.1

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 30 September 2020
"""

### Import packages
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import scipy.stats as stats

### Set parameters
regions = [r'SEA',r'India',r'SouthernOcean',r'NorthAtlanticWarmingHole']
variables = [r'T2M']
variplot = [r'PercLRP']
datasets = [r'XGHG',r'XAER',r'XBMB',r'XLULC',r'lens']
seasons = [r'annual']

### Set directories
directorydata = '/Users/zlabe/Documents/Research/InternalSignal/Data/'
directoryfigure = '/Users/zlabe/Desktop/SINGLE_v1.2/LRP_Charts/%s/' % variables[0]

def read_LRP(seasonq,variableq,variplot,regionq,directorydata):
    """
    Read time series of mean LRP over regions
    """
    
    ### Print parameters
    print('======== Reading LRP Time Series ========')
    print('------- %s -------' % seasonq) 
    print('------- %s -------' % variableq) 
    print('------- %s -------' % variplot) 
    print('------- %s -------' % regionq) 
    print('=========================================')
    
    ### Read in all data for XGHG
    yearghg,lrpghg = np.genfromtxt(directorydata + '%s_TimeSeries_%s_%s_%s_Globe_XGHG_landFalse_oceanFalse.txt' % (variplot,regionq,variableq,seasonq),
                                   unpack=True,usecols=[0,1])
    
    ### Read in all data for XAER
    yearaer,lrpaer = np.genfromtxt(directorydata + '%s_TimeSeries_%s_%s_%s_Globe_XAER_landFalse_oceanFalse.txt' % (variplot,regionq,variableq,seasonq),
                                   unpack=True,usecols=[0,1])
    
    ### Read in all data for XBMB
    yearbmb,lrpbmb = np.genfromtxt(directorydata + '%s_TimeSeries_%s_%s_%s_Globe_XBMB_landFalse_oceanFalse.txt' % (variplot,regionq,variableq,seasonq),
                                   unpack=True,usecols=[0,1])
    
    ### Read in all data for XLULC
    yearlulc,lrplulc = np.genfromtxt(directorydata + '%s_TimeSeries_%s_%s_%s_Globe_XLULC_landFalse_oceanFalse.txt' % (variplot,regionq,variableq,seasonq),
                                   unpack=True,usecols=[0,1])
    
    ### Read in all data for LENS
    yearlens,lrplens = np.genfromtxt(directorydata + '%s_TimeSeries_%s_%s_%s_Globe_lens_landFalse_oceanFalse.txt' % (variplot,regionq,variableq,seasonq),
                                   unpack=True,usecols=[0,1])
    
    ### Combine into lists
    years = [yearghg,yearaer,yearbmb,yearlulc,yearlens]
    data = [lrpghg,lrpaer,lrpbmb,lrplulc,lrplens]
    
    return years,data

###############################################################################
###############################################################################
###############################################################################
### Read functions to get data for SEA
years_sea,data_sea = read_LRP(seasons[0],variables[0],variplot[0],regions[0],directorydata)

### Read functions to get data for India
years_india,data_india = read_LRP(seasons[0],variables[0],variplot[0],regions[1],directorydata)

### Read functions to get data for the Southern Ocean
years_southern,data_southern = read_LRP(seasons[0],variables[0],variplot[0],regions[2],directorydata)

### Read functions to get data for North Atlantic Warming Hole
years_nawh,data_nawh = read_LRP(seasons[0],variables[0],variplot[0],regions[3],directorydata)

###############################################################################
###############################################################################
###############################################################################
### New parameters
variableforplot = variables[0]
regionsforplot = regions
lrpforplot = variplot[0]
seasonforplot = seasons[0]

###############################################################################
###############################################################################
###############################################################################
### Create subplots for LRP
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
        
fig = plt.figure(figsize=(8,7))
ax = plt.subplot(221)
        
adjust_spines(ax, ['left','bottom'])            
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none') 
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2) 
ax.tick_params('both',length=5.5,width=2,which='major',color='dimgrey')  

color=cmocean.cm.thermal_r(np.linspace(0.05,1,len(datasets)))
for i,c in zip(range(len(datasets)),color):
    if i == 4:
        plt.plot(years_sea[i],data_sea[i]*100.,c=c,linewidth=2.5,zorder=10,
                 label=r'\textbf{%s}' % datasets[i],linestyle='--',
                 dashes=(1,0.6))           
    else:
        plt.plot(years_sea[i],data_sea[i]*100.,c=c,linewidth=2.5,zorder=1,
                 label=r'\textbf{%s}' % datasets[i])   

plt.ylabel(r'\textbf{Percentile [\%]}',fontsize=10,color='dimgrey')
plt.yticks(np.arange(0,110,10),map(str,np.round(np.arange(0,110,10),2)),size=6)
plt.xticks(np.arange(1920,2101,20),map(str,np.arange(1920,2101,20)),size=6)
plt.xlim([1920,2100])   
plt.ylim([10,100])

leg = plt.legend(shadow=False,fontsize=5,loc='upper center',
            bbox_to_anchor=(0.5, 0.05),fancybox=True,ncol=5,frameon=False,
            handlelength=0,handletextpad=0)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
    
plt.text(2100,98,r'\textbf{SOUTHEAST ASIA}',color='dimgrey',fontsize=11,
         ha='right')

###############################################################################
###############################################################################
###############################################################################
ax = plt.subplot(222)
        
adjust_spines(ax, ['left','bottom'])            
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none') 
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2) 
ax.tick_params('both',length=5.5,width=2,which='major',color='dimgrey')  

color=cmocean.cm.thermal_r(np.linspace(0.05,1,len(datasets)))
for i,c in zip(range(len(datasets)),color):
    if i == 4:
        plt.plot(years_india[i],data_india[i]*100.,c=c,linewidth=2.5,zorder=10,
                 label=r'\textbf{%s}' % datasets[i],linestyle='--',
                 dashes=(1,0.6))           
    else:
        plt.plot(years_india[i],data_india[i]*100.,c=c,linewidth=2.5,zorder=1,
                 label=r'\textbf{%s}' % datasets[i]) 

plt.yticks(np.arange(0,110,10),map(str,np.round(np.arange(0,110,10),2)),size=6)
plt.xticks(np.arange(1920,2101,20),map(str,np.arange(1920,2101,20)),size=6)
plt.xlim([1920,2100])   
plt.ylim([10,100])

leg = plt.legend(shadow=False,fontsize=5,loc='upper center',
            bbox_to_anchor=(0.5, 0.05),fancybox=True,ncol=5,frameon=False,
            handlelength=0,handletextpad=0)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
    
plt.text(2100,98,r'\textbf{INDIA}',color='dimgrey',fontsize=11,
         ha='right')
        
###############################################################################
ax = plt.subplot(223)
        
adjust_spines(ax, ['left','bottom'])            
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none') 
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2) 
ax.tick_params('both',length=5.5,width=2,which='major',color='dimgrey')  

color=cmocean.cm.thermal_r(np.linspace(0.05,1,len(datasets)))
for i,c in zip(range(len(datasets)),color):
    if i == 4:
        plt.plot(years_southern[i],data_southern[i]*100.,c=c,linewidth=2.5,zorder=10,
                 label=r'\textbf{%s}' % datasets[i],linestyle='--',
                 dashes=(1,0.6))           
    else:
        plt.plot(years_southern[i],data_southern[i]*100.,c=c,linewidth=2.5,zorder=1,
                 label=r'\textbf{%s}' % datasets[i]) 

plt.ylabel(r'\textbf{Percentile [\%]}',fontsize=10,color='dimgrey')
plt.yticks(np.arange(0,110,10),map(str,np.round(np.arange(0,110,10),2)),size=6)
plt.xticks(np.arange(1920,2101,20),map(str,np.arange(1920,2101,20)),size=6)
plt.xlim([1920,2100])   
plt.ylim([10,100])

leg = plt.legend(shadow=False,fontsize=5,loc='upper center',
            bbox_to_anchor=(0.5, 0.05),fancybox=True,ncol=5,frameon=False,
            handlelength=0,handletextpad=0)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
    
plt.text(2100,98,r'\textbf{SOUTHERN OCEAN}',color='dimgrey',fontsize=11,
         ha='right')

###############################################################################
ax = plt.subplot(224)
        
adjust_spines(ax, ['left','bottom'])            
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none') 
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2) 
ax.tick_params('both',length=5.5,width=2,which='major',color='dimgrey')  

color=cmocean.cm.thermal_r(np.linspace(0.05,1,len(datasets)))
for i,c in zip(range(len(datasets)),color):
    if i == 4:
        plt.plot(years_nawh[i],data_nawh[i]*100.,c=c,linewidth=2.5,zorder=10,
                 label=r'\textbf{%s}' % datasets[i],linestyle='--',
                 dashes=(1,0.6))           
    else:
        plt.plot(years_nawh[i],data_nawh[i]*100.,c=c,linewidth=2.5,zorder=1,
                 label=r'\textbf{%s}' % datasets[i])  

plt.yticks(np.arange(0,110,10),map(str,np.round(np.arange(0,110,10),2)),size=6)
plt.xticks(np.arange(1920,2101,20),map(str,np.arange(1920,2101,20)),size=6)
plt.xlim([1920,2100])   
plt.ylim([10,100])

leg = plt.legend(shadow=False,fontsize=5,loc='upper center',
            bbox_to_anchor=(0.5, 0.05),fancybox=True,ncol=5,frameon=False,
            handlelength=0,handletextpad=0)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
    
plt.text(2100,98,r'\textbf{NORTH ATLANTIC}',color='dimgrey',fontsize=11,
     ha='right')

plt.tight_layout()
plt.savefig(directoryfigure + 'PercLRP_timeseries4_v1-1_%s_%s.png' % (seasonforplot,variableforplot),dpi=300)