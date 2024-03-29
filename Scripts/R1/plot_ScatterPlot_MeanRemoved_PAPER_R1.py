"""
Plot figures for paper showing scatter plots of predicting the year

Reference  : Barnes et al. [2019, GRL; 2020, JAMES]
Author    : Zachary M. Labe
Date      : 18 November 2020
"""

### Import packages
import numpy as np
import matplotlib.pyplot as plt
import palettable.cubehelix as cm
import cmocean as cmocean
import calc_Utilities as UT
import scipy.stats as stats

### Set parameters
variables = [r'T2M']
datasets = [r'XGHG',r'XAER',r'lens']
seasons = [r'annual',r'JFM',r'AMJ',r'JAS',r'OND']
letters = [r'a',r'b',r'c',r'd']
years = np.arange(1920,2080+1,1)
yearobs = np.arange(1920,2015+1,1)
ens = 20
training_ens = int(ens*0.8)
testing_ens = int(ens*0.2)

### Set directories
directorydata = '/Users/zlabe/Documents/Research/InternalSignal/Data/FINAL/'
directoryfigure = '/Users/zlabe/Desktop/PAPER/R1/'

###############################################################################
###############################################################################
###############################################################################
### Read in data
def readSeason(variableq,seasonq,datasets):
    """ 
    Read in observations and testing data
    """

    training = np.empty((len(datasets),len(years),training_ens))
    testing = np.empty((len(datasets),len(years),testing_ens))
    obs = np.empty((len(datasets),len(yearobs)))
    obyears = np.empty((len(datasets),len(yearobs)))
    for i,dataset in enumerate(datasets):
        ### Read in testing model data
        filename_testing = 'testing_%s_%s_rmmean.txt' % (dataset,seasonq)
        testing[i,:,:] = np.genfromtxt(directorydata + filename_testing,
                                       unpack=True)
        
        ### Read in testing observational data
        filename_obs = 'obs_%s_%s_rmmean.txt' % (dataset,seasonq)
        obs[i,:] = np.genfromtxt(directorydata + filename_obs,
                                 unpack=True)
        
        ### Read in actual years for observations
        filename_obyears = 'years_%s_%s_rmmean.txt' % (dataset,seasonq)
        obyears[i,:] = np.genfromtxt(directorydata + filename_obyears,
                                     unpack=True)
        
    modelpred = testing
    
    print('Completed: Read in all data for %s!' % seasonq)
    return modelpred,obs,obyears
###############################################################################
def calcRegressionObs(yearobs,obs):
    """
    Calculate regression line for observations
    """
    lines = np.empty((len(datasets),len(yearobs)))
    r_2 = np.empty((len(datasets)))
    for i in range(len(datasets)):
        slope,intercept,r_value,p_value,std_err = stats.linregress(yearobs,obs[i,:])
        lines[i,:] = slope*yearobs + intercept
        r_2[i] = r_value**2
    return lines,r_2
###############################################################################
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
###############################################################################
### Return all functions
modelpred_annual,obs_annual,obyears_annual = readSeason(variables[0],seasons[0],datasets)
modelpred_jfm,obs_jfm,obyears_jfm = readSeason(variables[0],seasons[1],datasets)
modelpred_amj,obs_amj,obyears_amj = readSeason(variables[0],seasons[2],datasets)
modelpred_jas,obs_jas,obyears_jas = readSeason(variables[0],seasons[3],datasets)
modelpred_ond,obs_ond,obyears_ond = readSeason(variables[0],seasons[4],datasets)

lines_annual,r2_annual = calcRegressionObs(yearobs,obs_annual)
lines_jfm,r2_jfm = calcRegressionObs(yearobs,obs_jfm)
lines_jas,r2_jas = calcRegressionObs(yearobs,obs_jas)

### Calculate statistics
std2_annual = np.nanstd(modelpred_annual,axis=2)*2.
mean_annual = np.nanmean(modelpred_annual,axis=2)
annual_minperc = np.percentile(modelpred_annual,100-95,axis=2)
annual_maxperc = np.percentile(modelpred_annual,95,axis=2)
annual_neg2std = mean_annual - std2_annual
annual_pos2std = mean_annual + std2_annual

std2_jfm = np.nanstd(modelpred_jfm,axis=2)*2.
mean_jfm = np.nanmean(modelpred_jfm,axis=2)
jfm_minperc = np.percentile(modelpred_jfm,100-95,axis=2)
jfm_maxperc = np.percentile(modelpred_jfm,95,axis=2)
jfm_neg2std = mean_jfm - std2_jfm
jfm_pos2std = mean_jfm + std2_jfm

std2_jas = np.nanstd(modelpred_jas,axis=2)*2.
mean_jas = np.nanmean(modelpred_jas,axis=2)
jas_minperc = np.percentile(modelpred_jas,100-95,axis=2)
jas_maxperc = np.percentile(modelpred_jas,95,axis=2)
jas_neg2std = mean_jas - std2_jas
jas_pos2std = mean_jas + std2_jas

###############################################################################
###############################################################################
###############################################################################
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
        
fig = plt.figure(figsize=(9,3))
ax = plt.subplot(131)

adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey',pad=1.1)

ax.fill_between(years,annual_minperc[0],annual_maxperc[0],color='deepskyblue',
                alpha=0.3,linewidth=0,clip_on=False)
plt.scatter(yearobs,obs_annual[0],s=12,color='crimson',clip_on=False,
            alpha=0.35,linewidths=0.5)
plt.plot(years,years,'-',color='black',linewidth=1,clip_on=False)
plt.plot(yearobs,lines_annual[0],linewidth=1.5,linestyle='--',dashes=(1,0.3),
          color='darkred')

plt.xlabel(r'\textbf{ACTUAL YEAR}',fontsize=10,color='dimgrey')
plt.ylabel(r'\textbf{PREDICTED YEAR}',fontsize=10,color='dimgrey')
plt.xticks(np.arange(years.min(),2101,20),map(str,np.arange(years.min(),2101,20)),size=6)
plt.yticks(np.arange(years.min(),2101,20),map(str,np.arange(years.min(),2101,20)),size=6)
plt.xlim([years.min(),years.max()])   
plt.ylim([years.min(),years.max()])

plt.text(2082,2084,r'\textbf{[a]}',fontsize=8,color='k',ha='right',va='center')
plt.text(1920,2074,r'\textbf{AER+}',fontsize=20,color='k',ha='left',va='center')
plt.text(2082,1921,r'\textbf{R$^{2}$=%s}' % np.round(r2_annual[0],2),fontsize=8,color='darkred',
          ha='right',va='center')

###############################################################################
ax = plt.subplot(132)

adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey',pad=1.1)

ax.fill_between(years,annual_minperc[1],annual_maxperc[1],color='deepskyblue',
                alpha=0.3,linewidth=0,clip_on=False,label=r'\textbf{Climate Models [Testing]}')
plt.scatter(yearobs,obs_annual[1],s=12,color='crimson',clip_on=False,
            alpha=0.35,linewidths=0.5,label=r'\textbf{Observations}')
plt.plot(years,years,'-',color='black',linewidth=1,clip_on=False)
plt.plot(yearobs,lines_annual[1],linewidth=1.5,linestyle='--',dashes=(1,0.3),
          color='darkred')

plt.xlabel(r'\textbf{ACTUAL YEAR}',fontsize=10,color='dimgrey')
# plt.ylabel(r'\textbf{PREDICTED YEAR}',fontsize=10,color='dimgrey')
plt.xticks(np.arange(years.min(),2101,20),map(str,np.arange(years.min(),2101,20)),size=6)
plt.yticks(np.arange(years.min(),2101,20),map(str,np.arange(years.min(),2101,20)),size=6)
plt.xlim([years.min(),years.max()])   
plt.ylim([years.min(),years.max()])

le = plt.legend(shadow=False,fontsize=7,loc='upper center',
            bbox_to_anchor=(0.5, 1.17),fancybox=True,ncol=2,frameon=False,
            labelspacing=0.2)
for text in le.get_texts():
    text.set_color('dimgrey') 

plt.text(2082,2084,r'\textbf{[b]}',fontsize=8,color='k',ha='right',va='center')
plt.text(1920,2074,r'\textbf{GHG+}',fontsize=20,color='k',ha='left',va='center')
plt.text(2082,1921,r'\textbf{R$^{2}$=%s}' % np.round(r2_annual[1],2),fontsize=8,color='darkred',
          ha='right',va='center')

###############################################################################
ax = plt.subplot(133)

adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey',pad=1.1)

ax.fill_between(years,annual_minperc[2],annual_maxperc[2],color='deepskyblue',
                alpha=0.3,linewidth=0,clip_on=False)
plt.scatter(yearobs,obs_annual[2],s=12,color='crimson',clip_on=False,
            alpha=0.35,linewidths=0.5)
plt.plot(years,years,'-',color='black',linewidth=1,clip_on=False)
plt.plot(yearobs,lines_annual[2],linewidth=1.5,linestyle='--',dashes=(1,0.3),
          color='darkred')

plt.xlabel(r'\textbf{ACTUAL YEAR}',fontsize=10,color='dimgrey')
# plt.ylabel(r'\textbf{PREDICTED YEAR}',fontsize=10,color='dimgrey')
plt.xticks(np.arange(years.min(),2101,20),map(str,np.arange(years.min(),2101,20)),size=6)
plt.yticks(np.arange(years.min(),2101,20),map(str,np.arange(years.min(),2101,20)),size=6)
plt.xlim([years.min(),years.max()])   
plt.ylim([years.min(),years.max()])

plt.text(2082,2084,r'\textbf{[c]}',fontsize=8,color='k',ha='right',va='center')
plt.text(1920,2074,r'\textbf{ALL}',fontsize=20,color='k',ha='left',va='center')
plt.text(2082,1921,r'\textbf{R$^{2}$=%s}' % np.round(r2_annual[2],2),fontsize=8,color='darkred',
          ha='right',va='center')

plt.subplots_adjust(bottom=0.17)
plt.savefig(directoryfigure + 'ScatterPrediction-MeanRemoved_Annual_T2M_PAPER_R1.png',
            dpi=600)