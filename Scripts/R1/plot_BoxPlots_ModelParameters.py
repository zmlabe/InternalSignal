"""
Plot results of the ANNs with different hyperparameters

Reference  : Barnes et al. [2020, JAMES] and Barnes et al. [2019, GRL]
Author     : Zachary M. Labe
Date       : 23 March 2021
Version    : R1
"""

### Import packages
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats 
import cmocean as cmocean

### Plotting defaults 
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

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directorydataLLS = '/Users/zlabe/Data/LENS/SINGLE/'
directorydataLLL = '/Users/zlabe/Data/LENS/monthly'
directorydataBB = '/Users/zlabe/Data/BEST/'
directorydataEE = '/Users/zlabe/Data/ERA5/'
directorydataoutput = '/Users/zlabe/Documents/Research/InternalSignal/Data/FINAL/R1/Parameters/'
datasetsingle = ['XGHG','XAER','lens']
simu = ['XGHG','XAER','lens']
simu = ['XGHG']
seasons = ['annual']
timexghg = np.arange(1920,2080+1,1)
timexaer = np.arange(1920,2080+1,1)
timelens = np.arange(1920,2080+1,1)
yearsall = [timexghg,timexaer,timelens]
directoriesall = [directorydataLLS,directorydataLLS,directorydataLLL]

### Parameters (setting random seeds everwhere for 10 iterations (counter))
l2_try = [0.0001,0.001,0.01,0.1,1,5]
epochs_try = [100,500,1500]
nodes_try = [8,20]
layers_try8 = [[8,8]],[[8,8,8]]
layers_try20 = [[20,20]],[[20,20,20]],[[20,20,20,20]]
layers_tryall = np.append(layers_try8,layers_try20)

###############################################################################
# for lay in range(len(layers_tryall)):
#     for epo in range(len(epochs_try)):
#         for ridg in range(len(l2_try)):   
#             for isample in range(SAMPLEQ): 
###############################################################################

### Read in data
for i in range(len(simu)):
    slopetrain = np.load(directorydataoutput + '%s_slopeTrain_annual_R1.npy' % simu[i],allow_pickle=True)
    slopetest = np.load(directorydataoutput + '%s_slopeTest_annual_R1.npy' % simu[i],allow_pickle=True)
    
    r2train = np.load(directorydataoutput + '%s_r2Train_annual_R1.npy' % simu[i],allow_pickle=True)
    r2test = np.load(directorydataoutput + '%s_r2Test_annual_R1.npy' % simu[i],allow_pickle=True)
    
    rmsePRE = np.load(directorydataoutput + '%s_rmsePRE_annual_R1.npy' % simu[i],allow_pickle=True)
    rmsePOS = np.load(directorydataoutput + '%s_rmsePOST_annual_R1.npy' % simu[i],allow_pickle=True)
    rmseALL = np.load(directorydataoutput + '%s_rmseYEARS_annual_R1.npy' % simu[i],allow_pickle=True)
    
    maePRE = np.load(directorydataoutput + '%s_maePRE_annual_R1.npy' % simu[i],allow_pickle=True)
    maePOS = np.load(directorydataoutput + '%s_maePOST_annual_R1.npy' % simu[i],allow_pickle=True)
    maeALL = np.load(directorydataoutput + '%s_maeYEARS_annual_R1.npy' % simu[i],allow_pickle=True)
    
    lay = np.load(directorydataoutput + '%s_numOfLayers_annual_R1.npy' % simu[i],allow_pickle=True)
    epo = np.load(directorydataoutput + '%s_numofEpochs_annual_R1.npy' % simu[i],allow_pickle=True)
    L2s = np.load(directorydataoutput + '%s_L2_annual_R1.npy' % simu[i],allow_pickle=True)
    
    isample = np.load(directorydataoutput + '%s_numOfSeedSamples_annual_R1.npy' % simu[i],allow_pickle=True)
    enstype = np.load(directorydataoutput + '%s_lensType_annual_R1.npy' % simu[i],allow_pickle=True)
    segseed = np.load(directorydataoutput + '%s_EnsembleSegmentSeed_annual_R1.npy' % simu[i],allow_pickle=True)
    annseed =np.load(directorydataoutput + '%s_annSegmentSeed_annual_R1.npy' % simu[i],allow_pickle=True)