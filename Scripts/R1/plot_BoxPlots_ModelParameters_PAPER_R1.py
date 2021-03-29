"""
Plot results of the ANNs with different hyperparameters for [epochs x seeds]

Reference  : Barnes et al. [2020, JAMES] and Barnes et al. [2019, GRL]
Author     : Zachary M. Labe
Date       : 23 March 2021
Version    : R1
"""

### Import packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats 

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
directoryfigure = '/Users/zlabe/Desktop/PAPER/R1/'
datasetsingle = ['XGHG','XAER','lens']
simuname = np.repeat([r'AER+',r'GHG+',r'ALL'],3)
simu = np.repeat(['XGHG','XAER','lens'],3)
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
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
layers_try8 = [8,8],[8,8,8]
layers_try20 = [20,20],[20,20,20],[20,20,20,20]
layersname = layers_try20
layers_tryall = np.append(layers_try8,layers_try20)
SAMPLEQ = 10

###############################################################################
# for lay in range(len(layers_tryall)):
#     for epo in range(len(epochs_try)):
#         for ridg in range(len(l2_try)):   
#             for isample in range(SAMPLEQ): 
###############################################################################

###############################################################################
###############################################################################
###############################################################################
### Read in data
fig = plt.figure()
for mm in range(len(simu)):
    maePRE = np.load(directorydataoutput + '%s_maePRE_annual_R1.npy' % simu[mm],
                      allow_pickle=True).squeeze().reshape(len(layers_tryall),len(epochs_try),len(l2_try),SAMPLEQ)
    maePOS = np.load(directorydataoutput + '%s_maePOST_annual_R1.npy' % simu[mm],
                      allow_pickle=True).squeeze().reshape(len(layers_tryall),len(epochs_try),len(l2_try),SAMPLEQ)
    
    lay = np.load(directorydataoutput + '%s_numOfLayers_annual_R1.npy' % simu[mm],
                  allow_pickle=True).squeeze().reshape(len(layers_tryall),len(epochs_try),len(l2_try),SAMPLEQ)
    epo = np.load(directorydataoutput + '%s_numofEpochs_annual_R1.npy' % simu[mm],
                  allow_pickle=True).squeeze().reshape(len(layers_tryall),len(epochs_try),len(l2_try),SAMPLEQ)
    L2s = np.load(directorydataoutput + '%s_L2_annual_R1.npy' % simu[mm],
                  allow_pickle=True).squeeze().reshape(len(layers_tryall),len(epochs_try),len(l2_try),SAMPLEQ)
    
    isample = np.load(directorydataoutput + '%s_numOfSeedSamples_annual_R1.npy' % simu[mm],
                      allow_pickle=True).squeeze().reshape(len(layers_tryall),len(epochs_try),len(l2_try),SAMPLEQ)
    enstype = np.load(directorydataoutput + '%s_lensType_annual_R1.npy' % simu[mm],
                      allow_pickle=True).squeeze().reshape(len(layers_tryall),len(epochs_try),len(l2_try),SAMPLEQ)
    segseed = np.load(directorydataoutput + '%s_EnsembleSegmentSeed_annual_R1.npy' % simu[mm],
                      allow_pickle=True).squeeze().reshape(len(layers_tryall),len(epochs_try),len(l2_try),SAMPLEQ)
    annseed =np.load(directorydataoutput + '%s_annSegmentSeed_annual_R1.npy' % simu[mm],
                      allow_pickle=True).squeeze().reshape(len(layers_tryall),len(epochs_try),len(l2_try),SAMPLEQ)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Align correct distributions   
    maePREd = np.swapaxes(maePRE,1,2)
    maePOSd = np.swapaxes(maePOS,1,2)
    
    layd = np.swapaxes(lay,1,2)
    epod = np.swapaxes(epo,1,2)
    L2sd = np.swapaxes(L2s,1,2)
    isampled = np.swapaxes(isample,1,2)
    enstyped = np.swapaxes(enstype,1,2)
    segseedd = np.swapaxes(segseed,1,2)
    annseedd = np.swapaxes(annseed,1,2)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Prepare distributions  
    maePREr = maePREd.reshape(len(layers_tryall),len(l2_try),len(epochs_try)*SAMPLEQ)
    maePOSr = maePOSd.reshape(len(layers_tryall),len(l2_try),len(epochs_try)*SAMPLEQ)
    
    layr = layd.reshape(len(layers_tryall),len(l2_try),len(epochs_try)*SAMPLEQ)
    epor = epod.reshape(len(layers_tryall),len(l2_try),len(epochs_try)*SAMPLEQ)
    L2sr = L2sd.reshape(len(layers_tryall),len(l2_try),len(epochs_try)*SAMPLEQ)
    isampler = isampled.reshape(len(layers_tryall),len(l2_try),len(epochs_try)*SAMPLEQ)
    enstyper = enstyped.reshape(len(layers_tryall),len(l2_try),len(epochs_try)*SAMPLEQ)
    segseedr = segseedd.reshape(len(layers_tryall),len(l2_try),len(epochs_try)*SAMPLEQ)
    annseedr = annseedd.reshape(len(layers_tryall),len(l2_try),len(epochs_try)*SAMPLEQ)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Graph for MAE 20 layers
    predplot = [maePREr[2,1:,:],maePREr[3,1:,:],maePREr[4,1:,:],
                maePREr[2,1:,:],maePREr[3,1:,:],maePREr[4,1:,:],
                maePREr[2,1:,:],maePREr[3,1:,:],maePREr[4,1:,:]]
    posdplot = [maePOSr[2,1:,:],maePOSr[3,1:,:],maePOSr[4,1:,:],
                maePOSr[2,1:,:],maePOSr[3,1:,:],maePOSr[4,1:,:],
                maePOSr[2,1:,:],maePOSr[3,1:,:],maePOSr[4,1:,:]]
    
    pree = predplot[mm].transpose() 
    post = posdplot[mm].transpose() 
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ax = plt.subplot(3,3,mm+1)
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=3,width=2,which='major',color='dimgrey',pad=2)
    ax.tick_params(axis="x",which="both",bottom = False,top=False,
                    labelbottom=False)
    
    def set_box_color(bp, color):
        plt.setp(bp['boxes'],color=color)
        plt.setp(bp['whiskers'], color=color,linewidth=1.5)
        plt.setp(bp['caps'], color='w',alpha=0)
        plt.setp(bp['medians'], color='w',linewidth=0.5)
    
    positionspre = np.array(range(pree.shape[1]))*2.0-0.3
    positionspost = np.array(range(post.shape[1]))*2.0+0.3
    bpl = plt.boxplot(pree,positions=positionspre,widths=0.5,
                      patch_artist=True,sym='')
    bpr = plt.boxplot(post,positions=positionspost, widths=0.5,
                      patch_artist=True,sym='')
    
    # Modify boxes
    cpree = 'deepskyblue'
    cpost = 'indianred'
    set_box_color(bpl,cpree)
    set_box_color(bpr,cpost)
    plt.plot([], c=cpree, label=r'\textbf{BEFORE 1980}')
    plt.plot([], c=cpost, label=r'\textbf{AFTER 1980}')
    
    for i in range(pree.shape[1]):
        y = pree[:,i]
        x = np.random.normal(positionspre[i], 0.04, size=len(y))
        plt.plot(x, y,color='darkblue', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=2,markeredgewidth=0)
    for i in range(post.shape[1]):
        y = post[:,i]
        x = np.random.normal(positionspost[i], 0.04, size=len(y))
        plt.plot(x, y,color='darkred', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=2,markeredgewidth=0)
        
    plt.yticks(np.arange(0,91,20),list(map(str,np.round(np.arange(0,91,20),2))),
                fontsize=5) 
    plt.ylim([0,80])
    plt.text(8.2,75,r'\textbf{[%s]}' % letters[mm],color='dimgrey',fontsize=7,
                          rotation=0,ha='center',va='center')
    if any([mm==2,mm==5,mm==8]):
        plt.text(9,38,r'\textbf{%s}' % simuname[mm],color='k',fontsize=13,rotation=0,
                 ha='left',va='center')
    if any([mm==0,mm==1,mm==2]):
        plt.text(4,99,r'\textbf{%s LAYERS$_{20}$}' % len(layersname[mm]),color='k',fontsize=13,rotation=0,
                 ha='center',va='center')
    
    plt.text(-0.7,-5,r'\textbf{L$_{2}$=%s}' % l2_try[1],fontsize=4,color='k',
              ha='left',va='center')
    plt.text(2.1,-5,r'\textbf{L$_{2}$= %s}' % l2_try[2],fontsize=4,color='k',
              ha='center',va='center')
    plt.text(4.7,-5,r'\textbf{L$_{2}$= %s}' % l2_try[3],fontsize=4,color='k',
              ha='right',va='center')
    plt.text(5.55,-5,r'\textbf{L$_{2}$= %s}' % l2_try[4],fontsize=4,color='k',
              ha='left',va='center')
    plt.text(8.0,-5,r'\textbf{L$_{2}$= %s}' % l2_try[5],fontsize=4,color='k',
              ha='center',va='center')
    
    if any([mm==3]):
        plt.ylabel(r'\textbf{Mean Absolute Error (MAE)}',fontsize=8,color='dimgrey') 
    if mm == 7:
        plt.xlabel(r'\textbf{Ridge Regularizations}',color='dimgrey',fontsize=8,labelpad=4)
    if mm == 7:
        l = plt.legend(shadow=False,fontsize=10,loc='upper center',
                    fancybox=True,frameon=False,ncol=2,bbox_to_anchor=(0.5,-0.20),
                    labelspacing=1,columnspacing=1,handletextpad=0.4)
    
plt.savefig(directoryfigure + 'MAEForHyperparameters_layer-%s_PAPER.png' % (20),dpi=300)