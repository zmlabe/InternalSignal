"""
Plot results of the ANNs with different hyperparameters of epochs for 20x20

Reference  : Barnes et al. [2020, JAMES] and Barnes et al. [2019, GRL]
Author     : Zachary M. Labe
Date       : 29 March 2021
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
simuname = np.repeat([r'GHG+',r'ALL'],3)
simu = np.repeat(['XAER','lens'],3)
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
seasons = ['annual']
timexghg = np.arange(1920,2080+1,1)
timexaer = np.arange(1920,2080+1,1)
timelens = np.arange(1920,2080+1,1)
yearsall = [timexghg,timexaer,timelens]
directoriesall = [directorydataLLS,directorydataLLS,directorydataLLL]

### Parameters (setting random seeds everwhere for 10 iterations (counter))
l2_try = [0.0001,0.001,0.01,0.1,1,5]
l2_tryname = [0.001,0.01,0.1,0.001,0.01,0.1]
epochs_try = [100,500,1500]
nodes_try = [8,20]
layers_try8 = [8,8],[8,8,8]
layers_try20 = [20,20],[20,20,20],[20,20,20,20]
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
    ax = plt.subplot(2,3,mm+1)
    
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
    ### Select epochs to show changes in distributions
    predplot = [maePREd[2,1,:,:],maePREd[2,2,:,:],maePREd[2,3,:,:],
                maePREd[2,1,:,:],maePREd[2,2,:,:],maePREd[2,3,:,:]]
    posdplot = [maePOSd[2,1,:,:],maePOSd[2,2,:,:],maePOSd[2,3,:,:],
                maePOSd[2,1,:,:],maePOSd[2,2,:,:],maePOSd[2,3,:,:]]
    
    pree = predplot[mm].transpose()
    post = posdplot[mm].transpose()

    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Graph for RMSE 8 layers
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
    ax.tick_params(axis="x",which="both",bottom = False,top=False,
                    labelbottom=False)
    
    ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.7,clip_on=False)
    
    def set_box_color(bp, color):
        plt.setp(bp['boxes'],color=color)
        plt.setp(bp['whiskers'], color=color,linewidth=1.5)
        plt.setp(bp['caps'], color='w',alpha=0)
        plt.setp(bp['medians'], color='w',linewidth=1)
    
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
    if mm == 1:
        l = plt.legend(shadow=False,fontsize=10,loc='upper center',
                    fancybox=True,frameon=False,ncol=2,bbox_to_anchor=(0.5,1.3),
                    labelspacing=1,columnspacing=1,handletextpad=0.4)
    
    for i in range(pree.shape[1]):
        y = pree[:,i]
        x = np.random.normal(positionspre[i], 0.04, size=len(y))
        plt.plot(x, y,color='darkblue', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0)
    for i in range(post.shape[1]):
        y = post[:,i]
        x = np.random.normal(positionspost[i], 0.04, size=len(y))
        plt.plot(x, y,color='darkred', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0)
        
    plt.yticks(np.arange(0,61,5),list(map(str,np.round(np.arange(0,61,5),2))),
                fontsize=6) 
    plt.ylim([0,20])
    
    plt.text(-0.26,0.5,r'\textbf{100}',fontsize=8,color='k',
              ha='left',va='center')
    plt.text(2.02,0.5,r'\textbf{500}',fontsize=8,color='k',
              ha='center',va='center')
    plt.text(4.61,0.5,r'\textbf{1500}',fontsize=8,color='k',
              ha='right',va='center')

    plt.xlabel(r'\textbf{Epochs for L$_{2}$=%s}'  % l2_tryname[mm],color='k',fontsize=6,labelpad=0.25)
    plt.text(4.6,19.05,r'\textbf{[%s]}' % letters[mm],color='dimgrey',fontsize=8,
                          rotation=0,ha='center',va='center')
    
    if any([mm==2,mm==5]):
        plt.text(5,10,r'\textbf{%s}' % simuname[mm],color='dimgrey',fontsize=13,rotation=0,
                 ha='left',va='center')
    
    ### Create x/y labels
    if any([mm==0]):
        plt.text(-3,-14.5,r'\textbf{Mean Absolute Error (MAE)}',fontsize=10,color='k',
                 rotation=90) 
    
    if mm == 0:
        ax.axes.xaxis.set_ticklabels([])
    elif any([mm==1,mm==2]):
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
    elif any([mm==4,mm==5]):
        ax.axes.yaxis.set_ticklabels([])
    
plt.savefig(directoryfigure + 'timeMAE_ForHyperparameters_Epochs_PAPER.png',dpi=600)