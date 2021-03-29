"""
Plot results of the ANNs with different hyperparameters

Reference  : Barnes et al. [2020, JAMES] and Barnes et al. [2019, GRL]
Author     : Zachary M. Labe
Date       : 25 March 2021
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
simuname = [r'AER+',r'GHG+',r'ALL']
simu = ['XGHG','XAER','lens']
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
for mm in range(len(simu)):
    slopetrain = np.load(directorydataoutput + '%s_slopeTrain_annual_R1.npy' % simu[mm],
                          allow_pickle=True).squeeze().reshape(len(layers_tryall),len(epochs_try),len(l2_try),SAMPLEQ)
    slopetest = np.load(directorydataoutput + '%s_slopeTest_annual_R1.npy' % simu[mm],
                        allow_pickle=True).squeeze().reshape(len(layers_tryall),len(epochs_try),len(l2_try),SAMPLEQ)
    
    r2train = np.load(directorydataoutput + '%s_r2Train_annual_R1.npy' % simu[mm],
                      allow_pickle=True).squeeze().reshape(len(layers_tryall),len(epochs_try),len(l2_try),SAMPLEQ)
    r2test = np.load(directorydataoutput + '%s_r2Test_annual_R1.npy' % simu[mm],
                      allow_pickle=True).squeeze().reshape(len(layers_tryall),len(epochs_try),len(l2_try),SAMPLEQ)
    
    rmsePRE = np.load(directorydataoutput + '%s_rmsePRE_annual_R1.npy' % simu[mm],
                      allow_pickle=True).squeeze().reshape(len(layers_tryall),len(epochs_try),len(l2_try),SAMPLEQ)
    rmsePOS = np.load(directorydataoutput + '%s_rmsePOST_annual_R1.npy' % simu[mm],
                      allow_pickle=True).squeeze().reshape(len(layers_tryall),len(epochs_try),len(l2_try),SAMPLEQ)
    rmseALL = np.load(directorydataoutput + '%s_rmseYEARS_annual_R1.npy' % simu[mm],
                      allow_pickle=True).squeeze().reshape(len(layers_tryall),len(epochs_try),len(l2_try),SAMPLEQ)
    
    maePRE = np.load(directorydataoutput + '%s_maePRE_annual_R1.npy' % simu[mm],
                      allow_pickle=True).squeeze().reshape(len(layers_tryall),len(epochs_try),len(l2_try),SAMPLEQ)
    maePOS = np.load(directorydataoutput + '%s_maePOST_annual_R1.npy' % simu[mm],
                      allow_pickle=True).squeeze().reshape(len(layers_tryall),len(epochs_try),len(l2_try),SAMPLEQ)
    maeALL = np.load(directorydataoutput + '%s_maeYEARS_annual_R1.npy' % simu[mm],
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
    slopetestd = np.swapaxes(slopetest,1,2)
    r2testd = np.swapaxes(r2test,1,2)
    
    rmsePREd = np.swapaxes(rmsePRE,1,2)
    rmsePOSd = np.swapaxes(rmsePOS,1,2)
    rmseALLd = np.swapaxes(rmseALL,1,2)
    
    maePREd = np.swapaxes(maePRE,1,2)
    maePOSd = np.swapaxes(maePOS,1,2)
    maeALLd = np.swapaxes(maeALL,1,2)
    
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
    comp8_1 = maePREd[0,1,:,:]
    comp8_2 = maePOSd[0,1,:,:]
    
    comp20_1 = maePREd[2,1,:,:]
    comp20_2 = maePOSd[2,1,:,:]
    
    comp8_1m = maePREd[0,2,:,:]
    comp8_2m = maePOSd[0,2,:,:]
    
    comp20_1m = maePREd[2,2,:,:]
    comp20_2m = maePOSd[2,2,:,:]
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Graph for RMSE 8 layers
    pree = comp8_1.transpose() # 8x8
    post = comp8_2.transpose() # 8x8
    
    fig = plt.figure(figsize=(10,4))
    ax = plt.subplot(141)
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
    ax.tick_params(axis="x",which="both",bottom = False,top=False,
                    labelbottom=False)
    
    ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.7)
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
    l = plt.legend(shadow=False,fontsize=10,loc='upper center',
                fancybox=True,frameon=False,ncol=2,bbox_to_anchor=(2.8,1.16),
                labelspacing=1,columnspacing=1,handletextpad=0.4)
    
    for i in range(pree.shape[1]):
        y = pree[:,i]
        x = np.random.normal(positionspre[i], 0.04, size=len(y))
        plt.plot(x, y,color='darkblue', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0)
    for i in range(post.shape[1]):
        y = post[:,i]
        x = np.random.normal(positionspost[i], 0.04, size=len(y))
        plt.plot(x, y,color='darkred', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0)
        
    plt.ylabel(r'\textbf{\underline{MAE} for layers = %s using %s}' % (layers_tryall[0],simuname[mm]),color='dimgrey',fontsize=8)
    plt.yticks(np.arange(0,61,5),list(map(str,np.round(np.arange(0,61,5),2))),
                fontsize=6) 
    plt.ylim([0,15])
    
    plt.text(-0.2,0.2,r'\textbf{100}',fontsize=8,color='k',
              ha='left',va='center')
    plt.text(2.02,0.2,r'\textbf{500}',fontsize=8,color='k',
              ha='center',va='center')
    plt.text(4.61,0.2,r'\textbf{1500}',fontsize=8,color='k',
              ha='right',va='center')

    plt.xlabel(r'\textbf{Epochs for L$_{2}$=%s}'  % l2_try[1],color='dimgrey',fontsize=8)
    
    ###########################################################################
    ###############################################################################
    ### Graph for RMSE 20 layers
    pree = comp20_1.transpose() # 20x20
    post = comp20_2.transpose() # 20x20
    
    ax = plt.subplot(142)
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
    ax.tick_params(axis="x",which="both",bottom = False,top=False,
                    labelbottom=False)
    
    ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.7)
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
    
    for i in range(pree.shape[1]):
        y = pree[:,i]
        x = np.random.normal(positionspre[i], 0.04, size=len(y))
        plt.plot(x, y,color='darkblue', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0)
    for i in range(post.shape[1]):
        y = post[:,i]
        x = np.random.normal(positionspost[i], 0.04, size=len(y))
        plt.plot(x, y,color='darkred', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0)
        
    plt.ylabel(r'\textbf{\underline{MAE} for layers = %s using %s}' % (layers_tryall[2],simuname[mm]),color='dimgrey',fontsize=8)
    plt.yticks(np.arange(0,61,5),list(map(str,np.round(np.arange(0,61,5),2))),
                fontsize=6) 
    plt.ylim([0,15])
    
    plt.text(-0.2,0.2,r'\textbf{100}',fontsize=8,color='k',
              ha='left',va='center')
    plt.text(2.02,0.2,r'\textbf{500}',fontsize=8,color='k',
              ha='center',va='center')
    plt.text(4.61,0.2,r'\textbf{1500}',fontsize=8,color='k',
              ha='right',va='center')

    plt.xlabel(r'\textbf{Epochs for L$_{2}$=%s}'  % l2_try[1],color='dimgrey',fontsize=8)
    
    ###########################################################################
    ###########################################################################
    ### Graph for RMSE 8 layers
    pree = comp8_1m.transpose() # 8x8
    post = comp8_2m.transpose() # 8x8
    
    ax = plt.subplot(143)
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
    ax.tick_params(axis="x",which="both",bottom = False,top=False,
                    labelbottom=False)
    
    ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.7)
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
    
    for i in range(pree.shape[1]):
        y = pree[:,i]
        x = np.random.normal(positionspre[i], 0.04, size=len(y))
        plt.plot(x, y,color='darkblue', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0)
    for i in range(post.shape[1]):
        y = post[:,i]
        x = np.random.normal(positionspost[i], 0.04, size=len(y))
        plt.plot(x, y,color='darkred', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0)
        
    plt.ylabel(r'\textbf{\underline{MAE} for layers = %s using %s}' % (layers_tryall[0],simuname[mm]),color='dimgrey',fontsize=8)
    plt.yticks(np.arange(0,61,5),list(map(str,np.round(np.arange(0,61,5),2))),
                fontsize=6) 
    plt.ylim([0,15])
    
    plt.text(-0.2,0.2,r'\textbf{100}',fontsize=8,color='k',
              ha='left',va='center')
    plt.text(2.02,0.2,r'\textbf{500}',fontsize=8,color='k',
              ha='center',va='center')
    plt.text(4.61,0.2,r'\textbf{1500}',fontsize=8,color='k',
              ha='right',va='center')

    plt.xlabel(r'\textbf{Epochs for L$_{2}$=%s}'  % l2_try[2],color='dimgrey',fontsize=8)
    
    ###########################################################################
    ###############################################################################
    ### Graph for RMSE 8 layers
    pree = comp20_1m.transpose() # 20x20
    post = comp20_2m.transpose() # 20x20
    
    ax = plt.subplot(144)
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
    ax.tick_params(axis="x",which="both",bottom = False,top=False,
                    labelbottom=False)
    
    ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.7)
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
    
    for i in range(pree.shape[1]):
        y = pree[:,i]
        x = np.random.normal(positionspre[i], 0.04, size=len(y))
        plt.plot(x, y,color='darkblue', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0)
    for i in range(post.shape[1]):
        y = post[:,i]
        x = np.random.normal(positionspost[i], 0.04, size=len(y))
        plt.plot(x, y,color='darkred', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0)
        
    plt.ylabel(r'\textbf{\underline{MAE} for layers = %s using %s}' % (layers_tryall[2],simuname[mm]),color='dimgrey',fontsize=8)
    plt.yticks(np.arange(0,61,5),list(map(str,np.round(np.arange(0,61,5),2))),
                fontsize=6) 
    plt.ylim([0,15])
    
    plt.text(-0.2,0.2,r'\textbf{100}',fontsize=8,color='k',
              ha='left',va='center')
    plt.text(2.02,0.2,r'\textbf{500}',fontsize=8,color='k',
              ha='center',va='center')
    plt.text(4.61,0.2,r'\textbf{1500}',fontsize=8,color='k',
              ha='right',va='center')

    plt.xlabel(r'\textbf{Epochs for L$_{2}$=%s}'  % l2_try[2],color='dimgrey',fontsize=8)
    
    plt.subplots_adjust(wspace=0.5)
    plt.savefig(directoryfigure + '%s_MAEForHyperparameters_Epochs.png' % (simu[mm]),dpi=300)