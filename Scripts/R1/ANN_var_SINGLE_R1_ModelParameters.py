"""
Try different hyperparameters for the overall ANN architecture

Reference  : Barnes et al. [2020, JAMES] and Barnes et al. [2019, GRL]
Author     : Zachary M. Labe
Date       : 16 March 2021
Version    : R1
"""

###############################################################################
### TURNED OFF SAVING FILES FOR NPZ AND H5!!!!!! TOO MUCH STORAGE SPACE ATM
###############################################################################

### Import packages
import math
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
from keras.layers import Dense, Activation
from keras import regularizers
from keras import metrics
from keras import optimizers
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import random
import scipy.stats as stats 
import cmocean as cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS

### Remove warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

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
seasons = ['annual']
timexghg = np.arange(1920,2080+1,1)
timexaer = np.arange(1920,2080+1,1)
timelens = np.arange(1920,2080+1,1)
yearsall = [timexghg,timexaer,timelens]
directoriesall = [directorydataLLS,directorydataLLS,directorydataLLL]

### Test script
datasetsingle = ['XAER']
directoriesall = [directorydataLLS]
yearsall = [timexaer]
l2_try = [0.01]
epochs_try = [500]
nodes_try = [20]
layers_try20 = [[20,20]]
layers_tryall = layers_try20
SAMPLEQ = 1

# ### Parameters (setting random seeds everwhere for 5 iterations (counter))
# l2_try = [0.0001,0.001,0.01,0.1,1,5]
# epochs_try = [100,500,1500]
# nodes_try = [8,20]
# layers_try8 = [[8,8]],[[8,8,8]]
# layers_try20 = [[20,20]],[[20,20,20]],[[20,20,20,20]]
# layers_tryall = np.append(layers_try8,layers_try20)
# SAMPLEQ = 10
 
### Begin model
slopetrain_tryModel = []
slopetest_tryModel = []
meanr_train_tryModel = []
meanr_test_tryModel = []

rma_test_tryModel = []
rmb_test_tryModel = []
rmall_test_tryModel = []

mae_testa_tryModel = []
mae_testb_tryModel = []
mae_testall_tryModel = []

layers_trysaveModel = []
epochs_trysaveModel = []
l2_trysaveModel = []

samples_trysaveModel = []
models_trysaveModel = []
ensembleseed_trysaveModel = []
annseed_trysaveModel = []

for sis,singlesimulation in enumerate(datasetsingle):
    slopetrain_try = []
    slopetest_try = []
    meanr_train_try = []
    meanr_test_try = []
    
    rma_test_try = []
    rmb_test_try = []
    rmall_test_try = []

    mae_testa_try = []
    mae_testb_try = []    
    mae_testall_try = []
    
    layers_trysave = []
    epochs_trysave = []
    l2_trysave = []
    
    samples_trysave = []
    models_trysave = []
    ensembleseed_trysave = []
    annseed_trysave = []
    for lay in range(len(layers_tryall)):
        for epo in range(len(epochs_try)):
            for ridg in range(len(l2_try)):   
                for isample in range(SAMPLEQ): 
                    ###############################################################################
                    ###############################################################################
                    ###############################################################################
                    ### ANN preliminaries
                    variq = 'T2M'
                    monthlychoice = seasons[0]
                    reg_name = 'Globe'
                    lat_bounds,lon_bounds = UT.regions(reg_name)
                    directoryfigure = '/Users/zlabe/Desktop/PAPER/R1/'
                    
                    experiment_result = pd.DataFrame(columns=['actual iters','hiddens','cascade',
                                                              'RMSE Train','RMSE Test',
                                                              'ridge penalty','zero mean',
                                                              'zero merid mean','land only?','ocean only?'])
                    
                    
                    ### Define primary dataset to use
                    dataset = singlesimulation
                    modelType = dataset
                    
                    ### Whether to test and plot the results using obs data
                    test_on_obs = True
                    dataset_obs = '20CRv3'
                    if dataset_obs == '20CRv3':
                        year_obsall = np.arange(yearsall[sis].min(),2015+1,1)
                    elif dataset_obs == 'ERA5':
                        year_obsall = np.arange(1979,2019+1,1)
                    if monthlychoice == 'DJF':
                        obsyearstart = year_obsall.min()+1
                        year_obs = year_obsall[1:]
                    else:
                        obsyearstart = year_obsall.min()
                        year_obs = year_obsall
                    
                    ### Remove the annual mean? True to subtract it from dataset ##########
                    rm_annual_mean = False #################################################
                    if rm_annual_mean == True:
                        directoryfigure = '/Users/zlabe/Desktop/PAPER/R1/'
                    
                    ### Remove the meridional mean? True to subtract it from dataset ######
                    rm_merid_mean = False #################################################
                    if rm_merid_mean == True:
                        directoryfigure = '/Users/zlabe/Desktop/PAPER/R1/'
                    
                    ### Calculate only over land? True if land ############################
                    land_only = False ######################################################
                    if land_only == True:
                        directoryfigure = '/Users/zlabe/Desktop/PAPER/R1/'
                    
                    ### Calculate only over ocean? True if ocean ##########################
                    ocean_only = False #####################################################
                    if ocean_only == True:
                        directoryfigure = '/Users/zlabe/Desktop/PAPER/R1/'
                    
                    ### Rove the ensemble mean? True to subtract it from dataset ##########
                    rm_ensemble_mean = False ##############################################
                    if rm_ensemble_mean == True:
                        directoryfigure = '/Users/zlabe/Desktop/PAPER/R1/'
                    
                    ### Split the data into training and testing sets? value of 1 will use all 
                    ### data as training, .8 will use 80% training, 20% testing; etc.
                    segment_data_factor = .8
                    
                    ### Hiddens corresponds to the number of hidden layers the nnet will use - 0 
                    ### for linear model, or a list [10, 20, 5] for multiple layers of nodes 
                    ### (10 nodes in first layer, 20 in second, etc); The "loop" part 
                    ### allows you to loop through multiple architectures. For example, 
                    ### hiddens_loop = [[2,4],[0],[1 1 1]] would produce three separate NNs, the 
                    ### first with 2 hidden layers of 2 and 4 nodes, the next the linear model,
                    ### and the next would be 3 hidden layers of 1 node each.
                    
                    ### Set useGPU to True to use the GPU, but only if you selected the GPU 
                    ### Runtime in the menu at the top of this page
                    useGPU = False
                    
                    ### Set Cascade to True to utilize the nnet's cascade function
                    cascade = False
                    
                    ### Plot within the training loop - may want to set to False when testing out 
                    ### larget sets of parameters
                    plot_in_train = False
                    
                    ###############################################################################
                    ###############################################################################
                    ###############################################################################
                    ### Read in model and observational/reanalysis data
                    
                    def read_primary_dataset(variq,dataset,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
                        data,lats,lons = df.readFiles(variq,dataset,monthlychoice)
                        datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
                        print('\nOur dataset: ',dataset,' is shaped',data.shape)
                        return datar,lats,lons
                      
                    def read_obs_dataset(variq,dataset_obs,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
                        data_obs,lats_obs,lons_obs = df.readFiles(variq,dataset_obs,monthlychoice)
                        data_obs,lats_obs,lons_obs = df.getRegion(data_obs,lats_obs,lons_obs,
                                                                lat_bounds,lon_bounds)
                        if dataset_obs == '20CRv3':
                            if monthlychoice == 'DJF':
                                year20cr = np.arange(1837,2015+1)
                            else:
                                  year20cr = np.arange(1836,2015+1)
                            year_obsall = np.arange(yearsall[sis].min(),yearsall[sis].max()+1,1)
                            yearqq = np.where((year20cr >= year_obsall.min()) & (year20cr <= year_obsall.max()))[0]
                            data_obs = data_obs[yearqq,:,:]
                        
                        print('our OBS dataset: ',dataset_obs,' is shaped',data_obs.shape)
                        return data_obs,lats_obs,lons_obs
                    
                    ###############################################################################
                    ###############################################################################
                    ###############################################################################
                    ### Select data to test, train on
                        
                    def segment_data(data,fac = segment_data_factor):
                      
                        global random_segment_seed,trainIndices, estIndices
                        if random_segment_seed == None:
                            random_segment_seed = int(int(np.random.randint(1, 100000)))
                        np.random.seed(random_segment_seed)
                        
                        if fac < 1 :
                            nrows = data.shape[0]
                            segment_train = int(np.round(nrows * fac))
                            segment_test = nrows - segment_train
                            print('Training on',segment_train,'ensembles, testing on',segment_test)
                    
                            ### Picking out random ensembles
                            i = 0
                            trainIndices = list()
                            while i < segment_train:
                                line = np.random.randint(0, nrows)
                                if line not in trainIndices:
                                    trainIndices.append(line)
                                    i += 1
                                else:
                                    pass
                        
                            i = 0
                            testIndices = list()
                            while i < segment_test:
                                line = np.random.randint(0, nrows)
                                if line not in trainIndices:
                                    if line not in testIndices:
                                        testIndices.append(line)
                                        i += 1
                                else:
                                    pass
                            
                            ### Random ensembles are picked
                            if debug:
                                print('Training on ensembles: ',trainIndices)
                                print('Testing on ensembles: ',testIndices)
                            
                            ### Training segment----------
                            data_train = ''
                            for ensemble in trainIndices:
                                this_row = data[ensemble, :, :, :]
                                this_row = this_row.reshape(-1,data.shape[1],data.shape[2],
                                                            data.shape[3])
                                if data_train == '':
                                    data_train = np.empty_like(this_row)
                                data_train = np.vstack((data_train,this_row))
                            data_train = data_train[1:, :, :, :]
                            
                            if debug:
                                print('org data - shape', data.shape)
                                print('training data - shape', data_train.shape)
                        
                            ### Reshape into X and T
                            Xtrain = data_train.reshape((data_train.shape[0] * data_train.shape[1]),
                                                        (data_train.shape[2] * data_train.shape[3]))
                            Ttrain = np.tile((np.arange(data_train.shape[1]) + yearsall[sis].min()).reshape(data_train.shape[1],1),
                                              (data_train.shape[0],1))
                            Xtrain_shape = (data_train.shape[0],data_train.shape[1])
                            
                            
                            ### Testing segment----------
                            data_test = ''
                            for ensemble in testIndices:
                                this_row = data[ensemble, :, :, :]
                                this_row = this_row.reshape(-1,data.shape[1],data.shape[2],
                                                            data.shape[3])
                                if data_test == '':
                                    data_test = np.empty_like(this_row)
                                data_test = np.vstack((data_test, this_row))
                            data_test = data_test[1:, :, :, :]
                            
                            if debug:
                                print('testing data', data_test.shape)
                              
                            ### Reshape into X and T
                            Xtest = data_test.reshape((data_test.shape[0] * data_test.shape[1]),
                                                      (data_test.shape[2] * data_test.shape[3]))
                            Ttest = np.tile((np.arange(data_test.shape[1]) + yearsall[sis].min()).reshape(data_test.shape[1],1),
                                            (data_test.shape[0], 1))   
                    
                        else:
                            trainIndices = np.arange(0,np.shape(data)[0])
                            testIndices = np.arange(0,np.shape(data)[0])    
                            print('Training on ensembles: ',trainIndices)
                            print('Testing on ensembles: ',testIndices)
                    
                            data_train = data
                            data_test = data
                        
                            Xtrain = data_train.reshape((data_train.shape[0] * data_train.shape[1]),
                                                        (data_train.shape[2] * data_train.shape[3]))
                            Ttrain = np.tile((np.arange(data_train.shape[1]) + yearsall[sis].min()).reshape(data_train.shape[1],1),
                                              (data_train.shape[0],1))
                            Xtrain_shape = (data_train.shape[0], data_train.shape[1])
                    
                        Xtest = data_test.reshape((data_test.shape[0] * data_test.shape[1]),
                                                  (data_test.shape[2] * data_test.shape[3]))
                        Ttest = np.tile((np.arange(data_test.shape[1]) + yearsall[sis].min()).reshape(data_test.shape[1],1),
                                        (data_test.shape[0],1))
                    
                        Xtest_shape = (data_test.shape[0], data_test.shape[1])
                        data_train_shape = data_train.shape[1]
                        data_test_shape = data_test.shape[1]
                      
                        ### 'unlock' the random seed
                        np.random.seed(None)
                      
                        return Xtrain,Ttrain,Xtest,Ttest,Xtest_shape,Xtrain_shape,data_train_shape,data_test_shape,testIndices
                    
                    ###############################################################################
                    ###############################################################################
                    ###############################################################################
                    ### Data management
                        
                    def shape_obs(data_obs,year_obs):
                        Xtest_obs = np.reshape(data_obs,(data_obs.shape[0],
                                                          (data_obs.shape[1]*data_obs.shape[2])))
                        Ttest_obs = np.tile(np.arange(data_obs.shape[0])+year_obs[0])
                        return Xtest_obs,Ttest_obs
                    
                    def consolidate_data():
                        '''function to delete data and data_obs since we have already sliced other 
                        variables from them.  Only run after segment_data and shape_obs!!!
                        will delete global variables and clear memory'''
                        global data
                        global data_obs
                        del data
                        del data_obs
                    
                    ###############################################################################
                    ###############################################################################
                    ###############################################################################
                    ### Plotting functions
                    
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
                    
                    def plot_prediction (Ttest, test_output, Ttest_obs, obs_output):
                        ### Predictions
                        
                        plt.figure(figsize=(16,4))
                        plt.subplot(1, 2, 1)
                        plt.title('Predicted vs Actual Year for Testing')
                        plt.xlabel('Actual Year')
                        plt.ylabel('Predicted Year')
                        plt.plot(Ttest, test_output, 'o', color='black', label='GCM')
                          
                        if test_on_obs == True:
                            plt.plot(Ttest_obs, obs_output,'o',color='deepskyblue',label='obs')
                        a = min(min(Ttest), min(test_output))
                        b = max(max(Ttest), max(test_output))
                        plt.plot((a,b), (a,b), '-', lw=3, alpha=0.7, color='gray')
                        #plt.axis('square')
                        plt.xlim(a * .995, b * 1.005)
                        plt.ylim(a * .995, b * 1.005)
                        plt.legend()
                        plt.show()
                      
                    def plot_training_error(nnet):
                        ### Training error (nnet)
                      
                        plt.subplot(1, 2, 2)
                        plt.plot(nnet.getErrors(), color='black')
                        plt.title('Training Error per Itobstion')
                        plt.xlabel('Training Itobstion')
                        plt.ylabel('Training Error')
                        plt.show()
                     
                    def plot_rmse(train_output,Ttrain,test_output,Ttest,data_train_shape,data_test_shape):
                        ### rmse (train_output, Ttrain, test_output, Ttest, data_train_shape, data_test_shape)
                      
                        plt.figure(figsize=(16, 4))
                        plt.subplot(1, 2, 1)
                        rmse_by_year_train = np.sqrt(np.mean(((train_output - Ttrain)**2).reshape(Xtrain_shape),
                                                              axis=0))
                        xs_train = (np.arange(data_train_shape) + yearsall[sis].min())
                        rmse_by_year_test = np.sqrt(np.mean(((test_output - Ttest)**2).reshape(Xtest_shape),
                                                            axis=0))
                        xs_test = (np.arange(data_test_shape) + yearsall[sis].min())
                        plt.title('RMSE by year')
                        plt.xlabel('year')
                        plt.ylabel('error')
                        plt.plot(xs_train,rmse_by_year_train,label = 'training error',
                                  color='gold',linewidth=1.5)
                        plt.plot(xs_test,rmse_by_year_test,labe ='test error',
                                  color='forestgreen',linewidth=0.7)
                        plt.legend()
                    
                        if test_on_obs == True:
                            plt.subplot(1,2,2)
                            error_by_year_test_obs = obs_output - Ttest_obs
                            plt.plot(Ttest_obs,error_by_year_test_obs,label='obs error',
                                  color='deepskyblue',linewidth=2.)            
                            plt.title('Error by year for obs')
                            plt.xlabel('year')
                            plt.ylabel('error')
                            plt.legend()
                            plt.plot((1979,2020), (0,0), color='gray', linewidth=2.)
                            plt.xlim(1979,2020)
                        plt.show()
                        
                    
                    def plot_weights(nnet, lats, lons, basemap):
                        # plot maps of the NN weights
                        plt.figure(figsize=(16, 6))
                        ploti = 0
                        nUnitsFirstLayer = nnet.layers[0].nUnits
                        
                        for i in range(nUnitsFirstLayer):
                            ploti += 1
                            plt.subplot(np.ceil(nUnitsFirstLayer/3), 3, ploti)
                            maxWeightMag = nnet.layers[0].W[1:, i].abs().max().item() 
                            df.drawOnGlobe(((nnet.layers[0].W[1:, i]).cpu().data.numpy()).reshape(len(lats),
                                                                                                  len(lons)),
                                            lats,lons,basemap,vmin=-maxWeightMag,vmax=maxWeightMag,
                                            cmap=cmocean.cm.balance)
                            if(hiddens[0]==0):
                                plt.title('Linear Weights')
                            else:
                                plt.title('First Layer, Unit {}'.format(i+1))
                          
                        if(cascade is True and hiddens[0]!=0):
                            plt.figure(figsize=(16, 6))
                            ploti += 1
                            plt.subplot(np.ceil(nUnitsFirstLayer/3), 3, ploti)
                            maxWeightMag = nnet.layers[-1].W[1:Xtrain.shape[1]+1, 0].abs().max().item()
                            df.drawOnGlobe(((nnet.layers[-1].W[1:Xtrain.shape[1]+1, 0]).cpu().data.numpy()).reshape(len(lats),
                                                                                                                    len(lons)),
                                            lats,lons,basemap,vmin=-maxWeightMag,
                                            vmax=maxWeightMag,cmap=cmocean.cm.balance)
                            plt.title('Linear Weights')
                        plt.tight_layout()
                      
                    def plot_results(plots = 4): 
                        ### Calls all our plot functions together
                        global nnet,train_output,test_output,obs_output,Ttest,Ttrain,Xtrain_shape,Xtest_shape,data_train_shape,data_test_shape,Ttest_obs,lats,lons,basemap
                        
                        if plots >=1:
                            plot_prediction(Ttest, test_output, Ttest_obs, obs_output)
                        if plots >= 2:
                            plot_training_error(nnet)
                            plot_rmse(train_output, Ttrain, test_output, Ttest, data_train_shape, data_test_shape)
                        if plots == 4:
                            plot_weights(nnet, lats, lons, basemap)
                        plt.show()
                       
                    def plot_classifier_output(class_prob,test_class_prob,Xtest_shape,Xtrain_shape):
                        prob = class_prob[-1].reshape(Xtrain_shape)
                        
                        plt.figure(figsize=(14, 6))
                        plt.plot((np.arange(Xtest_shape[1]) + yearsall[sis].min()),
                                  prob[:,:,1].T, '-',alpha = .7)
                        plt.plot((np.arange(Xtest_shape[1]) + yearsall[sis].min()),
                                  (np.mean(prob[:, :, 1], axis = 0).reshape(180, -1)),
                                  'b-',linewidth=3.5, alpha = .5, label = 'ensemble avobsge')
                        plt.title('Classifier Output by Ensemble using Training Data')
                        plt.xlabel('year')
                        plt.yticks((0, 1), ['Pre-Baseline', 'Post-Baseline'])
                        plt.legend()
                        plt.show()
                    
                        tprob = test_class_prob[0].reshape(Xtest_shape)
                        
                        plt.figure(figsize=(14, 6))
                        plt.plot(((np.arange(Xtest_shape[1]) + yearsall[sis].min())),tprob[:,:,1].T,'-',
                                  alpha = .7)
                        plt.plot((np.arange(Xtest_shape[1]) + yearsall[sis].min()), 
                                  (np.mean(tprob[:, :, 1], axis = 0).reshape(180, -1)),
                                  'r-',linewidth=4,alpha = .5,label = 'ensemble avobsge')
                        plt.title('Classifier Output by Ensemble using Test Data')
                        plt.xlabel('year')
                        plt.yticks((0, 1), ['Pre-Baseline', 'Post-Baseline'])
                        plt.legend()
                        plt.show()
                        
                    def beginFinalPlot(YpredTrain,YpredTest,Ytrain,Ytest,testIndices,years,yearsObs,YpredObs):
                        """
                        Plot prediction of year
                        """
                        
                        # plt.rc('text',usetex=True)
                        # plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
                                
                        # fig = plt.figure()
                        # ax = plt.subplot(111)
                        
                        # adjust_spines(ax, ['left', 'bottom'])
                        # ax.spines['top'].set_color('none')
                        # ax.spines['right'].set_color('none')
                        # ax.spines['left'].set_color('dimgrey')
                        # ax.spines['bottom'].set_color('dimgrey')
                        # ax.spines['left'].set_linewidth(2)
                        # ax.spines['bottom'].set_linewidth(2)
                        # ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
                    
                        train_output_rs = YpredTrain.reshape(len(trainIndices),
                                                          len(years))
                        test_output_rs = YpredTest.reshape(len(testIndices),
                                                      len(years))
                    
                        xs_train = (np.arange(np.shape(train_output_rs)[1]) + yearsall[sis].min())
                        xs_test = (np.arange(np.shape(test_output_rs)[1]) + yearsall[sis].min())
                    
                        # for i in range(0,train_output_rs.shape[0]):
                        #     if i == train_output_rs.shape[0]-1:
                        #         plt.plot(xs_test,train_output_rs[i,:],'o',
                        #                     markersize=4,color='lightgray',clip_on=False,
                        #                     alpha=0.4,markeredgecolor='k',markeredgewidth=0.4,
                        #                     label=r'\textbf{%s - Training Data}' % singlesimulation)
                        #     else:
                        #         plt.plot(xs_test,train_output_rs[i,:],'o',
                        #                     markersize=4,color='lightgray',clip_on=False,
                        #                     alpha=0.4,markeredgecolor='k',markeredgewidth=0.4)
                        # for i in range(0,test_output_rs.shape[0]):
                        #     if i == test_output_rs.shape[0]-1:
                        #         plt.plot(xs_test,test_output_rs[i,:],'o',
                        #                 markersize=4,color='crimson',clip_on=False,alpha=0.3,
                        #                 markeredgecolor='crimson',markeredgewidth=0.4,
                        #                 label=r'\textbf{%s - Testing Data}' % singlesimulation)
                        #     else:
                        #         plt.plot(xs_test,test_output_rs[i,:],'o',
                        #                 markersize=4,color='crimson',clip_on=False,alpha=0.3,
                        #                 markeredgecolor='crimson',markeredgewidth=0.4)
                        
                        # if rm_ensemble_mean == False:
                        #     iy = np.where(yearsObs>=obsyearstart)[0]
                        #     plt.plot(yearsObs[iy],YpredObs[iy],'x',color='deepskyblue',
                        #               label=r'\textbf{Reanalysis}',clip_on=False)
                      
                        
                        # plt.xlabel(r'\textbf{ACTUAL YEAR}',fontsize=10,color='dimgrey')
                        # plt.ylabel(r'\textbf{PREDICTED YEAR}',fontsize=10,color='dimgrey')
                        # plt.plot(np.arange(yearsall[sis].min(),yearsall[sis].max()+1,1),np.arange(yearsall[sis].min(),yearsall[sis].max()+1,1),'-',
                        #           color='black',linewidth=2,clip_on=False)
                        
                        # plt.xticks(np.arange(yearsall[sis].min(),2101,20),map(str,np.arange(yearsall[sis].min(),2101,20)),size=6)
                        # plt.yticks(np.arange(yearsall[sis].min(),2101,20),map(str,np.arange(yearsall[sis].min(),2101,20)),size=6)
                        # plt.xlim([yearsall[sis].min(),yearsall[sis].max()])   
                        # plt.ylim([yearsall[sis].min(),yearsall[sis].max()])
                        
                        # plt.title(r'\textbf{[ %s ] $\bf{\longrightarrow}$ RMSE Train = %s; RMSE Test = %s}' % (variq,np.round(dSS.rmse(YpredTrain[:,],
                        #                                                                 Ytrain[:,0]),1),np.round(dSS.rmse(YpredTest[:,],
                        #                                                                                                       Ytest[:,0]),
                        #                                                                                                       decimals=1)),
                        #                                                                                                   color='k',
                        #                                                                                                   fontsize=15)
                        
                        # iyears = np.where(Ytest<1980)[0]
                        # plt.text(yearsall[sis].max(),yearsall[sis].min()+5, r'\textbf{Test RMSE before 1980 = %s}' % (np.round(dSS.rmse(YpredTest[iyears,],
                        #                                                                     Ytest[iyears,0]),
                        #                                                               decimals=1)),
                        #           fontsize=5,ha='right')
                        
                        # iyears = np.where(Ytest>=1980)[0]
                        # (np.round(dSS.rmse(YpredTest[iyears,],Ytest[iyears,0]),decimals=1))
                        # plt.text(yearsall[sis].max(),yearsall[sis].min(), r'\textbf{Test RMSE after 1980 = %s}' % (np.round(dSS.rmse(YpredTest[iyears,],
                        #                                                                       Ytest[iyears,0]),
                        #                                                                   decimals=1)),
                        #           fontsize=5,ha='right')
                        
                        # plt.legend(shadow=False,fontsize=7,loc='upper left',
                        #               bbox_to_anchor=(-0.01,1),fancybox=True,ncol=1,frameon=False,
                        #               handlelength=1,handletextpad=0.5)
                        # savefigName = modelType+'_'+variq+'_scatterPred_'+savename 
                        # plt.savefig(directoryfigure+savefigName+'_%s_land%s_ocean%s.png' % (monthlychoice,land_only,ocean_only),
                        #             dpi=300)      
                        print(np.round(np.corrcoef(yearsObs,YpredObs)[0,1],2),'= correlation for obs')
                        
                        #######################################################
                        #######################################################
                        #######################################################
                        #######################################################
                        #######################################################
                        #######################################################
                        #######################################################
                        #######################################################
                        #######################################################
                        ### Try considering order of years as parameter
                        slopetrain = np.empty((train_output_rs.shape[0]))
                        r_valuetrain = np.empty((train_output_rs.shape[0]))
                        for talin in range(train_output_rs.shape[0]):
                            slopetrain[talin],intercepttrain,r_valuetrain[talin],p_valuetrain,std_errtrain = stats.linregress(xs_train,train_output_rs[talin,:])
 
                        slopetest = np.empty((test_output_rs.shape[0]))
                        r_valuetest = np.empty((test_output_rs.shape[0]))
                        for telin in range(test_output_rs.shape[0]):
                            slopetest[telin],intercepttest,r_valuetest[telin],p_valuetest,std_errtest = stats.linregress(xs_test,test_output_rs[telin,:])

                        #######################################################
                        #######################################################
                        #######################################################
                        ### Try considering RMSE before/after 1980                         
                        iyearstesta = np.where(Ytest<1980)[0]
                        rma_test = np.round(dSS.rmse(YpredTest[iyearstesta,],Ytest[iyearstesta,0]),decimals=1)
                        
                        iyearstestb = np.where(Ytest>=1980)[0]
                        rmb_test = np.round(dSS.rmse(YpredTest[iyearstestb,],Ytest[iyearstestb,0]),decimals=1)
                        
                        #######################################################
                        #######################################################
                        #######################################################
                        ### Try considering RMSE for all years 
                        rmall_test = np.round(dSS.rmse(YpredTest[:],Ytest[:,0]),decimals=1)
                        
                        #######################################################
                        #######################################################
                        #######################################################
                        ### Try considering RMSE before/after 1980    
                        mae_testa = np.round(mean_absolute_error(YpredTest[iyearstesta],Ytest[iyearstesta,0]),1)
                        mae_testb = np.round(mean_absolute_error(YpredTest[iyearstestb],Ytest[iyearstestb,0]),1)
                        
                        #######################################################
                        #######################################################
                        #######################################################
                        ### Try considering MAE for all years 
                        mae_testall = np.round(mean_absolute_error(YpredTest[:],Ytest[:,0]),1)
                        
                        #######################################################
                        #######################################################
                        #######################################################
                        ### Stats on testing
                        meanslope_train = np.round(np.nanmean(slopetrain),3)
                        meanslope_test = np.round(np.nanmean(slopetest),3)
                        
                        meanr_train = np.round(np.nanmean(r_valuetrain)**2,3)
                        meanr_test = np.round(np.nanmean(r_valuetest)**2,3)

                        slopetrain_try.append(meanslope_train)
                        slopetest_try.append(meanslope_test)
                        meanr_train_try.append(meanr_train)
                        meanr_test_try.append(meanr_test)
                        
                        rma_test_try.append(rma_test)
                        rmb_test_try.append(rmb_test)
                        rmall_test_try.append(rmall_test)
                        
                        mae_testa_try.append(mae_testa)
                        mae_testb_try.append(mae_testb)
                        mae_testall_try.append(mae_testall)
                        
                        layers_trysave.append(layers_tryall[lay])
                        epochs_trysave.append(epochs_try[epo])
                        l2_trysave.append(l2_try[ridg])
                        
                        samples_trysave.append(isample)
                        models_trysave.append(singlesimulation)
                        ensembleseed_trysave.append(random_segment_seed)
                        annseed_trysave.append(random_network_seed)

                        #######################################################
                        #######################################################
                        #######################################################
                        ### Saving training and testing data predictions
                        # np.savetxt(directorydataoutput + 'training_%s.txt' % (singlesimulation),train_output_rs)
                        # np.savetxt('directorydataoutput + 'testing_%s.txt' % (singlesimulation),test_output_rs)
                        
                        #######################################################
                        #######################################################
                        #######################################################
                        #######################################################
                        #######################################################
                        #######################################################
                        #######################################################
                        #######################################################
                        #######################################################
                        return 
                    
                    ###############################################################################
                    ###############################################################################
                    ###############################################################################
                    ### Neural Network Creation & Training
                    
                    def movingAverageInputMaps(data,avgHalfChunk):
                        print(np.shape(data))
                        dataAvg = np.zeros(data.shape)
                        halfChunk = 2
                    
                        for iy in np.arange(0,data.shape[1]):
                            yRange = np.arange(iy-halfChunk,iy+halfChunk+1)
                            yRange[yRange<0] = -99
                            yRange[yRange>=data.shape[1]] = -99
                            yRange = yRange[yRange>=0]
                            dataAvg[:,iy,:,:] = np.nanmean(data[:,yRange,:,:],axis=1)
                        return dataAvg
                    
                    
                    class TimeHistory(keras.callbacks.Callback):
                        def on_train_begin(self, logs={}):
                            self.times = []
                    
                        def on_epoch_begin(self, epoch, logs={}):
                            self.epoch_time_start = time.time()
                    
                        def on_epoch_end(self, epoch, logs={}):
                            self.times.append(time.time() - self.epoch_time_start)
                    
                    def defineNN(hidden, input_shape, output_shape, ridgePenalty):        
                       
                        model = Sequential()
                        ### Initialize first layer
                        if hidden[0]==0:
                            ### Model is linear
                            model.add(Dense(1,input_shape=(input_shape,),
                                            activation='linear',use_bias=True,
                                            kernel_regularizer=regularizers.l1_l2(l1=0.00,l2=ridgePenalty),
                                            bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                                    kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
                            print('\nTHIS IS A LINEAR NN!\n')
                        else:
                            ### Model is a single node with activation function
                            model.add(Dense(hidden[0],input_shape=(input_shape,),
                                            activation=actFun, use_bias=True,
                                            kernel_regularizer=regularizers.l1_l2(l1=0.00,l2=ridgePenalty),
                                            bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                                    kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
                    
                            ### Initialize other layers
                            for layer in hidden[1:]:
                                model.add(Dense(layer,activation=actFun,
                                                use_bias=True,
                                                kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.00),
                                                bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                                    kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
                                
                            print('\nTHIS IS A ANN!\n')
                    
                        #### Initialize output layer
                        model.add(Dense(output_shape,activation=None,use_bias=True,
                                        kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.0),
                                        bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                                    kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
                    
                        ### Add softmax layer at the end
                        model.add(Activation('softmax'))
                        
                        return model
                    
                    def trainNN(model, Xtrain, Ytrain, niter=500, verbose=False):
                      
                        global lr_here, batch_size
                        lr_here = .01
                        model.compile(optimizer=optimizers.SGD(lr=lr_here,
                                                                momentum=0.9,nesterov=True),  #Adadelta .Adam()
                                      loss = 'binary_crossentropy',
                                      metrics=[metrics.categorical_accuracy],)
                    
                        ### Declare the relevant model parameters
                        batch_size = 32 # np.shape(Xtrain)[0] ### This doesn't seem to affect much in this case
                    
                        print('----ANN Training: learning rate = '+str(lr_here)+'; activation = '+actFun+'; batch = '+str(batch_size) + '----')    
                        time_callback = TimeHistory()
                        history = model.fit(Xtrain,Ytrain,batch_size=batch_size,epochs=niter,
                                            shuffle=True,verbose=verbose,
                                            callbacks=[time_callback],
                                            validation_split=0.)
                        print('******** done training ***********')
                    
                        return model, history
                    
                    def test_train_loopClass(Xtrain,Ytrain,Xtest,Ytest,iterations,ridge_penalty,hiddens,plot_in_train=True):
                        """or loops to iterate through training iterations, ridge penalty, 
                        and hidden layer list
                        """
                        results = {}
                        global nnet,random_network_seed
                      
                        for niter in iterations:
                            for penalty in ridge_penalty:
                                for hidden in hiddens:
                                    
                                    ### Check / use random seed
                                    if random_network_seed == None:
                                      np.random.seed(None)
                                      random_network_seed = int(np.random.randint(1, 100000))
                                    np.random.seed(random_network_seed)
                                    random.seed(random_network_seed)
                                    tf.set_random_seed(0)
                    
                                    ### Standardize the data
                                    Xtrain,Xtest,stdVals = dSS.standardize_data(Xtrain,Xtest)
                                    Xmean,Xstd = stdVals
                                    
                                    ### Define the model
                                    model = defineNN(hidden,
                                                      input_shape=np.shape(Xtrain)[1],
                                                      output_shape=np.shape(Ytrain)[1],
                                                      ridgePenalty=penalty)  
                                   
                                    ### Train the net
                                    model, history = trainNN(model,Xtrain,
                                                              Ytrain,niter=niter,verbose=0)
                    
                                    ### After training, use the network with training data to 
                                    ### check that we don't have any errors and output RMSE
                                    rmse_train = dSS.rmse(convert_fuzzyDecade_toYear(Ytrain,startYear,
                                                                                  classChunk),
                                                      convert_fuzzyDecade_toYear(model.predict(Xtrain),
                                                                                  startYear,
                                                                                  classChunk))
                                    if type(Ytest) != bool:
                                        rmse_test = 0.
                                        rmse_test = dSS.rmse(convert_fuzzyDecade_toYear(Ytest,
                                                                                    startYear,classChunk),
                                                          convert_fuzzyDecade_toYear(model.predict(Xtest),
                                                                                    startYear,
                                                                                    classChunk))
                                    else:
                                        rmse_test = False
                    
                                    this_result = {'iters': niter, 
                                                    'hiddens' : hidden, 
                                                    'RMSE Train' : rmse_train, 
                                                    'RMSE Test' : rmse_test, 
                                                    'ridge penalty': penalty, 
                                                    'zero mean' : rm_annual_mean,
                                                    'zero merid mean' : rm_merid_mean,
                                                    'land only?' : land_only,
                                                    'ocean only?' : ocean_only,
                                                    'Segment Seed' : random_segment_seed,
                                                    'Network Seed' : random_network_seed }
                                    results.update(this_result)
                    
                                    global experiment_result
                                    experiment_result = experiment_result.append(results,
                                                                                  ignore_index=True)
                    
                                    #if True to plot each iter's graphs.
                                    if plot_in_train == True:
                                        plt.figure(figsize = (16,6))
                    
                                        plt.subplot(1,2,1)
                                        plt.plot(history.history['loss'],label = 'training')
                                        plt.title(history.history['loss'][-1])
                                        plt.xlabel('epoch')
                                        plt.xlim(2,len(history.history['loss'])-1)
                                        plt.legend()
                    
                                        plt.subplot(1,2,2)
                                        
                                        plt.plot(convert_fuzzyDecade_toYear(Ytrain,startYear,
                                                                            classChunk),
                                                  convert_fuzzyDecade_toYear(model.predict(Xtrain),
                                                                            startYear,
                                                                            classChunk),'o',
                                                                              color='gray')
                                        plt.plot(convert_fuzzyDecade_toYear(Ytest,startYear,
                                                                            classChunk),
                                                  convert_fuzzyDecade_toYear(model.predict(Xtest),
                                                                            startYear,
                                                                            classChunk),'x', 
                                                                            color='red')
                                        plt.plot([startYear,yearsall[sis].max()],[startYear,yearsall[sis].max()],'--k')
                                        plt.yticks(np.arange(yearsall[sis].min(),yearsall[sis].max(),10))
                                        plt.xticks(np.arange(yearsall[sis].min(),yearsall[sis].max(),10))
                                        
                                        plt.grid(True)
                                        plt.show()
                    
                                    #'unlock' the random seed
                                    np.random.seed(None)
                                    random.seed(None)
                                    tf.set_random_seed(None)
                      
                        return experiment_result, model
                    
                    def convert_fuzzyDecade(data,startYear,classChunk):
                        years = np.arange(startYear-classChunk*2,yearsall[sis].max()+classChunk*2)
                        chunks = years[::int(classChunk)] + classChunk/2
                        
                        labels = np.zeros((np.shape(data)[0],len(chunks)))
                        
                        for iy,y in enumerate(data):
                            norm = stats.uniform.pdf(years,loc=y-classChunk/2.,scale=classChunk)
                            
                            vec = []
                            for sy in years[::classChunk]:
                                j=np.logical_and(years>sy,years<sy+classChunk)
                                vec.append(np.sum(norm[j]))
                            vec = np.asarray(vec)
                            vec[vec<.0001] = 0. # This should not matter
                            
                            vec = vec/np.sum(vec)
                            
                            labels[iy,:] = vec
                        return labels, chunks
                    
                    def convert_fuzzyDecade_toYear(label,startYear,classChunk):
                        years = np.arange(startYear-classChunk*2,yearsall[sis].max()+classChunk*2)
                        chunks = years[::int(classChunk)] + classChunk/2
                        
                        return np.sum(label*chunks,axis=1)
                    
                    def invert_year_output(ypred,startYear):
                        if(option4):
                            inverted_years = convert_fuzzyDecade_toYear(ypred,startYear,classChunk)
                        else:
                            inverted_years = invert_year_outputChunk(ypred,startYear)
                        
                        return inverted_years
                    
                    def invert_year_outputChunk(ypred,startYear):
                        
                        if(len(np.shape(ypred))==1):
                            maxIndices = np.where(ypred==np.max(ypred))[0]
                            if(len(maxIndices)>classChunkHalf):
                                maxIndex = maxIndices[classChunkHalf]
                            else:
                                maxIndex = maxIndices[0]
                    
                            inverted = maxIndex + startYear - classChunkHalf
                    
                        else:    
                            inverted = np.zeros((np.shape(ypred)[0],))
                            for ind in np.arange(0,np.shape(ypred)[0]):
                                maxIndices = np.where(ypred[ind]==np.max(ypred[ind]))[0]
                                if(len(maxIndices)>classChunkHalf):
                                    maxIndex = maxIndices[classChunkHalf]
                                else:
                                    maxIndex = maxIndices[0]
                                inverted[ind] = maxIndex + startYear - classChunkHalf
                        
                        return inverted
                    
                    ###############################################################################
                    ###############################################################################
                    ###############################################################################
                    ### Results
                        
                    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                                  inter_op_parallelism_threads=1)
                    
                    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
                    K.set_session(sess)
                    K.clear_session()
                    
                    ### Parameters
                    debug = True
                    NNType = 'ANN'
                    classChunkHalf = 5
                    classChunk = 10
                    iSeed = 8
                    avgHalfChunk = 0
                    option4 = True
                    biasBool = False
                    
                    ridge_penalty = [l2_try[ridg]]
                    iterations = [epochs_try[epo]]
                    hiddensList = [layers_tryall[lay]]
                    actFun = 'relu'
                    
                    expList = [(0)] # (0,1)
                    expN = np.size(expList)
                    random_segment = True
                    foldsN = 1
                    
                    for avgHalfChunk in (0,): # ([1,5,10]):#([1,2,5,10]):
                        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                                      inter_op_parallelism_threads=1)
                        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
                        K.set_session(sess)
                        # K.get_session()
                        K.clear_session()
                        
                        for loop in ([0]): # (0,1,2,3,4,5):
                            # get info about the region
                            lat_bounds,lon_bounds = UT.regions(reg_name)
                            data_all,lats,lons = read_primary_dataset(variq,dataset,
                                                                                  lat_bounds,
                                                                                  lon_bounds)
                            data_obs_all,lats_obs,lons_obs = read_obs_dataset(variq,dataset_obs,
                                                                                lat_bounds,
                                                                                lon_bounds)
                            test_output_mat = np.empty((np.max(expList)+1,
                                                        foldsN,180*int(np.round(np.shape(data_all)[0]*(1.0-segment_data_factor)))))
                    
                            for exp in expList:  
                                # get the data together
                                data, data_obs, = data_all, data_obs_all,
                                if rm_annual_mean == True:
                                    data, data_obs = dSS.remove_annual_mean(data,data_obs,
                                                                        lats,lons,
                                                                        lats_obs,lons_obs)
                                    print('*Removed annual mean*')
                    
                                if rm_merid_mean == True:
                                    data, data_obs = dSS.remove_merid_mean(data,data_obs,
                                                                        lats,lons,
                                                                        lats_obs,lons_obs)
                                    print('*Removed meridian mean*')  
                                if rm_ensemble_mean == True:
                                    data = dSS.remove_ensemble_mean(data)
                                    print('*Removed ensemble mean*')
                                    
                                if land_only == True:
                                    data, data_obs = dSS.remove_ocean(data,data_obs) 
            
                                if ocean_only == True:
                                    data, data_obs = dSS.remove_land(data,data_obs) 
                    
                                for ih in np.arange(0,len(hiddensList)):
                                    hiddens = [hiddensList[ih]]
                                    if hiddens[0][0]==0:
                                        annType = 'linear'
                                    elif hiddens[0][0]==1 and len(hiddens)==1:
                                        annType = 'layers1'
                                    else:
                                        annType = 'layers10x10'
                    
                                if(avgHalfChunk!=0):
                                    data = movingAverageInputMaps(data,avgHalfChunk)
                    
                            #     ### Loop over folds
                                for loop in np.arange(0,foldsN): 
                    
                                    K.clear_session()
                                    #---------------------------
                                    random_segment_seed = None #34515
                                    #---------------------------
                                    Xtrain,Ytrain,Xtest,Ytest,Xtest_shape,Xtrain_shape,data_train_shape,data_test_shape,testIndices = segment_data(data,segment_data_factor)
                    
                                    # Convert year into decadal class
                                    startYear = Ytrain[0] # define startYear for GLOBAL USE
                                    YtrainClassMulti, decadeChunks = convert_fuzzyDecade(Ytrain,
                                                                                          startYear,
                                                                                          classChunk)  
                                    YtestClassMulti, __ = convert_fuzzyDecade(Ytest,
                                                                              startYear,
                                                                              classChunk)  
                    
                                    # For use later
                                    XtrainS,XtestS,stdVals = dSS.standardize_data(Xtrain,Xtest)
                                    Xmean, Xstd = stdVals      
                    
                                    #---------------------------
                                    random_network_seed = None #87750
                                    #---------------------------
                    
                                    # Create and train network
                                    exp_result,model = test_train_loopClass(Xtrain,
                                                                            YtrainClassMulti,
                                                                            Xtest,
                                                                            YtestClassMulti,
                                                                            iterations=iterations,
                                                                            ridge_penalty=ridge_penalty,
                                                                            hiddens=hiddensList,
                                                                            plot_in_train=plot_in_train)
                                    model.summary()  
                                    
                                    ################################################################################################################################################                
                                    # save the model
                                    dirname = '/Users/zlabe/Documents/Research/InternalSignal/savedModels/'
                                    savename = modelType+'_'+variq+'_kerasMultiClassBinaryOption4_Chunk'+ str(classChunk)+'_' + NNType + '_L2_'+ str(ridge_penalty[0])+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(iterations[0]) + '_' + str(hiddensList[0][0]) + 'x' + str(hiddensList[0][-1]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
                                    savenameModelTestTrain = modelType+'_'+variq+'_modelTrainTest_SegSeed'+str(random_segment_seed)+'_NetSeed'+str(random_network_seed)
                    
                                    if(reg_name=='Globe'):
                                        regSave = ''
                                    else:
                                        regSave = '_' + reg_name
                                    
                                    if(rm_annual_mean==True):
                                        savename = savename + '_AnnualMeanRemoved' 
                                        savenameModelTestTrain = savenameModelTestTrain + '_AnnualMeanRemoved'
                                    if(rm_ensemble_mean==True):
                                        savename = savename + '_EnsembleMeanRemoved' 
                                        savenameModelTestTrain = savenameModelTestTrain + '_EnsembleMeanRemoved'
                                    if(avgHalfChunk!=0):
                                        savename = savename + '_avgHalfChunk' + str(avgHalfChunk)
                                        savenameModelTestTrain = savenameModelTestTrain + '_avgHalfChunk' + str(avgHalfChunk)
                    
                                    savename = savename + regSave    
                                    # model.save(dirname + savename + '.h5')
                                    # np.savez(dirname + savenameModelTestTrain + '.npz',trainModels=trainIndices,testModels=testIndices,Xtrain=Xtrain,Ytrain=Ytrain,Xtest=Xtest,Ytest=Ytest,Xmean=Xmean,Xstd=Xstd,lats=lats,lons=lons)
                    
                                    print('saving ' + savename)
                                    
                                    ###############################################################
                                    ### Make final plot
                                    ### Get obs
                                    dataOBSERVATIONS = data_obs
                                    latsOBSERVATIONS = lats_obs
                                    lonsOBSERVATIONS = lons_obs
                    
                                    def findStringMiddle(start,end,s):
                                        return s[s.find(start)+len(start):s.rfind(end)]
                    
                                    if(avgHalfChunk!=0):
                                        dataOBSERVATIONS = movingAverageInputMaps(dataOBSERVATIONS,avgHalfChunk)
                                    Xobs = dataOBSERVATIONS.reshape(dataOBSERVATIONS.shape[0],dataOBSERVATIONS.shape[1]*dataOBSERVATIONS.shape[2])
                                    yearsObs = np.arange(dataOBSERVATIONS.shape[0]) + obsyearstart
                    
                                    annType = 'class'
                                    if monthlychoice == 'DJF':
                                        startYear = yearsall[sis].min()+1
                                        endYear = yearsall[sis].max()
                                    else:
                                        startYear = yearsall[sis].min()
                                        endYear = yearsall[sis].max()
                                    years = np.arange(startYear,endYear+1,1)    
                                    Xmeanobs = np.nanmean(Xobs,axis=0)
                                    Xstdobs = np.nanstd(Xobs,axis=0)  
                                    
                                    XobsS = (Xobs-Xmeanobs)/Xstdobs
                                    XobsS[np.isnan(XobsS)] = 0
                                    
                                    if(annType=='class'):
                                        ### Chunk by individual year
                                        YpredObs = convert_fuzzyDecade_toYear(model.predict(XobsS),
                                                                              startYear,
                                                                              classChunk)
                                        YpredTrain = convert_fuzzyDecade_toYear(model.predict((Xtrain-Xmean)/Xstd),
                                                                                startYear,
                                                                                classChunk)
                                        YpredTest = convert_fuzzyDecade_toYear(model.predict((Xtest-Xmean)/Xstd),
                                                                                startYear,
                                                                                classChunk)
                                        
                                        ### Chunk by multidecadal
                                        Ytrainchunk = model.predict((Xtrain-Xmean)/Xstd)
                                        Ytestchunk = model.predict((Xtest-Xmean)/Xstd)
                                        YObschunk = model.predict(XobsS)
                    
                                        YtrainClassMulti = YtrainClassMulti
                                        YtestClassMulti = YtestClassMulti
                                        
                                    ### Create final plot
                                    beginFinalPlot(YpredTrain,YpredTest,Ytrain,Ytest,
                                                testIndices,years,
                                                yearsObs,YpredObs)
                                        
                    ##############################################################################
                    ##############################################################################
                    ##############################################################################
                    ### Analyzing parameters
            
                    ### Define variable for analysis
                    print('\n\n------------------------')
                    print(variq,'= Variable!')
                    print(monthlychoice,'= Time!')
                    print(reg_name,'= Region!')
                    print(lat_bounds,lon_bounds)
                    print(dataset,'= Model!')
                    print(dataset_obs,'= Observations!\n')
                    print(rm_annual_mean,'= rm_annual_mean') 
                    print(rm_merid_mean,'= rm_merid_mean') 
                    print(rm_ensemble_mean,'= rm_ensemble_mean') 
                    print(land_only,'= land_only')
                    print(ocean_only,'= ocean_only')                    
                    print('\n\n<<<<<<<<<< COMPLETED ITERATION = %s >>>>>>>>>>>\n\n' % (isample+1))

    slopetrain_tryModel.append(slopetrain_try)
    slopetest_tryModel.append(slopetest_try)
    meanr_train_tryModel.append(meanr_train_try)
    meanr_test_tryModel.append(meanr_test_try)
    
    rma_test_tryModel.append(rma_test_try)
    rmb_test_tryModel.append(rmb_test_try)
    rmall_test_tryModel.append(rmall_test_try)

    mae_testa_tryModel.append(mae_testa_try)
    mae_testb_tryModel.append(mae_testb_try)    
    mae_testall_tryModel.append(mae_testall_try)
    
    layers_trysaveModel.append(layers_trysave)
    epochs_trysaveModel.append(epochs_trysave)
    l2_trysaveModel.append(l2_trysave)
    
    samples_trysaveModel.append(samples_trysave)
    models_trysaveModel.append(models_trysave)
    ensembleseed_trysaveModel.append(ensembleseed_trysave)
    annseed_trysaveModel.append(annseed_trysave)
    
###############################################################################
###############################################################################
###############################################################################
### Save parameters for later analysis
# np.savetxt(directorydataoutput + 'slopeTrain_annual_R1.txt',slopetrain_tryModel)
# np.savetxt(directorydataoutput + 'slopeTest_annual_R1.txt',slopetest_tryModel)
# np.savetxt(directorydataoutput + 'r2Train_annual_R1.txt',meanr_train_tryModel)
# np.savetxt(directorydataoutput + 'r2Test_annual_R1.txt',meanr_test_tryModel)
# np.savetxt(directorydataoutput + 'rmsePRE_annual_R1.txt',rma_test_tryModel)
# np.savetxt(directorydataoutput + 'rmsePOST_annual_R1.txt',rmb_test_tryModel)

# np.savetxt(directorydataoutput + 'numOfLayers_annual_R1.txt',layers_trysaveModel)
# np.savetxt(directorydataoutput + 'numofEpochs_annual_R1.txt',epochs_trysaveModel)
# np.savetxt(directorydataoutput + 'L2_annual_R1.txt',l2_trysaveModel)

# np.savetxt(directorydataoutput + 'numOfSeedSamples_annual_R1.txt',samples_trysaveModel)
# np.savetxt(directorydataoutput + 'lensType_annual_R1.txt',models_trysaveModel)
# np.savetxt(directorydataoutput + 'EnsembleSegmentSeed_annual_R1.txt',ensembleseed_trysaveModel)
# np.savetxt(directorydataoutput + 'annSegmentSeed_annual_R1.txt',annseed_trysaveModel)

np.save(directorydataoutput + 'slopeTrain_annual_R1.npy',slopetrain_tryModel)
np.save(directorydataoutput + 'slopeTest_annual_R1.npy',slopetest_tryModel)
np.save(directorydataoutput + 'r2Train_annual_R1.npy',meanr_train_tryModel)
np.save(directorydataoutput + 'r2Test_annual_R1.npy',meanr_test_tryModel)

np.save(directorydataoutput + 'rmsePRE_annual_R1.npy',rma_test_tryModel)
np.save(directorydataoutput + 'rmsePOST_annual_R1.npy',rmb_test_tryModel)
np.save(directorydataoutput + 'rmseYEARS_annual_R1.npy',rmall_test_tryModel)

np.save(directorydataoutput + 'maePRE_annual_R1.npy',mae_testa_tryModel)
np.save(directorydataoutput + 'maePOST_annual_R1.npy',mae_testb_tryModel)
np.save(directorydataoutput + 'maeYEARS_annual_R1.npy',mae_testall_tryModel)

np.save(directorydataoutput + 'numOfLayers_annual_R1.npy',layers_trysaveModel)
np.save(directorydataoutput + 'numofEpochs_annual_R1.npy',epochs_trysaveModel)
np.save(directorydataoutput + 'L2_annual_R1.npy',l2_trysaveModel)

np.save(directorydataoutput + 'numOfSeedSamples_annual_R1.npy',samples_trysaveModel)
np.save(directorydataoutput + 'lensType_annual_R1.npy',models_trysaveModel)
np.save(directorydataoutput + 'EnsembleSegmentSeed_annual_R1.npy',ensembleseed_trysaveModel)
np.save(directorydataoutput + 'annSegmentSeed_annual_R1.npy',annseed_trysaveModel)