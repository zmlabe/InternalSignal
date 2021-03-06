"""
Test model to understand if ANN can detect a forced signal using only internal
variability from LENS and observations from BEST

Reference  : Barnes et al. [2020, JAMES preprint on ArXiv]
Author    : Zachary M. Labe
Date      : 15 June 2018
"""

### Import packages
import math
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
from keras.layers import Dense, Activation
from keras import regularizers
from keras import metrics
from keras import optimizers
from keras.utils import to_categorical
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import innvestigate
import random
from sklearn.preprocessing import MultiLabelBinarizer
import scipy.stats as stats
import copy as copy
import cartopy as ct
import cmocean as cmocean
import cmocean.plots
from mpl_toolkits.axes_grid1 import make_axes_locatable
import calc_Utilities as UT
import calc_dataFunctions as df
import cmocean

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

### Prevent tensorflow 2.+ deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directorydataLL = '/Users/zlabe/Data/LENS/monthly/'
directorydataBB = '/Users/zlabe/Data/BEST/'
directoryfigure = '/Users/zlabe/Desktop/TestIntSignal/'

###############################################################################
###############################################################################
###############################################################################
### ANN preliminaries 
experiment_result = pd.DataFrame(columns=['actual iters','hiddens','cascade',
                                          'RMSE Train','RMSE Test',
                                          'ridge penalty','zero mean',
                                          'zero merid mean','land only?'])

### Define primary dataset to use
dataset = 'lens'

### Name of the region of interest
reg_name = 'Globe'
lat_bounds, lon_bounds = UT.regions(reg_name)

### whether to test and plot the results using BEST data
test_on_obs = True
dataset_obs = 'best'
year_obs = np.arange(1956,2019+1,1)

### Remove the annual mean? True to subtract it from dataset
rm_annual_mean = False

### Remove the merid mean? True to subtract it from dataset
rm_merid_mean = False

### Use land locations only
land_only = False

### Split the data into training and testing sets? value of 1 will use all 
### data as training, .8 will use 80% training, 20% testing; etc.
segment_data_factor = .8

### iterations is for the # of sample runs the model will use.  Must be a 
### list, but can be a list with only one object
iterations = [150]

### Hiddens corresponds to the number of hidden layers the nnet will use - 0 
### for linear model, or a list [10, 20, 5] for multiple layers of nodes 
### (10 nodes in first layer, 20 in second, etc); The "loop" part 
### allows you to loop through multiple architectures. For example, 
### hiddens_loop = [[2,4],[0],[1 1 1]] would produce three separate NNs, the 
### first with 2 hidden layers of 2 and 4 nodes, the next the linear model,
### and the next would be 3 hidden layers of 1 node each.
hiddens = [[5], [5, 5]]

### List of different ridge penalties to iterate through. Must be a list, but 
### can be a list with only one object
ridge_penalty = [100] #+ list(range(1000,))

### Set useGPU to True to use the GPU, but only if you selected the GPU 
### Runtime in the menu at the top of this page
useGPU = False

### Set Cascade to True to utilize the nnet's cascade function
cascade = False

### Plot within the training loop - may want to set to False when testing out 
### larget sets of parameters
plot_in_train = False

### Number of plots returned (valid values are 0, 1,2 or 4 (maps))
plots = 4

### Colors
train_color = 'navy'
test_color = 'fuchsia'
obs_color = 'darkorange'
map_colors = cmocean.cm.balance

###############################################################################
###############################################################################
###############################################################################
### Create utility functions

def rmse(a,b):
    '''calculates the root mean squared error
    takes two variables, a and b, and returns value'''
    return np.sqrt(np.mean((a - b)**2))

def read_primary_dataset(dataset,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
    data,lats,lons,ensmean = df.readFiles(dataset)
    data,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    ensmean,lats,lons = df.getRegion(ensmean,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return data,lats,lons,ensmean
  
def read_obs_dataset(dataset_obs,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
    data_obs,lats_obs,lons_obs,ensmean = df.readFiles(dataset_obs)
    data_obs,lats_obs,lons_obs = df.getRegion(data_obs,lats_obs,lons_obs,
                                            lat_bounds,lon_bounds)
    print('our OBS dataset: ',dataset_obs,' is shaped',data_obs.shape)
    return data_obs,lats_obs,lons_obs

def remove_annual_mean(data,data_obs,lats,lons,lats_obs,lons_obs):
    lons2,lats2 = np.meshgrid(lons,lats)
    lons2_obs,lats2_obs = np.meshgrid(lons_obs,lats_obs)
    data = data - UT.calc_weightedAve(data,lats2)[:,:,np.newaxis,np.newaxis]
    data_obs = data_obs - UT.calc_weightedAve(data_obs,lats2_obs)[:,np.newaxis,np.newaxis]
    return data,data_obs

def remove_merid_mean(data, data_obs):
  data = data - np.nanmean(data,axis=2)[:,:,np.newaxis,:]
  data_obs = data_obs - np.nanmean(data_obs,axis=1)[:,np.newaxis,:]
  return data, data_obs

def segment_data(data,fac = segment_data_factor):
  
    global random_segment_seed,trainIndices, estIndices
    if random_segment_seed == None:
        random_segment_seed = int(int(np.random.randint(1, 100000)))
    np.random.seed(random_segment_seed)
    
    if fac < 1 :
        nrows = data.shape[0]
        segment_train = int(np.round(nrows * fac))
        segment_test = nrows - segment_train
        print('Training on',segment_train, 'ensembles, testing on',segment_test)

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
            print('org data', data.shape)
            print('training data', data_train.shape)
    
        ### Reshape into X and T
        Xtrain = data_train.reshape((data_train.shape[0] * data_train.shape[1]),
                                    (data_train.shape[2] * data_train.shape[3]))
        Ttrain = np.tile((np.arange(data_train.shape[1]) + 1920).reshape(data_train.shape[1],1),
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
        Ttest = np.tile((np.arange(data_test.shape[1]) + 1920).reshape(data_test.shape[1],1),
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
        Ttrain = np.tile((np.arange(data_train.shape[1]) + 1920).reshape(data_train.shape[1],1),
                         (data_train.shape[0],1))
        Xtrain_shape = (data_train.shape[0], data_train.shape[1])

    Xtest = data_test.reshape((data_test.shape[0] * data_test.shape[1]),
                              (data_test.shape[2] * data_test.shape[3]))
    Ttest = np.tile((np.arange(data_test.shape[1]) + 1920).reshape(data_test.shape[1],1),
                    (data_test.shape[0],1))

    Xtest_shape = (data_test.shape[0], data_test.shape[1])
    data_train_shape = data_train.shape[1]
    data_test_shape = data_test.shape[1]
  
    ### 'unlock' the random seed
    np.random.seed(None)
  
    return Xtrain,Ttrain,Xtest,Ttest,Xtest_shape,Xtrain_shape,data_train_shape,data_test_shape,testIndices
  
  
def shape_obs(data_obs,year_obs):
    Xtest_obs = np.reshape(data_obs,(data_obs.shape[0],
                                     (data_obs.shape[1]*data_obs.shape[2])))
    Ttest_obs = np.tile(np.arange(data_obs.shape[0])+year_obs[0])
    return Xtest_obs,Ttest_obs

###############################################################################
###############################################################################
###############################################################################
### Data management

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

def drawOnGlobe(axes,data,lats,lons,region,cmap='coolwarm',vmin=None,vmax=None,inc=None):
    '''Usage: drawOnGlobe(data, lats, lons, basemap, cmap)
          data: nLats x nLons
          lats: 1 x nLats
          lons: 1 x nLons
          basemap: returned from getRegion
          cmap
          vmin
          vmax'''

    data_cyc, lons_cyc = data, lons
    ### Fixes white line by adding point
    data_cyc, lons_cyc = ct.util.add_cyclic_point(data, coord=lons)
   
    image = plt.pcolormesh(lons_cyc,lats,data_cyc,transform=ct.crs.PlateCarree(),
                           vmin=vmin,vmax=vmax,cmap=cmap,shading='flat')

    axes.coastlines(color = 'black', linewidth = 1.2)

    divider = make_axes_locatable(axes)
    ax_cb = divider.new_horizontal(size="2%", pad=0.1, axes_class=plt.Axes)

    plt.gcf().add_axes(ax_cb)
#     if vmin is not None:
#       cb = plt.colorbar(image, cax=ax_cb,
#                           boundaries=np.arange(vmin,vmax+inc,inc))
#     else:
#       cb = plt.colorbar(image, cax=ax_cb)
    cb = plt.colorbar(image, cax=ax_cb)  
    #cb.set_label('units', fontsize=20)

    plt.sca(axes)  # in case other calls, like plt.title(...), will be made
    
    ### Return image
    return cb,image

plt.rcdefaults()

def plot_prediction (Ttest, test_output, Ttest_obs, obs_output):
    ### Predictions
    
    plt.figure(figsize=(16,4))
    plt.subplot(1, 2, 1)
    plt.title('Predicted vs Actual Year for Testing')
    plt.xlabel('Actual Year')
    plt.ylabel('Predicted Year')
    plt.plot(Ttest, test_output, 'o', color='black', label='GCM')
      
    if test_on_obs == True:
      plt.plot(Ttest_obs, obs_output, 'o', color=obs_color, label='obs-BEST')
    a = min(min(Ttest), min(test_output))
    b = max(max(Ttest), max(test_output))
    plt.plot((a,b), (a,b), '-', lw=3, alpha=0.7, color='gray')
    #plt.axis('square')
    plt.xlim(a * .995, b * 1.005)
    plt.ylim(a * .995, b * 1.005)
    plt.legend()
    plt.show()
  
def plot_training_error(nnet):
    #training error (nnet)
  
    plt.subplot(1, 2, 2)
    plt.plot(nnet.getErrors(), color='black')
    plt.title('Training Error per Itobstion')
    plt.xlabel('Training Itobstion')
    plt.ylabel('Training Error')
    plt.show()
 
def plot_rmse(train_output,Ttrain,test_output,Ttest,data_train_shape,data_test_shape):
    #rmse (train_output, Ttrain, test_output, Ttest, data_train_shape, data_test_shape)
  
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 2, 1)
    rmse_by_year_train = np.sqrt(np.mean(((train_output - Ttrain)**2).reshape(Xtrain_shape),
                                         axis=0))
    xs_train = (np.arange(data_train_shape) + 1920)
    rmse_by_year_test = np.sqrt(np.mean(((test_output - Ttest)**2).reshape(Xtest_shape),
                                        axis=0))
    xs_test = (np.arange(data_test_shape) + 1920)
    plt.title('RMSE by year')
    plt.xlabel('year')
    plt.ylabel('error')
    plt.plot(xs_train,rmse_by_year_train,label = 'training error',
             color=train_color,linewidth=1.5)
    plt.plot(xs_test,rmse_by_year_test,labe ='test error',
             color=test_color,linewidth=1.)
    plt.legend()

    if test_on_obs == True:
        plt.subplot(1,2,2)
        error_by_year_test_obs = obs_output - Ttest_obs
        plt.plot(Ttest_obs,error_by_year_test_obs,label='obs-BEST error',
             color=obs_color,linewidth=2.)            
        plt.title('Error by year for obs-BEST')
        plt.xlabel('year')
        plt.ylabel('error')
        plt.legend()
        plt.plot((1975,2020), (0,0), color='gray', linewidth=2.)
        plt.xlim(1975,2020)
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
                       cmap=map_colors)
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
                       vmax=maxWeightMag,cmap=map_colors)
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
    display(results)
   
def plot_classifier_output(class_prob,test_class_prob,Xtest_shape,Xtrain_shape):
    prob = class_prob[-1].reshape(Xtrain_shape)
    
    plt.figure(figsize=(14, 6))
    plt.plot((np.arange(Xtest_shape[1]) + 1920),
             prob[:,:,1].T, '-',alpha = .7)
    plt.plot((np.arange(Xtest_shape[1]) + 1920),
             (np.mean(prob[:, :, 1], axis = 0).reshape(180, -1)),
             'b-',linewidth=3.5, alpha = .5, label = 'ensemble avobsge')
    plt.title('Classifier Output by Ensemble using Training Data')
    plt.xlabel('year')
    plt.yticks((0, 1), ['Pre-Baseline', 'Post-Baseline'])
    plt.legend()
    plt.show()

    tprob = test_class_prob[0].reshape(Xtest_shape)
    
    plt.figure(figsize=(14, 6))
    plt.plot(((np.arange(Xtest_shape[1]) + 1920)),tprob[:,:,1].T,'-',
             alpha = .7)
    plt.plot((np.arange(Xtest_shape[1]) + 1920), 
             (np.mean(tprob[:, :, 1], axis = 0).reshape(180, -1)),
             'r-',linewidth=4,alpha = .5,label = 'ensemble avobsge')
    plt.title('Classifier Output by Ensemble using Test Data')
    plt.xlabel('year')
    plt.yticks((0, 1), ['Pre-Baseline', 'Post-Baseline'])
    plt.legend()
    plt.show()

###############################################################################
###############################################################################
###############################################################################
### Neural Network Creation & Training
import time

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

def standardize_data(Xtrain,Xtest):

    Xmean = np.nanmean(Xtrain,axis=0)
    Xstd = np.nanstd(Xtrain,axis=0)
    Xtest = (Xtest - Xmean)/Xstd
    Xtrain = (Xtrain - Xmean)/Xstd
    
    global stdVals
    stdVals = (Xmean,Xstd)
    stdVals = stdVals[:]
    
    return Xtrain, Xtest

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def defineNN(hidden, input_shape, output_shape, ridgePenalty = 0.):        
   
    model = Sequential()
    ### Initialize first layer
    if hidden[0]==0:
        ### Model is linear
        model.add(Dense(1,input_shape=(input_shape,),
                        activation='linear',use_bias=True,
                        kernel_regularizer=regularizers.l1_l2(l1=0.00,l2=ridgePenalty),
                        bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
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
#     model.compile(optimizer=optimizers.SGD(lr=lr_here, momentum=0.9, nesterov=True),  #Adadelta .Adam()
#                  loss = 'categorical_crossentropy',
#                  metrics=[metrics.categorical_accuracy],)

#     model.compile(optimizer=optimizers.Adam(0.001),
#                  loss = 'categorical_crossentropy',
#                  metrics=[metrics.categorical_accuracy],)

    ### Declare the relevant model parameters
    batch_size = 32 #np.shape(Xtrain)[0] ### This doesn't seem to affect much in this case

    print('----ANN Training: learning rate = '+str(lr_here)+'; activation = '+actFun+'; batch = '+str(batch_size) + '----')    
    time_callback = TimeHistory()
    history = model.fit(Xtrain,Ytrain,batch_size=batch_size,epochs=niter,
                        shuffle=True,verbose=verbose,
                        callbacks=[time_callback],
                        validation_split=0.)
    print('******** done training ***********')

    return model, history

def test_train_loopClass(Xtrain,Ytrain,Xtest,Ytest,iterations,ridge_penalty,hiddens,plot_in_train=True):
    '''for loops to iterate through training iterations, ridge penalty, and hidden layer list'''
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
#                 print('not standardizing the data here')
                Xtrain,Xtest = standardize_data(Xtrain,Xtest)
                Xmean, Xstd = stdVals
                
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
                rmse_train = rmse(convert_fuzzyDecade_toYear(Ytrain,startYear,
                                                             classChunk),
                                  convert_fuzzyDecade_toYear(model.predict(Xtrain),
                                                             startYear,
                                                             classChunk))
                if type(Ytest) != bool:
                    rmse_test = 0.
                    rmse_test = rmse(convert_fuzzyDecade_toYear(Ytest,
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
                                'Segment Seed' : random_segment_seed,
                                'Network Seed' : random_network_seed }
                results.update(this_result)

                global experiment_result
                experiment_result = experiment_result.append(results, ignore_index=True)
                display(experiment_result.tail(1))

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
                    plt.plot([startYear,2100],[startYear,2100],'--k')
                    plt.yticks(np.arange(1920,2100,10))
                    plt.xticks(np.arange(1920,2100,10))
                    
                    plt.grid(True)
                    plt.show()

                #'unlock' the random seed
                np.random.seed(None)
                random.seed(None)
                tf.set_random_seed(None)
  
    return experiment_result, model

def makeScatter(ax,train_output_rs,test_output_rs, testColor=''):
    MS0 = 3
    MS1 = 3
    FS = 10

    #--------------------------------------------------------------------------
    #Predicted plots
    xs_test = (np.arange(np.shape(test_output_rs)[1]) + 1920)

    ax.set_xlabel('actual year', fontsize = FS)
    #ax.xaxis.label.set_size(FS)
    ax.set_ylabel('predicted year',fontsize = FS)
    ax.plot([1800,2130],[1800,2130],'-',color='black',linewidth=2)
    ax.set_xticks(np.arange(1820,2130,20))
    ax.set_yticks(np.arange(1820,2130,20))

    p=plt.plot(xs_test,train_output_rs[0,:],'o',
               markersize=MS1,color = 'gray',label='GCM training data')
    p=plt.plot(xs_test,test_output_rs[0,:],'o',
               markersize=MS0,color = 'black',label='GCM testing data')

    for i in np.arange(0,np.shape(train_output_rs)[0]):
        p=plt.plot(xs_test,train_output_rs[i,:],'o',
                   markersize=MS1,
                   color='gray')
        clr = p[0].get_color()

    for i in np.arange(0,np.shape(test_output_rs)[0]):
        if not testColor:
            p=plt.plot(xs_test,test_output_rs[i,:],'o',markersize=MS0)
        else:
            p=plt.plot(xs_test,test_output_rs[i,:],'o',
                       markersize=MS0,color=testColor)
        clr = p[0].get_color()
  
    plt.xlim(1910,2110)   
    plt.ylim(1910,2110)
    plt.legend(fontsize=FS)
    plt.grid(True)

def convert_fuzzyDecade(data,startYear,classChunk):
    years = np.arange(startYear-classChunk*2,2100+classChunk*2)
    chunks = years[::int(classChunk)] + classChunk/2
    
    labels = np.zeros((np.shape(data)[0],len(chunks)))
    
    for iy,y in enumerate(data):
        norm = stats.uniform.pdf(years,loc=y-classChunk/2.,scale=classChunk)
        
        vec = []
        for sy in years[::classChunk]:
            j=np.logical_and(years>sy,years<sy+classChunk)
            vec.append(np.sum(norm[j]))
        vec = np.asarray(vec)
        vec[vec<.0001] = 0. #this should not matter
        
#         plt.figure(figsize=(3,3))
#         plt.plot(chunks,vec)
#         plt.xlim(1920,1990)
#         plt.show()
#         print(np.sum(vec))
        
        vec = vec/np.sum(vec)
        
        labels[iy,:] = vec
    return labels, chunks

def convert_fuzzyDecade_toYear(label,startYear,classChunk):
    years = np.arange(startYear-classChunk*2,2100+classChunk*2)
    chunks = years[::int(classChunk)] + classChunk/2
    
    return np.sum(label*chunks,axis=1)

# index = 2
# val, decadeChunks = convert_fuzzyDecade(Ytrain[index],startYear,classChunk=10)
# pred = convert_fuzzyDecade_toYear(val,startYear,classChunk=10)
# print(Ytrain[index],pred)
# print(val)
# print(decadeChunks)
# YtrainClass, decadeChunks = convert_fuzzyDecade(Ytrain,startYear,classChunk=10)
# outval = convert_fuzzyDecade_toYear(YtrainClass,startYear,classChunk=10)
# print(outval[2])
# print(Ytrain[2])

# plt.figure(figsize=(3,3))
# plt.plot(Ytrain,outval,'.',color='gray')
# plt.plot([1920,2100],[1920,2100],'--',color='orange')
# plt.show()

###############################################################################
###############################################################################
###############################################################################
### Results
# print(Error)


### Figure 1: pred/actual curves (1-1)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
from keras import backend as K
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
# K.get_session()
K.clear_session()
#tf.Session.close
#==============================================================================
debug = True
reg_name = 'Globe'
actFun = 'relu'
ridge_penalty = [0.01] # .01
classChunk = 10

hiddensList = [[20,20]]  # linear model
hiddensList = [[1]]  # one hidden layer and node
hiddensList = [[20,20]] 

expList = [(0)] # (0,1)
expN = np.size(expList)

iterations = [500] # [500]#[1500]
random_segment = True
foldsN = 1
#==============================================================================

for avgHalfChunk in (0,):#([1,5,10]):#([1,2,5,10]):
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    # K.get_session()
    K.clear_session()
    
    for loop in ([0]):#(0,1,2,3,4,5):

        #------------------------------------------------
        #define primary dataset to use

        if(loop==0):
            modelType = 'lens'
            var = 'T2M'
            dataset = 'lens'
            dataset_obs = 'best'
        else:
            ValueError('no such dataset')
        #------------------------------------------------

        # get info about the region
        lat_bounds,lon_bounds = UT.regions(reg_name)
        data_all,lats,lons,ensmean = read_primary_dataset(dataset,
                                                             lat_bounds,
                                                             lon_bounds)
        data_obs_all,lats_obs,lons_obs = read_obs_dataset(dataset_obs,
                                                            lat_bounds,
                                                            lon_bounds)
        test_output_mat = np.empty((np.max(expList)+1,
                                    foldsN,180*int(np.round(np.shape(data_all)[0]*(1.0-segment_data_factor)))))

        for exp in expList:  

            # loop over experiments
            if exp == 0:
                rm_annual_mean == True
                rm_merid_mean = False
                land_only = False

            # get the data together
            data, data_obs, = data_all, data_obs_all,
            if rm_annual_mean == True:
                data, data_obs = remove_annual_mean(data,data_obs,
                                                    lats,lons,
                                                    lats_obs,lons_obs)
                print('removed annual mean')

            if rm_merid_mean == True:
                data, data_obs = remove_merid_mean(data,data_obs,
                                                   lats,lons,
                                                   lats_obs,lons_obs)
                print('removed meridian mean')    

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

            ### Loop over folds
            for loop in np.arange(0,foldsN): 

                K.clear_session()
                #---------------------------
                random_segment_seed = 34515#45827#96715#45827#96715#45827#96715#None#45827#None#45827#45827#81558#None
#                 segment_data_factor = 1.
                #---------------------------
                Xtrain,Ytrain,Xtest,Ytest,Xtest_shape,Xtrain_shape,data_train_shape,data_test_shape,testIndices = segment_data(data,segment_data_factor)

                # convert year into decadal class
                startYear = Ytrain[0] # define startYear for GLOBAL USE
                YtrainClassMulti, decadeChunks = convert_fuzzyDecade(Ytrain,
                                                                     startYear,
                                                                     classChunk)  
                YtestClassMulti, __ = convert_fuzzyDecade(Ytest,
                                                          startYear,
                                                          classChunk)  

                # for use later
                XtrainS,XtestS = standardize_data(Xtrain,Xtest)
                Xmean, Xstd = stdVals      

                #---------------------------
                random_network_seed = 87750#None#84256#53985#84256#53985#None#84256#None#84256#84256#8453#None#None
                tf.set_random_seed(0)
                #---------------------------

                #create and train network
                exp_result,model = test_train_loopClass(Xtrain,
                                                        YtrainClassMulti,
                                                        Xtest,
                                                        YtestClassMulti,
                                                        iterations=iterations,
                                                        ridge_penalty=ridge_penalty,
                                                        hiddens=hiddensList,
                                                        plot_in_train = True)
                model.summary()  
                
                ################################################################################################################################################                
                # save the model
                dirname = '/Users/zlabe/Documents/Research/InternalSignal/Scripts/savedModels/'+var +'/'
                savename = modelType+'_'+var+'_kerasMultiClassBinaryOption4_Chunk'+ str(classChunk)+'_Linear_L2_'+ str(ridge_penalty[0])+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(iterations[0]) + '_' + str(hiddensList[0][0]) + 'x' + str(hiddensList[0][-1]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'                     + str(random_network_seed) 
                savenameModelTestTrain = modelType+'_'+var+'_modelTrainTest_SegSeed'+str(random_segment_seed)+'_NetSeed'+str(random_network_seed)

                if(reg_name=='Globe'):
                    regSave = ''
                else:
                    regSave = '_' + reg_name
                
                if(rm_annual_mean==True):
                    savename = savename + '_meanRemoved' 
                    savenameModelTestTrain = savenameModelTestTrain + '_meanRemoved'
                if(avgHalfChunk!=0):
                    savename = savename + '_avgHalfChunk' + str(avgHalfChunk)
                    savenameModelTestTrain = savenameModelTestTrain + '_avgHalfChunk' + str(avgHalfChunk)

                savename = savename + regSave    
                model.save(dirname + savename + '.h5')
                np.savez(dirname + savenameModelTestTrain + '.npz',trainModels=trainIndices,testModels=testIndices,Xtrain=Xtrain,Ytrain=Ytrain,Xtest=Xtest,Ytest=Ytest,Xmean=Xmean,Xstd=Xstd,lats=lats,lons=lons)

                print('saving ' + savename)
                
                ###############################################################
                ### Make final plot
                ### Get obs
                dataBEST = data_obs
                latsBEST = lats_obs
                lonsBEST = lons_obs

                def findStringMiddle(start,end,s):
                    return s[s.find(start)+len(start):s.rfind(end)]

                if(var=='T2M'):
                    if(avgHalfChunk!=0):
                        dataBEST = movingAverageInputMaps(dataBEST,avgHalfChunk)
                    Xobs = dataBEST.reshape(dataBEST.shape[0],dataBEST.shape[1]*dataBEST.shape[2])
                    yearsObs = np.arange(dataBEST.shape[0]) + 1956
                # if(rm_annual_mean==True):
                #     Xobs = Xobs - np.nanmean(Xobs,axis=1)[:,np.newaxis]

                annType = 'class'
                startYear = 1920
                endYear = 2100
                years = np.arange(startYear,endYear+1,1)                    
                XobsS = (Xobs-Xmean)/Xstd
                XobsS[np.isnan(XobsS)] = 0
                
                if(annType=='class'):
                    YpredObs = convert_fuzzyDecade_toYear(model.predict(XobsS),
                                                          startYear,
                                                          classChunk)
                    YpredObsPDF = model.predict(XobsS)
                # elif(annType=='reg'):    
                #     YpredObs = model.predict(XobsS)*Ystd + Ymean
                #     YpredObsPDF = np.nan

                if(annType=='class'):
                    YpredTrain = convert_fuzzyDecade_toYear(model.predict((Xtrain-Xmean)/Xstd),
                                                            startYear,
                                                            classChunk)
                    YpredTest = convert_fuzzyDecade_toYear(model.predict((Xtest-Xmean)/Xstd),
                                                           startYear,
                                                           classChunk)
                # elif(annType=='reg'):
                #     YpredTrain = model.predict((Xtrain-Xmean)/Xstd)*Ystd + Ymean
                #     YpredTest = model.predict((Xtest-Xmean)/Xstd)*Ystd + Ymean

                ###############################################################
                plt.figure(figsize=(7,5))
                ax = plt.subplot(1,1,1)
                makeScatter(ax,YpredTrain.reshape(len(trainIndices),
                                                  len(years)),
                            YpredTest.reshape(len(testIndices),
                                              len(years)))
                plt.title('LENS-'+var+'\nRMSE Train = '+ str(np.round(rmse(YpredTrain[:,],
                                                                               Ytrain[:,0]),
                                                                    decimals=1))+ '; RMSE Test = '+str(np.round(rmse(YpredTest[:,],
                                                                                                                     Ytest[:,0]),
                                                                                                                     decimals=1)))

                iyears = np.where(Ytest<1980)[0]
                plt.text(2064,1922, 'Test RMSE before 1980 = ' + str(np.round(rmse(YpredTest[iyears,],
                                                                                   Ytest[iyears,0]),
                                                                              decimals=1)),
                         fontsize=5)
                iyears = np.where(Ytest>=1980)[0]
                plt.text(2064,1922+5, 'Test RMSE after   1980 = ' + str(np.round(rmse(YpredTest[iyears,],
                                                                                      Ytest[iyears,0]),
                                                                                 decimals=1)),
                         fontsize=5)

                iy = np.where(yearsObs>=1956)[0]
                plt.plot(yearsObs[iy],YpredObs[iy],'o',color='black',markerfacecolor='white',
                         markeredgecolor = 'black',
                         markeredgewidth=1.5,
                         label='BEST Observations')

                plt.legend()
                savefigName = modelType+'_'+var+'_scatterPred_'+savename 
                plt.annotate(savename,(0,.98),xycoords='figure fraction',
                             fontsize=5,
                             color='gray')
                plt.savefig(directoryfigure+var+'/'+savefigName+'.png',
                            dpi=300,bbox_inches = 'tight')
                plt.show()        
                print(np.round(np.corrcoef(yearsObs[iy],YpredObs[iy])[0,1],2))
                
### Next to Run
# * temp, 300, 10x10, L2 = 0.01, mean retained

model.summary()
model.layers[0].get_config()

# test_output = YpredTest
# Ttrain = Ytrain
# Ttest = Ytest
# Ttest_obs = YpredObs
# obs_output = yearsObs
# nnet = model
# plot_results(plots = 1)




