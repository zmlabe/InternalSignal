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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.preprocessing import MultiLabelBinarizer
import scipy.stats as stats
import copy as copy
import cartopy as ct
import cmocean as cmocean
import cmocean.plots
from mpl_toolkits.axes_grid1 import make_axes_locatable
import calc_Utilities as UT
import read_LENS as LL
import read_BEST as BB

### Preliminaries 
directorydataLL = '/Users/zlabe/Data/LENS/monthly/'
directorydataBB = '/Users/zlabe/Data/BEST/'
directoryfigure = '/Users/zlabe/Desktop/TestIntSignal/'
datareader = True

if datareader == True:
    ### Read in observations
    sliceperiodBB = 'annual'
    sliceyearBB = np.arange(1956,2019+1,1)
    sliceshapeBB = 3
    slicenanBB = 'nan'
    addclimoBB = True
    latb,lonb,best = BB.read_BEST(directorydataBB,sliceperiodBB,sliceyearBB,
                            sliceshapeBB,addclimoBB,slicenanBB)
    
    ### Read in model data
    variLL = 'T2M'
    sliceperiodLL = 'annual'
    slicebaseLL = np.arange(1951,1980+1,1)
    sliceshapeLL = 4
    slicenanLL = 'nan'
    addclimoLL = True
    takeEnsMeanLL = True
    latl,lonl,lens,lensmean = LL.read_LENS(directorydataLL,variLL,sliceperiodLL,
                                       slicebaseLL,sliceshapeLL,addclimoLL,
                                       slicenanLL,takeEnsMeanLL)

# cmip5NamesPR = ['ACCESS1-0',
#  'ACCESS1-3',
#  'CMCC-CMS',
#  'CNRM-CM5',
#  'CSIRO-Mk3-6-0',
#  'CanESM2',
#  'GFDL-CM3',
#  'GFDL-ESM2G',
#  'GFDL-ESM2M',
#  'GISS-E2-H-CC',
#  'GISS-E2-H',
#  'GISS-E2-R-CC',
#  'GISS-E2-R',
#  'HadGEM2-CC',
#  'HadGEM2-ES',
#  'MIROC-ESM-CHEM',
#  'MIROC-ESM',
#  'MIROC5',
#  'MRI-CGCM3',
#  'NorESM1-M',
#  'NorESM1-ME',
#  'inmcm4']

# cmip5NamesTAS = ['ACCESS1-0',
#  'ACCESS1-3',
#  'CCSM4',
#  'CESM1-BGC',
#  'CESM1-CAM5',
#  'CMCC-CMS',
#  'CNRM-CM5',
#  'CSIRO-Mk3-6-0',
#  'CanESM2',
#  'GFDL-CM3',
#  'GFDL-ESM2G',
#  'GFDL-ESM2M',
#  'GISS-E2-H-CC',
#  'GISS-E2-H',
#  'GISS-E2-R-CC',
#  'GISS-E2-R',
#  'HadGEM2-AO',
#  'HadGEM2-CC',
#  'HadGEM2-ES',
#  'IPSL-CM5A-LR',
#  'IPSL-CM5A-MR',
#  'MIROC-ESM-CHEM',
#  'MIROC-ESM',
#  'MIROC5',
#  'MPI-ESM-MR',
#  'MRI-CGCM3',
#  'NorESM1-M',
#  'NorESM1-ME',
#  'inmcm4']


# # ### Colors

# # In[5]:


# #Color Definitions

# train_color = 'navy'
# test_color = 'fuchsia'
# era_color = 'darkorange'
# map_colors = 'seismic'#'RdYlBu_r'

# #Figure dpi
# import matplotlib as mpl
# mpl.rcParams['figure.dpi']= 200


# # ### Regional Definitions

# # In[6]:


# def get_region_bounds(reg_name):
#   if reg_name is 'Globe':
#     lat_bounds = (-90.,90.)
#     lon_bounds = (0., 360.)
#   elif reg_name is 'GlobeNoSP':
#     lat_bounds = (-66.,90.)
#     lon_bounds = (0., 360.)
#   elif reg_name is 'NH':
#     lat_bounds = (0.,90.)
#     lon_bounds = (0., 360.)
#   elif reg_name is 'SH':
#     lat_bounds = (-90.,0.)
#     lon_bounds = (0., 360.)
#   elif reg_name is 'Tropics':
#     lat_bounds = (-30., 30.)
#     lon_bounds = (0., 360.)  
#   elif reg_name is 'narrowTropics':
#     lat_bounds = (-20., 20.)
#     lon_bounds = (0., 360.)  
#   elif reg_name is 'wideNH':
#     lat_bounds = (20., 90.)
#     lon_bounds = (0., 360.)  
#   elif reg_name is 'wideSH':
#     lat_bounds = (-90., -20.)
#     lon_bounds = (0., 360.)  
#   elif reg_name is 'Arctic':
#     lat_bounds = (60.,90.)
#     lon_bounds = (0., 360.)  
#   elif reg_name is 'Antarctic':
#     lat_bounds = (-90.,-60.)
#     lon_bounds = (0., 360.)  
#   elif reg_name is 'NHExtra':
#     lat_bounds = (30.,60.)
#     lon_bounds = (0., 360.)  
#   elif reg_name is 'SHExtra':
#     lat_bounds = (-60.,-30.)
#     lon_bounds = (0., 360.)  
#   elif reg_name is 'SriLanka':
#     lat_bounds = (6., 9.)
#     lon_bounds = (79., 82.5)
#   elif reg_name is 'SriLanka_big':
#     lat_bounds = (6.-3., 9.+3.)
#     lon_bounds = (79.-3., 82.5+3.)
#   elif reg_name is 'UK':
#     lat_bounds = (50.,60.)
#     lon_bounds = (360-11., 360.)
#   elif reg_name is 'US':
#     lat_bounds = (24, 60)
#     lon_bounds = (235, 290)
#   elif reg_name is 'NAext':
#     lat_bounds = (25, 90)
#     lon_bounds = (200,330)         
#   elif reg_name is 'NAprop':
#     lat_bounds = (10, 80)
#     lon_bounds = (180, 360)
#   elif reg_name is 'A':
#     lat_bounds = (50, 70)
#     lon_bounds = (40, 60)         
#   elif reg_name is 'B':
#     lat_bounds = (-10, 10)
#     lon_bounds = (20,40)         
#   elif reg_name is 'CentralAfrica':
#     lat_bounds = (-10, 30)
#     lon_bounds = (0,40)         
#   elif reg_name is 'C':
#     lat_bounds = (30, 50)
#     lon_bounds = (260, 280)         
#   elif reg_name is 'D':
#     lat_bounds = (-50, -30)
#     lon_bounds = (340,360)         
#   elif reg_name is 'E':
#     lat_bounds = (10, 30)
#     lon_bounds = (340,360) 
#   elif reg_name is 'Indonesia':
#     lat_bounds = (-9,5)
#     lon_bounds = (95,140)     
#   elif reg_name is 'SEAsia':
#     lat_bounds = (15,35)
#     lon_bounds = (90,115)     
    
#   else:
#     print('no region by that name')
    
#   print('Region = ' + reg_name + ': lat' + str(lat_bounds) + ', lon' + str(lon_bounds))
  
#   return lat_bounds, lon_bounds
    


# # ## Define default parameter values

# # In[7]:


# experiment_result = pd.DataFrame(columns=['actual iters', 'hiddens', 'cascade', 'RMSE Train', 'RMSE Test', 'ridge penalty', 'zero mean', 'zero merid mean', 'land only?'])

# #define primary dataset to use
# dataset = lens_hist_temp_90

# # name of the region of interest
# reg_name = 'Globe'
# lat_bounds, lon_bounds = get_region_bounds(reg_name)

# # whether to test and plot the results using ERA-interim data
# test_on_era = True
# dataset_era = erai_temp_90  # ERA-Interim file to use for the comparison

# #remove the annual mean? True to subtract it from dataset
# rm_annual_mean = True

# #remove the merid mean? True to subtract it from dataset
# rm_merid_mean = False

# #use land locations only
# land_only = False

# #split the data into training and testing sets? value of 1 will use all data as training, .8 will use 80% training, 20% testing; etc.
# segment_data_factor = .8

# #iterations is for the # of sample runs the model will use.  Must be a list, but can be a list with only one object
# iterations = [150]

# #hiddens corresponds to the number of hidden layers the nnet will use - 0 for linear model, or a list [10, 20, 5] for multiple layers of nodes (10 nodes in first layer, 20 in second, etc); The "loop" part 
# #allows you to loop through multiple architectures. For example, hiddens_loop = [[2,4],[0],[1 1 1]] would produce three separate NNs, the first with 2 hidden layers of 2 and 4 nodes, the next the linear model,
# #and the next would be 3 hidden layers of 1 node each.
# hiddens = [[5], [5, 5]]

# #list of different ridge penalties to iterate through. Must be a list, but can be a list with only one object
# ridge_penalty = [100] #+ list(range(1000,))

# #set useGPU to True to use the GPU, but only if you selected the GPU Runtime in the menu at the top of this page
# useGPU = False

# #Set Cascade to True to utilize the nnet's cascade function
# cascade = False

# #plot within the training loop - may want to set to False when testing out larget sets of parameters
# plot_in_train = False

# #number of plots returned (valid values are 0, 1,2 or 4 (maps))
# plots = 4


# # # Functions

# # ### Data Intake and Segmentation

# # In[8]:


# def rmse(a, b):
#   '''calculates the root mean squared error
#   takes two variables, a and b, and returns value'''
#   return np.sqrt(np.mean((a - b)**2))


# def read_primary_dataset(dataset, lat_bounds = lat_bounds, lon_bounds = lon_bounds):
#   data, lats, lons = df.readFiles(dataset)
#   data, lats, lons, basemap = df.getRegion(data, lats, lons, lat_bounds, lon_bounds)
#   print('our dataset: ',dataset, ' is shaped', data.shape)
#   return data, lats, lons, basemap
  
  
# def read_era_dataset(dataset_era, lat_bounds = lat_bounds, lon_bounds = lon_bounds):
#   data_era, lats_era, lons_era = df.readFiles(dataset_era)
#   data_era, lats_era, lons_era, __ = df.getRegion(data_era, lats_era, lons_era, lat_bounds, lon_bounds)
#   print('our ERA dataset: ',dataset_era, ' is shaped', data_era.shape)
#   return data_era, lats_era, lons_era


# def remove_annual_mean(data, data_era):
#   data = data - data.mean((2,3))[:,:,np.newaxis, np.newaxis]
#   data_era = data_era - data_era.mean((2,3))[:,:,np.newaxis, np.newaxis]
#   return data, data_era


# def remove_merid_mean(data, data_era):
#   data = data - data.mean((3))[:,:,:,np.newaxis]
#   data_era = data_era - data_era.mean((3))[:,:,:, np.newaxis]
#   return data, data_era


# def land_only_segment(data, data_era):
#   if(dataset.find('240') != -1):
#     landfrac, lats_land, lons_land = df.readFiles('data/LENS_landfraction_r240x120.npz')
#   else:
#     landfrac, lats_land, lons_land = df.readFiles('data/LENS_landfraction_r90x45.npz')
#   landfrac[landfrac<1.] = 0.
#   landfrac, __, __, __ = df.getRegion(landfrac, lats_land, lons_land, lat_bounds, lon_bounds)
#   data = data*landfrac
#   data_era = data_era*landfrac
#   return data, data_era


# # In[9]:


# def segment_data(data, fac = segment_data_factor):
  
#   global random_segment_seed, trainIndices, testIndices
#   if random_segment_seed == None:
#     random_segment_seed = int(int(np.random.randint(1, 100000)))
#   np.random.seed(random_segment_seed)
    
#   if fac < 1 :
#     nrows = data.shape[0]
#     segment_train = int(np.round(nrows * fac))
#     segment_test = nrows - segment_train
#     print('Training on', segment_train, 'ensembles, testing on', segment_test)

#     #picking out random ensembles
#     i = 0
#     trainIndices = list()
#     while i < segment_train:
#       line = np.random.randint(0, nrows)
#       if line not in trainIndices:
#         trainIndices.append(line)
#         i += 1
#       else:
#         pass
    
#     i = 0
#     testIndices = list()
#     while i < segment_test:
#       line = np.random.randint(0, nrows)
#       if line not in trainIndices:
#         if line not in testIndices:
#           testIndices.append(line)
#           i += 1
#       else:
#         pass
    
#     #random ensembles are picked
#     if debug:
#       print('Training on ensembles: ', trainIndices)
#       print('Testing on ensembles: ', testIndices)
    
#     #training segment----------
#     data_train = ''
#     for ensemble in trainIndices:
#       this_row = data[ensemble, :, :, :]
#       this_row = this_row.reshape(-1, data.shape[1], data.shape[2] , data.shape[3])
#       if data_train == '':
#         data_train = np.empty_like(this_row)
#       data_train = np.vstack((data_train, this_row))
#     data_train = data_train[1:, :, :, :]
    
#     if debug:
#       print('org data', data.shape)
#       print('training data', data_train.shape)

#     #reshape into X and T
#     Xtrain = data_train.reshape((data_train.shape[0] * data_train.shape[1]), (data_train.shape[2] * data_train.shape[3]))
#     Ttrain = np.tile((np.arange(data_train.shape[1]) + 1920).reshape(data_train.shape[1], 1), (data_train.shape[0], 1))
#     Xtrain_shape = (data_train.shape[0], data_train.shape[1])
    
    
#     #testing segment----------
#     data_test = ''
#     for ensemble in testIndices:
#       this_row = data[ensemble, :, :, :]
#       this_row = this_row.reshape(-1, data.shape[1], data.shape[2] , data.shape[3])
#       if data_test == '':
#         data_test = np.empty_like(this_row)
#       data_test = np.vstack((data_test, this_row))
#     data_test = data_test[1:, :, :, :]
    
#     if debug:
#       print('testing data', data_test.shape)
      
#     #reshape into X and T
#     Xtest = data_test.reshape((data_test.shape[0] * data_test.shape[1]), (data_test.shape[2] * data_test.shape[3]))
#     Ttest = np.tile((np.arange(data_test.shape[1]) + 1920).reshape(data_test.shape[1], 1), (data_test.shape[0], 1))
    

#   else:
#     trainIndices = np.arange(0,np.shape(data)[0])
#     testIndices = np.arange(0,np.shape(data)[0])    
#     print('Training on ensembles: ', trainIndices)
#     print('Testing on ensembles: ', testIndices)

#     data_train = data
#     data_test = data
    
#     Xtrain = data_train.reshape((data_train.shape[0] * data_train.shape[1]), (data_train.shape[2] * data_train.shape[3]))
#     Ttrain = np.tile((np.arange(data_train.shape[1]) + 1920).reshape(data_train.shape[1], 1), (data_train.shape[0], 1))
#     Xtrain_shape = (data_train.shape[0], data_train.shape[1])

#     Xtest = data_test.reshape((data_test.shape[0] * data_test.shape[1]), (data_test.shape[2] * data_test.shape[3]))
#     Ttest = np.tile((np.arange(data_test.shape[1]) + 1920).reshape(data_test.shape[1], 1), (data_test.shape[0], 1))

#   Xtest_shape = (data_test.shape[0], data_test.shape[1])
#   data_train_shape = data_train.shape[1]
#   data_test_shape = data_test.shape[1]
  
#   #'unlock' the random seed
#   np.random.seed(None)
  
#   return Xtrain, Ttrain, Xtest, Ttest, Xtest_shape, Xtrain_shape, data_train_shape, data_test_shape
  
  
# def shape_era(data_era):
#     Xtest_era = data_era.reshape((data_era.shape[0] * data_era.shape[1]), (data_era.shape[2] * data_era.shape[3]))
#     Ttest_era = np.tile((np.arange(data_era.shape[1]) + 1979).reshape(data_era.shape[1], 1), (data_era.shape[0], 1))
#     return Xtest_era, Ttest_era

# #data management

# def consolidate_data():
#     '''function to delete data and data_era since we have already sliced other variables from them.  Only run after segment_data and shape_era!!!
#     will delete global variables and clear memory'''
#     global data
#     global data_era
#     del data
#     del data_era
#     gc.collect()
  


# # ### Plotting and Visualization

# # In[10]:


# def drawOnGlobe(axes,data, lats, lons, region, cmap='coolwarm', vmin=None, vmax=None, inc=None):
#     '''Usage: drawOnGlobe(data, lats, lons, basemap, cmap)
#          data: nLats x nLons
#          lats: 1 x nLats
#          lons: 1 x nLons
#          basemap: returned from getRegion
#          cmap
#          vmin
#          vmax'''

#     data_cyc, lons_cyc = data, lons
#     data_cyc, lons_cyc = ct.util.add_cyclic_point(data, coord=lons) #fixes white line by adding point
   
#     image = plt.pcolormesh(lons_cyc, lats, data_cyc, transform=ct.crs.PlateCarree(), vmin=vmin, vmax=vmax, cmap = cmap, shading='flat')

#     axes.coastlines(color = 'black', linewidth = 1.2)

#     divider = make_axes_locatable(axes)
#     ax_cb = divider.new_horizontal(size="2%", pad=0.1, axes_class=plt.Axes)

#     plt.gcf().add_axes(ax_cb)
# #     if vmin is not None:
# #       cb = plt.colorbar(image, cax=ax_cb, boundaries=np.arange(vmin,vmax+inc,inc))
# #     else:
# #       cb = plt.colorbar(image, cax=ax_cb)
#     cb = plt.colorbar(image, cax=ax_cb)  
#     #cb.set_label('units', fontsize=20)

#     plt.sca(axes)  # in case other calls, like plt.title(...), will be made
    
#     #return image
#     return cb,image


# # In[11]:


# #plt.rcParams['axes.facecolor'] = 'white'
# plt.rcdefaults()

# def plot_prediction (Ttest, test_output, Ttest_era, era_output):
#   #predictions
  
#   plt.figure(figsize=(16,4))
#   plt.subplot(1, 2, 1)
#   plt.title('Predicted vs Actual Year for Testing')
#   plt.xlabel('Actual Year')
#   plt.ylabel('Predicted Year')
#   plt.plot(Ttest, test_output, 'o', color='black', label='GCM')

#   if test_on_era == True:
#     plt.plot(Ttest_era, era_output, 'o', color=era_color, label='ERA-Interim')
#   a = min(min(Ttest), min(test_output))
#   b = max(max(Ttest), max(test_output))
#   plt.plot((a,b), (a,b), '-', lw=3, alpha=0.7, color='gray')
#   #plt.axis('square')
#   plt.xlim(a * .995, b * 1.005)
#   plt.ylim(a * .995, b * 1.005)
#   plt.legend()
#   plt.show()
  

# def plot_training_error(nnet):
#   #training error (nnet)
  
#   plt.subplot(1, 2, 2)
#   plt.plot(nnet.getErrors(), color='black')
#   plt.title('Training Error per Iteration')
#   plt.xlabel('Training Iteration')
#   plt.ylabel('Training Error')
#   plt.show()

  
# def plot_rmse(train_output, Ttrain, test_output, Ttest, data_train_shape, data_test_shape):
#   #rmse (train_output, Ttrain, test_output, Ttest, data_train_shape, data_test_shape)
  
#   plt.figure(figsize=(16, 4))
#   plt.subplot(1, 2, 1)
#   rmse_by_year_train = np.sqrt(np.mean(((train_output - Ttrain)**2).reshape(Xtrain_shape), axis=0))
#   xs_train = (np.arange(data_train_shape) + 1920)
#   rmse_by_year_test = np.sqrt(np.mean(((test_output - Ttest)**2).reshape(Xtest_shape), axis=0))
#   xs_test = (np.arange(data_test_shape) + 1920)
#   plt.title('RMSE by year')
#   plt.xlabel('year')
#   plt.ylabel('error')
#   plt.plot(xs_train, rmse_by_year_train, label = 'training error', color=train_color, linewidth=1.5)
#   plt.plot(xs_test, rmse_by_year_test, label = 'test error', color=test_color, linewidth=1.)
#   plt.legend()

#   if test_on_era == True:
#     plt.subplot(1,2,2)
#     error_by_year_test_era = era_output - Ttest_era
#     plt.plot(Ttest_era, error_by_year_test_era, label = 'ERA-Interim error', color=era_color, linewidth=2.)            
#     plt.title('Error by year for ERA-Interim')
#     plt.xlabel('year')
#     plt.ylabel('error')
#     plt.legend()
#     plt.plot((1975,2020), (0,0), color='gray', linewidth=2.)
#     plt.xlim(1975,2020)
#   plt.show()
    

# def plot_weights(nnet, lats, lons, basemap):
#   # plot maps of the NN weights
#   plt.figure(figsize=(16, 6))
#   ploti = 0
#   nUnitsFirstLayer = nnet.layers[0].nUnits
  
#   for i in range(nUnitsFirstLayer):
#     ploti += 1
#     plt.subplot(np.ceil(nUnitsFirstLayer/3), 3, ploti)
#     maxWeightMag = nnet.layers[0].W[1:, i].abs().max().item() 
#     df.drawOnGlobe(((nnet.layers[0].W[1:, i]).cpu().data.numpy()).reshape(len(lats), len(lons)), lats, lons, basemap, vmin=-maxWeightMag, vmax=maxWeightMag, cmap=map_colors)
#     if(hiddens[0]==0):
#       plt.title('Linear Weights')
#     else:
#       plt.title('First Layer, Unit {}'.format(i+1))

#   if(cascade is True and hiddens[0]!=0):
#     plt.figure(figsize=(16, 6))
#     ploti += 1
#     plt.subplot(np.ceil(nUnitsFirstLayer/3), 3, ploti)
#     maxWeightMag = nnet.layers[-1].W[1:Xtrain.shape[1]+1, 0].abs().max().item()
#     df.drawOnGlobe(((nnet.layers[-1].W[1:Xtrain.shape[1]+1, 0]).cpu().data.numpy()).reshape(len(lats), len(lons)), lats, lons, basemap, vmin=-maxWeightMag, vmax=maxWeightMag, cmap=map_colors)
#     plt.title('Linear Weights')
#   plt.tight_layout()
  
  
  
# def plot_results(plots = 4): 
#   #calls all our plot functions together
#   global nnet, train_output, test_output, era_output, Ttest, Ttrain, Xtrain_shape, Xtest_shape, data_train_shape, data_test_shape, Ttest_era, lats, lons, basemap
  
#   if plots >=1:
#     plot_prediction (Ttest, test_output, Ttest_era, era_output)
#   if plots >= 2:
#     plot_training_error(nnet)
#     plot_rmse(train_output, Ttrain, test_output, Ttest, data_train_shape, data_test_shape)
#   if plots == 4:
#     plot_weights(nnet, lats, lons, basemap)
#   plt.show()
#   display(results)
  
  
# def plot_classifier_output(class_prob, test_class_prob, Xtest_shape, Xtrain_shape):
#   prob = class_prob[-1].reshape(Xtrain_shape)

#   plt.figure(figsize=(14, 6))
#   plt.plot((np.arange(Xtest_shape[1]) + 1920), prob[:,:,1].T, '-', alpha = .7)
#   plt.plot((np.arange(Xtest_shape[1]) + 1920), (np.mean(prob[:, :, 1], axis = 0).reshape(180, -1)), 'b-', linewidth=3.5, alpha = .5, label = 'ensemble average')
#   plt.title('Classifier Output by Ensemble using Training Data')
#   plt.xlabel('year')
#   plt.yticks((0, 1), ['Pre-Baseline', 'Post-Baseline'])
#   plt.legend()
#   plt.show()

#   tprob = test_class_prob[0].reshape(Xtest_shape)

#   plt.figure(figsize=(14, 6))
#   plt.plot(((np.arange(Xtest_shape[1]) + 1920)), tprob[:,:,1].T, '-', alpha = .7)
#   plt.plot((np.arange(Xtest_shape[1]) + 1920), (np.mean(tprob[:, :, 1], axis = 0).reshape(180, -1)), 'r-', linewidth=4, alpha = .5, label = 'ensemble average')
#   plt.title('Classifier Output by Ensemble using Test Data')
#   plt.xlabel('year')
#   plt.yticks((0, 1), ['Pre-Baseline', 'Post-Baseline'])
#   plt.legend()
#   plt.show()


# # ### Neural Network Creation & Training

# # In[29]:


# import time

# def movingAverageInputMaps(data,avgHalfChunk):
#     print(np.shape(data))
#     dataAvg = np.zeros(data.shape)
#     halfChunk = 2

#     for iy in np.arange(0,data.shape[1]):
#         yRange = np.arange(iy-halfChunk,iy+halfChunk+1)
#         yRange[yRange<0] = -99
#         yRange[yRange>=data.shape[1]] = -99
#         yRange = yRange[yRange>=0]
#         dataAvg[:,iy,:,:] = np.nanmean(data[:,yRange,:,:],axis=1)
#     return dataAvg

# def standardize_data(Xtrain,Xtest):

#     Xmean = np.mean(Xtrain,axis=0)
#     Xstd = np.std(Xtrain,axis=0)
#     Xtest = (Xtest - Xmean)/Xstd
#     Xtrain = (Xtrain - Xmean)/Xstd
    
#     global stdVals
#     stdVals = (Xmean,Xstd)
#     stdVals = stdVals[:]
    
#     return Xtrain, Xtest

# class TimeHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.times = []

#     def on_epoch_begin(self, epoch, logs={}):
#         self.epoch_time_start = time.time()

#     def on_epoch_end(self, epoch, logs={}):
#         self.times.append(time.time() - self.epoch_time_start)

# def defineNN(hidden, input_shape, output_shape, ridgePenalty = 0.):        
   
#     model = Sequential()
#     # initialize first layer
#     if hidden[0]==0:
#         #model is linear
#         model.add(Dense(1, input_shape=(input_shape,), activation='linear', use_bias=True, kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=ridgePenalty),bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
#                 kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
#     else:
#         #model is a single node with activation function
#         model.add(Dense(hidden[0], input_shape=(input_shape,), activation=actFun, use_bias=True, kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=ridgePenalty),bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
#                 kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))

#         #initialize other layers
#         for layer in hidden[1:]:
#             model.add(Dense(layer, activation=actFun, use_bias=True, kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.00),bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
#                 kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))


#     #initialize output layer
#     model.add(Dense(output_shape, activation=None, use_bias=True, kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.0),bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
#                 kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))

#     # add softmax layer at the end
#     model.add(Activation('softmax'))
    
#     return model

# def trainNN(model, Xtrain, Ytrain, niter=500, verbose=False):
  
#     global lr_here, batch_size
#     lr_here = .01
#     model.compile(optimizer=optimizers.SGD(lr=lr_here, momentum=0.9, nesterov=True),  #Adadelta .Adam()
#                  loss = 'binary_crossentropy',
#                  metrics=[metrics.categorical_accuracy],)
# #     model.compile(optimizer=optimizers.SGD(lr=lr_here, momentum=0.9, nesterov=True),  #Adadelta .Adam()
# #                  loss = 'categorical_crossentropy',
# #                  metrics=[metrics.categorical_accuracy],)

# #     model.compile(optimizer=optimizers.Adam(0.001),
# #                  loss = 'categorical_crossentropy',
# #                  metrics=[metrics.categorical_accuracy],)

#     #Declare the relevant model parameters
#     batch_size = 32 #np.shape(Xtrain)[0] #This doesn't seem to affect much in this case

#     print('----ANN Training: learning rate = ' + str(lr_here) + '; activation = ' + actFun + '; batch = ' + str(batch_size) + '----')    
#     time_callback = TimeHistory()
#     history = model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=niter, shuffle=True, verbose=verbose, callbacks=[time_callback], validation_split=0.)
#     print('******** done training ***********')

#     return model, history


# # In[13]:


# def test_train_loopClass(Xtrain, Ytrain, Xtest, Ytest, iterations, ridge_penalty, hiddens, plot_in_train = True):
#     '''for loops to iterate through training iterations, ridge penalty, and hidden layer list'''
#     results = {}
#     global nnet, random_network_seed
  
#     for niter in iterations:
#         for penalty in ridge_penalty:
#             for hidden in hiddens:
                
#                 # check / use random seed
#                 if random_network_seed == None:
#                   np.random.seed(None)
#                   random_network_seed = int(np.random.randint(1, 100000))
#                 np.random.seed(random_network_seed)
#                 random.seed(random_network_seed)
#                 tf.set_random_seed(0)

#                 # standardize the data
# #                 print('not standardizing the data here')
#                 Xtrain,Xtest = standardize_data(Xtrain,Xtest)
#                 Xmean, Xstd = stdVals
                
#                 # define the model
#                 model = defineNN(hidden, input_shape = np.shape(Xtrain)[1], output_shape = np.shape(Ytrain)[1], ridgePenalty=penalty)  
               
#                 #train the net
#                 model, history = trainNN(model, Xtrain, Ytrain, niter=niter, verbose=0)

#                 #after training, use the network with training data to check that we don't have any errors and output RMSE
#                 rmse_train = rmse(convert_fuzzyDecade_toYear(Ytrain,startYear, classChunk), convert_fuzzyDecade_toYear(model.predict(Xtrain),startYear, classChunk))
#                 if type(Ytest) != bool:
#                     rmse_test = 0.
#                     rmse_test = rmse(convert_fuzzyDecade_toYear(Ytest,startYear, classChunk), convert_fuzzyDecade_toYear(model.predict(Xtest),startYear, classChunk))
#                 else:
#                     rmse_test = False

#                 this_result = {'iters': niter, 
#                                'hiddens' : hidden, 
#                                'RMSE Train' : rmse_train, 
#                                'RMSE Test' : rmse_test, 
#                                'ridge penalty': penalty, 
#                                'zero mean' : rm_annual_mean,
#                                'zero merid mean' : rm_merid_mean,
#                                'land only?' : land_only,
#                                'Segment Seed' : random_segment_seed,
#                                'Network Seed' : random_network_seed }
#                 results.update(this_result)

#                 global experiment_result
#                 experiment_result = experiment_result.append(results, ignore_index=True)
#                 display(experiment_result.tail(1))

#                 #if True to plot each iter's graphs.
#                 if plot_in_train == True:
#                     plt.figure(figsize = (16,6))

#                     plt.subplot(1,2,1)
#                     plt.plot(history.history['loss'],label = 'training')
#                     plt.title(history.history['loss'][-1])
#                     plt.xlabel('epoch')
#                     plt.xlim(2,len(history.history['loss'])-1)
#                     plt.legend()

#                     plt.subplot(1,2,2)
                    
#                     plt.plot(convert_fuzzyDecade_toYear(Ytrain,startYear, classChunk),convert_fuzzyDecade_toYear(model.predict(Xtrain),startYear, classChunk),'o',color='gray')
#                     plt.plot(convert_fuzzyDecade_toYear(Ytest,startYear, classChunk),convert_fuzzyDecade_toYear(model.predict(Xtest),startYear, classChunk),'x', color='red')
#                     plt.plot([startYear,2100],[startYear,2100],'--k')
#                     plt.yticks(np.arange(1920,2100,10))
#                     plt.xticks(np.arange(1920,2100,10))
                    
#                     plt.grid(True)
#                     plt.show()

#                 #'unlock' the random seed
#                 np.random.seed(None)
#                 random.seed(None)
#                 tf.set_random_seed(None)
  
#     return experiment_result, model


# # In[14]:


# def makeScatter(ax,train_output_rs,test_output_rs, testColor=''):
#     mpl.rcParams['figure.dpi']= 150
#     MS0 = 3
#     MS1 = 3
#     FS = 10

#     #-------------------------------------------------------------------------------
#     #Predicted plots
#     xs_test = (np.arange(np.shape(test_output_rs)[1]) + 1920)

#     ax.set_xlabel('actual year', fontsize = FS)
#     #ax.xaxis.label.set_size(FS)
#     ax.set_ylabel('predicted year', fontsize = FS)
#     ax.plot([1800,2130],[1800,2130],'-',color='black',linewidth=2)
#     ax.set_xticks(np.arange(1820,2130,20))
#     ax.set_yticks(np.arange(1820,2130,20))

#     p=plt.plot(xs_test,train_output_rs[0,:],'o', markersize=MS1, color = 'gray',label='GCM training data')
#     p=plt.plot(xs_test,test_output_rs[0,:],'o', markersize=MS0, color = 'black',label='GCM testing data')

#     for i in np.arange(0,np.shape(train_output_rs)[0]):
#         p=plt.plot(xs_test,train_output_rs[i,:],'o', markersize=MS1, color='gray')
#         clr = p[0].get_color()

#     for i in np.arange(0,np.shape(test_output_rs)[0]):
#         if not testColor:
#             p=plt.plot(xs_test,test_output_rs[i,:],'o', markersize=MS0)
#         else:
#             p=plt.plot(xs_test,test_output_rs[i,:],'o', markersize=MS0, color=testColor)
#         clr = p[0].get_color()
  
#     plt.xlim(1910,2110)   
#     plt.ylim(1910,2110)
#     plt.legend(fontsize=FS)
#     plt.grid(True)


# # In[15]:


# def convert_fuzzyDecade(data,startYear,classChunk):
#     years = np.arange(startYear-classChunk*2,2100+classChunk*2)
#     chunks = years[::int(classChunk)] + classChunk/2
    
#     labels = np.zeros((np.shape(data)[0],len(chunks)))
    
#     for iy,y in enumerate(data):
#         norm = stats.uniform.pdf(years,loc=y-classChunk/2.,scale=classChunk)
        
#         vec = []
#         for sy in years[::classChunk]:
#             j=np.logical_and(years>sy,years<sy+classChunk)
#             vec.append(np.sum(norm[j]))
#         vec = np.asarray(vec)
#         vec[vec<.0001] = 0. #this should not matter
        
# #         plt.figure(figsize=(3,3))
# #         plt.plot(chunks,vec)
# #         plt.xlim(1920,1990)
# #         plt.show()
# #         print(np.sum(vec))
        
#         vec = vec/np.sum(vec)
        
#         labels[iy,:] = vec
#     return labels, chunks

# def convert_fuzzyDecade_toYear(label,startYear,classChunk):
#     years = np.arange(startYear-classChunk*2,2100+classChunk*2)
#     chunks = years[::int(classChunk)] + classChunk/2
    
#     return np.sum(label*chunks,axis=1)


# # In[16]:


# # index = 2
# # val, decadeChunks = convert_fuzzyDecade(Ytrain[index],startYear,classChunk=10)
# # pred = convert_fuzzyDecade_toYear(val,startYear,classChunk=10)
# # print(Ytrain[index],pred)
# # print(val)
# # print(decadeChunks)
# # YtrainClass, decadeChunks = convert_fuzzyDecade(Ytrain,startYear,classChunk=10)
# # outval = convert_fuzzyDecade_toYear(YtrainClass,startYear,classChunk=10)
# # print(outval[2])
# # print(Ytrain[2])

# # plt.figure(figsize=(3,3))
# # plt.plot(Ytrain,outval,'.',color='gray')
# # plt.plot([1920,2100],[1920,2100],'--',color='orange')
# # plt.show()


# # # Results

# # In[17]:


# print(Error)


# # ## Figure 1: pred/actual curves (1-1)

# # In[32]:


# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
#                               inter_op_parallelism_threads=1)
# from keras import backend as K

# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)
# # K.get_session()
# K.clear_session()
# #tf.Session.close
# #====================================================================================
# debug = True
# reg_name = 'Globe'#'Globe'
# actFun = 'ReLu'
# ridge_penalty = [0.01] #.01
# classChunk = 10

# hiddensList = [[20,20]]  #linear model
# #hiddensList = [[1]]  #one hidden layer and node
# # hiddensList = [[20,20]] 

# expList = [(0)]#(0,1)
# expN = np.size(expList)

# iterations = [500]#[500]#[1500]
# random_segment = True
# foldsN = 1
# #====================================================================================

# for avgHalfChunk in (0,):#([1,5,10]):#([1,2,5,10]):
#     session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
#                                   inter_op_parallelism_threads=1)
#     sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#     K.set_session(sess)
#     # K.get_session()
#     K.clear_session()
    
#     for loop in ([1]):#(0,1,2,3,4,5):

#         #------------------------------------------------
#         #define primary dataset to use

#         if(loop==0):
#             modelType = 'control'
#             var = 'temp'
#             dataset_cmip5 = lens_cont_temp_90
#             dataset = lens_cont_temp_90
#             dataset_era = erai_temp_90  # ERA-Interim file to use for the comparison
#             dataset_twentyCR = twentyCR_temp_90
#             dataset_best = best_temp_90
#         elif(loop==1):
#             modelType = 'cmip5'
#             var = 'temp'
#             dataset_cmip5 = cmip5_hist_temp_90
#             dataset = cmip5_hist_temp_90
#             dataset_era = erai_temp_90  # ERA-Interim file to use for the comparison
#             dataset_twentyCR = twentyCR_temp_90
#             dataset_best = best_temp_90
#         elif(loop==2):
#             modelType = 'lens'
#             var = 'temp'
#             dataset_cmip5 = lens_hist_temp_90
#             dataset = lens_hist_temp_90
#             dataset_era = erai_temp_90  # ERA-Interim file to use for the comparison
#             dataset_twentyCR = twentyCR_temp_90
#             dataset_best = best_temp_90
#         elif(loop==3):
#             modelType = 'cmip5'
#             var = 'precip'
#             dataset_cmip5 = cmip5_hist_precip_90
#             dataset = cmip5_hist_precip_90
#             dataset_era = merra2_precip_90#erai_precip_90  # merra2_precip_90
#             dataset_twentyCR = twentyCR_precip_90
#             dataset_best = gpcp_precip_90#best_temp_90
#         elif(loop==4):
#             modelType = 'control'
#             var = 'precip'
#             dataset_cmip5 = lens_cont_precip_90
#             dataset = lens_cont_precip_90
#             dataset_era = merra2_precip_90#erai_precip_90  # ERA-Interim file to use for the comparison
#             dataset_twentyCR = twentyCR_precip_90
#             dataset_best = gpcp_precip_90
#         elif(loop==5):
#             modelType = 'lens'
#             var = 'precip'
#             dataset_cmip5 = lens_hist_precip_90
#             dataset = lens_hist_precip_90
#             dataset_era = merra2_precip_90#erai_precip_90  # ERA-Interim file to use for the comparison
#             dataset_twentyCR = twentyCR_precip_90
#             dataset_best = gpcp_precip_90
#         elif(loop==6):
#             modelType = 'cmip5Control'
#             var = 'temp'
#             dataset_cmip5 = lens_cont_temp_90
#             dataset = cmip5_hist_temp_90
#             dataset_era = erai_temp_90#erai_precip_90  # ERA-Interim file to use for the comparison
#             dataset_twentyCR = twentyCR_temp_90
#             dataset_best = best_temp_90
#         elif(loop==7):
#             modelType = 'cmip5Control'
#             var = 'precip'
#             dataset_cmip5 = lens_cont_precip_90
#             dataset = cmip5_hist_precip_90
#             dataset_era = merra2_precip_90#erai_precip_90  # merra2_precip_90
#             dataset_twentyCR = twentyCR_precip_90
#             dataset_best = gpcp_precip_90#best_temp_90
#         else:
#             error('no such dataset')
#         #------------------------------------------------

#         # get info about the region
#         lat_bounds, lon_bounds = get_region_bounds(reg_name)
#         data_all, lats, lons, basemap = read_primary_dataset(dataset, lat_bounds, lon_bounds)
#         data_era_all, lats_era, lons_era = read_era_dataset(dataset_era, lat_bounds, lon_bounds)
#         data_best_all, lats_best, lons_best = read_era_dataset(dataset_best, lat_bounds, lon_bounds)
#         data_cmip5_all, lats_cmip5_all, lons_cmip5_all, ___ = read_primary_dataset(dataset_cmip5, lat_bounds, lon_bounds)

#         test_output_mat = np.empty((np.max(expList)+1,foldsN,180*int(np.round(np.shape(data_all)[0]*(1.0-segment_data_factor)))))

#         for exp in expList:  

#             # loop over experiments
#             if exp == 0:
#                 rm_annual_mean = False
#                 rm_merid_mean = False
#                 land_only = False
#             elif exp == 3:
#                 rm_annual_mean = False
#                 rm_merid_mean = False
#                 land_only = True
#             elif exp == 1:
#                 rm_annual_mean = True
#                 rm_merid_mean = False
#                 land_only = False
#             elif exp == 2:
#                 rm_annual_mean = True
#                 rm_merid_mean = True
#                 land_only = False

#             # get the data together
#             data, data_era, data_cmip5, data_best = data_all, data_era_all, data_cmip5_all, data_best_all
#             if rm_annual_mean == True:
#                 data, data_era = remove_annual_mean(data, data_era)
#                 data, data_best = remove_annual_mean(data, data_best)
#                 data_cmip5, ___ = remove_annual_mean(data_cmip5, data_era)
#                 print('removed annual mean')

#             if rm_merid_mean == True:
#                 data, data_era = remove_merid_mean(data, data_era)
#                 data, data_best = remove_merid_mean(data, data_best)
#                 data_cmip5, ___ = remove_merid_mean(data_cmip5, data_era)
#                 print('removed meridian mean')

#             if land_only == True:
#                 data, data_era = land_only_segment(data, data_era)
#                 data, data_best = land_only_segment(data, data_best)
#                 data_cmip5, ___ = land_only_segment(data_cmip5, data_era)
#                 print('land only')     

#             for ih in np.arange(0,len(hiddensList)):
#                 hiddens = [hiddensList[ih]]
#                 if hiddens[0][0]==0:
#                     annType = 'linear'
#                 elif hiddens[0][0]==1 and len(hiddens)==1:
#                     annType = 'layers1'
#                 else:
#                     annType = 'layers10x10'

#             if(avgHalfChunk!=0):
#                 data = movingAverageInputMaps(data,avgHalfChunk)

#             # loop over folds
#             for loop in np.arange(0,foldsN): 

#                 K.clear_session()
#                 #---------------------------
#                 random_segment_seed = 34515#45827#96715#45827#96715#45827#96715#None#45827#None#45827#45827#81558#None
# #                 segment_data_factor = 1.
#                 #---------------------------
#                 Xtrain, Ytrain, Xtest, Ytest, Xtest_shape, Xtrain_shape, data_train_shape, data_test_shape = segment_data(data, segment_data_factor)

#                 # convert year into decadal class
#                 startYear = Ytrain[0] # define startYear for GLOBAL USE
#                 YtrainClassMulti, decadeChunks = convert_fuzzyDecade(Ytrain,startYear,classChunk)  
#                 YtestClassMulti, __ = convert_fuzzyDecade(Ytest,startYear,classChunk)  

#                 # for use later
#                 XtrainS,XtestS = standardize_data(Xtrain,Xtest)
#                 Xmean, Xstd = stdVals      

#                 #---------------------------
#                 random_network_seed = 87750#None#84256#53985#84256#53985#None#84256#None#84256#84256#8453#None#None
#                 tf.set_random_seed(0)
#                 #---------------------------

#                 #create and train network
#                 exp_result, model = test_train_loopClass(Xtrain, YtrainClassMulti, Xtest, YtestClassMulti, iterations=iterations, ridge_penalty=ridge_penalty, hiddens = hiddensList, plot_in_train = True)
#                 model.summary()  
                
#                 ################################################################################################################################################                
#                 # save the model
#                 dirname = '/Users/eabarnes/GoogleDrive/WORK/RESEARCH/2019/TOE_Paper_II/savedModels/' + var + '/'
#                 savename = modelType + '_' + var + '_kerasMultiClassBinaryOption4_Chunk' + str(classChunk) + '_Linear_L2_' + str(ridge_penalty[0])                     + '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(iterations[0]) + '_' + str(hiddensList[0][0]) + 'x' + str(hiddensList[0][-1]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'                     + str(random_network_seed) 
#                 savenameModelTestTrain = modelType + '_' + var + '_modelTrainTest_SegSeed' + str(random_segment_seed) + '_NetSeed' + str(random_network_seed)

#                 if(reg_name=='Globe'):
#                     regSave = ''
#                 else:
#                     regSave = '_' + reg_name
                
#                 if(rm_annual_mean==True):
#                     savename = savename + '_meanRemoved' 
#                     savenameModelTestTrain = savenameModelTestTrain + '_meanRemoved'
#                 if(avgHalfChunk!=0):
#                     savename = savename + '_avgHalfChunk' + str(avgHalfChunk)
#                     savenameModelTestTrain = savenameModelTestTrain + '_avgHalfChunk' + str(avgHalfChunk)

#                 savename = savename + regSave    
#                 model.save(dirname + savename + '.h5')
#                 np.savez(dirname + savenameModelTestTrain + '.npz', trainModels=trainIndices, testModels=testIndices, Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, Xmean=Xmean, Xstd=Xstd, lats=lats, lons=lons)

#                 print('saving ' + savename)
                
#                 ################################################################################################################################################
#                 # make final plot
#                 # get obs
#                 data_dir = '/Users/eabarnes/GoogleDrive/WORK/RESEARCH/2018/TOE/TOE_Models/data/'

#                 #best data
#                 best_temp_90 = data_dir + 'BEST_temperature_annualmean_r90x45.npz'
#                 dataBEST, latsBEST, lonsBEST = df.readFiles(best_temp_90)

#                 #gpcp data
#                 gpcp_precip_90 = data_dir + 'GPCP_precip_annualmean_r90x45.npz'
#                 dataGPCP, latsGPCP, lonsGPCP = df.readFiles(gpcp_precip_90)

#                 def findStringMiddle(start,end,s):
#                     return s[s.find(start)+len(start):s.rfind(end)]

#                 if(var=='temp'):
#                     if(avgHalfChunk!=0):
#                         dataBEST = movingAverageInputMaps(dataBEST,avgHalfChunk)
#                     Xobs = dataBEST.reshape(dataBEST.shape[1],dataBEST.shape[2]*dataBEST.shape[3])
#                     yearsObs = np.arange(dataBEST.shape[1]) + 1850
#                 elif(var=='precip'):
#                     if(avgHalfChunk!=0):
#                         dataGPCP = movingAverageInputMaps(dataGPCP,avgHalfChunk)    
#                     Xobs = dataGPCP.reshape(dataGPCP.shape[1],dataGPCP.shape[2]*dataGPCP.shape[3])
#                     yearsObs = np.arange(dataGPCP.shape[1]) + 1979

#                 if(rm_annual_mean==True):
#                     Xobs = Xobs - np.nanmean(Xobs,axis=1)[:,np.newaxis]

#                 annType = 'class'
#                 startYear = 1920
#                 endYear = 2099
#                 years = np.arange(startYear,endYear+1,1)                    
#                 XobsS = (Xobs-Xmean)/Xstd
                
#                 if(annType=='class'):
#                     YpredObs = convert_fuzzyDecade_toYear(model.predict(XobsS),startYear,classChunk)
#                     YpredObsPDF = model.predict(XobsS)
#                 elif(annType=='reg'):    
#                     YpredObs = model.predict(XobsS)*Ystd + Ymean
#                     YpredObsPDF = np.nan

#                 if(annType=='class'):
#                     YpredTrain = convert_fuzzyDecade_toYear(model.predict((Xtrain-Xmean)/Xstd),startYear,classChunk)
#                     YpredTest = convert_fuzzyDecade_toYear(model.predict((Xtest-Xmean)/Xstd),startYear,classChunk)
#                 elif(annType=='reg'):
#                     YpredTrain = model.predict((Xtrain-Xmean)/Xstd)*Ystd + Ymean
#                     YpredTest = model.predict((Xtest-Xmean)/Xstd)*Ystd + Ymean

#                 ################################################################################################################################################
#                 plt.figure(figsize=(7,5))
#                 ax = plt.subplot(1,1,1)
#                 makeScatter(ax,YpredTrain.reshape(len(trainIndices),len(years)),YpredTest.reshape(len(testIndices),len(years)))
#                 plt.title('CMIP5 ' + var + '\nRMSE Train = ' + str(np.round(rmse(YpredTrain[:,], Ytrain[:,0]), decimals=1))           + '; RMSE Test = ' + str(np.round(rmse(YpredTest[:,], Ytest[:,0]), decimals=1)))

#                 iyears = np.where(Ytest<1980)[0]
#                 plt.text(2064,1922, 'Test RMSE before 1980 = ' + str(np.round(rmse(YpredTest[iyears,], Ytest[iyears,0]), decimals=1)), fontsize=5)
#                 iyears = np.where(Ytest>=1980)[0]
#                 plt.text(2064,1922+5, 'Test RMSE after   1980 = ' + str(np.round(rmse(YpredTest[iyears,], Ytest[iyears,0]), decimals=1)), fontsize=5)

#                 iy = np.where(yearsObs>=1950)[0]
#                 plt.plot(yearsObs[iy],YpredObs[iy],'o', color='black',                         markerfacecolor = 'white', markeredgecolor = 'black', markeredgewidth=1.5, label='BEST Observations')


#                 savefigName = modelType + '_' + var + '_scatterPred_' + savename 
#                 plt.annotate(savename,(0,.98),xycoords='figure fraction', fontsize=5, color='gray')
#                 plt.savefig('testingFigures/' + var + '/' + savefigName + '.png', dpi=300, bbox_inches = 'tight')
#                 plt.show()        
#                 print(np.round(np.corrcoef(yearsObs[iy],YpredObs[iy])[0,1],2))
                


# # # Next to Run
# # * temp, 300, 10x10, L2 = 0.01, mean retained
# # 

# # # Notes

# # In[27]:


# model.summary()
# model.layers[0].get_config()


# # In[ ]:




