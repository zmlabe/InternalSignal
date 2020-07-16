"""
Functions are useful statistical untilities for data processing in the NN
 
Notes
-----
    Author : Zachary Labe
    Date   : 15 July 2020
    
Usage
-----
    [1] rmse(a,b)
    [2] remove_annual_mean(data,data_obs,lats,lons,lats_obs,lons_obs)
    [3] remove_merid_mean(data, data_obs)
    [4] remove_ensemble_mean(data,ensmean)
    [5] standardize_data(Xtrain,Xtest)
"""

def rmse(a,b):
    """calculates the root mean squared error
    takes two variables, a and b, and returns value
    """
    
    ### Import modules
    import numpy as np
    
    ### Calculate RMSE
    rmse_stat = np.sqrt(np.mean((a - b)**2))
    
    return rmse_stat

def remove_annual_mean(data,data_obs,lats,lons,lats_obs,lons_obs):
    """
    Removes annual mean from data set
    """
    
    ### Import modulates
    import numpy as np
    import calc_Utilities as UT
    
    ### Create 2d grid
    lons2,lats2 = np.meshgrid(lons,lats)
    lons2_obs,lats2_obs = np.meshgrid(lons_obs,lats_obs)
    
    ### Calculate weighted average and remove mean
    data = data - UT.calc_weightedAve(data,lats2)[:,:,np.newaxis,np.newaxis]
    data_obs = data_obs - UT.calc_weightedAve(data_obs,lats2_obs)[:,np.newaxis,np.newaxis]
    
    return data,data_obs

def remove_merid_mean(data, data_obs):
    """
    Removes annual mean from data set
    """
    
    ### Import modulates
    import numpy as np
    
    ### Move mean of latitude
    data = data - np.nanmean(data,axis=2)[:,:,np.newaxis,:]
    data_obs = data_obs - np.nanmean(data_obs,axis=1)[:,np.newaxis,:]

    return data,data_obs

def remove_ensemble_mean(data,ensmean):
    """
    Removes ensemble mean
    """
    
    ### Import modulates
    import numpy as np
    
    ### Remove ensemble mean
    datameangone = data - ensmean
    
    return datameangone

def standardize_data(Xtrain,Xtest):
    """
    Standardizes training and testing data
    """
    
    ### Import modulates
    import numpy as np

    Xmean = np.nanmean(Xtrain,axis=0)
    Xstd = np.nanstd(Xtrain,axis=0)
    Xtest = (Xtest - Xmean)/Xstd
    Xtrain = (Xtrain - Xmean)/Xstd
    
    stdVals = (Xmean,Xstd)
    stdVals = stdVals[:]
    
    return Xtrain,Xtest,stdVals