"""
Functions are useful untilities for data processing in the NN
 
Notes
-----
    Author : Zachary Labe
    Date   : 8 July 2020
    
Usage
-----
    [1] readFiles(dataset)
    [2] getRegion(data,lat1,lon1,lat_bounds,lon_bounds)
"""

def readFiles(dataset):
    """
    Function reads in data for selected dataset

    Parameters
    ----------
    dataset : string
        name of data set for primary data
        
    Returns
    -------
    data : numpy array
        data from selected data set
    lat1 : 1d numpy array
        latitudes
    lon1 : 1d numpy array
        longitudes

    Usage
    -----
    data,lat1,lon1 = readFiles(dataset)
    """
    print('\n>>>>>>>>>> Using readFiles function!')
    
    ### Import modules
    import numpy as np
    import read_LENS as LL
    import read_BEST as BB
    
    if dataset == 'lens':
        directorydataLL = '/Users/zlabe/Data/LENS/monthly/'
        variLL = 'T2M'
        sliceperiodLL = 'annual'
        slicebaseLL = np.arange(1951,1980+1,1)
        sliceshapeLL = 4
        slicenanLL = 'nan'
        addclimoLL = True
        takeEnsMeanLL = True
        lat1,lon1,data = LL.read_LENS(directorydataLL,variLL,
                                               sliceperiodLL,slicebaseLL,
                                               sliceshapeLL,addclimoLL,
                                               slicenanLL,takeEnsMeanLL)
    elif dataset == 'best':
        directorydataBB = '/Users/zlabe/Data/BEST/'
        sliceperiodBB = 'annual'
        sliceyearBB = np.arange(1956,2019+1,1)
        sliceshapeBB = 3
        slicenanBB = 'nan'
        addclimoBB = True
        lat1,lon1,data = BB.read_BEST(directorydataBB,sliceperiodBB,
                                      sliceyearBB,sliceshapeBB,addclimoBB,
                                      slicenanBB)
    else:
        ValueError('WRONG DATA SET SELECTED!')
        
    print('>>>>>>>>>> Completed: Finished readFiles function!')
    return data,lat1,lon1   

def getRegion(data,lat1,lon1,lat_bounds,lon_bounds):
    """
    Function masks out region for data set

    Parameters
    ----------
    data : 3d+ numpy array
        original data set
    lat1 : 1d array
        latitudes
    lon1 : 1d array
        longitudes
    lat_bounds : 2 floats
        (latmin,latmax)
    lon_bounds : 2 floats
        (lonmin,lonmax)
        
    Returns
    -------
    data : numpy array
        MASKED data from selected data set
    lat1 : 1d numpy array
        latitudes
    lon1 : 1d numpy array
        longitudes

    Usage
    -----
    data,lats,lons = getRegion(data,lat1,lon1,lat_bounds,lon_bounds)
    """
    print('\n>>>>>>>>>> Using get_region function!')
    
    ### Import modules
    import numpy as np
    
    ### Note there is an issue with 90N latitude (fixed!)
    lat1 = np.round(lat1,3)
    
    ### Mask latitudes
    if data.ndim == 3:
        latq = np.where((lat1 >= lat_bounds[0]) & (lat1 <= lat_bounds[1]))[0]
        latn = lat1[latq]
        datalatq = data[:,latq,:]
        
        ### Mask longitudes
        lonq = np.where((lon1 >= lon_bounds[0]) & (lon1 <= lon_bounds[1]))[0]
        lonn = lon1[lonq]
        datalonq = datalatq[:,:,lonq]
    elif data.ndim == 4:
        latq = np.where((lat1 >= lat_bounds[0]) & (lat1 <= lat_bounds[1]))[0]
        latn = lat1[latq]
        datalatq = data[:,:,latq,:]
        
        ### Mask longitudes
        lonq = np.where((lon1 >= lon_bounds[0]) & (lon1 <= lon_bounds[1]))[0]
        lonn = lon1[lonq]
        datalonq = datalatq[:,:,:,lonq]
    elif data.ndim == 6:
        latq = np.where((lat1 >= lat_bounds[0]) & (lat1 <= lat_bounds[1]))[0]
        latn = lat1[latq]
        datalatq = data[:,:,:,latq,:]
        
        ### Mask longitudes
        lonq = np.where((lon1 >= lon_bounds[0]) & (lon1 <= lon_bounds[1]))[0]
        lonn = lon1[lonq]
        datalonq = datalatq[:,:,:,:,lonq]
    
    ### New variable name
    datanew = datalonq
    
    print('>>>>>>>>>> Completed: getRegion function!')
    return datanew,latn,lonn   