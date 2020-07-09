"""
Function(s) reads in monthly data from LENS for different variables using # of 
ensemble members
 
Notes
-----
    Author : Zachary Labe
    Date   : 6 July 2020
    
Usage
-----
    [1] read_LENS(directory,vari,sliceperiod,
                  slicebase,sliceshape,addclimo,
                  slicenan,takeEnsMean)
"""

def read_LENS(directory,vari,sliceperiod,slicebase,sliceshape,addclimo,slicenan,takeEnsMean):
    """
    Function reads monthly data from LENS
    
    Parameters
    ----------
    directory : string
        path for data
    vari : string
        variable for analysis
    sliceperiod : string
        how to average time component of data
    sliceyear : string
        how to slice number of years for data
    sliceshape : string
        shape of output array
    addclimo : binary
        True or false to add climatology
    slicenan : string or float
        Set missing values
    takeEnsMean : binary
        whether to take ensemble mean
        
    Returns
    -------
    lat : 1d numpy array
        latitudes
    lon : 1d numpy array
        longitudes
    var : numpy array
        processed variable
    ENSmean : numpy array
        ensemble mean
        
    Usage
    -----
    read_LENS(directory,vari,sliceperiod,
                  slicebase,sliceshape,addclimo,
                  slicenan,takeEnsMean)
    """
    print('\n>>>>>>>>>> STARTING read_LENS function!')
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    
    ###########################################################################
    ### Parameters
    time = np.arange(1920,2100+1,1)
    mon = 12
    ens1 = np.arange(1,35+1,1)
    ens2 = np.arange(101,105+1,1)
    allens = np.append(ens1,ens2)
    ens = list(map('{:03d}'.format, allens))
    
    ###########################################################################
    ### Read in data
    membersvar = []
    for i,ensmember in enumerate(ens):
        filename = directory + '%s/%s_%s_1920-2100.nc' % (vari,vari,ensmember)
        data = Dataset(filename,'r')
        lat1 = data.variables['latitude'][:]
        lon1 = data.variables['longitude'][:]
        var = data.variables['%s' % vari][:,:,:]
        data.close()
        
        print('Completed: read ensemble --%s--' % ensmember)
        membersvar.append(var)
        del var
    membersvar = np.asarray(membersvar)
    ensvarKelvin = np.reshape(membersvar,(len(ens),time.shape[0],mon,
                                    lat1.shape[0],lon1.shape[0]))
    del membersvar
    print('Completed: read all members!\n')
    
    ###########################################################################
    ### Convert to degrees C
    ensvar = ensvarKelvin - 273.15
    print('Completed: Kevin to degrees C!')
    
    ###########################################################################
    ### Calculate anomalies or not
    if addclimo == True:
        ensvalue = ensvar
        print('Completed: calculated absolute temperature!')
    elif addclimo == False:
        yearsq = np.where((time >= slicebase.min()) & (time <= slicebase.max()))[0]
        yearssel = time[yearsq]
        
        mean = np.nanmean(ensvar[:,yearsq,:,:,:])
        ensvalue = ensvar - mean
        print('Completed: calculated anomalies from',
              slicebase.min(),'to',slicebase.max())
        
    ###########################################################################
    ### Slice over months (currently = [ens,yr,mn,lat,lon])
    ### Shape of output array
    if sliceperiod == 'annual':
        ensvalue = np.nanmean(ensvalue,axis=2)
        if sliceshape == 1:
            ensshape = ensvalue.ravel()
        elif sliceshape == 4:
            ensshape = ensvalue
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: ANNUAL MEAN!')
    elif sliceperiod == 'none':
        if sliceshape == 1:
            ensshape = ensvalue.ravel()
        elif sliceshape == 3:
            ensshape= np.reshape(ensvalue,(ensvalue.shape[0]*ensvalue.shape[1],
                                             ensvalue.shape[2],ensvalue.shape[3]))
        elif sliceshape == 5:
            ensshape = ensvalue
        print('Shape of output =', ensshape.shape, [[ensshape.ndim]])
        print('Completed: ALL MONTHS!')
        
    ###########################################################################
    ### Change missing values
    if slicenan == 'nan':
        ensshape[np.where(np.isnan(ensshape))] = np.nan
        print('Completed: missing values are =',slicenan)
    else:
        ensshape[np.where(np.isnan(ensshape))] = slicenan

    ###########################################################################
    ### Take ensemble mean
    if takeEnsMean == True:
        ENSmean = np.nanmean(ensshape,axis=0)
        print('Ensemble mean AVAILABLE!')
    elif takeEnsMean == False:
        ENSmean = np.nan
        print('Ensemble mean NOT available!')
    else:
        ValueError('WRONG OPTION!')
        
    print('>>>>>>>>>> ENDING read_LENS function!')
    return lat1,lon1,ensshape
        

# ### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# directory = '/Users/zlabe/Data/LENS/monthly/'
# vari = 'T2M'
# sliceperiod = 'annual'
# slicebase = np.arange(1951,1980+1,1)
# sliceshape = 4
# slicenan = 'nan'
# addclimo = True
# takeEnsMean = True
# lat,lon,var,ENSmean = read_LENS(directory,vari,sliceperiod,
#                         slicebase,sliceshape,addclimo,
#                         slicenan,takeEnsMean)