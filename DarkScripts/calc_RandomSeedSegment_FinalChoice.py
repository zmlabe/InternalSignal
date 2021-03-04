"""
Select seeds for use of final figure in paper

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 17 November
"""

### Import packages
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import scipy.stats as stats

### Set parameters
variables = [r'T2M']
datasets = [r'XGHG',r'XAER',r'lens']
seasons = [r'annual']
SAMPLEQ = 100

### Set directories
directorydata = '/Users/zlabe/Documents/Research/InternalSignal/Data/FINAL/'
directoryfigure = '/Users/zlabe/Desktop/PAPER/'

### Read in slope data
filename_slope = 'Slopes_20CRv3-Obs_XGHG-XAER-LENS_%s_RANDOMSEED_20ens.txt' % SAMPLEQ
slopes = np.genfromtxt(directorydata + filename_slope,unpack=True)
ghg_slopes = slopes[:,0]
aer_slopes = slopes[:,1]
lens_slopes = slopes[:,2]

### Read in R2 data
filename_R2= 'R2_20CRv3-Obs_XGHG-XAER-LENS_%s_RANDOMSEED_20ens.txt' % SAMPLEQ
slopes = np.genfromtxt(directorydata + filename_R2,unpack=True)
ghg_r2 = slopes[:,0]
aer_r2 = slopes[:,1]
lens_r2 = slopes[:,2]

### Read in SegmentSeeds data
filename_R2= 'SegmentSeed_20CRv3-Obs_XGHG-XAER-LENS_%s_RANDOMSEED_20ens.txt' % SAMPLEQ
segment = np.genfromtxt(directorydata + filename_R2,unpack=True)
ghg_segment = segment[:,0]
aer_segment = segment[:,1]
lens_segment = segment[:,2]

### Read in NetworkSeeds data (Model seeds - constant)
filename_R2= 'ModelSeed_20CRv3-Obs_XGHG-XAER-LENS_%s_RANDOMSEED_20ens.txt' % SAMPLEQ
network = np.genfromtxt(directorydata + filename_R2,unpack=True)
ghg_network = network[:,0]
aer_network = network[:,1]
lens_network = network[:,2]

###############################################################################
### Select seeds based on XAER
aerq = np.where((aer_slopes > 0.99) & (aer_slopes < 1.01))[0]
newslopes_aer = aer_slopes[aerq]
newr2_aer = aer_r2[aerq]
maxr2_aer = np.nanmax(newr2_aer)
wherer2_aer = np.where((maxr2_aer == newr2_aer))
segment_aer = aer_segment[wherer2_aer]
np.savetxt(directorydata + 'AER_SelectedSegmentSeed.txt',segment_aer)