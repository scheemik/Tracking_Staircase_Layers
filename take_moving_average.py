"""
Author: Mikhail Schee
Created: 2022-06-17

This script is set up to take the moving average of Arctic Ocean profile data
that has been formatted into netcdfs by the `make_netcdf` function, adding
the variables 'ma_iT', 'ma_CT', 'ma_SP', 'ma_SA', and 'ma_sigma' to those files.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions in source code must retain the accompanying copyright notice, this list of conditions, and the following disclaimer.
    2. Redistributions in binary form must reproduce the accompanying copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Names of the copyright holders must not be used to endorse or promote products derived from this software without prior written permission from the copyright holders.
    4. If any files are modified, you must cause the modified files to carry prominent notices stating that you changed the files and the date of any change.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

import scipy.ndimage as ndimage
from scipy import interpolate

# Import the Thermodynamic Equation of Seawater 2010 (TEOS-10) from GSW
# For calculating density anomaly
import gsw

# The moving average window in dbar
c3 = 50#100

################################################################################
# Main execution
################################################################################
# Select the netcdfs to modify
ncs_to_modify = [
                 'netcdfs/ITP_001.nc',
                 'netcdfs/ITP_002.nc',
                 'netcdfs/ITP_003.nc',
                 'netcdfs/ITP_004.nc',
                 'netcdfs/ITP_005.nc',
                 'netcdfs/ITP_006.nc',
                #  'netcdfs/ITP_007.nc', # Not in BGR
                 'netcdfs/ITP_008.nc',
                #  'netcdfs/ITP_009.nc', # Not in BGR
                #  'netcdfs/ITP_009.nc', # Not in BGR
                #  'netcdfs/ITP_010.nc', # Not in BGR
                 'netcdfs/ITP_011.nc',
                #  'netcdfs/ITP_012.nc', # Not in BGR
                 'netcdfs/ITP_013.nc',
                #  'netcdfs/ITP_014.nc', # Not in BGR
                #  'netcdfs/ITP_015.nc', # Not in BGR
                #  'netcdfs/ITP_016.nc', # Not in BGR
                #  'netcdfs/ITP_017.nc', # Not in BGR
                 'netcdfs/ITP_018.nc',
                #  'netcdfs/ITP_019.nc', # Not in BGR
                 'netcdfs/ITP_021.nc',
                #  'netcdfs/ITP_022.nc', # Not in BGR
                #  'netcdfs/ITP_023.nc', # Not in BGR
                #  'netcdfs/ITP_024.nc', # Not in BGR
                 'netcdfs/ITP_025.nc',
                #  'netcdfs/ITP_026.nc', # Not in BGR
                #  'netcdfs/ITP_027.nc', # Not in BGR
                #  'netcdfs/ITP_028.nc', # Not in BGR
                #  'netcdfs/ITP_029.nc', # Not in BGR
                 'netcdfs/ITP_030.nc',
                 'netcdfs/ITP_032.nc',
                 'netcdfs/ITP_033.nc',
                 'netcdfs/ITP_034.nc',
                 'netcdfs/ITP_035.nc',
                #  'netcdfs/ITP_036.nc', # Not in BGR
                #  'netcdfs/ITP_037.nc', # Not in BGR
                #  'netcdfs/ITP_038.nc', # Not in BGR
                 'netcdfs/ITP_041.nc',
                 'netcdfs/ITP_042.nc',
                 'netcdfs/ITP_043.nc',
                #  'netcdfs/ITP_047.nc', # Not in BGR
                #  'netcdfs/ITP_048.nc', # Not in BGR
                #  'netcdfs/ITP_049.nc', # Not in BGR
                #  'netcdfs/ITP_051.nc', # Not in BGR
                 'netcdfs/ITP_052.nc',
                 'netcdfs/ITP_053.nc',
                 'netcdfs/ITP_054.nc',
                 'netcdfs/ITP_055.nc',

                 'netcdfs/ITP_056.nc',
                 'netcdfs/ITP_057.nc',
                 'netcdfs/ITP_058.nc',
                 'netcdfs/ITP_059.nc',
                 'netcdfs/ITP_060.nc',
                 'netcdfs/ITP_061.nc',
                 'netcdfs/ITP_062.nc',
                 'netcdfs/ITP_063.nc',
                 'netcdfs/ITP_064.nc',
                 'netcdfs/ITP_065.nc',
                 'netcdfs/ITP_068.nc',
                 'netcdfs/ITP_069.nc',

                 'netcdfs/ITP_070.nc',
                 'netcdfs/ITP_072.nc',
                 'netcdfs/ITP_073.nc',
                 'netcdfs/ITP_074.nc',
                 'netcdfs/ITP_075.nc',
                 'netcdfs/ITP_076.nc',
                 'netcdfs/ITP_077.nc',
                 'netcdfs/ITP_078.nc',
                 'netcdfs/ITP_079.nc',
                 'netcdfs/ITP_080.nc',
                 'netcdfs/ITP_081.nc',
                 'netcdfs/ITP_082.nc',

                 'netcdfs/ITP_083.nc',
                 'netcdfs/ITP_084.nc',
                 'netcdfs/ITP_085.nc',
                 'netcdfs/ITP_086.nc',
                 'netcdfs/ITP_087.nc',
                 'netcdfs/ITP_088.nc',
                 'netcdfs/ITP_089.nc',
                 'netcdfs/ITP_090.nc',
                 'netcdfs/ITP_091.nc',
                 'netcdfs/ITP_092.nc',
                 'netcdfs/ITP_094.nc',
                 'netcdfs/ITP_095.nc',

                 'netcdfs/ITP_097.nc',
                 'netcdfs/ITP_098.nc',
                 'netcdfs/ITP_099.nc',
                 'netcdfs/ITP_100.nc',
                 'netcdfs/ITP_101.nc',
                 'netcdfs/ITP_102.nc',
                 'netcdfs/ITP_103.nc',
                 'netcdfs/ITP_104.nc',
                 'netcdfs/ITP_105.nc',
                 'netcdfs/ITP_107.nc',
                 'netcdfs/ITP_108.nc',
                 'netcdfs/ITP_109.nc',

                 'netcdfs/ITP_110.nc',
                 'netcdfs/ITP_113.nc',
                 'netcdfs/ITP_114.nc',
                 'netcdfs/ITP_116.nc',

                 'netcdfs/ITP_117.nc',
                 'netcdfs/ITP_118.nc',
                 'netcdfs/ITP_120.nc',
                 'netcdfs/ITP_121.nc',
                 'netcdfs/ITP_122.nc',
                 'netcdfs/ITP_123.nc',
                 'netcdfs/ITP_125.nc',
                 'netcdfs/ITP_128.nc',
                 ]

# Get a list of all ITP netcdfs that have 3 digits in the file name
# ncs_to_modify = [] 
# ncs_available = os.listdir('netcdfs/')
# for nc in ncs_available:
#     itp_number = ''.join(filter(str.isdigit, nc))
#     if len(itp_number) == 3 and 'ITP' in nc:
#         ncs_to_modify.append('netcdfs/'+nc)
# print(ncs_to_modify)
# exit(0)

# Loop through the netcdfs to modify
for my_nc in ncs_to_modify:
    print('')
    print('Reading',my_nc)
    # Load in with xarray
    ds = xr.load_dataset(my_nc)

    gattrs_to_print = ['Last modified', 'Last modification', 'Moving average window']

    # See the variables before
    for attr in gattrs_to_print:
        print('\t',attr+':',ds.attrs[attr])

    print('making changes')

    ## Get the moving average profiles
    # Convert a subset of the dataset to a dataframe
    df = ds[['press','iT','CT','PT','SP','SA','prof_no']].squeeze().to_dataframe()
    # Get the original dimensions of the data to reshape the arrays later
    len0 = len(df.index.get_level_values(0).unique())
        # Had issues with ITP11, so I had to add `prof_no` to the dataframe and count
        #   the unique values
    len0 = len(list(set(df['prof_no'].values)))
    len1 = len(df.index.get_level_values(1).unique())
    # Use the pandas `rolling` function to get the moving average
    #   center=True makes the first and last window/2 of the profiles are masked
    #   win_type='boxcar' uses a rectangular window shape
    #   on='press' means it will take `press` as the index column
    #   .mean() takes the average of the rolling
    df1 = df.rolling(window=c3, center=True, win_type='boxcar', on='press').mean()
    # Put the moving average profiles for temperature, salinity, and density into the dataset
    ds['ma_iT'].values = df1['iT'].values.reshape((len0, len1))
    ds['ma_CT'].values = df1['CT'].values.reshape((len0, len1))
    ds['ma_PT'].values = df1['PT'].values.reshape((len0, len1))
    ds['ma_SP'].values = df1['SP'].values.reshape((len0, len1))
    ds['ma_SA'].values = df1['SA'].values.reshape((len0, len1))
    ds['ma_sigma'].values= gsw.sigma1(ds['ma_SP'], ds['ma_CT'])

    # Update the global variables:
    ds.attrs['Last modified'] = str(datetime.now())
    ds.attrs['Last modification'] = 'Added moving averages with '+str(c3)+' dbar window'
    ds.attrs['Moving average window'] = str(c3)+' dbar'

    # Write out to netcdf
    print('Writing data to',my_nc)
    ds.to_netcdf(my_nc, 'w')

    # Load in with xarray
    ds2 = xr.load_dataset(my_nc)

    # See the variables after
    for attr in gattrs_to_print:
        print('\t',attr+':',ds2.attrs[attr])
