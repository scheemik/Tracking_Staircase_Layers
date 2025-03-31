"""
Author: Mikhail Schee
Created: 2022-04-11

This script is set up to subsample Arctic Ocean profile data that has been
formatted into netcdfs by the `make_netcdf` function, adding them to those files.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions in source code must retain the accompanying copyright notice, this list of conditions, and the following disclaimer.
    2. Redistributions in binary form must reproduce the accompanying copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Names of the copyright holders must not be used to endorse or promote products derived from this software without prior written permission from the copyright holders.
    4. If any files are modified, you must cause the modified files to carry prominent notices stating that you changed the files and the date of any change.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
# For formatting data into dataframes
import pandas as pd
# For formatting date objects
from datetime import datetime
# For reading netcdf files
import xarray as xr
# For taking moving averages
import scipy.ndimage as ndimage
# For interpolating
from scipy import interpolate

################################################################################

# Select the subsampling scheme
# ss_scheme = 'AIDJEX PDF'
ss_scheme = 'Keep every'
# ss_scheme = 'Interpolate'
# Select integer x to Keep every x points
n_pts = 4
# Define the spacing (in dbar) for interpolation
interp_spacing = 2.0

################################################################################

def keep_every(vert, n_pts):
    """
    Sub-samples one profile, keeping only every <n_pts> points, skipping the rest
    Returns a mask which, when applied to the given vertical array, results in a
        subsampled vertical profile. Doesn't change any of the values

    vert    An array of vertical values to subsample (pressure or depth)
    n_pts   An integer so that only every n_pts point is not masked
    """
    # Get the length of the vertical array
    vert_len = len(vert)
    # Remove all nan values from vert assuming they will only be on the end
    vert = [x for x in vert if np.isnan(x)==False]
    # Check to make sure there is data to use
    if vert_len < 1:
        return [None]*vert_len
    else:
        # Make a blank array in which to put the mask
        ss_mask = np.array([None]*vert_len)
    #
    # Some really convenient syntax [::x] which returns a view of every x point
    #   Not a new array, changing it will change the original
    keep_these = ss_mask[::n_pts]
    # Unmask every n_pts point
    keep_these[:] = 1
    # print('in keep_every(), ss_mask:',ss_mask)
    return ss_mask

################################################################################
# Main execution
################################################################################
# Select the netcdf to modify
ncs_to_modify = [
                 'netcdfs/ITP_001.nc',
                 'netcdfs/ITP_002.nc', # Not in time period
                 'netcdfs/ITP_003.nc',
                 'netcdfs/ITP_004.nc',
                 'netcdfs/ITP_005.nc',
                 'netcdfs/ITP_006.nc',
                #  'netcdfs/ITP_007.nc', # Not in BGR
                 'netcdfs/ITP_008.nc',
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
                #  'netcdfs/ITP_056.nc', # Not in BGR
                #  'netcdfs/ITP_057.nc', # Not in BGR
                #  'netcdfs/ITP_058.nc', # Not in BGR
                #  'netcdfs/ITP_059.nc', # Not in BGR
                #  'netcdfs/ITP_060.nc', # Not in BGR
                #  'netcdfs/ITP_061.nc', # Not in BGR
                 'netcdfs/ITP_062.nc',
                #  'netcdfs/ITP_063.nc', # Not in BGR
                 'netcdfs/ITP_064.nc',
                 'netcdfs/ITP_065.nc',
                 'netcdfs/ITP_068.nc',
                 'netcdfs/ITP_069.nc',
                 'netcdfs/ITP_070.nc',
                #  'netcdfs/ITP_072.nc', # Not in BGR
                #  'netcdfs/ITP_073.nc', # Not in BGR
                #  'netcdfs/ITP_074.nc', # Not in BGR
                #  'netcdfs/ITP_075.nc', # Not in BGR
                #  'netcdfs/ITP_076.nc', # Not in BGR
                 'netcdfs/ITP_077.nc',
                 'netcdfs/ITP_078.nc',
                 'netcdfs/ITP_079.nc',
                 'netcdfs/ITP_080.nc',
                 'netcdfs/ITP_081.nc',
                 'netcdfs/ITP_082.nc',
                #  'netcdfs/ITP_083.nc', # Not in BGR
                 'netcdfs/ITP_084.nc',
                 'netcdfs/ITP_085.nc',
                 'netcdfs/ITP_086.nc',
                 'netcdfs/ITP_087.nc',
                 'netcdfs/ITP_088.nc',
                 'netcdfs/ITP_089.nc',
                #  'netcdfs/ITP_090.nc', # Not in BGR
                #  'netcdfs/ITP_091.nc', # Not in BGR
                #  'netcdfs/ITP_092.nc', # Not in BGR
                #  'netcdfs/ITP_094.nc', # Not in BGR
                #  'netcdfs/ITP_095.nc', # Not in BGR
                 'netcdfs/ITP_097.nc',
                #  'netcdfs/ITP_098.nc', # Not in BGR
                 'netcdfs/ITP_099.nc',
                 'netcdfs/ITP_100.nc',
                 'netcdfs/ITP_101.nc',
                #  'netcdfs/ITP_102.nc', # Not in BGR
                 'netcdfs/ITP_103.nc',
                 'netcdfs/ITP_104.nc',
                 'netcdfs/ITP_105.nc',
                 'netcdfs/ITP_107.nc',
                 'netcdfs/ITP_108.nc',
                 'netcdfs/ITP_109.nc',
                 'netcdfs/ITP_110.nc',
                 'netcdfs/ITP_113.nc',
                 'netcdfs/ITP_114.nc',
                #  'netcdfs/ITP_116.nc', # Not in BGR
                 'netcdfs/ITP_117.nc',
                 'netcdfs/ITP_118.nc',
                 'netcdfs/ITP_120.nc',
                 'netcdfs/ITP_121.nc',
                 'netcdfs/ITP_122.nc',
                 'netcdfs/ITP_123.nc',
                #  'netcdfs/ITP_125.nc',
                #  'netcdfs/ITP_128.nc',
                 ]

# Loop through the netcdfs to modify
for my_nc in ncs_to_modify:
    print('Reading',my_nc)
    print('ss scheme:', ss_scheme)
    # Load in with xarray
    ds = xr.load_dataset(my_nc)

    # Define global attributes to print
    gattrs_to_print = ['Last modified', 'Last modification', 'Sub-sample scheme']
    print('')
    for attr in gattrs_to_print:
        print('\t',attr+':',ds.attrs[attr])

    print('making changes')

    ## Update the subsample mask
    # Find the original vertical variable (pressure or depth)
    vert_var = ds.attrs['Original vertical measure']

    # Loop over all the profiles
    if ss_scheme == 'AIDJEX_PDF':
        ss_scheme_str = ss_scheme
        for i in range(len(ds[vert_var].values)):
            # Add subsample mask for this profile
            ds['ss_mask'][i] = AIDJEX_PDF(ds[vert_var].values[i])
    elif ss_scheme == 'Keep every':
        ss_scheme_str = ss_scheme + ' '+str(n_pts)+' points'
        for i in range(len(ds[vert_var].values)):
            # Add subsample mask for this profile
            ds['ss_mask'][i] = keep_every(ds[vert_var].values[i], n_pts)
    else:
        print('Sub-sample scheme \"',ss_scheme,'\" not supported')
        exit(0)
    # print('ds[ss_mask]:',ds['ss_mask'].values)

    # Update the global variables:
    ds.attrs['Last modified'] = str(datetime.now())
    ds.attrs['Last modification'] = 'Modified sub-sample scheme'
    ds.attrs['Sub-sample scheme'] = ss_scheme_str

    # Write out to netcdf
    print('Writing data to',my_nc)
    ds.to_netcdf(my_nc, 'w')

    # Load in with xarray
    ds2 = xr.load_dataset(my_nc)
    # print('ds2[ss_mask]:',ds2['ss_mask'].values)

    for attr in gattrs_to_print:
        print('\t',attr+':',ds2.attrs[attr])
    print('\t ss_mask:')
    print('\t',ds2.where(ds2.prof_no==7, drop=True).squeeze().ss_mask.values)
