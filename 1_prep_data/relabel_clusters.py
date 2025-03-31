"""
Author: Mikhail Schee
Created: 2024-07-18

This script will modify the specified netcdf files, relabeling the cluster ID numbers based on
the specified method.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions in source code must retain the accompanying copyright notice, this list of conditions, and the following disclaimer.
    2. Redistributions in binary form must reproduce the accompanying copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Names of the copyright holders must not be used to endorse or promote products derived from this software without prior written permission from the copyright holders.
    4. If any files are modified, you must cause the modified files to carry prominent notices stating that you changed the files and the date of any change.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

import analysis_helper_functions as ahf
import BGR_params as bps
import matplotlib.pyplot as plt

################################################################################

# Select the relabeling scheme
relab_scheme = 'SA_divs'

################################################################################

def relabel_clusters_SA_divs(df):
    """
    Relabel the clusters based on the SA divisions

    df: pandas DataFrame to relabel
    """
    print('\t\t- Relabeling clusters based on SA divisions')
    # Check to make sure there is a 'cluster' column
    if 'cluster' not in df.columns:
        raise ValueError('`df` must have a column named "cluster"')
    # Get the SA divider values
    SA_divs = bps.BGR_HPC_SA_divs
    ## For the first divider
    # Find all the rows with SA values less than the first divider and are not noise
    df_this_div = df[(df['SA'] < SA_divs[0]) & (df['cluster'] != -1)]
    # Relabel the cluster IDs
    df.loc[df_this_div.index, 'cluster'] = 0
    ## For the middle dividers
    # Loop through the dividers
    for i in range(1,len(SA_divs)):
        # Find all the rows with SA values less than the divider and greater than the previous divider
        df_this_div = df[(df['SA'] < SA_divs[i]) & (df['SA'] >= SA_divs[i-1]) & (df['cluster'] != -1)]
        # Relabel the cluster IDs
        df.loc[df_this_div.index, 'cluster'] = i
    ## For the last divider
    # Find all the rows with SA values greater than the last divider
    df_this_div = df[(df['SA'] >= SA_divs[-1]) & (df['cluster'] != -1)]
    # Relabel the cluster IDs
    df.loc[df_this_div.index, 'cluster'] = len(SA_divs)

    # Make a test plot
    if False:
        # Find the list of cluster ids 
        clstr_ids  = np.unique(np.array(df['cluster'].values, dtype=int)).tolist()
        # Remove the noise cluster id
        if -1 in clstr_ids: 
            clstr_ids.remove(-1)
        # Make a figure to plot onto
        fig, ax = plt.subplots()
        # Loop through each cluster
        for i in clstr_ids:
            # Get just the rows for this cluster
            this_df_clstr = df[df['cluster']==i]
            # Check to see whether the cluster is empty
            if len(this_df_clstr) == 0:
                continue
            # Decide on the color and symbol, don't go off the end of the arrays
            my_clr = ahf.distinct_clrs[i%len(ahf.distinct_clrs)]
            # Plot the histogram
            n, bins, patches = ax.hist(this_df_clstr['SA'], bins=100, color=my_clr, alpha=0.5)
            # Find the maximum value for this histogram
            h_max = n.max()
            # Find where that max value occured
            bin_h_max = bins[np.where(n == h_max)][0]
            # Mark the cluster ID
            ax.scatter(bin_h_max, h_max, color=my_clr, s=80, marker=r"${}$".format(str(i)), zorder=10)
            # Plot a backing circle for the cluster number if it's a light color
            if my_clr in [ahf.jackson_clr[13], ahf.jackson_clr[14]]:
                ax.scatter(bin_h_max, h_max, color='k', s=100, marker='o', alpha=0.5, zorder=9)
        # Set x axis bounds
        ax.set_xlim(bps.S_range_LHW_AW)
        # Make the y axis log scale
        ax.set_yscale('log')
        # Add vertical lines at every divider
        for div in SA_divs:
            ax.axvline(div, color='b', linestyle=':')
        plt.show()
    return df

################################################################################
# Main execution
################################################################################
# Select the netcdf to modify
ncs_to_modify = []
new_nc_names = []
# Loop through each period
for this_BGR in    ['BGR0506',
                    'BGR0607',
                    'BGR0708',
                    'BGR0809',
                    'BGR0910',
                    'BGR1011',
                    'BGR1112',
                    'BGR1213',
                    'BGR1314',
                    'BGR1415',
                    'BGR1516',
                    'BGR1617',
                    'BGR1718',
                    'BGR1819',
                    'BGR1920',
                    'BGR2021',
                    'BGR2122']:
    ncs_to_modify.append('netcdfs/HPC_'+this_BGR+'_clstrd_unrelab.nc')
    new_nc_names.append('netcdfs/HPC_'+this_BGR+'_clstrd_'+relab_scheme+'.nc')

print('- Relabeling scheme:',relab_scheme)
# Loop through the netcdfs to modify
for i in range(len(ncs_to_modify)):
    my_nc = ncs_to_modify[i]
    new_nc_name = new_nc_names[i]
    print('- Reading',my_nc)
    # Load in with xarray
    ds = xr.load_dataset(my_nc)

    # Define global attributes to print
    gattrs_to_print = ['Last modified', 'Last modification']
    print('')
    for attr in gattrs_to_print:
        print('\t',attr+':',ds.attrs[attr])

    print('\t- Making changes')

    ## Get the moving average profiles
    # Convert a subset of the dataset to a dataframe
    df = ds[['SA','prof_no','cluster']].squeeze().to_dataframe()
    # Get the original dimensions of the data to reshape the arrays later
    len0 = len(df.index.get_level_values(0).unique())
        # Had issues with ITP11, so I had to add `prof_no` to the dataframe and count
        #   the unique values
    # len0 = len(list(set(df['prof_no'].values)))
    len1 = len(df.index.get_level_values(1).unique())
    # Apply the relabeling scheme
    if relab_scheme == 'SA_divs':
        df = relabel_clusters_SA_divs(df)
    # Put the new cluster ID values back into the dataset
    ds['cluster'].values = df['cluster'].values.reshape((len0, len1))

    # Update the global variables:
    ds.attrs['Last modified'] = str(datetime.now())
    ds.attrs['Last modification'] = 'Relabeled cluster IDs with '+relab_scheme+' scheme'

    # Write out to netcdf
    print('Writing data to',new_nc_name)
    ds.to_netcdf(new_nc_name, 'w')

    # Load in with xarray
    ds2 = xr.load_dataset(new_nc_name)

    # See the variables after
    for attr in gattrs_to_print:
        print('\t',attr+':',ds2.attrs[attr])
