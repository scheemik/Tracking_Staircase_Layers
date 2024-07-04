"""
Author: Mikhail Schee
Created: 2024-07-03

This script will calculate cluster properties for the BGR data set per period and save the 
pandas dataframe to a pickle file. 

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions in source code must retain the accompanying copyright notice, this list of conditions, and the following disclaimer.
    2. Redistributions in binary form must reproduce the accompanying copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Names of the copyright holders must not be used to endorse or promote products derived from this software without prior written permission from the copyright holders.
    4. If any files are modified, you must cause the modified files to carry prominent notices stating that you changed the files and the date of any change.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
# For adding subplot labels a, b, c, ...
import string
import analysis_helper_functions as ahf
import dill as pl

# For common BGR parameters
import BGR_params as bps
# For common BGR objects
import BGR_objects as bob

# Decide whether to calculate the per profile cluster properties
calc_per_pf = False

################################################################################
# Declare variables for plotting
################################################################################
dark_mode = True

# Enable dark mode plotting
if dark_mode:
    plt.style.use('dark_background')
    std_clr = 'w'
    alt_std_clr = 'yellow'
    clr_ocean = 'k'
    clr_land  = 'grey'
    clr_lines = 'w'
else:
    std_clr = 'k'
    alt_std_clr = 'olive'
    clr_ocean = 'w'
    clr_land  = 'grey'
    clr_lines = 'k'

# Set some plotting styles
grid_alpha    = 0.3

# Define array of linestyles to cycle through
l_styles = ['-', '--', '-.', ':']

################################################################################

# Make the profile filters
pfs_0 = ahf.Profile_Filters()
# Make the plot parameters
# these_vars_to_keep = ['dt_start', 'lon', 'lat', 'cluster', 'press', 'SA', 'CT', 'sigma', 'alpha', 'beta']
these_vars_to_keep = ['dt_start', 'cluster', 'SA']
# pp_ = ahf.Plot_Parameters(x_vars=['cRL'], y_vars=['clst_prob'], extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':False, 'mark_outliers':False, 'extra_vars_to_keep':these_vars_to_keep}, legend=False)
pp_ = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['clst_prob'], extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':False, 'mark_outliers':False, 'extra_vars_to_keep':these_vars_to_keep}, legend=False)

# Start a list of the data frames
dfs = []
# Loop through each period
for this_BGR in ['BGR0506', 'BGR0607', 'BGR0708', 'BGR0809', 'BGR0910', 'BGR1011', 'BGR1112', 'BGR1213', 'BGR1314', 'BGR1415', 'BGR1516', 'BGR1617', 'BGR1718', 'BGR1819', 'BGR1920', 'BGR2021', 'BGR2122']:
    # Make the data set
    ds_this_BGR = ahf.Data_Set(bps.BGR_HPC_unrelab_dict[this_BGR], bob.dfs_all)
    # Make the subplot groups
    group_ = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_)
    # Concatonate all the pandas data frames together
    df = pd.concat(group_.data_frames)
    # Add column for the dataset label
    df['dataset'] = this_BGR

    calc_pcs_vars = False
    if calc_pcs_vars:
        # Make a deep copy of the data frame to calculate cluster properties per profile
        df_copy = df.copy(deep=True)

        # Make a list of cluster properties per profile to calculate
        calc_vars = []
        for var in ['press', 'SA', 'CT', 'sigma']:
            # calc_vars.append('pca_'+var)
            calc_vars.append('pcs_'+var)

        # Calculate new cluster variables
        print('Calculating cluster spans per profile')
        # df_copy = ahf.calc_extra_cl_vars(df_copy, calc_vars)
        df_copy = ahf.calc_extra_cl_vars(df_copy, ['pcs_press'])
        # Drop duplicates to get one row per cluster per profile
        df_per_pf = df_copy.drop_duplicates(subset=['cluster','instrmt-prof_no'])
        print(df_per_pf.columns)

        # Pickle the data frame to a file
        print('Pickling to outputs/'+this_BGR+'_pf_cluster_properties_unrelab.pickle')
        pl.dump(df_per_pf, open('outputs/'+this_BGR+'_pf_cluster_properties_unrelab.pickle', 'wb'))
    # else:
    #     # Load in temp file
    #     print('Loading from to outputs/'+this_BGR+'_pf_cluster_properties_unrelab.pickle')
    #     df_per_pf = pl.load(open('outputs/'+this_BGR+'_pf_cluster_properties_unrelab.pickle', 'rb'))

#     # Make a list of variables to calculate
#     calc_vars = []
#     for var in ['press']:#, 'SA', 'CT', 'sigma']:
#         calc_vars.append('trd_pcs_'+var)
#         calc_vars.append('nztrd_pcs_'+var)
#         calc_vars.append('ca_pcs_'+var)
#         calc_vars.append('nzca_pcs_'+var)
#     # Calculate the above variables for the cluster spans per profile
#     print('Calculating cluster averages in the cluster spans per profile')
#     df_per_pf = ahf.calc_extra_cl_vars(df_per_pf, calc_vars)
#     print(df_per_pf.columns)
#     # Drop duplicates to get one row per cluster
#     df_per_pf = df_per_pf.drop_duplicates(subset=['cluster'])
#     # Drop Time dimension
#     df_per_pf = df_per_pf.droplevel('Time')
#     # Drop the columns for dt_start and entry
#     df_per_pf = df_per_pf.drop(columns=['dt_start', 'entry'])
# 
#     # Calculate fit variables over lat-lon for each cluster
#     print('Calculating fit variables over lat-lon for each cluster')
#     add_new_cols = True
#     for i in df['cluster'].unique():
#         # Find the data from this cluster
#         df_this_cluster = df[df['cluster']==i]
#         # Calculate the fit variables for this cluster
#         df_this_cluster = ahf.calc_fit_vars(df_this_cluster, ('press-fit', 'SA-fit', 'CT-fit', 'sigma-fit', 'alpha-fit', 'beta-fit'), ['lon','lat'])
#         # Add the new columns to the main data frame
#         if add_new_cols:
#             # Get list of columns in new data frame
#             new_cols = df_this_cluster.columns
#             # Ged list of columns in main data frame
#             main_cols = df.columns
#             # Find the columns that are in the new data frame but not in the main data frame
#             for col in new_cols:
#                 if col not in main_cols:
#                     df[col] = None
#             add_new_cols = False
#         # Update the data in the main data frame with the new data
#         df[df['cluster']==i] = df_this_cluster
    print(df.columns)

    # Make a list of variables to calculate
    calc_vars = []#'n_points', 'cRL']
    # for var in ['press-fit', 'SA-fit', 'CT-fit', 'sigma-fit', 'alpha-fit', 'beta-fit']:
    #     calc_vars.append('trd_'+var)
    for var in ['SA']:#, 'press', 'CT', 'sigma', 'alpha', 'beta']:
        calc_vars.append('ca_'+var)
        # calc_vars.append('cs_'+var)
        # calc_vars.append('csd_'+var)
        # calc_vars.append('nir_'+var)
        # calc_vars.append('trd_'+var)
    # Calculate new cluster variables
    print('Calculating cluster properties')
    df = ahf.calc_extra_cl_vars(df, calc_vars)

    # Drop duplicates to get one row per cluster
    df = df.drop_duplicates(subset=['cluster'])
    # Drop Time dimension
    df = df.droplevel('Time')
    # Drop the columns for dt_start and entry
    # df = df.drop(columns=['dt_start', 'entry'])
    # Replace the values in dt_start with December 31st of the same year
    #   Build the appropriate date string, taking BGR[XX]YY and converting it to 20[XX]-12-31 00:00:00
    dec_31 = '20'+this_BGR[3:5]+'-12-31 00:00:00'
    df['dt_start'] = dec_31

    # # Add the values from pf data frame to the main data frame
    # print('Adding values from pf data frame to the main data frame')
    # df = df.merge(df_per_pf, left_index=True, right_index=True, how='outer', suffixes=('', '_y'))
    # df.drop(df.filter(regex='_y$').columns, axis=1, inplace=True)

    print('df:')
    print(df)
    print(df.columns)
    # Add the data frame to the list
    dfs.append(df)

# Concatonate all the pandas data frames together
df = pd.concat(dfs)
# Pickle the data frame to a file
pl.dump(df, open('outputs/BGR_all_cluster_properties_unrelab.pickle', 'wb'))
