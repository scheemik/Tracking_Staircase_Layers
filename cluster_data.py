"""
Author: Mikhail Schee
Created: 2023-03-06

This script is set up to run the HDBSCAN clustering algorithm on a set of ITP 
data given certain inputs. It will fill in values in the 'cluster' and 
'clst_prob' columns, as well as for the global attributes:
'Last clustered'
'Clustering x-axis'
'Clustering y-axis'
'Clustering m_pts'
'Clustering ranges'
'Clustering DBCV'

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

# Import the Thermodynamic Equation of Seawater 2010 (TEOS-10) from GSW
# For converting from depth to pressure
import gsw

# For custom analysis functions
import analysis_helper_functions as ahf

################################################################################
# Main execution
################################################################################

dfs0 = ahf.Data_Filters()
dfs1 = ahf.Data_Filters(min_press=400)

LHW_S_range = [34.366, 35.5]
pfs_fltrd = ahf.Profile_Filters(lon_range=[-152.9,-133.7], lat_range=[72.6,77.4], SA_range=LHW_S_range, lt_pCT_max=True)
pfs_fltrd_ss = ahf.Profile_Filters(lon_range=[-152.9,-133.7], lat_range=[72.6,77.4], SA_range=LHW_S_range, lt_pCT_max=True, subsample=True)

## Reproducing figures from Timmermans et al. 2008
ITP2_clstr_dict = {#'netcdf_file':'netcdfs/ITP_2.nc',
                   'sources_dict':{'ITP_2':'all'},
                   'data_filters':dfs0,
                   'pfs_object':ahf.Profile_Filters(SP_range=[34.05,34.75]),
                   'cl_x_var':'SP',
                   'cl_y_var':'la_CT',
                   'm_pts':170
                   }

## Reproducing figures from Lu et al. 2022
ITP3_clstr_dict = {#'netcdf_file':'netcdfs/ITP_3.nc',
                   'sources_dict':{'ITP_3':'all'},
                   'data_filters':dfs0,
                   'SP_range':[34.21,34.82],
                   'cl_x_var':'SP',
                   'cl_y_var':'la_CT',
                   'm_pts':580
                   }

## AIDJEX BigBear
ABB_clstr_dict =  {'netcdf_file':'netcdfs/AIDJEX_BigBear.nc',
                   'sources_dict':{'AIDJEX_BigBear':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_fltrd,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'm_pts':300
                   }

## AIDJEX
AJX_clstr_dict =  {'netcdf_file':'netcdfs/AIDJEX_mpts_500_ell_010.nc',
                   'sources_dict':{'AIDJEX_BigBear':'all','AIDJEX_BlueFox':'all','AIDJEX_Caribou':'all','AIDJEX_Snowbird':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_fltrd,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'm_pts':500
                   }
AJX_clstr_dict =  {'netcdf_file':'netcdfs/AIDJEX_mpts_300_ell_100.nc',
                   'sources_dict':{'AIDJEX_BigBear':'all','AIDJEX_BlueFox':'all','AIDJEX_Caribou':'all','AIDJEX_Snowbird':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_fltrd,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'm_pts':300
                   }

## BGOS
BGOS_clstr_dict = {'netcdf_file':'netcdfs/BGOS_mpts_440_ell_010.nc',
                   'sources_dict':{'ITP_33':'all','ITP_34':'all','ITP_35':'all','ITP_41':'all','ITP_42':'all','ITP_43':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_fltrd,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'm_pts':440
                   }
BGOS_clstr_dict = {'netcdf_file':'netcdfs/BGOS_mpts_360_ell_100.nc',
                   'sources_dict':{'ITP_33':'all','ITP_34':'all','ITP_35':'all','ITP_41':'all','ITP_42':'all','ITP_43':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_fltrd,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'm_pts':360
                   }

## BGOSss
BGOSss_clstr_dict = {'netcdf_file':'netcdfs/BGOSss_mpts_260_ell_010.nc',
                   'sources_dict':{'ITP_33':'all','ITP_34':'all','ITP_35':'all','ITP_41':'all','ITP_42':'all','ITP_43':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_fltrd_ss,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'm_pts':260
                   }
BGOSss_clstr_dict = {'netcdf_file':'netcdfs/BGOSss_mpts_220_ell_100.nc',
                   'sources_dict':{'ITP_33':'all','ITP_34':'all','ITP_35':'all','ITP_41':'all','ITP_42':'all','ITP_43':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_fltrd_ss,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'm_pts':220
                   }

# for clstr_dict in [ITP2_clstr_dict]:#, ITP3_clstr_dict]:
for clstr_dict in [AJX_clstr_dict, BGOSss_clstr_dict, BGOS_clstr_dict]:
    gattrs_to_print =  ['Last modified',
                        'Last modification',
                        'Last clustered',
                        'Clustering x-axis',
                        'Clustering y-axis',
                        'Clustering m_pts',
                        'Clustering filters',
                        'Clustering DBCV']

    gattrs_to_copy = {        # Note: can't store datetime objects in netcdfs
                    'Title':'',
                    'Source':'',
                    'Instrument':'',
                    'Data Attribution':'',
                    'Data Source':'',
                    'Data Organizer':'',
                    'Reference':'',
                    'Original vertical measure':'',
                    'Original temperature measure':'',
                    'Original salinity measure':'',
                    'Sub-sample scheme':'',
                    'Moving average window':''
                    }
    # Find the netcdf to which to write the clustering results
    try:
        my_nc = clstr_dict['netcdf_file']
    except:
        my_nc = False
    # Make new netcdf to store clustering results
    if my_nc != False:
        # Will make a new netcdf to store the clustering results
        ds_var_info = {}
        # ds_coord_info = {}
        # Create data set object
        ds_object = ahf.Data_Set(clstr_dict['sources_dict'], clstr_dict['data_filters'])
        # Copy down the global attributes to put into the new netcdf later
        # Loop through each dataset (one for each instrument)
        for ds in ds_object.arr_of_ds:
            # Loop through the variables to get the information
            # print(ds.variables)
            # exit(0)
            for var in ds.variables:
                var_ = ds.variables[var]
                # print('Variable:',var,'attrs:',var_.attrs)
                ds_var_info[var] = var_.attrs
            # print(ds_var_info)
            # print('')
            # print(ds.coords)
            # for coord in ds.coords:
            #     coord_ = ds.coords[coord]
            #     print(coord_.attrs)
            # exit(0)
            # Loop through each of that dataset's global attributes
            for attr in ds.attrs:
                # See if it's an attribute we want to copy
                if attr in gattrs_to_copy:
                    this_attr = str(ds.attrs[attr])
                    # See whether we've already copied it in
                    if this_attr not in gattrs_to_copy[attr]:
                        gattrs_to_copy[attr] = gattrs_to_copy[attr] + ds.attrs[attr] + ' '
                    #
                #
            #
        #
        # print(gattrs_to_copy)
        # exit(0)
        # Create profile filter object
        pfs_object = clstr_dict['pfs_object']
        # Create plot parameters object
        pp_clstr = ahf.Plot_Parameters(x_vars=[clstr_dict['cl_x_var']], y_vars=[clstr_dict['cl_y_var']], clr_map='cluster', extra_args={'b_a_w_plt':True, 'cl_x_var':clstr_dict['cl_x_var'], 'cl_y_var':clstr_dict['cl_y_var'], 'm_pts':clstr_dict['m_pts'], 'extra_vars_to_keep':['BL_yn', 'up_cast', 'lon', 'lat', 'press_max', 'press_CT_max', 'alpha', 'beta', 'ma_CT']}, legend=True)
        # Create analysis group
        group_test_clstr = ahf.Analysis_Group(ds_object, pfs_object, pp_clstr)
        # Make a figure to run clustering algorithm and check results
        ahf.make_figure([group_test_clstr])

        # Does this netcdf already exist?
        try:
            ds = xr.load_dataset(my_nc)
            print('- Previously,',my_nc)
            # See the variables before
            for attr in gattrs_to_print:
                print('\t',attr+':',ds.attrs[attr])
        except:
            foo = 2
        
        # Pull the dataframes from the analysis group
        dfs = group_test_clstr.data_frames
        # print('len(group_test_clstr.data_frames):',len(dfs))
        # exit(0)
        # Loop through each dataframe each (avoiding index collisions)
        if False:
        # if 'AIDJEX' in gattrs_to_copy['Source']:
            print('Avoiding index collisions')
            for df in dfs:
                # Get list of instruments
                instrmts = np.unique(np.array(df['instrmt']))
                # Need to make the `Time` values unique for each instrument
                # Can only change values of columns, not indexes, so reset Time
                df.reset_index(inplace=True, level='Time')
                # Loop across each instrument
                i = 0
                for instrmt in instrmts:
                    i += 1
                    # Add i seconds to all the time values
                    df.loc[df['instrmt'] == instrmt, ['Time']] = df.loc[df['instrmt'] == instrmt, ['Time']] + np.timedelta64(i, 's')
            new_df = pd.concat(dfs)
            # Make `Time` and index again
            new_df = new_df.set_index('Time', append=True)
            # Turn `Vertical` on and then off again as index to reset index order
            new_df.reset_index(inplace=True, level='Vertical')
            new_df = new_df.set_index('Vertical', append=True)
        else:
            print('Avoiding index collisions for ITPSs by adding `instrmt` as an index')
            new_df = pd.concat(dfs)
            new_df = new_df.set_index('instrmt', append=True)
        # print(new_df)
        # print('Duplicated indices')
        # print(new_df.index[new_df.index.duplicated()].unique())
        # exit(0)
        # Convert back to xarray dataset
        ds = new_df.to_xarray()
        # Put the global attributes back in
        for attr in gattrs_to_copy:
            ds.attrs[attr] = gattrs_to_copy[attr]
        # Update the global variables:
        ds.attrs['Creation date'] = str(datetime.now())
        ds.attrs['Last modified'] = str(datetime.now())
        ds.attrs['Last modification'] = 'Updated clustering'
        ds.attrs['Last clustered'] = str(datetime.now())
        ds.attrs['Clustering x-axis'] = clstr_dict['cl_x_var']
        ds.attrs['Clustering y-axis'] = clstr_dict['cl_y_var']
        ds.attrs['Clustering m_pts'] = clstr_dict['m_pts']
        ds.attrs['Clustering filters'] = ahf.print_profile_filters(clstr_dict['pfs_object'])
        ds.attrs['Clustering DBCV'] = group_test_clstr.data_set.arr_of_ds[0].attrs['Clustering DBCV']
        # Add the variable attributes back in
        for var in ds.variables:
            if var in ds_var_info.keys():
                ds.variables[var].attrs = ds_var_info[var]
        # Write out to netcdf
        print('Writing data to',my_nc)
        ds.to_netcdf(my_nc, 'w')
        # Load in with xarray
        ds2 = xr.load_dataset(my_nc)
        # See the variables after
        for attr in gattrs_to_print:
            print('\t',attr+':',ds2.attrs[attr])
        #
    # Overwrite existing netcdf with clustering results
    else:
        # Make sure only one, complete source file was specified
        if len(clstr_dict['sources_dict']) > 1:
            print('Must specify `netcdf_file` in cluster dictionary when using multiple sources')
            exit(0)
        else:
            # Find the netcdf file name of the original
            # Get a list which should be: (name_of_source, 'all')
            this_source = next(iter(clstr_dict['sources_dict'].items()))
            if this_source[1] == 'all':
                my_nc = 'netcdfs/' + this_source[0] + '.nc'
            else:
                print('Must specify `all` from a source to overwrite netcdf file')
                exit(0)
        # Create data set object
        ds_object = ahf.Data_Set(clstr_dict['sources_dict'], clstr_dict['data_filters'])
        # Find the xarray dataset
        ds = ds_object.xarrs[0]
        # Convert a subset of the dataset to a dataframe
        og_df = ds[['cluster', 'clst_prob']].squeeze().to_dataframe()
        # Get the original dimensions of the data to reshape the arrays later
        len0 = len(og_df.index.get_level_values(0).unique())
        len1 = len(og_df.index.get_level_values(1).unique())
        # See the variables before
        for attr in gattrs_to_print:
            print('\t',attr+':',ds.attrs[attr])

        # Create profile filter object
        pfs_object = clstr_dict['pfs_object']
        # Create plot parameters object
        pp_clstr = ahf.Plot_Parameters(x_vars=[clstr_dict['cl_x_var']], y_vars=[clstr_dict['cl_y_var']], clr_map='cluster', extra_args={'b_a_w_plt':False, 'cl_x_var':clstr_dict['cl_x_var'], 'cl_y_var':clstr_dict['cl_y_var'], 'm_pts':clstr_dict['m_pts']}, legend=False)
        # Create analysis group
        group_test_clstr = ahf.Analysis_Group(ds_object, pfs_object, pp_clstr)
        # Make a figure to run clustering algorithm and check results
        ahf.make_figure([group_test_clstr])

        print('making changes')
        # Pull the now modified dataframe from the analysis group
        new_df = group_test_clstr.data_frames[0][['cluster','clst_prob']]
        # Update the original dataframe from the netcdf with the filtered dataframe from plotting
        og_df.update(new_df)
        # Put the clustering variables back into the dataset
        ds['cluster'].values   = og_df['cluster'].values.reshape((len0, len1))
        ds['clst_prob'].values = og_df['clst_prob'].values.reshape((len0, len1))
        # Update the global variables:
        ds.attrs['Last modified'] = str(datetime.now())
        ds.attrs['Last modification'] = 'Updated clustering'
        ds.attrs['Last clustered'] = str(datetime.now())
        ds.attrs['Clustering x-axis'] = clstr_dict['cl_x_var']
        ds.attrs['Clustering y-axis'] = clstr_dict['cl_y_var']
        ds.attrs['Clustering m_pts'] = clstr_dict['m_pts']
        ds.attrs['Clustering filters'] = ahf.print_profile_filters(clstr_dict['pfs_object'])
        ds.attrs['Clustering DBCV'] = group_test_clstr.data_set.arr_of_ds[0].attrs['Clustering DBCV']
        # Write out to netcdf
        print('Writing data to',my_nc)
        ds.to_netcdf(my_nc, 'w')
        # Load in with xarray
        ds2 = xr.load_dataset(my_nc)
        # See the variables after
        for attr in gattrs_to_print:
            print('\t',attr+':',ds2.attrs[attr])



# 
#     exit(0)
#     # Load in with xarray
#     xarrs, var_attr_dicts = ahf.list_xarrays(clstr_dict['sources_dict'])
#     if len(xarrs) == 1:
#         ds = xarrs[0]
#     else: 
#         # Find the maximum length of the `Vertical` dimension between all xarrays
#         max_vertical = 0
#         for i in range(len(xarrs)):
#             this_vertical = xarrs[i].dims['Vertical']
#             print(this_vertical)
#             max_vertical = max(max_vertical, this_vertical)
#         print('max_vertical:',max_vertical)
#         # exit(0)
#         print(xarrs[0])
#         print('')
#         # Get the list of variables that just have the `Time` dimension
#         list_of_time_vars = []
#         for var in xarrs[0].variables:
#             these_dims = xarrs[0][var].dims
#             if these_dims == ('Time',):
#                 # print(var,xarrs[0][var].dims)
#                 list_of_time_vars.append(var)
#         list_of_time_vars.remove('Time')
#         # print(list_of_time_vars)
#         # Convert to a pandas data frame
#         test_df = xarrs[0].to_dataframe()
#         # print('test_df:')
#         # print(test_df)
#         print('')
#         # Convert back to xarray
#         test_ds = test_df.to_xarray()
#         # Reset the variables to have jus the dimensions they did originally
#         for var in list_of_time_vars:
#             test_ds[var] = test_ds[var].isel(Vertical=0, drop=True)
#         print('test_ds:')
#         print(test_ds)
#         # print(xarrs[0].dims)
#         # Concatonate the xarrays
#         # ds = xr.concat(xarrs)
#         # print('')
#         # print('ds to df:')
#         # print(ds.dims)
#         # print(ds.to_dataframe())
#         exit(0)
# 
#     # Convert a subset of the dataset to a dataframe
#     og_df = ds[['cluster', 'clst_prob']].squeeze().to_dataframe()
#     # Get the original dimensions of the data to reshape the arrays later
#     len0 = len(og_df.index.get_level_values(0).unique())
#     len1 = len(og_df.index.get_level_values(1).unique())
#     # print('Dataframe from netcdf, selected columns:')
#     # print(og_df)
#     # print('')
#     # print('Dataframe from netcdf, non-NaN rows:')
#     # print(og_df[~og_df.isnull().any(axis=1)])
#     # print('')
#     # See the variables before
#     for attr in gattrs_to_print:
#         print('\t',attr+':',ds.attrs[attr])
# 
#     
# 
#     print('making changes')
#     # Pull the now modified dataframe from the analysis group
#     new_df = group_test_clstr.data_frames[0][['cluster','clst_prob']]
#     # print('Dataframe from plotting, selected columns:')
#     # print(new_df)
#     # print('')
#     # print('Dataframe from plotting, non-NaN rows:')
#     # print(new_df[~new_df.isnull().any(axis=1)])
#     # Update the original dataframe from the netcdf with the filtered dataframe from plotting
#     og_df.update(new_df)
#     # print('Dataframe from netcdf, updated:')
#     # print(og_df)
#     # print('')
#     # print('Dataframe from netcdf, non-NaN rows:')
#     # print(og_df[~og_df.isnull().any(axis=1)])
#     # Put the clustering variables back into the dataset
#     ds['cluster'].values   = og_df['cluster'].values.reshape((len0, len1))
#     ds['clst_prob'].values = og_df['clst_prob'].values.reshape((len0, len1))
#     # Update the global variables:
#     ds.attrs['Last modified'] = str(datetime.now())
#     ds.attrs['Last modification'] = 'Updated clustering'
#     ds.attrs['Last clustered'] = str(datetime.now())
#     ds.attrs['Clustering x-axis'] = clstr_dict['cl_x_var']
#     ds.attrs['Clustering y-axis'] = clstr_dict['cl_y_var']
#     ds.attrs['Clustering m_pts'] = clstr_dict['m_pts']
#     ds.attrs['Clustering filters'] = ahf.print_profile_filters(clstr_dict['pfs_object'])
#     ds.attrs['Clustering DBCV'] = group_test_clstr.data_set.arr_of_ds[0].attrs['Clustering DBCV']
#     # Write out to netcdf
#     print('Writing data to',my_nc)
#     ds.to_netcdf(my_nc, 'w')
#     # Load in with xarray
#     ds2 = xr.load_dataset(my_nc)
#     # See the variables after
#     for attr in gattrs_to_print:
#         print('\t',attr+':',ds2.attrs[attr])
