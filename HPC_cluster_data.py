"""
Author: Mikhail Schee
Created: 2023-11-15

This script is set up to cluster data into new netcdfs on the Niagara HPC cluster

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions in source code must retain the accompanying copyright notice, this list of conditions, and the following disclaimer.
    2. Redistributions in binary form must reproduce the accompanying copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Names of the copyright holders must not be used to endorse or promote products derived from this software without prior written permission from the copyright holders.
    4. If any files are modified, you must cause the modified files to carry prominent notices stating that you changed the files and the date of any change.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""

NOTE: Run in parallel with 
$ mpirun -np 2 python3 HPC_cluster_data.py
where the `-np` flag denotes the number of processes
"""

# For running in parallel
from mpi4py import MPI
# Get MPI variables set up
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Only use the root process for clustering
if rank == 0:

    # import libraries
    import numpy as np
    import xarray as xr
    # For formatting data into dataframes
    import pandas as pd
    # For custom analysis functions
    import analysis_helper_functions as ahf
    # For formatting date objects
    from datetime import datetime
    # Import the Thermodynamic Equation of Seawater 2010 (TEOS-10) from GSW
    # For converting from depth to pressure
    import gsw
    # 
    # # Change the matplotlib configure directory to somewhere writable to avoid warnings
    # import os
    # os.environ['MPLCONFIGDIR'] = 'scratch/n/ngrisoua/mschee/.config/matplotlib'

################################################################################

    ## Data Filters
    # Generic
    dfs1 = ahf.Data_Filters(min_press=400)
    # Specific
    dfs1_BGR0508 = ahf.Data_Filters(min_press=this_min_press, date_range=['2005/08/15 00:00:00','2008/08/15 00:00:00'])

    # Profile Filters
    lon_BGR = [-160,-130]
    lat_BGR = [73,81.5]
    test_S_range = [34.4, 34.6]
    pfs_test = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR, p_range=[1000,5], SA_range=test_S_range, lt_pCT_max=True)

    # Cluster dictionaries
    ## BGR ITPs 04
    BGR04_clstr_dict = {'netcdf_file':'netcdfs/BGR04_HPC_clstrd.nc',
                    'sources_dict':{'ITP_002':'all'},
                    'data_filters':dfs1,
                    'pfs_object':pfs_test,
                    'cl_x_var':'SA',
                    'cl_y_var':'la_CT',
                    'cl_z_var':'None',
                    'm_pts':'auto',
                    #    'm_pts':'None',
                    #    'm_pts':190,
                    #    'm_pts':250,
                    }

    ## BGR ITPs 0508
    BGR0508_clstr_dict = {'netcdf_file':'netcdfs/BGR0508.nc',
                    'sources_dict':{'ITP_001':'all','ITP_003':'all','ITP_004':'all','ITP_005':'all','ITP_006':'all','ITP_008':'all','ITP_013':'all','ITP_018':'all','ITP_021':'all','ITP_030':'all'},
                    'data_filters':dfs1_BGR0508,
                    'pfs_object':pfs_test,
                    'cl_x_var':'SA',
                    'cl_y_var':'la_CT',
                    'cl_z_var':'None',
                    'm_pts':'auto',
                    #    'm_pts':'None',
                    #    'm_pts':350,
                    }

################################################################################
# Run the clustering process
################################################################################

    for clstr_dict in [BGR0508_clstr_dict]:
    # for clstr_dict in [BGR0506_clstr_dict, BGR0607_clstr_dict, BGR0708_clstr_dict]:
        gattrs_to_print =  ['Last modified',
                            'Last modification',
                            'Last clustered',
                            'Clustering x-axis',
                            'Clustering y-axis',
                            'Clustering z-axis',
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
            print('Preparing to write to',my_nc)
            # Will make a new netcdf to store the clustering results
            ds_var_info = {}
            # Create data set object
            ds_object = ahf.Data_Set(clstr_dict['sources_dict'], clstr_dict['data_filters'])
            # Copy down the global attributes to put into the new netcdf later
            # Loop through each dataset (one for each instrument)
            for ds in ds_object.arr_of_ds:
                # Loop through the variables to get the information
                for var in ds.variables:
                    var_ = ds.variables[var]
                    ds_var_info[var] = var_.attrs
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
            # Create profile filter object
            pfs_object = clstr_dict['pfs_object']
            # Create plot parameters object
            if clstr_dict['m_pts'] == 'None':
                keep_these_vars = ['ma_CT', 'entry', 'prof_no', 'CT', 'SA']
                pp_clstr = ahf.Plot_Parameters(x_vars=[clstr_dict['cl_x_var']], y_vars=[clstr_dict['cl_y_var']], clr_map='clr_by_instrmt', extra_args={'extra_vars_to_keep':keep_these_vars}, legend=True)
            else:
                # keep_these_vars = ['la_CT', 'entry', 'prof_no', 'CT', 'SA']
                keep_these_vars = ['entry', 'prof_no', 'BL_yn', 'dt_start', 'dt_end', 'lon', 'lat', 'region', 'up_cast', 'press_max', 'CT_max', 'press_CT_max', 'SA_CT_max', 'R_rho', 'press', 'depth', 'iT', 'CT', 'PT', 'SP', 'SA', 'sigma', 'alpha', 'beta', 'aCT', 'BSA', 'ss_mask', 'ma_iT', 'ma_CT', 'ma_PT', 'ma_SP', 'ma_SA', 'ma_sigma', 'la_iT', 'la_CT', 'la_PT', 'la_SP', 'la_SA', 'la_sigma']
                pp_clstr = ahf.Plot_Parameters(x_vars=[clstr_dict['cl_x_var']], y_vars=[clstr_dict['cl_y_var']], clr_map='cluster', extra_args={'b_a_w_plt':True, 'cl_x_var':clstr_dict['cl_x_var'], 'cl_y_var':clstr_dict['cl_y_var'], 'm_pts':clstr_dict['m_pts'], 'extra_vars_to_keep':keep_these_vars}, legend=True)
            # Create analysis group
            print('creating analysis group')
            group_test_clstr = ahf.Analysis_Group(ds_object, pfs_object, pp_clstr)
            # Make a figure to run clustering algorithm and check results
            print('making figure')
            ahf.make_figure([group_test_clstr], filename='test.png')

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
            new_df = pd.concat(dfs)
            if clstr_dict['m_pts'] == 'None':
                # Remove unnecessary variables
                print('variables in new_df:',list(new_df))
                only_need_these_vars = ['entry','prof_no','CT','ma_CT','SA',]
                for var in list(new_df):
                    if not var in only_need_these_vars:
                        new_df = new_df.drop(var, axis=1)
            print('after, vars in new_df:',list(new_df))
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
            ds.attrs['Clustering z-axis'] = clstr_dict['cl_z_var']
            ds.attrs['Clustering m_pts'] = group_test_clstr.data_set.arr_of_ds[0].attrs['Clustering m_pts']
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