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
    # For formatting data into dataframes
    import pandas as pd
    # For working with netcdfs
    import xarray as xr
    # For formatting date objects
    from datetime import datetime
    # Import the Thermodynamic Equation of Seawater 2010 (TEOS-10) from GSW
    # For converting from depth to pressure
    import gsw
    # 
    # # Change the matplotlib configure directory to somewhere writable to avoid warnings
    # import os
    # os.environ['MPLCONFIGDIR'] = 'scratch/n/ngrisoua/mschee/.config/matplotlib'
    # For custom analysis functions
    import analysis_helper_functions as ahf
    # For common BGR parameters
    import BGR_params as bps
    # For common BGR objects
    import BGR_objects as bob

################################################################################
# Defining parameters
################################################################################

    # Define prefix for netcdf file name
    file_prefix = 'HPC_'
    # Choose BGR period
    this_BGR = 'BGR0506'
    # Define the clustering dictionary
    this_BGR_clstr_dict = bob.build_clustering_dict(file_prefix, this_BGR, m_pts='auto')

################################################################################
# Run the clustering process
################################################################################

    for clstr_dict in [this_BGR_clstr_dict]:
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
                keep_these_vars = ['source', 'instrmt', 'entry', 'prof_no', 'SA', 'CT', 'ma_CT']
                pp_clstr = ahf.Plot_Parameters(x_vars=[clstr_dict['cl_x_var']], y_vars=[clstr_dict['cl_y_var']], clr_map='clr_by_instrmt', extra_args={'extra_vars_to_keep':keep_these_vars}, legend=True)
            else:
                keep_these_vars = ['source', 'instrmt', 'entry', 'prof_no', 'SA', 'CT', 'ma_CT']
                # keep_these_vars = ['entry', 'prof_no', 'BL_yn', 'dt_start', 'dt_end', 'lon', 'lat', 'region', 'up_cast', 'press_max', 'press_min', 'CT_TC_max', 'CT_TC_min', 'press_TC_max', 'press_TC_min', 'SA_TC_max', 'SA_TC_min', 'sig_TC_max', 'sig_TC_min', 'R_rho', 'press', 'depth', 'iT', 'CT', 'PT', 'SP', 'SA', 'sigma', 'alpha', 'beta', 'aCT', 'BSA', 'ss_mask', 'ma_iT', 'ma_CT', 'ma_PT', 'ma_SP', 'ma_SA', 'ma_sigma', 'la_iT', 'la_CT', 'la_PT', 'la_SP', 'la_SA', 'la_sigma']
                pp_clstr = ahf.Plot_Parameters(x_vars=[clstr_dict['cl_x_var']], y_vars=[clstr_dict['cl_y_var']], clr_map='cluster', extra_args={'b_a_w_plt':True, 'cl_x_var':clstr_dict['cl_x_var'], 'cl_y_var':clstr_dict['cl_y_var'], 'm_pts':clstr_dict['m_pts'], 'm_cls':clstr_dict['m_cls'], 'extra_vars_to_keep':keep_these_vars, 'relab_these':clstr_dict['relab_these']}, legend=True)
            # Create analysis group
            print('Creating analysis group')
            group_test_clstr = ahf.Analysis_Group(ds_object, pfs_object, pp_clstr)
            # Make a figure to run clustering algorithm and check results
            print('making figure')
            ahf.make_figure([group_test_clstr], filename=file_prefix+clstr_dict['name']+'_mpts_'+str(clstr_dict['m_pts'])+'.png')

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
                only_need_these_vars = keep_these_vars # ['entry','prof_no','CT','ma_CT','SA',]
                for var in list(new_df):
                    if not var in only_need_these_vars:
                        new_df = new_df.drop(var, axis=1)
            print('after, vars in new_df:',list(new_df))
            # Convert back to xarray dataset
            ds = new_df.to_xarray()
            pf_vars = ahf.pf_vars + ['instrmt', 'source', 'notes', 'SA_CT_max', 'press_max']
            # Get the values of the dimensions
            time_vals, vert_vals = ds.indexes.values()
            # Loop across the Time dimension
            for this_time in time_vals:
                for var in ds.variables:
                    if var in pf_vars:
                        # print('Adjusting this var:',var)
                        # Get the array of values for this var for this Time
                        these_vals = ds[var].sel(Time=this_time)
                        if var in ['dt_start', 'dt_end', 'region', 'instrmt', 'source', 'notes']:
                            these_vals1 = [x for x in np.array(these_vals) if not isinstance(x, float)]
                        else:
                            these_vals1 = np.unique(these_vals[~np.isnan(these_vals)])
                        if len(these_vals1) > 0:
                            this_val = these_vals1[0]
                        else:
                            this_val = np.nan
                        # print('\tthis_val:',this_val)
                        # Copy this value to all `Vertical` entries for this variable
                        ds[var].loc[dict(Time=this_time)] = this_val
                    #
                #
            # Drop the Vertical dimension for variables which vary in Time only
            for var in ds.variables:
                if var in pf_vars:
                    # print('Dropping Vertical dimension for this var:',var)
                    ds[var] = ds[var].isel(Vertical=0, drop=True)
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
            ds.attrs['Clustering m_cls'] = group_test_clstr.data_set.arr_of_ds[0].attrs['Clustering m_cls']
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