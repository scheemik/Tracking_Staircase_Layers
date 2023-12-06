"""
Author: Mikhail Schee
Created: 2023-10-03

This script is set up to run parameter sweeps on the Niagara HPC cluster

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
$ mpirun -np 2 python3 HPC_param_sweep.py
where the `-np` flag denotes the number of processes
"""

import numpy as np
# For formatting data into dataframes
import pandas as pd
# For custom analysis functions
import analysis_helper_functions as ahf
# For running in parallel
from mpi4py import MPI
# For formatting date objects
from datetime import datetime
# 
# # Change the matplotlib configure directory to somewhere writable to avoid warnings
# import os
# os.environ['MPLCONFIGDIR'] = 'scratch/n/ngrisoua/mschee/.config/matplotlib'

# Title
this_plot_title = 'BGR0708'

# Get MPI variables set up
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Only handle file I/O on the root process
if rank == 0:
    # Open a text file to record values from the parameter sweep
    sweep_txt_file = 'outputs/'+this_plot_title+'_ps.csv'
    f = open(sweep_txt_file,'w')
    f.write('Parameter Sweep for '+this_plot_title+'\n')
    f.write(datetime.now().strftime("%I:%M%p on %B %d, %Y")+'\n')
    f.write('m_pts,ell_size,n_clusters,DBCV\n')
    f.close()

# Reduce the number of active processes
rf = 4
if rank%rf == 0:

    ITP2_S_range = [34.05,34.75]
    LHW_S_range = [34.366, 35.5]
    test_S_range = [34.4, 34.6]

################################################################################
# Make dictionaries for what data to load in and analyze
################################################################################

    ## Preclustered files
    BGR04   = {'minimal_BGR04':'all'}
    BGR0506 = {'minimal_BGR0506':'all'}
    BGR0607 = {'minimal_BGR0607':'all'}
    BGR0708 = {'minimal_BGR0708':'all'}
    BGR0809 = {'minimal_BGR0809':'all'}
    BGR0910 = {'minimal_BGR0910':'all'}
    BGR1011 = {'minimal_BGR1011':'all'}
    BGR0508 = {'BGR0508':'all'}
    ITP2 = {'ITP_002':'all'}

################################################################################
# Create data filtering objects
# print('- Creating data filtering objects')
################################################################################

    this_min_press = 400

    dfs_all = ahf.Data_Filters(keep_black_list=True, cast_direction='any')
    dfs0 = ahf.Data_Filters()
    dfs1 = ahf.Data_Filters(min_press=this_min_press)

################################################################################
# Create data sets by combining filters and the data to load in
# print('- Creating data sets')
################################################################################

    # ds_this_BGR = ahf.Data_Set(BGR04, dfs_all)
    # ds_this_BGR = ahf.Data_Set(BGR0506, dfs_all)
    # ds_this_BGR = ahf.Data_Set(BGR0607, dfs_all)
    ds_this_BGR = ahf.Data_Set(BGR0708, dfs_all)
    # ds_this_BGR = ahf.Data_Set(BGR0809, dfs_all)
    # ds_this_BGR = ahf.Data_Set(BGR0910, dfs_all)
    # ds_this_BGR = ahf.Data_Set(BGR1011, dfs_all)
    # ds_this_BGR = ahf.Data_Set(BGR0508, dfs_all)
    # ds_this_BGR = ahf.Data_Set(ITP2, dfs0)

################################################################################
# Create profile filtering objects
# print('- Creating profile filtering objects')
################################################################################

    pfs_0 = ahf.Profile_Filters()
    pfs_1 = ahf.Profile_Filters(every_nth_row=4)

    test_p_range = [400,200]
    pfs_ell  = ahf.Profile_Filters(p_range=test_p_range)
    pfs_ell  = ahf.Profile_Filters(SA_range=test_S_range)

################################################################################

    ## Preclustered
    # pfs_this_BGR = pfs_0
    # pfs_this_BGR = pfs_1

    pfs_this_BGR = pfs_ell

################################################################################
### Figures

    ## Clustering parameter sweeps
    # ## Parameter sweep for BGR ITP data
    if True:
        print('')
        print('- Creating clustering parameter sweep for BGR ITP data')
        test_mpts = 360
        # Make the Plot Parameters
        pp = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'cl_ps_tuple':[10,801,10], 'mpi_run':True}) #[10,721,10]
        # Make the subplot groups
        group_mpts_param_sweep = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp, plot_title=this_plot_title)
        # Run the parameter sweep
        df = ahf.make_subplot(None, group_mpts_param_sweep, None, None)
        # Grab just the datasets from the analysis group
        arr_of_ds = group_mpts_param_sweep.data_set.arr_of_ds
    # Build the array for the x_var axis
    cl_ps_tuple = pp.extra_args['cl_ps_tuple']
    x_var_array = np.arange(cl_ps_tuple[0], cl_ps_tuple[1], cl_ps_tuple[2])

    # Divide the runs among the processes
    x_var_array = x_var_array[int(rank/rf)::int(size/rf)]

################################################################################
# Start the parameter sweeps

    # Set the main x and y data keys
    x_key = pp.x_vars[0]
    y_key = pp.y_vars[0]
    print('\trank',rank,'plotting these x values of',x_key,':',x_var_array)
    # Get cluster arguments
    m_pts, min_s, cl_x_var, cl_y_var, cl_z_var, plot_slopes, b_a_w_plt = ahf.get_cluster_args(pp)
    # If limiting the number of pfs, find total number of pfs in the given df
    #   In the multi-index of df, level 0 is 'Time'
    if x_key == 'n_pfs':
        pf_nos = np.unique(np.array(df['prof_no'].values, dtype=type('')))
        number_of_pfs = len(pf_nos)
        print('\tNumber of profiles:',number_of_pfs)
        x_var_array = x_var_array[x_var_array <= number_of_pfs]
    #
    x_len = len(x_var_array)
    lines = []
    for x in x_var_array:
        # Set initial values for some variables
        zlabel = None
        this_df = df.copy()
        # Set parameters based on variables selected
        #   NOTE: need to run `ell_size` BEFORE `n_pfs`
        if x_key == 'ell_size':
            # Need to apply moving average window to original data, before
            #   the data filters were applied, so make a new Analysis_Group
            a_group.profile_filters.m_avg_win = x
            new_a_group = Analysis_Group(a_group.data_set, a_group.profile_filters, a_group.plt_params)
            this_df = pd.concat(new_a_group.data_frames)
            xlabel = r'$\ell$ (dbar)'
        if x_key == 'n_pfs':
            this_df = this_df[this_df['prof_no'] <= pf_nos[x-1]].copy()
            xlabel = 'Number of profiles included'
        if x_key == 'm_pts':
            # min cluster size must be an integer
            m_pts = int(x)
            xlabel = r'$m_{pts}$'
        elif x_key == 'min_samps':
            # min samples must be an integer, or None
            if not isinstance(x, type(None)):
                min_s = int(x)
            else:
                min_s = x
            xlabel = 'Minimum samples'
        # Run the HDBSCAN algorithm on the provided dataframe
        try:
            new_df, rel_val, m_pts, ell = ahf.HDBSCAN_(arr_of_ds, this_df, cl_x_var, cl_y_var, cl_z_var, m_pts, min_samp=min_s, param_sweep=True)
        except:
            print('rank',rank,'failed to run HDBSCAN for',x_key,'=',x)
            break
        # Record outputs to output object
        lines.append(str(m_pts)+','+str(ell)+','+str(new_df['cluster'].max()+1)+','+str(rel_val)+'\n')
else:
    lines = ''
################################################################################

# Gather the data from all the processes
lines = comm.gather(lines, root=0)
if rank == 0:
    output_lines = []
    for entry in lines:
        output_lines.append(''.join(entry))
    print(output_lines)
    f = open(sweep_txt_file,'a')
    f.write(''.join(output_lines))
    f.close()


