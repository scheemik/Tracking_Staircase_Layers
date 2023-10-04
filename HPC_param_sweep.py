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

# Change the matplotlib configure directory to somewhere writable to avoid warnings
import os
os.environ['MPLCONFIGDIR'] = 'scratch/n/ngrisoua/mschee/.config/matplotlib'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Only do the data filtering on the root process
if rank == 0:
    LHW_S_range = [34.366, 35.5]

################################################################################
# Make dictionaries for what data to load in and analyze
################################################################################

    # Sets of ITPs within the BGR that appear within certain time ranges
    ## 2004-08-20 00:00:01 to 2004-09-29 00:00:05, duration: 41 days
    BGR_ITPs_0a = {'ITP_002':'all'}
    ## 2005-08-16 06:00:01 to 2007-01-08 18:00:03, gap: 320 days, duration: 511 days
    BGR_ITPs_0b = { 'ITP_001':'all',
                    # 'ITP_002':'all', # Not in this time range
                    'ITP_003':'all'
                    }
    ## 2006-09-03 06:00:01 to 2008-05-13 00:00:02, overlap: 128 days, duration: 619 days
    BGR_ITPs_0c = { 'ITP_004':'all',
                    'ITP_005':'all',
                    'ITP_006':'all'
                    }
    ## 2007-08-13 00:00:01 to 2008-09-15 00:00:04, overlap: 275 days, duration: 400 days
    BGR_ITPs_0d = { # 'ITP_007':'all', # Not in BGR
                    'ITP_008':'all',
                    # 'ITP_009':'all', # Not in BGR
                    # 'ITP_010':'all', # Not in BGR
                    # 'ITP_011':'all', # Not in this time period, can't time slice
                    # 'ITP_012':'all', # Not in BGR
                    'ITP_013':'all',  # Can't time slice
                    # 'ITP_014':'all', # Not in BGR
                    # 'ITP_015':'all', # Not in BGR
                    # 'ITP_016':'all', # Not in BGR
                    # 'ITP_017':'all', # Not in BGR
                    'ITP_018':'all',
                    # 'ITP_019':'all', # Not in BGR
                    'ITP_021':'all',
                    # 'ITP_022':'all', # Not in BGR
                    # 'ITP_023':'all', # Not in BGR
                    # 'ITP_024':'all', # Not in BGR
                    # 'ITP_025':'all', # Not in this time period
                    # 'ITP_026':'all', # Not in BGR
                    # 'ITP_027':'all', # Not in BGR
                    # 'ITP_028':'all', # Not in BGR
                    # 'ITP_029':'all', # Not in BGR
                    'ITP_030':'all',
                    }
    ## 2008-09-21 00:00:05 to 2009-08-31 00:00:08, gap: 7 days, duration: 345 days
    BGR_ITPs_0e = { 'ITP_008':'all',
                    # 'ITP_009':'all', # Not in BGR
                    # 'ITP_010':'all', # Not in BGR
                    'ITP_011':'all', # Can't time slice
                    # 'ITP_012':'all', # Not in BGR
                    # 'ITP_013':'all',  # Not in this time period, can't time slice
                    # 'ITP_014':'all', # Not in BGR
                    # 'ITP_015':'all', # Not in BGR
                    # 'ITP_016':'all', # Not in BGR
                    # 'ITP_017':'all', # Not in BGR
                    # 'ITP_018':'all', # Not in this time period
                    # 'ITP_019':'all', # Not in BGR
                    'ITP_021':'all', 
                    # 'ITP_022':'all', # Not in BGR
                    # 'ITP_023':'all', # Not in BGR
                    # 'ITP_024':'all', # Not in BGR
                    'ITP_025':'all', 
                    # 'ITP_026':'all', # Not in BGR
                    # 'ITP_027':'all', # Not in BGR
                    # 'ITP_028':'all', # Not in BGR
                    # 'ITP_029':'all', # Not in BGR
                    # 'ITP_030':'all', # Not in this time period
                    }
    ## 2009-10-04 06:00:02 to 2010-06-08 00:00:07, gap: 33 days, duration: 248 days
    BGR_ITPs_0f = { 'ITP_032':'all',
                    'ITP_033':'all',
                    'ITP_034':'all',
                    'ITP_035':'all',
                    }
    ## 2010-06-16 00:00:06 to 2011-06-08 00:00:06, gap: 8 days, duration: 358 days
    BGR_ITPs_0g = { 'ITP_033':'all',
                    # 'ITP_036':'all', # Not in BGR
                    # 'ITP_037':'all', # Not in BGR
                    # 'ITP_038':'all', # Not in BGR
                    'ITP_041':'all',
                    'ITP_042':'all',
                    'ITP_043':'all'
                    }
    ## 2011-06-11 00:00:05 to 2012-08-06 06:00:05, gap: 2 days, duration: 423 days
    BGR_ITPs_0h = { 'ITP_041':'all',
                    # 'ITP_042':'all', # Not in this time period
                    # 'ITP_043':'all', # Not in this time period
                    # 'ITP_047':'all', # Not in BGR
                    # 'ITP_048':'all', # Not in BGR
                    # 'ITP_049':'all', # Not in BGR
                    # 'ITP_051':'all', # Not in BGR
                    'ITP_052':'all',
                    'ITP_053':'all',
                    'ITP_054':'all',
                    'ITP_055':'all'
                    }
    ## 2012-08-08 00:00:07 to 2013-08-23 12:02:02, gap: 1 day, duration: 381 days
    BGR_ITPs_0i = { 'ITP_041':'all',
                    # 'ITP_042':'all', # Not in this time period
                    # 'ITP_043':'all', # Not in this time period
                    # 'ITP_047':'all', # Not in BGR
                    # 'ITP_048':'all', # Not in BGR
                    # 'ITP_049':'all', # Not in BGR
                    # 'ITP_051':'all', # Not in BGR
                    # 'ITP_052':'all', # Not in this time period
                    # 'ITP_053':'all', # Not in this time period
                    # 'ITP_054':'all', # Not in this time period
                    # 'ITP_055':'all', # Not in this time period
                    # 'ITP_056':'all', # Not in BGR
                    # 'ITP_057':'all', # Not in BGR
                    # 'ITP_058':'all', # Not in BGR
                    # 'ITP_059':'all', # Not in BGR
                    # 'ITP_060':'all', # Not in BGR
                    # 'ITP_061':'all', # Not in BGR
                    'ITP_062':'all',
                    # 'ITP_063':'all', # Not in BGR
                    'ITP_064':'all',
                    'ITP_065':'all',
                    }
    ## 2013-08-26 00:02:01 to 2014-10-01 00:02:01, gap: 2 days, duration: 402 days
    BGR_ITPs_0j = { 'ITP_068':'all',
                    'ITP_069':'all',
                    'ITP_070':'all',
                    # 'ITP_072':'all', # Not in BGR
                    # 'ITP_073':'all', # Not in BGR
                    # 'ITP_074':'all', # Not in BGR
                    # 'ITP_075':'all', # Not in BGR
                    # 'ITP_076':'all', # Not in BGR
                    'ITP_077':'all',
                    'ITP_078':'all',
                    'ITP_079':'all'
                    }
    ## 2014-08-14 00:02:01 to 2015-09-04 00:02:00, overlap: 49 days, duration: 387 days
    BGR_ITPs_0k = { 'ITP_080':'all',
                    'ITP_081':'all',
                    'ITP_082':'all',
                    # 'ITP_083':'all', # Not in BGR
                    'ITP_084':'all',
                    'ITP_085':'all',
                    'ITP_086':'all',
                    'ITP_087':'all'
                    }
    ## 2015-09-06 00:02:01 to 2016-05-02 00:02:01, gap: 1 day, duration: 240 days
    BGR_ITPs_0l = { 'ITP_082':'all',
                    # 'ITP_083':'all', # Not in BGR
                    # 'ITP_084':'all', # Not in this time period
                    # 'ITP_085':'all', # Not in this time period
                    # 'ITP_086':'all', # Not in this time period
                    # 'ITP_087':'all', # Not in this time period
                    'ITP_088':'all',
                    'ITP_089':'all',
                    # 'ITP_090':'all', # Not in BGR
                    # 'ITP_091':'all', # Not in BGR
                    # 'ITP_092':'all', # Not in BGR
                    # 'ITP_094':'all', # Not in BGR
                    # 'ITP_095':'all', # Not in BGR
                    }
    ## 2016-10-03 00:02:02 to 2017-08-04 00:02:01, gap: 153 days, duration: 306 days
    BGR_ITPs_0m = { 'ITP_097':'all',
                    # 'ITP_098':'all', # Just barely outside BGR
                    'ITP_099':'all',
                    }
    ## 2017-09-17 00:02:02 to 2018-07-21 00:02:02, gap: 43 days, duration: 308 days
    BGR_ITPs_0n = { 'ITP_097':'all',
                    'ITP_100':'all',
                    'ITP_101':'all',
                    # 'ITP_102':'all', # Not in BGR
                    # 'ITP_103':'all', # Not in this time period
                    # 'ITP_104':'all', # Not in this time period
                    # 'ITP_105':'all', # Not in this time period
                    # 'ITP_107':'all', # Not in this time period
                    'ITP_108':'all',
                    }
    ## 2018-09-19 00:02:03 to 2019-04-14 00:02:02, gap: 59 days, duration: 208 days
    BGR_ITPs_0o = { 'ITP_103':'all',
                    'ITP_104':'all',
                    'ITP_105':'all',
                    'ITP_107':'all',
                    # 'ITP_108':'all', # Not in this time period
                    'ITP_109':'all',
                    'ITP_110':'all',
                    }
    ## 2019-05-01 00:02:02 to 2019-08-31 00:02:03, gap: 16 days, duration: 123 days
    BGR_ITPs_0p = { 'ITP_103':'all',
                    # 'ITP_104':'all', # Not in this time period
                    'ITP_105':'all',
                    'ITP_107':'all',
                    # 'ITP_108':'all', # Not in this time period
                    # 'ITP_109':'all', # Not in this time period
                    'ITP_110':'all',
                    # 'ITP_111':'all', # Not in BGR
                    }
    ## 2019-09-20 00:02:03 to 2020-04-02 17:33:46, gap: 19 days, duration: 196 days
    BGR_ITPs_0q = { 'ITP_113':'all',
                    'ITP_114':'all',
                    # 'ITP_116':'all', # Not in BGR
                    'ITP_117':'all',
                    'ITP_118':'all',
                    }
    ## 2020-05-20 17:33:46 to 2021-08-31 00:02:02, gap: 47 days, duration: 469 days
    BGR_ITPs_0r = { 'ITP_113':'all',
                    'ITP_114':'all',
                    # 'ITP_116':'all', # Not in BGR
                    # 'ITP_117':'all', # Not in this time period
                    # 'ITP_118':'all', # Not in this time period
                    'ITP_120':'all',
                    'ITP_121':'all',
                    }
    ## 
    BGR_ITPs_0s = { 'ITP_113':'all',
                    'ITP_114':'all',
                    # 'ITP_116':'all', # Not in BGR
                    'ITP_117':'all',
                    'ITP_118':'all',
                    'ITP_120':'all',
                    'ITP_121':'all',
                    'ITP_122':'all',
                    'ITP_123':'all',
                    'ITP_125':'all',
                    'ITP_128':'all',
                    }

    ## All
    # BGR_ITPs_all = {**BGR_ITPs_0a, **BGR_ITPs_0b, **BGR_ITPs_0c, **BGR_ITPs_0d, **BGR_ITPs_0e, **BGR_ITPs_0f, **BGR_ITPs_0g, **BGR_ITPs_0h, **BGR_ITPs_0i}

################################################################################
    # Create data filtering objects
    print('- Creating data filtering objects')
################################################################################

    this_min_press = 400

    dfs_all = ahf.Data_Filters(keep_black_list=True, cast_direction='any')
    dfs0 = ahf.Data_Filters()
    dfs1 = ahf.Data_Filters(min_press=this_min_press)

    # Beaufort Gyre Region
    dfs1_BGR_0d = ahf.Data_Filters(min_press=this_min_press, date_range=['2007/08/11 00:00:00','2008/09/18 00:00:00'])
    dfs1_BGR_0e = ahf.Data_Filters(min_press=this_min_press, date_range=['2008/09/18 00:00:00','2009/09/01 00:00:00'])
    dfs1_BGR_0f = ahf.Data_Filters(min_press=this_min_press, date_range=['2009/08/31 00:00:08','2010/06/11 00:00:00'])
    dfs1_BGR_0g = ahf.Data_Filters(min_press=this_min_press, date_range=['2010/06/11 00:00:00','2011/06/09 00:00:00'])
    dfs1_BGR_0h = ahf.Data_Filters(min_press=this_min_press, date_range=['2011/06/09 00:00:00','2012/08/07 00:00:00'])
    dfs1_BGR_0i = ahf.Data_Filters(min_press=this_min_press, date_range=['2012/08/07 00:00:00','2013/08/25 00:00:00'])
    dfs1_BGR_0j = ahf.Data_Filters(min_press=this_min_press, date_range=['2013/08/25 00:00:00','2014/10/10 00:00:00'])
    dfs1_BGR_0k = ahf.Data_Filters(min_press=this_min_press, date_range=['2013/08/25 00:00:00','2015/09/05 00:00:00'])
    dfs1_BGR_0l = ahf.Data_Filters(min_press=this_min_press, date_range=['2015/09/05 00:00:00','2016/07/10 00:00:00'])
    dfs1_BGR_0m = ahf.Data_Filters(min_press=this_min_press, date_range=['2016/07/10 00:00:00','2017/08/29 00:00:00'])
    dfs1_BGR_0n = ahf.Data_Filters(min_press=this_min_press, date_range=['2017/08/29 00:00:00','2018/08/17 00:00:00'])
    dfs1_BGR_0o = ahf.Data_Filters(min_press=this_min_press, date_range=['2018/08/17 00:00:00','2019/04/23 00:00:00'])
    dfs1_BGR_0p = ahf.Data_Filters(min_press=this_min_press, date_range=['2019/04/23 00:00:00','2019/09/10 00:00:00'])
    dfs1_BGR_0q = ahf.Data_Filters(min_press=this_min_press, date_range=['2019/09/10 00:00:00','2020/04/22 00:00:00'])
    dfs1_BGR_0r = ahf.Data_Filters(min_press=this_min_press, date_range=['2020/04/22 00:00:00','2021/09/03 00:00:00'])
    dfs1_BGR_0s = ahf.Data_Filters(min_press=this_min_press, date_range=['2020/07/01 00:00:00','2022/01/01 00:00:00'])

################################################################################
    # Create data sets by combining filters and the data to load in
    print('- Creating data sets')
################################################################################

    ## ITP

    # ds_all_ITP = ahf.Data_Set(all_ITPs, dfs_all)

    ## BGR ITP datasets, by time period
    # ds_BGR_ITPs_all = ahf.Data_Set(BGR_ITPs_all, dfs1)
    ds_BGR_ITPs_0a = ahf.Data_Set(BGR_ITPs_0a, dfs1)
    # ds_BGR_ITPs_0b = ahf.Data_Set(BGR_ITPs_0b, dfs1)
    # ds_BGR_ITPs_0c = ahf.Data_Set(BGR_ITPs_0c, dfs1)
    # ds_BGR_ITPs_0d = ahf.Data_Set(BGR_ITPs_0d, dfs1_BGR_0d)
    # ds_BGR_ITPs_0e = ahf.Data_Set(BGR_ITPs_0e, dfs1_BGR_0e)
    # ds_BGR_ITPs_0f = ahf.Data_Set(BGR_ITPs_0f, dfs1_BGR_0f)
    # ds_BGR_ITPs_0g = ahf.Data_Set(BGR_ITPs_0g, dfs1_BGR_0g)
    # ds_BGR_ITPs_0h = ahf.Data_Set(BGR_ITPs_0h, dfs1_BGR_0h)
    # ds_BGR_ITPs_0i = ahf.Data_Set(BGR_ITPs_0i, dfs1_BGR_0i)
    # ds_BGR_ITPs_0j = ahf.Data_Set(BGR_ITPs_0j, dfs1_BGR_0j)
    # ds_BGR_ITPs_0k = ahf.Data_Set(BGR_ITPs_0k, dfs1_BGR_0k)
    # ds_BGR_ITPs_0l = ahf.Data_Set(BGR_ITPs_0l, dfs1_BGR_0l)
    # ds_BGR_ITPs_0m = ahf.Data_Set(BGR_ITPs_0m, dfs1_BGR_0m)
    # ds_BGR_ITPs_0n = ahf.Data_Set(BGR_ITPs_0n, dfs1_BGR_0n)
    # ds_BGR_ITPs_0o = ahf.Data_Set(BGR_ITPs_0o, dfs1_BGR_0o)
    # ds_BGR_ITPs_0p = ahf.Data_Set(BGR_ITPs_0p, dfs1_BGR_0p)
    # ds_BGR_ITPs_0q = ahf.Data_Set(BGR_ITPs_0q, dfs1_BGR_0q)
    # ds_BGR_ITPs_0r = ahf.Data_Set(BGR_ITPs_0r, dfs1_BGR_0r)

################################################################################
    # Create profile filtering objects
    print('- Creating profile filtering objects')
################################################################################

    pfs_0 = ahf.Profile_Filters()

    # Beaufort Gyre Region (BGR), see Shibley2022
    lon_BGR = [-160,-130]
    lat_BGR = [73,81.5]
    pfs_BGR = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR)
    pfs_BGR1 = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR, p_range=[1000,5], SA_range=LHW_S_range, lt_pCT_max=True)
    pfs_BGR1_4 = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR, p_range=[1000,5], SA_range=LHW_S_range, lt_pCT_max=True, every_nth_row=4)

################################################################################

    # Use these things
    # pfs_this_BGR = pfs_BGR1
    pfs_this_BGR = pfs_BGR1_4
    # ds_this_BGR = ds_BGR_ITPs_all
    ds_this_BGR = ds_BGR_ITPs_0a
    # ds_this_BGR = ds_BGR_ITPs_0b
    # ds_this_BGR = ds_BGR_ITPs_0c
    # ds_this_BGR = ds_BGR_ITPs_0d
    # ds_this_BGR = ds_BGR_ITPs_0e
    # ds_this_BGR = ds_BGR_ITPs_0f
    # ds_this_BGR = ds_BGR_ITPs_0g
    # ds_this_BGR = ds_BGR_ITPs_0h
    # ds_this_BGR = ds_BGR_ITPs_0i
    # ds_this_BGR = ds_BGR_ITPs_0j
    # ds_this_BGR = ds_BGR_ITPs_0k
    # ds_this_BGR = ds_BGR_ITPs_0l
    # ds_this_BGR = ds_BGR_ITPs_0m
    # ds_this_BGR = ds_BGR_ITPs_0n
    # ds_this_BGR = ds_BGR_ITPs_0o
    # ds_this_BGR = ds_BGR_ITPs_0p
    # ds_this_BGR = ds_BGR_ITPs_0q
    # ds_this_BGR = ds_BGR_ITPs_0r

################################################################################
### Figures

    ## Clustering parameter sweeps
    # ## Parameter sweep for BGR ITP data
    this_plot_title = 'BGRa_n4'
    if True:
        print('')
        print('- Creating clustering parameter sweep for BGR ITP data')
        test_mpts = 360
        # Make the Plot Parameters
        pp_mpts_param_sweep = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'cl_ps_tuple':[10,721,10], 'mpi_run':True}) #[10,721,10]
        # Make the subplot groups
        group_mpts_param_sweep = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_mpts_param_sweep, plot_title=this_plot_title)
        # Run the parameter sweep
        df = ahf.make_subplot(None, group_mpts_param_sweep, None, None)
        # print(df)
    # Open a text file to record values from the parameter sweep
    sweep_txt_file = 'outputs/'+this_plot_title+'_ps.csv'
    f = open(sweep_txt_file,'w')
    f.write('Parameter Sweep for '+this_plot_title+'\n')
    f.write(datetime.now().strftime("%I:%M%p on %B %d, %Y")+'\n')
    f.write('m_pts,ell_size,n_clusters,DBCV\n')
    f.close()
    # Build the array for the x_var axis
    cl_ps_tuple =pp_mpts_param_sweep.extra_args['cl_ps_tuple']
    x_var_array = np.arange(cl_ps_tuple[0], cl_ps_tuple[1], cl_ps_tuple[2])
else:
    df = None
    pp_mpts_param_sweep = None
    group_mpts_param_sweep = None
    x_var_array = None

# Broadcast the filtered dataframe to all processes
df = comm.bcast(df, root=0)
print('this is rank',rank,'with df of length',len(df))
pp = comm.bcast(pp_mpts_param_sweep, root=0)
a_group = comm.bcast(group_mpts_param_sweep, root=0)
x_var_array = comm.bcast(x_var_array, root=0)
# Divide the runs among the processes
x_var_array = x_var_array[rank::size]

################################################################################
# Start the parameter sweep

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
        new_df, rel_val, m_pts, ell = ahf.HDBSCAN_(a_group, this_df, cl_x_var, cl_y_var, cl_z_var, m_pts, min_samp=min_s, param_sweep=True)
    except:
        print('rank',rank,'failed to run HDBSCAN for',x_key,'=',x)
        break
    # Record outputs to output object
    lines.append(str(m_pts)+','+str(ell)+','+str(new_df['cluster'].max()+1)+','+str(rel_val)+'\n')

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


