"""
Author: Mikhail Schee
Created: 2022-08-18

This script is set up to make figures of Arctic Ocean profile data that has been
formatted into netcdfs by the `make_netcdf` function.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions in source code must retain the accompanying copyright notice, this list of conditions, and the following disclaimer.
    2. Redistributions in binary form must reproduce the accompanying copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Names of the copyright holders must not be used to endorse or promote products derived from this software without prior written permission from the copyright holders.
    4. If any files are modified, you must cause the modified files to carry prominent notices stating that you changed the files and the date of any change.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""
Note: You unfortunately cannot pickle plots with parasite axes. So, the pickling
functionality won't work if you try to make a plot with the box and whisker plot
in an inset.

The Jupyter notebook Create_Figures.ipynb offers detailed explanations of each 
plot created by this script.

NOTE: BEFORE YOU RUN THIS SCRIPT

This script expects that the following files exist:
netcdfs/ITP_2.nc
netcdfs/ITP_3.nc

and that they have been created by running the following scripts in this order:
make_netcdf.py
take_moving_average.py
cluster_data.py
"""

# For custom analysis functions
import analysis_helper_functions as ahf
# For common BGR parameters
import BGR_params as bps
# For common BGR objects
import BGR_objects as bob

# Parameters
test_S_range = bps.test_S_range

### Filters for reproducing plots from Timmermans et al. 2008
ITP2_p_range = [185,300]
ITP2_S_range = [34.05,34.75]
ITP2_m_pts = 170
# Timmermans 2008 Figure 4 depth range
T2008_fig4_y_lims = {'y_lims':[260,220]}
# Timmermans 2008 Figure 4 shows profile 185
T2008_fig4_pfs = [183, 185, 187]
# Filters used in Timmermans 2008 T-S and aT-BS plots
T2008_p_range = [180,300]
T2008_fig5a_x_lims  = {'x_lims':[34.05,34.75]}
T2008_fig5a_ax_lims = {'x_lims':[34.05,34.75], 'y_lims':[-1.3,0.5]}
T2008_fig6a_ax_lims = {'x_lims':[0.027002,0.027042], 'y_lims':[-13e-6,3e-6]}
# The actual limits are above, but need to adjust the x lims for some reason
T2008_fig6a_ax_lims = {'x_lims':[0.026838,0.026878], 'y_lims':[-13e-6,3e-6]}

# A list of many profiles to plot from ITP2
start_pf = 191
import numpy as np
n_pfs_to_plot = 7
ITP2_some_pfs = list(np.arange(start_pf, start_pf+(n_pfs_to_plot*2), 2))

# For showing multiple layers grouped into one cluster
ITP2_some_pfs_0 = [87, 89, 95, 97, 99, 101, 103, 105, 109, 111]
ITP2_some_pfs_ax_lims_0 = {'y_lims':[245,220]}
# For showing one layer split into multiple clusters
ITP2_some_pfs_1 = [67, 69, 73, 75, 81, 83, 91, 93, 97, 99]
ITP2_some_pfs_ax_lims_1 = {'y_lims':[295,270]}

# For showing examples of shallow profiles
ITP35_some_pfs0 = [7,15,23,47,55,63,71]
# For showing examples of middle incomplete profiles
ITP35_some_pfs1 = [3,11,19,27,51,59,67]
# For showing examples of full profiles
ITP35_some_pfs2 = [1,9,17,25,49,57,65,73]

### Filters for reproducing plots from Lu et al. 2022
Lu2022_p_range = [200,355]
Lu2022_T_range = [-1.0,0.9]
Lu2022_S_range = [34.21,34.82]
Lu2022_m_pts = 580

################################################################################
# Make dictionaries for what data to load in and analyze
################################################################################

# A list of many profiles to plot from ITP3
start_pf = 169
import numpy as np
n_pfs_to_plot = 6
ITP3_some_pfs_1 = list(np.arange(start_pf, start_pf+(n_pfs_to_plot*2), 2))
ITP3_pfs1  = {'ITP_003':ITP3_some_pfs_1}
ITP35_pfs0 = {'ITP_35':ITP35_some_pfs0}
ITP35_pfs1 = {'ITP_35':ITP35_some_pfs1}
ITP35_pfs2 = {'ITP_35':ITP35_some_pfs2}

# Example profiles
# Coincident_pfs0 = {'AIDJEX_Snowbird':[138,140,142], 'SHEBA_Seacat':['SH36200'], 'ITP_33':[779, 781, 783]}
# ex_pfs1 = {'ITP_002':[188, 189, 208, 209]}

## Example profiles that appeared in other studies
ex_pfs1 = {
            'ITP_001':[1257],               # Shibley2017 Figure 3b, 800-0 m, 235-200 m
            # 'ITP_002':[113],                # Bebieva2019a Figure 1, 750-0 m, 385-230 m
            # 'ITP_002':[185],                # Timmermans2008 Figure 4, 260-220 m
            'ITP_003':[1073],               # Timmermans2008 Figure 2, 740-0 m, 290-240 m
            'ITP_004':[453],                # Lu2022 Figure 2, 750-0 m, 350-270 m
            'ITP_006':[475, 747],           # Toole2011 Figure 6, 300-0 m
            # 'ITP_008':[1301],               # Shibley2017 Figure 3a, 800-0 m, 260-230 m
            'ITP_041':[515],                # Bebieva2017(unpub) Figure 1, 750-0 m, 390-230 m
            'ITP_064':[377],                # vanderBoog2021a Figure A1, 750-250 m
          }
ex_pfs1_zoom_range = [280,250]

## Example profiles spaced roughly evenly in time
ex_pfs2 = {
            'ITP_001':[1,365,1457],
            'ITP_003':[701,1057],
            'ITP_004':[331,693],
            'ITP_005':[205],
            'ITP_006':[505],
            'ITP_008':[739],
            'ITP_011':[949,1345],
            'ITP_013':[209,621],
            'ITP_018':[389],
            'ITP_021':[391,747],
            'ITP_033':[263,441,625],
            'ITP_035':[297],
            'ITP_041':[271,449,631],
            'ITP_042':[83],
          }
## test quarterly slices
dfs1_q = ahf.Data_Filters(date_range=['2011/05/14 18:00:00','2011/05/16 00:00:00'])

ITP2_ex_pfs   = {'ITP_002':ITP2_some_pfs}
BGR04_ex_pfs  = {'BGR04':ITP2_some_pfs}

ITP2_ex_pfs_0 = {'ITP_002':ITP2_some_pfs_0}

################################################################################
# Create data filtering objects
print('- Creating data filtering objects')
################################################################################

this_min_press = bps.this_min_press
dfs1 = ahf.Data_Filters(min_press=this_min_press)
dfs2 = ahf.Data_Filters(min_press_TC_max=400)

dfs_CB = ahf.Data_Filters(geo_extent='CB', keep_black_list=True, cast_direction='any')
dfs1_CB_0a = ahf.Data_Filters(geo_extent='CB', min_press=this_min_press)
dfs1_CB_0b = ahf.Data_Filters(geo_extent='CB', min_press=this_min_press)#, date_range=['2005/08/15 00:00:00','2010/07/22 00:00:00'])


# To filter pre-clustered files to just certain cluster labels
# dfs_clstr_lbl = ahf.Data_Filters(clstr_labels=[[-1, 0, 1, 2]])
dfs_clstr_lbl = ahf.Data_Filters(clstr_labels=[[16],[18],[14]])
dfs_clstr_lbl = ahf.Data_Filters(clstr_labels=[[29],[17],[20]])
# fg corresponding
dfs_clstr_lbl = ahf.Data_Filters(clstr_labels=[[16],[18]])
# fg NOT corresponding
dfs_clstr_lbl = ahf.Data_Filters(clstr_labels=[[16],[17]])
# fg 3 clusters
dfs_clstr_lbl = ahf.Data_Filters(clstr_labels=[[12,13,17]])
dfs_clstr_lbl1 = ahf.Data_Filters(clstr_labels=[[11,12,13]])
# No noise points
dfs_no_noise = ahf.Data_Filters(clstr_labels='no_noise')

################################################################################
# Create data sets by combining filters and the data to load in
print('- Creating data sets')
################################################################################

# ds_all_sources_all  = ahf.Data_Set(all_sources, dfs_all)
# ds_all_sources_up   = ahf.Data_Set(all_sources, dfs0)
# ds_all_sources_pmin = ahf.Data_Set(all_sources, dfs1)

## Example profiles
# ds_all_sources_ex_pfs = ahf.Data_Set(Coincident_pfs0, dfs_all)

## ITP

# ds_this_BGR = ahf.Data_Set(BGR0607, dfs_all)

# ds_all_BGOS = ahf.Data_Set(all_BGOS, dfs_all)
# ds_all_BGOS = ahf.Data_Set(all_BGOS, dfs0)
# ds_BGOS = ahf.Data_Set(all_BGOS, dfs1)

# ds_all_ITP = ahf.Data_Set(all_ITPs, dfs_all)
# ds_all_ITP = ahf.Data_Set(all_ITPs, dfs_CB)

# ds_ITP = ahf.Data_Set(all_ITPs, dfs1_CB)

# ds_ITP2 = ahf.Data_Set(ITP002_all, dfs1)
# ds_ITP3 = ahf.Data_Set(ITP003_all, dfs1)
# ds_this_BGR = ds_ITP2

# ds_ITP_test = ahf.Data_Set({'ITP_098':'all'}, dfs_all)

# by different time periods
# this_BGR = 'BGR04'
this_BGR = 'BGR0506'
# this_BGR = 'BGR0607'
# this_BGR = 'BGR0708'
# this_BGR = 'BGR0809'
# this_BGR = 'BGR0910'
# this_BGR = 'BGR1011'
# this_BGR = 'BGR1112'
# this_BGR = 'BGR1213'
# this_BGR = 'BGR1314'
# this_BGR = 'BGR1415'
# this_BGR = 'BGR1516'
# this_BGR = 'BGR1617'
# this_BGR = 'BGR1718'
# this_BGR = 'BGR1819'
# this_BGR = 'BGR1920'
# this_BGR = 'BGR2021'
# this_BGR = 'BGR2122'
# this_BGR = 'BGR2223'
# this_BGR = 'BGR_all'
ds_this_BGR = ahf.Data_Set(bps.BGRITPs_dict[this_BGR], bob.dfs1_BGR_dict[this_BGR])

# ds_this_BGR = ahf.Data_Set(BGRITPs0506, dfs1_BGR0506)
# ds_this_BGR = ahf.Data_Set(BGRITPs0511, dfs1_BGR0511)

# ds_this_BGR = ahf.Data_Set(BGRITPsAll, dfs1)

# Data Sets without filtering based on TC_max
# ds_ITP22_all  = ahf.Data_Set(ITP22_all, dfs_all)
# ds_ITP23_all  = ahf.Data_Set(ITP23_all, dfs_all)
# ds_ITP32_all  = ahf.Data_Set(ITP32_all, dfs_all)
# ds_ITP33_all  = ahf.Data_Set(ITP33_all, dfs_all)
# ds_ITP34_all  = ahf.Data_Set(ITP34_all, dfs_all)
# ds_ITP35_all  = ahf.Data_Set(ITP35_all, dfs_all)
# ds_ITP41_all  = ahf.Data_Set(ITP41_all, dfs_all)
# ds_ITP42_all  = ahf.Data_Set(ITP42_all, dfs_all)
# ds_ITP43_all  = ahf.Data_Set(ITP43_all, dfs_all)

# Data Sets filtered based on TC_max
# ds_ITP2 = ahf.Data_Set(ITP2_all, dfs0)

# ds_ITP33 = ahf.Data_Set(ITP33_all, dfs1)
# ds_ITP34 = ahf.Data_Set(ITP34_all, dfs1)
# ds_ITP35 = ahf.Data_Set(ITP35_all, dfs1)
# ds_ITP41 = ahf.Data_Set(ITP41_all, dfs1)
# ds_ITP42 = ahf.Data_Set(ITP42_all, dfs1)
# ds_ITP43 = ahf.Data_Set(ITP43_all, dfs1)
# 
# ds_ITP35_some_pfs0 = ahf.Data_Set(ITP35_pfs0, dfs0)
# ds_ITP35_some_pfs1 = ahf.Data_Set(ITP35_pfs1, dfs0)
# ds_ITP35_some_pfs2 = ahf.Data_Set(ITP35_pfs2, dfs0)

## Pre-clustered
# By year
# ds_BGR04   = ahf.Data_Set(BGR04, dfs_to_use)
# ds_BGR0506 = ahf.Data_Set(BGR0506, dfs_to_use)
# ds_BGR0607 = ahf.Data_Set(BGR0607, dfs_to_use)
# ds_BGR0708 = ahf.Data_Set(BGR0708, dfs_to_use)
# ds_BGR0809 = ahf.Data_Set(BGR0809, dfs_to_use)
# ds_BGR0910 = ahf.Data_Set(BGR0910, dfs_to_use)
# ds_BGR1011 = ahf.Data_Set(BGR1011, dfs_to_use)

# ds_this_BGR = ahf.Data_Set(bps.BGR_auto_mpts_dict[this_BGR], bob.dfs0)

# ds_BGR04_clstrs_456 = ahf.Data_Set(BGR04_clstrs_456, dfs_to_use)
# ds_BGR0508 = ahf.Data_Set(BGR0508, dfs_to_use)
# ds_BGR050607 = ahf.Data_Set(BGR050607, dfs_to_use)
# ds_BGR05060708 = ahf.Data_Set(BGR05060708, dfs_to_use)
# ds_BGR05060708_no_noise = ahf.Data_Set(BGR05060708, dfs_no_noise)
# ds_BGR05060708_clstrs_456 = ahf.Data_Set(BGR05060708_clstrs_456, dfs_all)
# ds_BGR_all = ahf.Data_Set(BGR_all, dfs_to_use)

# ds_this_BGR = ahf.Data_Set(BGR_all, dfs_to_use)

################################################################################
# Create profile filtering objects
print('- Creating profile filtering objects')
################################################################################

pfs_0 = ahf.Profile_Filters()
pfs_1 = ahf.Profile_Filters(SA_range=[34.4,34.8])#ITP2_S_range)
pfs_2 = ahf.Profile_Filters(p_range=[400,200])

# Canada Basin (CB), see Peralta-Ferriz2015
lon_CB = [-155,-130]
lat_CB = [72,84]
pfs_CB = ahf.Profile_Filters(lon_range=lon_CB,lat_range=lat_CB)
pfs_CB1 = ahf.Profile_Filters(lon_range=lon_CB,lat_range=lat_CB, p_range=[1000,5])
# Beaufort Gyre Region (BGR), see Shibley2022
lon_BGR = bps.lon_BGR
lat_BGR = bps.lat_BGR
pfs_BGR = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR)
pfs_BGR1 = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR, p_range=[1000,5], SA_range=test_S_range, lt_pTC_max=True)
pfs_BGR1_n = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR, p_range=[1000,5], SA_range=test_S_range, lt_pTC_max=True, every_nth_row=12)

pfs_BGR_test = bob.pfs_BGR_test

pfs_test = ahf.Profile_Filters(every_nth_row=4)

# Profile filters
test_p_range = [400,200]
pfs_ell_10  = ahf.Profile_Filters(p_range=test_p_range, m_avg_win=10)
pfs_ell_50  = ahf.Profile_Filters(p_range=test_p_range, m_avg_win=50)
pfs_ell_100 = ahf.Profile_Filters(p_range=test_p_range, m_avg_win=100)
pfs_ell_150 = ahf.Profile_Filters(p_range=test_p_range, m_avg_win=150)


################################################################################
# Create plotting parameter objects

# print('- Creating plotting parameter objects')
################################################################################

# Use these things
# pfs_this_BGR = pfs_0
pfs_this_BGR = pfs_BGR_test
# pfs_this_BGR = pfs_BGR1_n
# pfs_this_BGR = pfs_test

# ds_this_BGR = ds_ITP2
# ds_this_BGR = ds_ITP3
# ds_this_BGR = ds_BGR_ITPs_all

# by year
# ds_this_BGR = ds_BGR04
# ds_this_BGR = ds_BGR04_clstrs_456
# ds_this_BGR = ds_BGR0506
# ds_this_BGR = ds_BGR0607
# ds_this_BGR = ds_BGR0708
# ds_this_BGR = ds_BGR_all
# ds_this_BGR = ds_BGR0508
# ds_this_BGR = ds_BGR050607
# ds_this_BGR = ds_BGR05060708
# ds_this_BGR = ds_BGR05060708_clstrs_456

################################################################################
################################################################################
### Figures
################################################################################
################################################################################

################################################################################
## Basic TS plots
################################################################################
# TS plot
if False:
    print('')
    print('- Creating TS plot')
    # Make the Plot Parameters
    pp_TS = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='clr_all_same')
    # Make the Analysis Group
    group_TS_plot = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_TS)
    # Make the figure
    ahf.make_figure([group_TS_plot])

## TS plot by instrument
if False:
    print('')
    print('- Creating TS plot, colored by instrument')
    # Make the Plot Parameters
    pp_TS_instrmt = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='instrmt')
    # Make the Analysis Group pfs_fltrd pfs_AOA1
    group_TS_plot = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_TS_instrmt)
    # Make the figure
    ahf.make_figure([group_TS_plot])

################################################################################
## Overviews
################################################################################
# ITP number vs time
if False:
    print('')
    print('- Creating plot of ITP number vs time')
    # Make the Plot Parameters
    pp_ITP_vs_time = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['instrmt'], clr_map='instrmt', legend=False)
    # Make the Analysis Group
    group_ITP_vs_time = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_ITP_vs_time)
    # Make the figure
    ahf.make_figure([group_ITP_vs_time])

# Output summary
if False:
    # Make the Plot Parameters
    pp_test = ahf.Plot_Parameters(extra_args={'extra_vars_to_keep':['dt_start']})
    # Make the Analysis Group
    group_test = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_test)
    ahf.txt_summary([group_test])

# Cluster stats
if False:
    # Make the Plot Parameters
    pp_test = ahf.Plot_Parameters(extra_args={'extra_vars_to_keep':['SA','cluster']})
    # Make the Analysis Group
    group_test = ahf.Analysis_Group(ds_BGRmn, pfs_0, pp_test)
    ahf.cluster_stats([group_test], stat_vars=['SA'])

################################################################################
## Maps and distances
################################################################################
pp_map = ahf.Plot_Parameters(plot_type='map', clr_map='instrmt', extra_args={'map_extent':'Western_Arctic'})
pp_map_full_Arctic = ahf.Plot_Parameters(plot_type='map', clr_map='instrmt', extra_args={'map_extent':'Full_Arctic'})#, legend=False, add_grid=False)
pp_map_by_date = ahf.Plot_Parameters(plot_type='map', clr_map='dt_start', extra_args={'map_extent':'Western_Arctic'})
## Map of all profiles for all sources
if False:
    print('')
    print('- Creating a map of all profiles from all sources')
    # Make the subplot groups
    group_map_full_Arctic = ahf.Analysis_Group(ds_all_ITP, pfs_0, pp_map_full_Arctic, plot_title='')
    group_map = ahf.Analysis_Group(ds_all_ITP, pfs_0, pp_map, plot_title='')
    # Make the figure
    ahf.make_figure([group_map_full_Arctic, group_map], use_same_x_axis=False, use_same_y_axis=False, filename='Figure_1.pickle')
## Map of just in the Beaufort Gyre Region
if False:
    print('')
    print('- Creating a map of profiles in the Beaufort Gyre Region')
    # Make the subplot groups
    group_map = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_map, plot_title='')
    # Make the figure
    ahf.make_figure([group_map], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_1.pickle')
## Map of all upgoing profiles for all sources
if False:
    print('')
    print('- Creating a map of all upgoing profiles from all sources')
    # Make the subplot groups
    group_map_full_Arctic = ahf.Analysis_Group(ds_all_sources_up, pfs_0, pp_map_full_Arctic, plot_title='')
    group_map = ahf.Analysis_Group(ds_all_sources_up, pfs_0, pp_map, plot_title='')
    # Make the figure
    ahf.make_figure([group_map_full_Arctic, group_map], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_1.pickle')
## Map of all upgoing profiles for all sources with a max pressure > 400 dbar
if False:
    print('')
    print('- Creating a map of all upgoing profiles from all sources that go below 400 dbar')
    # Make the subplot groups
    group_map_full_Arctic = ahf.Analysis_Group(ds_all_sources_pmin, pfs_0, pp_map_full_Arctic, plot_title='')
    group_map = ahf.Analysis_Group(ds_all_sources_pmin, pfs_0, pp_map, plot_title='')
    # Make the figure
    ahf.make_figure([group_map_full_Arctic, group_map], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_1.pickle') 
### Date and Distance spans
## Map of all BGR profiles
if False:
    print('')
    print('- Creating map all BGR profiles')
    # Make the subplot groups
    group_BGR_map_by_date = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_map_by_date)
    # Make the figure
    ahf.make_figure([group_BGR_map_by_date], use_same_x_axis=False, use_same_y_axis=False)
    # Find the maximum distance between any two profiles for each data set in the group
    # ahf.find_max_distance([group_BGR_map_by_date])

################################################################################
## Tracking the top and bottom of the thermocline
################################################################################
# For these plots, I'm specifically not using a pre-clustered file because the 
#   thermocline lies outside the salinity range I used when clustering the data

# Tracking press_max, press_TC_max, and CT_TC_max over time
if False:
    print('')
    print('- Creating a plot of TC_max over time')
    # Make the Plot Parameters
    across_x_var = 'dt_start'
    pp_press_max    = ahf.Plot_Parameters(x_vars=[across_x_var], y_vars=['press_max'], legend=True)
    pp_TC_max       = ahf.Plot_Parameters(x_vars=[across_x_var], y_vars=['press_TC_max'], legend=True, extra_args={'plot_slopes':True})
    pp_press_TC_max = ahf.Plot_Parameters(x_vars=[across_x_var], y_vars=['CT_TC_max', 'SA_TC_max'], legend=True, extra_args={'plot_slopes':True})
    # Make the subplot groups
    group_press_max    = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_press_max, plot_title=this_BGR)
    group_TC_max       = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_TC_max, plot_title='')
    group_press_TC_max = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_press_TC_max, plot_title='')
    # Make the figure
    ahf.make_figure([group_press_max, group_TC_max, group_press_TC_max], row_col_list=[3,1, 0.45, 1.4])
# Tracking press_min, press_TC_min, and CT_TC_min over time
if False:
    print('')
    print('- Creating a plot of TC_min over time')
    # Make the Plot Parameters
    across_x_var = 'dt_start'
    pp_press_min    = ahf.Plot_Parameters(x_vars=[across_x_var], y_vars=['press_min'], legend=True)
    pp_TC_min       = ahf.Plot_Parameters(x_vars=[across_x_var], y_vars=['press_TC_min'], legend=True, extra_args={'plot_slopes':True})
    pp_press_TC_min = ahf.Plot_Parameters(x_vars=[across_x_var], y_vars=['CT_TC_min', 'SA_TC_min'], legend=True, extra_args={'plot_slopes':True})
    # Make the subplot groups
    group_press_min    = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_press_min, plot_title=this_BGR)
    group_TC_min       = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_TC_min, plot_title='')
    group_press_TC_min = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_press_TC_min, plot_title='')
    # Make the figure
    ahf.make_figure([group_press_min, group_TC_min, group_press_TC_min], row_col_list=[3,1, 0.45, 1.4])
# Tracking top and bottom of the thermocline over time
if False:
    print('')
    print('- Creating a plot of p(TC_max/TC_min) over time')
    # Make dataset
    # df_mod = bob.dfs1_BGR_dict[this_BGR]
    # df_mod.min_press_TC_min = 100
    df_mod = ahf.Data_Filters(min_press_TC_min=100, min_press=this_min_press, date_range=['2005/08/15 00:00:00','2006/08/15 00:00:00'])
    ds_this_BGR_mod = ahf.Data_Set(bps.BGRITPs_dict[this_BGR], df_mod)
    # Make the Plot Parameters
    across_x_var = 'dt_start'
    y_span = 200-45
    pp_TC_min_v_time = ahf.Plot_Parameters(x_vars=[across_x_var], y_vars=['press_TC_min'], legend=True, extra_args={'plot_slopes':True}, ax_lims={'y_lims':[200,200-y_span]})
    pp_TC_max_v_time = ahf.Plot_Parameters(x_vars=[across_x_var], y_vars=['press_TC_max'], legend=True, extra_args={'plot_slopes':True}, ax_lims={'y_lims':[450,450-y_span]})
    # Make the subplot groups
    group_TC_min_v_time = ahf.Analysis_Group(ds_this_BGR_mod, pfs_0, pp_TC_min_v_time, plot_title=this_BGR)
    group_TC_max_v_time = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_TC_max_v_time, plot_title='')
    # Make the figure
    ahf.make_figure([group_TC_min_v_time, group_TC_max_v_time], row_col_list=[2,1, 0.45, 1.4])
# Density histograms of the top and bottom of the thermocline
if False:
    print('')
    print('- Creating histograms of sigma(TC_max/TC_min) over time')
    # Make the Plot Parameters
    pp_sig_TC_min_hist = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['sig_TC_min'], legend=True, extra_args={'plot_slopes':False})
    pp_sig_TC_max_hist = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['sig_TC_max'], legend=True, extra_args={'plot_slopes':False})
    # Make the subplot groups
    group_sig_TC_min_hist = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_sig_TC_min_hist, plot_title=this_BGR)
    group_sig_TC_max_hist = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_sig_TC_max_hist, plot_title=this_BGR)
    # Make the figure
    ahf.make_figure([group_sig_TC_min_hist, group_sig_TC_max_hist])

################################################################################
## Isopycnal tracking
################################################################################
# For these plots, I'm specifically not using a pre-clustered file so that I can
#   use all the points (including noise) to track isopycnals

def get_iso_sig_ranges(iso_sig_values, iso_sig_width):
    """
    Get the ranges of sigma values that correspond to the given isopycnals using 
    the given width

    iso_sig_values      list of floats, The isopycnal values to use
    iso_sig_width       float, The width to make the isopycnals
    """
    sig_ranges = []
    for this_sig in iso_sig_values:
        sig_ranges.append([this_sig+iso_sig_width, this_sig-iso_sig_width])
    return sig_ranges

# Plotting density on a pressure vs. time plot, testing different widths of isopycnals
if False:
    print('')
    print('- Creating test plots of sigma')
    lat_lon_groups_to_plot = []
    hist_groups_to_plot = []
    press_vs_time_uncorr = []
    press_vs_time_corrtd = []
    sig_vals_widths = [[32.33, 0.00125], [32.33, 0.0025], [32.33, 0.005]]
    # exit(0)
    for this_sig_val_width in sig_vals_widths:
        # Make the title
        this_sig_val   = this_sig_val_width[0]
        this_sig_width = this_sig_val_width[1]
        that_title = r'$\sigma_1=$'+str(this_sig_val)+r', $\Delta\sigma_1=$'+str(this_sig_width)
        # Make profile filters
        pfs_sigma_test = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR, p_range=[1000,5], SA_range=test_S_range, lt_pTC_max=True, sig_range=get_iso_sig_ranges([this_sig_val], this_sig_width)[0])
        # Make the Plot Parameters
        pp_lat_lon_sigma = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='press', legend=True, extra_args={'plot_slopes':True, 'extra_vars_to_keep':['sigma']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR})
        pp_sigma_hist = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['press'], legend=True, extra_args={'plot_slopes':False, 'extra_vars_to_keep':['sigma']})
        pp_p_v_lat = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['press'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['sigma']})
        pp_minus_fit = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['press-fit'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['sigma'], 'fit_vars':['lon','lat']})
        # Make the subplot groups
        # lat_lon_groups_to_plot.append(ahf.Analysis_Group(ds_this_BGR, pfs_sigma_test, pp_lat_lon_sigma, plot_title=that_title))
        # hist_groups_to_plot.append(ahf.Analysis_Group(ds_this_BGR, pfs_sigma_test, pp_sigma_hist, plot_title=that_title))
        press_vs_time_uncorr.append(ahf.Analysis_Group(ds_this_BGR, pfs_sigma_test, pp_p_v_lat, plot_title=r'Uncorrected '+that_title))
        press_vs_time_corrtd.append(ahf.Analysis_Group(ds_this_BGR, pfs_sigma_test, pp_minus_fit, plot_title=r'Corrected '+that_title))
    # Make the figure
    # ahf.make_figure(lat_lon_groups_to_plot + hist_groups_to_plot)
    ahf.make_figure(press_vs_time_uncorr + press_vs_time_corrtd)

# The isopycnals that correspond to the top and bottom of the thermocline
iso_TC_min_val = 31.45
iso_TC_min_width = 0.075
iso_TC_max_val = 32.608
iso_TC_max_width = 0.006
# Tracking the isopycnals that correspond to the top and bottom of the thermocline
if False:
    print('')
    print('- Tracking the isopycnals that correspond to the top and bottom of the thermocline')
    TC_groups_to_plot = []
    sig_vals_widths = [[iso_TC_min_val, iso_TC_min_width, 'Top of Thermocline'], [iso_TC_max_val, iso_TC_max_width, 'Bottom of Thermocline']]
    for i in range(len(sig_vals_widths)):
        # Make the title
        this_sig_val   = sig_vals_widths[i][0]
        this_sig_width = sig_vals_widths[i][1]
        that_title = r'$\sigma_1=$'+str(this_sig_val)+r', $\Delta\sigma_1=$'+str(this_sig_width)
        # Make profile filters
        pfs_sigma_test = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR, p_range=[1000,5], lt_pTC_max=True, sig_range=get_iso_sig_ranges([this_sig_val], this_sig_width)[0])
        # Make the Plot Parameters
        pp_iso_vs_time = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['press'], legend=True, extra_args={'plot_slopes':'OLS', 'extra_vars_to_keep':['sigma']})
        pp_fit = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='press', legend=True, extra_args={'plot_slopes':True, 'extra_vars_to_keep':['sigma']}, ax_lims={'x_lims':bps.lon_BGR, 'y_lims':bps.lat_BGR, 'c_lims':None})
        pp_iso_vs_time_minus_fit = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['press-fit'], legend=True, extra_args={'plot_slopes':'OLS', 'extra_vars_to_keep':['sigma'], 'fit_vars':['lon','lat']})
        # Make the subplot groups
        TC_groups_to_plot.append(ahf.Analysis_Group(ds_this_BGR, pfs_sigma_test, pp_iso_vs_time, plot_title=r'Uncorrected '+that_title))
        TC_groups_to_plot.append(ahf.Analysis_Group(ds_this_BGR, pfs_sigma_test, pp_fit, plot_title=sig_vals_widths[i][2]+' '+this_BGR))
        TC_groups_to_plot.append(ahf.Analysis_Group(ds_this_BGR, pfs_sigma_test, pp_iso_vs_time_minus_fit, plot_title=r'Corrected '+that_title))
    # Make the figure
    ahf.make_figure(TC_groups_to_plot)

################################################################################
## Density ratio plots
################################################################################
# Map and histogram of vertical density ratio per profile NOT WORKING CURRENTLY
if False:
    print('')
    print('- Creating map of vertical density ratio')
    # Make the Plot Parameters
    pp_map_R_rho = ahf.Plot_Parameters(plot_type='map', clr_map='R_rho', ax_lims={'c_lims':[0,10]}, extra_args={'map_extent':'Western_Arctic'})
    pp_hist_R_rho = ahf.Plot_Parameters(plot_scale='by_pf', x_vars=['R_rho'], y_vars=['hist'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'extra_vars_to_keep':['SA']})
    # Make the subplot groups
    # group_map_R_rho = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_map_R_rho)
    group_hist_R_rho = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_hist_R_rho)
    # Make the figure
    # ahf.make_figure([group_map_R_rho, group_hist_R_rho])
    ahf.make_figure([group_hist_R_rho])

################################################################################
## Basic plots
################################################################################
# TS 
if False:
    print('')
    print('- Creating TS plot')
    # Make the Plot Parameters
    pp_TS = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='press', ax_lims={'c_lims':[180,240]})
    # Make the subplot groups
    group_TS = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_TS)
    # Make the figure
    ahf.make_figure([group_TS])

# SA vs time
if False:
    print('')
    print('- Creating TS plot')
    # Make the Plot Parameters
    pp_SA_vs_time = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['SA'], clr_map='clr_all_same')
    # Make the subplot groups
    group_SA_vs_time = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_SA_vs_time)
    # Make the figure
    ahf.make_figure([group_SA_vs_time], row_col_list=[1,1, 0.27, 1.0])
# TS and map
if False:
    print('')
    print('- Creating TS plot')
    # Make the Plot Parameters
    pp_TS = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='clr_all_same')
    # Make the subplot groups
    group_TS = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_TS)
    group_map = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_map)
    # Make the figure
    ahf.make_figure([group_TS, group_map], row_col_list=[1,2, 0.31, 1.1])

################################################################################
## Filter effects
################################################################################
# Profiles filtered out by p_max > 400 dbar
if False:
    print('')
    print('- Creating figure for profiles eliminated by p_max > 400 dbar filter')
    # Make the data set
    # ds_pmax_lt_400 = ahf.Data_Set(pmax_lt_400_ITP_35, dfs_all) # So many profiles
    ds_pmax_lt_400 = ahf.Data_Set(pmax_lt_400_ITP_41, dfs_all) # A lot of profiles
    # ds_pmax_lt_400 = ahf.Data_Set(pmax_lt_400_ITP_42, dfs_all) # A few profiles
    # Make the Plot Parameters
    # pp_pmax_lt_400 = ahf.Plot_Parameters(x_vars=['SA','CT'], y_vars=['press'], plot_type='profiles')
    pp_pmax_lt_400 = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='press')
    # Make the Analysis Group
    group_pmax_lt_400 = ahf.Analysis_Group(ds_pmax_lt_400, pfs_0, pp_pmax_lt_400, plot_title=r'Profiles with $p_{max} < 400$ dbar')
    # Make the figure
    ahf.make_figure([group_pmax_lt_400])
## Histograms of pressure max
if False:
    print('')
    print('- Creating histograms of pressure max in each profile')
    # Define the Profile Filters
    pfs_BGR1_all = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR, p_range=[1000,5])#, SA_range=test_S_range, lt_pTC_max=False)
    # Make the Plot Parameters
    pp_max_press_hist = ahf.Plot_Parameters(plot_scale='by_pf', x_vars=['max_press'], y_vars=['hist'], clr_map='clr_all_same')
    # Make the subplot groups
    group_all_pfs    = ahf.Analysis_Group(ds_ITP3, pfs_BGR1_all, pp_max_press_hist, plot_title=r'All profiles')
    group_lt_pTC_max = ahf.Analysis_Group(ds_ITP3, pfs_BGR1, pp_max_press_hist, plot_title=r'Profiles with $p < p(\Theta_{max})$')
    # Make the figure
    ahf.make_figure([group_all_pfs, group_lt_pTC_max])
    # ahf.make_figure([group_lt_pTC_max])
## Histograms of TC_max data
if False:
    print('')
    print('- Creating histograms of pressure at temperature max in each profile')
    # Make the Plot Parameters
    pp_press_TC_max_hist = ahf.Plot_Parameters(x_vars=['press_TC_max'], y_vars=['hist'], clr_map='clr_all_same')
    # Make the subplot groups
    group_press_TC_max_hist = ahf.Analysis_Group(ds_ITP3, pfs_0, pp_press_TC_max_hist)
    # # Make the figure
    ahf.make_figure([group_press_TC_max_hist])
## Plots of SA_TC_max vs press_TC_max
if False:
    print('')
    print('- Creating plots of the info about TC_max')
    # Make the Plot Parameters
    pp_SA_TC_max = ahf.Plot_Parameters(x_vars=['SA_TC_max'], y_vars=['press_TC_max'], clr_map='CT_TC_max')
    # Make the subplot groups
    group_SA_TC_max = ahf.Analysis_Group(ds_ITP3, pfs_0, pp_SA_TC_max)
    # # Make the figure
    ahf.make_figure([group_SA_TC_max])

################################################################################
## Plots in la_CT vs. SA space with different values of ell
################################################################################
# BGR
if False:
    print('')
    print('- Creating plots in la_CT--SA space with different values of ell')
    # Select the dataset to use
    this_ds = ds_this_BGR
    # Make the profile filter objects for the different values of ell
    pfs_ell_010 = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR, p_range=[1000,5], SA_range=test_S_range, lt_pTC_max=True, m_avg_win=10)
    pfs_ell_050 = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR, p_range=[1000,5], SA_range=test_S_range, lt_pTC_max=True, m_avg_win=50)
    pfs_ell_100 = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR, p_range=[1000,5], SA_range=test_S_range, lt_pTC_max=True, m_avg_win=100)
    pfs_ell_150 = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR, p_range=[1000,5], SA_range=test_S_range, lt_pTC_max=True, m_avg_win=150)
    # Make the Plot Parameters
    pp_TS = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='clr_all_same')
    # Make the subplot groups
    group_test1 = ahf.Analysis_Group(this_ds, pfs_ell_010, pp_TS, plot_title=r'ell = 10')
    group_test2 = ahf.Analysis_Group(this_ds, pfs_ell_050, pp_TS, plot_title=r'ell = 50')
    group_test3 = ahf.Analysis_Group(this_ds, pfs_ell_100, pp_TS, plot_title=r'ell = 100')
    group_test4 = ahf.Analysis_Group(this_ds, pfs_ell_150, pp_TS, plot_title=r'ell = 150')
    # # Make the figure
    ahf.make_figure([group_test1, group_test2, group_test3, group_test4], use_same_y_axis=True)
# ITP2 (NOTE: using SP instead of SA)
if False:
    print('')
    print('- Creating plots in la_CT--SA space with different values of ell')
    # Select the dataset to use
    this_ds = ds_ITP2
    # Make the profile filter objects for the different values of ell
    pfs_ell_010 = ahf.Profile_Filters(p_range=[1000,5], SP_range=ITP2_S_range, m_avg_win=10)
    pfs_ell_050 = ahf.Profile_Filters(p_range=[1000,5], SP_range=ITP2_S_range, m_avg_win=50)
    pfs_ell_100 = ahf.Profile_Filters(p_range=[1000,5], SP_range=ITP2_S_range, m_avg_win=100)
    pfs_ell_150 = ahf.Profile_Filters(p_range=[1000,5], SP_range=ITP2_S_range, m_avg_win=150)
    # Make the Plot Parameters
    pp_TS = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['la_CT'], clr_map='clr_all_same')
    # Make the subplot groups
    group_1 = ahf.Analysis_Group(this_ds, pfs_ell_010, pp_TS, plot_title=r'ITP2 $\ell=10$ dbar')
    group_2 = ahf.Analysis_Group(this_ds, pfs_ell_050, pp_TS, plot_title=r'ITP2 $\ell=50$ dbar')
    group_3 = ahf.Analysis_Group(this_ds, pfs_ell_100, pp_TS, plot_title=r'ITP2 $\ell=100$ dbar')
    group_4 = ahf.Analysis_Group(this_ds, pfs_ell_150, pp_TS, plot_title=r'ITP2 $\ell=150$ dbar')
    # # Make the figure
    ahf.make_figure([group_1, group_2, group_3, group_4])
# ITP2, clustered (NOTE: using SP instead of SA)
if False:
    print('')
    print('- Creating clustered plots in la_CT--SA space with different values of ell')
    # Select the dataset to use
    this_ds = ds_ITP2
    # Make the profile filter objects for the different values of ell
    pfs_ell_010 = ahf.Profile_Filters(p_range=[1000,5], SP_range=ITP2_S_range, m_avg_win=10)
    pfs_ell_050 = ahf.Profile_Filters(p_range=[1000,5], SP_range=ITP2_S_range, m_avg_win=50)
    pfs_ell_100 = ahf.Profile_Filters(p_range=[1000,5], SP_range=ITP2_S_range, m_avg_win=100)
    pfs_ell_150 = ahf.Profile_Filters(p_range=[1000,5], SP_range=ITP2_S_range, m_avg_win=150)
    # Make the Plot Parameters
    #   Note: values of m_pts determined from parameter sweep, see Figure S.4
    pp_ell_010 = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':230, 'b_a_w_plt':False, 'extra_vars_to_keep':['press']}, legend=True)
    pp_ell_050 = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':200, 'b_a_w_plt':False, 'extra_vars_to_keep':['press']}, legend=True)
    pp_ell_100 = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':170, 'b_a_w_plt':False, 'sort_clstrs':False, 'extra_vars_to_keep':['press']}, legend=True)
    pp_ell_150 = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':140, 'b_a_w_plt':False, 'extra_vars_to_keep':['press']}, legend=True)
    # Make the Analysis Groups
    group_ell_010 = ahf.Analysis_Group(this_ds, pfs_ell_010, pp_ell_010, plot_title=r'ITP2 $\ell=10$ dbar')
    group_ell_050 = ahf.Analysis_Group(this_ds, pfs_ell_050, pp_ell_050, plot_title=r'ITP2 $\ell=50$ dbar')
    group_ell_100 = ahf.Analysis_Group(this_ds, pfs_ell_100, pp_ell_100, plot_title=r'ITP2 $\ell=100$ dbar')
    group_ell_150 = ahf.Analysis_Group(this_ds, pfs_ell_150, pp_ell_150, plot_title=r'ITP2 $\ell=150$ dbar')
    # Make the figure
    ahf.make_figure([group_ell_010, group_ell_050, group_ell_100, group_ell_150])

################################################################################
## Example profile plots
################################################################################
# Example profile plots, CT and SA
if False:
    print('')
    print('- Creating figure of example profiles')
    # Make the data set
    # ds_ITP_ex_pfs = ahf.Data_Set(ITP2_ex_pfs, bob.dfs0)
    ds_ITP_ex_pfs = ahf.Data_Set(ITP3_pfs1, bob.dfs0)
    # Make the Plot Parameters
    pp_pfs = ahf.Plot_Parameters(x_vars=['CT','SA'], y_vars=['press'], plot_type='profiles', extra_args={'plot_pts':False, 'mark_thermocline':'min'}, legend=True, ax_lims={'y_lims':[200,0]})
    # Make the Analysis Groups
    group_example_profiles1 = ahf.Analysis_Group(ds_ITP_ex_pfs, pfs_0, pp_pfs, plot_title='')
    # Make the figure
    ahf.make_figure([group_example_profiles1], use_same_y_axis=False, row_col_list=[1,1, 0.3, 1.4])
# Example profile plots, CT and SA, full and zoomed
if False:
    print('')
    print('- Creating figure of an example profile')
    # Make the data set
    # ds_ITP_ex_pfs = ahf.Data_Set(ITP2_ex_pfs, bob.dfs0)
    ds_ITP_ex_pfs = ahf.Data_Set(ITP3_pfs1, bob.dfs0)
    # Make the Plot Parameters
    pp_pfs_full = ahf.Plot_Parameters(x_vars=['CT','SA'], y_vars=['press'], plot_type='profiles', extra_args={'plot_pts':False, 'mark_thermocline':True}, legend=False)
    pp_pfs_zoom = ahf.Plot_Parameters(x_vars=['CT','SA'], y_vars=['press'], plot_type='profiles', extra_args={'plot_pts':False})#, ax_lims={'y_lims':ex_pfs1_zoom_range})
    # Make the Analysis Groups
    group_example_profiles1 = ahf.Analysis_Group(ds_ITP_ex_pfs, pfs_0, pp_pfs_full, plot_title='')
    group_example_profiles2 = ahf.Analysis_Group(ds_ITP_ex_pfs, pfs_BGR_test, pp_pfs_zoom, plot_title='')
    # Make the figure
    ahf.make_figure([group_example_profiles1, group_example_profiles2], use_same_y_axis=False, row_col_list=[2,1, 0.45, 1.4])
# Example profile plot, iT and SP, full and zoomed
if False:
    print('')
    print('- Creating figure of an example profile')
    # Make the data set
    ds_ITP_ex_pfs = ahf.Data_Set(ex_pfs1, dfs_all)
    # Make the Plot Parameters
    pp_pfs_full = ahf.Plot_Parameters(x_vars=['iT','SP'], y_vars=['press'], plot_type='profiles', extra_args={'shift_pfs':True})
    pp_pfs_zoom = ahf.Plot_Parameters(x_vars=['iT','SP'], y_vars=['press'], plot_type='profiles', ax_lims={'y_lims':ex_pfs1_zoom_range})
    # Make the Analysis Groups
    group_example_profiles1 = ahf.Analysis_Group(ds_ITP_ex_pfs, pfs_0, pp_pfs_full, plot_title='')
    group_example_profiles2 = ahf.Analysis_Group(ds_ITP_ex_pfs, pfs_0, pp_pfs_zoom, plot_title='')
    # Make the figure
    ahf.make_figure([group_example_profiles1, group_example_profiles2], use_same_y_axis=False)
# Clustering a single example profile plot
if False:
    print('')
    print('- Creating figure of an example profile')
    # Make the data set
    ds_ITP_ex_pfs = ahf.Data_Set(ex_pfs1, dfs_all)
    # Make the Plot Parameters
    zoom_range = [260,220]
    this_m_pts = 4
    pp_pfs_clstr = ahf.Plot_Parameters(x_vars=['CT'], y_vars=['SA'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':this_m_pts, 'b_a_w_plt':False, 'plot_centroid':False})
    pp_pfs_zoomed = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['press'], plot_type='profiles', clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':this_m_pts, 'b_a_w_plt':False, 'plot_centroid':False})#, ax_lims={'y_lims':zoom_range}, add_grid=False)
    # Make the Analysis Group
    this_ds = ds_ITP_ex_pfs
    group_example_profiles1 = ahf.Analysis_Group(this_ds, pfs_1, pp_pfs_clstr)#, plot_title='')
    group_example_profiles2 = ahf.Analysis_Group(this_ds, pfs_1, pp_pfs_zoomed)#, plot_title='')
    # Make the figure
    ahf.make_figure([group_example_profiles1, group_example_profiles2])
## Example profiles with clustering marked
if False:
    print('')
    print('- Creating figure of clustered example profiles')
    # Make the Plot Parameters
    pp_pfs_SA = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['press'], clr_map='cluster', plot_type='profiles', extra_args={'plot_pts':True, 'extra_vars_to_keep':['dt_start'], 'dts_to_plot':['2005-08-16 06:00:01', '2006-08-15 06:00:03', '2005-11-15 06:00:03', '2006-05-15 06:00:02', '2006-02-15 06:00:03']}, legend=False)
    pp_pfs_CT = ahf.Plot_Parameters(x_vars=['CT'], y_vars=['press'], clr_map='cluster', plot_type='profiles', extra_args={'plot_pts':True, 'extra_vars_to_keep':['dt_start'], 'dts_to_plot':['2005-08-16 06:00:01', '2006-08-15 06:00:03', '2005-11-15 06:00:03', '2006-05-15 06:00:02', '2006-02-15 06:00:03']}, legend=False)
    # Make the Analysis Groups
    group_ex_pfs_SA = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_pfs_SA, plot_title='')
    group_ex_pfs_CT = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_pfs_CT, plot_title='')
    # Make the figure
    ahf.make_figure([group_ex_pfs_SA, group_ex_pfs_CT], use_same_y_axis=False, row_col_list=[2,1, 0.45, 1.4])

################################################################################
## Waterfall plots
################################################################################
# ITP2 Example profiles waterfall plot
if False:
    print('')
    print('- Creating a waterfall figure of example profiles')
    # Define the Profile Filters
    pfs_2 = ahf.Profile_Filters(p_range=[600,200])
    # Make the data set
    ds_ITP_ex_pfs = ahf.Data_Set(ITP2_ex_pfs, dfs_all)
    # Make the Plot Parameters
    pp_pfs_full = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['press'], plot_type='waterfall', extra_args={'plot_pts':False}, legend=False)
    # Make the Analysis Groups pfs_BGR1 pfs_0
    group_example_profiles1 = ahf.Analysis_Group(ds_ITP_ex_pfs, pfs_2, pp_pfs_full)
    # Make the figure
    ahf.make_figure([group_example_profiles1])
# ITP2 Waterfall plot with cluster points colored
if False:
    print('')
    print('- Creating figure of an example profile')
    # Define the Profile Filters
    # pfs_2 = ahf.Profile_Filters(p_range=[220,210])
    # Make the data set
    ds_ex_pfs = ahf.Data_Set(BGR04_ex_pfs, dfs_all)
    # Make the Plot Parameters
    pp_pfs_full = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['press'], clr_map='cluster', plot_type='waterfall', legend=False, add_grid=False)
    # Make the Analysis Groups pfs_BGR1 pfs_0
    group_example_profiles1 = ahf.Analysis_Group(ds_ex_pfs, pfs_2, pp_pfs_full)
    # Make the figure
    ahf.make_figure([group_example_profiles1], row_col_list=[1,1, 0.8, 1.5])
# BGR Example profiles waterfall plot
if False:
    print('')
    print('- Creating a waterfall figure of example profiles')
    # Define the Profile Filters
    pfs_2 = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR, p_range=[410,210], lt_pTC_max=True)#, SA_range=test_S_range)
    # Make the data set
    # ds_ex_pfs2 = ahf.Data_Set(ex_pfs2, dfs_all)
    # Make the Plot Parameters
    pp_pfs_full = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['press-fit'], clr_map='cluster', plot_type='waterfall', extra_args={'plot_pts':True, 'extra_vars_to_keep':['dt_start'], 'dts_to_plot':[13011.250011574075, 13375.250034722223, 13102.250034722223, 13283.250023148148, 13194.250034722223, 13559.000104166667, 13740.00019675926, 13467.000023148148, 13648.000023148148, 14106.000034722223, 14379.016851851851, 14198.016863425926, 13832.016863425926, 14014.016863425926, 13924.016863425926, 14468.00008101852, 14290.00008101852, 14744.000069444444, 14836.000069444444, 14655.000069444444, 14563.00162037037, 15109.00008101852, 15200.000069444444, 15020.00008101852, 14928.00008101852], 'fit_vars':['lon','lat']}, legend=False)
    # pp_pfs_full = ahf.Plot_Parameters(plot_scale='by_pf', x_vars=['dt_start'], y_vars=['prof_no'], clr_map='instrmt', legend=True)
    # Make the Analysis Groups 
    group_example_profiles2 = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_pfs_full)
    # group_example_profiles2 = ahf.Analysis_Group(ds_ex_pfs2, pfs_2, pp_pfs_full)
    # Make the figure
    ahf.make_figure([group_example_profiles2])

    # # Make the data set
    # ds_ITP_ex_pfs = ahf.Data_Set(ex_pfs2, dfs_all)
    # # Make the Plot Parameters
    # pp_pfs_full = ahf.Plot_Parameters(x_vars=['CT','SA'], y_vars=['press'], plot_type='profiles', extra_args={'plot_pts':False}, legend=False)
    # pp_pfs_zoom = ahf.Plot_Parameters(x_vars=['SA','CT'], y_vars=['press'], plot_type='profiles', extra_args={'plot_pts':False})#, ax_lims={'y_lims':ex_pfs1_zoom_range})
    # pp_pfs_map  = ahf.Plot_Parameters(plot_type='map', clr_map='instrmt', extra_args={'map_extent':'Western_Arctic'})#, ax_lims={'y_lims':ex_pfs1_zoom_range})
    # # Make the Analysis Groups
    # group_example_profiles1 = ahf.Analysis_Group(ds_ITP_ex_pfs, pfs_0, pp_pfs_full, plot_title='')
    # group_example_profiles2 = ahf.Analysis_Group(ds_ITP_ex_pfs, pfs_BGR_test, pp_pfs_zoom, plot_title='')
    # group_example_profiles3 = ahf.Analysis_Group(ds_ITP_ex_pfs, pfs_BGR_test, pp_pfs_map, plot_title='')
    # # Make the figure
    # ahf.make_figure([group_example_profiles1, group_example_profiles2, group_example_profiles3], use_same_y_axis=False)

################################################################################
## Separating profiles from an example area by instrument
################################################################################
# Selecting the example area
if False:
    print('')
    print('- Creating a map showing the example area')
    # Make the Plot Parameters
    pp_ex_area_map = ahf.Plot_Parameters(plot_type='map', clr_map='clr_all_same', extra_args={'map_extent':'Western_Arctic'}, legend=True)
    # Make the Analysis Groups pfs_BGR1 pfs_0
    group_ex_area_map = ahf.Analysis_Group(ds_this_BGR, bob.pfs_ex_area, pp_ex_area_map)
    # Make the figure
    ahf.make_figure([group_ex_area_map])
# Plotting the profiles from the example area, separated by instrument
if True:
    print('')
    print('- Creating profile plots that separate by instrmt')
    # Make the Plot Parameters
    pp_pfs_by_instrmt = ahf.Plot_Parameters(plot_type='profiles', x_vars=['SA'], y_vars=['press'], clr_map='clr_all_same', extra_args={'plot_pts':False, 'separate_instrmts':True}, legend=True)
    # Make the Analysis Groups pfs_BGR1 pfs_0
    group_pfs_by_instrmt = ahf.Analysis_Group(ds_this_BGR, bob.pfs_ex_area, pp_pfs_by_instrmt)
    # Make the figure
    ahf.make_figure([group_pfs_by_instrmt])

################################################################################
## Subsampling
################################################################################
## Subsampling ITP data
# Histograms of the first differences for data sets
if False:
    print('')
    print('- Creating histograms of first differences in pressures')
    # Make the Plot Parameters
    pp_first_dfs_press = ahf.Plot_Parameters(x_vars=['press'], y_vars=['hist'], clr_map='clr_all_same', first_dfs=[True,False])
    # Make the subplot groups
    group_first_dfs_press_hist = ahf.Analysis_Group(ds_ITP3, pfs_BGR1, pp_first_dfs_press)
    # # Make the figure
    ahf.make_figure([group_first_dfs_press_hist])#, use_same_x_axis=False, use_same_y_axis=False)
# Comparing ITP3 to ssITP3
if False:
    print('')
    print('- Creating TS plots to compare full profiles vs. subsampled')
    # Define the Profile Filters
    every_n = 12
    pfs_BGR1_n = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR, p_range=[1000,5], SA_range=test_S_range, lt_pTC_max=True, every_nth_row=every_n)
    # Make the Plot Parameters
    m_pts1 = 'auto'#500
    m_pts2 = 'auto'#100
    these_pfs = [313,315,317,319,321]
    pp_TS1 = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':m_pts1, 'b_a_w_plt':True, 'plot_centroid':True})
    pp_TS2 = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':m_pts2, 'b_a_w_plt':True, 'plot_centroid':True})
    pp_pfs1 = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['press'], plot_type='profiles', ax_lims={'y_lims':[240,225]}, clr_map='cluster', extra_args={'pfs_to_plot':these_pfs, 'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':m_pts1, 'b_a_w_plt':False})
    pp_pfs2 = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['press'], plot_type='profiles', ax_lims={'y_lims':[240,225]}, clr_map='cluster', extra_args={'pfs_to_plot':these_pfs, 'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':m_pts2, 'b_a_w_plt':False})
    # Make the subplot groups
    group_ITP3_full = ahf.Analysis_Group(ds_ITP3, pfs_BGR1, pp_TS1, plot_title=r'ITP3 full')
    group_ITP3_ss = ahf.Analysis_Group(ds_ITP3, pfs_BGR1_n, pp_TS2, plot_title=r'ITP3 subsampled, every '+str(every_n))
    group_ITP3_full_pfs = ahf.Analysis_Group(ds_ITP3, pfs_BGR1, pp_pfs1, plot_title=r'ITP3 full')
    group_ITP3_ss_pfs = ahf.Analysis_Group(ds_ITP3, pfs_BGR1_n, pp_pfs2, plot_title=r'ITP3 subsampled, every '+str(every_n))
    # # Make the figure
    ahf.make_figure([group_ITP3_full, group_ITP3_ss, group_ITP3_full_pfs, group_ITP3_ss_pfs], use_same_x_axis=False)

################################################################################
## Clustering parameter sweeps
################################################################################
# Parameter sweep for BGR ITP data
if False:
    print('')
    print('- Creating clustering parameter sweep for BGR ITP data')
    test_mpts = 360
    # Make the Plot Parameters
    pp_mpts_param_sweep = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'm_cls':'auto', 'cl_ps_tuple':[10,801,10]}) #[10,721,10]
    # pp_ell_param_sweep  = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'cl_ps_tuple':[10,271,10]}) 
    # Make the subplot groups
    group_mpts_param_sweep = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_mpts_param_sweep)#, plot_title='BGR04')
    # group_ell_param_sweep  = ahf.Analysis_Group(ds_BGOS, pfs_fltrd, pp_ell_param_sweep, plot_title='BGR04')
    # # Make the figure
    # ahf.make_figure([group_mpts_param_sweep, group_ell_param_sweep], filename='test_param_sweep_BGR.pickle')
    ahf.make_figure([group_mpts_param_sweep])#, filename='BGRa_ps.pickl')
## Parameter sweep for 4 different ell values
if False:
    print('')
    print('- Creating TS plots')
    # Make the Plot Parameters
    pp_mpts_ps = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':100, 'cl_ps_tuple':[10,721,10]})
    # Make the Analysis Groups
    group_ell_010 = ahf.Analysis_Group(ds_ITP2, pfs_ell_10, pp_mpts_ps, plot_title=r'ITP2 $\ell=10$ dbar')
    group_ell_050 = ahf.Analysis_Group(ds_ITP2, pfs_ell_50, pp_mpts_ps, plot_title=r'ITP2 $\ell=50$ dbar')
    group_ell_100 = ahf.Analysis_Group(ds_ITP2, pfs_ell_100,pp_mpts_ps, plot_title=r'ITP2 $\ell=100$ dbar')
    group_ell_150 = ahf.Analysis_Group(ds_ITP2, pfs_ell_150,pp_mpts_ps, plot_title=r'ITP2 $\ell=150$ dbar')
    # Make the figure
    ahf.make_figure([group_ell_010, group_ell_050, group_ell_100, group_ell_150], filename='4_ell_value_ps.pickle')

################################################################################
## Test clustering (live)
################################################################################
# test clustering
if False:
    print('')
    print('- Creating clustering plot')
    # Make the Plot Parameters
    # pp_live_clstr = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':190, 'm_cls':'auto', 'b_a_w_plt':True, 'relab_these':{1:2, 2:3, 5:6}, 'extra_vars_to_keep':['CT', 'ma_CT']})
    pp_live_clstr = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':205, 'm_cls':'auto', 'b_a_w_plt':True, 'extra_vars_to_keep':['CT', 'ma_CT']})
    # Make the subplot groups
    group_clstrd = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_live_clstr)
    # Make the figure
    ahf.make_figure([group_clstrd])#, filename='test_clstr.pickle')

################################################################################
################################################################################
## Plotting clusterings from pre-clustered files
################################################################################
################################################################################

################################################################################
## Pre-clustered la_CT vs. SA plots
################################################################################
# Make the Plot Parameters
pp_pre_clstrd = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'b_a_w_plt':True})
# relab_these_test = {2:3,3:4,4:5,5:6,6:7,7:8,8:9,9:10,10:11,11:12,12:13,13:14}
relab_these_test = {}
# BGR ITP clustering
if False:
    print('')
    print('- Creating plot of pre-clustered BGR ITP data')
    # pp_pre_clstrd = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'sort_clstrs':False, 'b_a_w_plt':True, 'relab_these':relab_these_test}, ax_lims={'x_lims':test_S_range})
    pp_pre_clstrd = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'sort_clstrs':False, 'b_a_w_plt':True}, ax_lims={'x_lims':test_S_range})
    # Make the subplot groups
    group_pre_clstrd = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_pre_clstrd)
    # Plot the figure
    ahf.make_figure([group_pre_clstrd], use_same_y_axis=False)
# Multiple BGR ITP clustering
if False:
    print('')
    print('- Creating plots of pre-clustered BGR ITP data')
    pp_pre_clstrd = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'b_a_w_plt':True}, ax_lims={'x_lims':test_S_range}, legend=True)
    # Make the subplot groups
    # group_pre_clstrd_0 = ahf.Analysis_Group(ds_BGR04, pfs_0, pp_pre_clstrd)
    group_pre_clstrd_1 = ahf.Analysis_Group(ds_BGR0506, pfs_0, pp_pre_clstrd)
    group_pre_clstrd_2 = ahf.Analysis_Group(ds_BGR0607, pfs_0, pp_pre_clstrd)
    group_pre_clstrd_3 = ahf.Analysis_Group(ds_BGR0708, pfs_0, pp_pre_clstrd)
    # Plot the figure
    ahf.make_figure([group_pre_clstrd_1, group_pre_clstrd_2, group_pre_clstrd_3])

################################################################################
## Pre-clustered plots vs. time
################################################################################
# Salinity vs. time
if False:
    print('')
    print('- Creating plots of salinity vs time')
    # Make the Plot Parameters
    pp_SA_vs_dt = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['SA'], clr_map='cluster', extra_args={'plt_noise':True})
    # Make the subplot groups 
    group_SA_vs_dt = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_SA_vs_dt)
    # Make the figure
    ahf.make_figure([group_SA_vs_dt])
# Temperature vs. time and pressure vs. time
if False:
    print('')
    print('- Creating plots of temperature and pressure vs time')
    # Make the Plot Parameters
    pp_CT_vs_dt = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['CT'], clr_map='cluster', extra_args={'plt_noise':False, 'extra_vars_to_keep':['SA','cluster']}, legend=False)
    pp_press_vs_dt = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['press'], clr_map='cluster', extra_args={'plt_noise':False, 'extra_vars_to_keep':['SA','cluster']}, legend=False)
    # Make the subplot groups ds_BGOS_m280_e050 ds_BGOS
    group_CT_vs_dt = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_CT_vs_dt)
    group_press_vs_dt = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_press_vs_dt)
    # Make the figure
    ahf.make_figure([group_CT_vs_dt, group_press_vs_dt], row_col_list=[2,1, 0.45, 1.4])

################################################################################
# cRL and nir_SA for all clusters
if False:
    print('')
    print('- Creating plots of lateral density ratio and normalized intercluster range vs pressure')
    # Make the Plot Parameters
    pp_nir_SA = ahf.Plot_Parameters(x_vars=['nir_SA'], y_vars=['ca_press'], clr_map='cluster', extra_args={'b_a_w_plt':False, 'plot_noise':False})
    pp_cRL = ahf.Plot_Parameters(x_vars=['cRL'], y_vars=['ca_press'], clr_map='cluster', extra_args={'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':True})
    # Make the subplot groups
    group_nir_SA = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_nir_SA)
    group_cRL = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_cRL)
    # Make the figure
    ahf.make_figure([group_cRL, group_nir_SA])

################################################################################
## Pre-clustered spans and averages in variables
################################################################################
# Spans in pressure, individual time periods
if False:
    print('')
    print('- Creating plots of cluster span in pressure')
    # Make the Plot Parameters
    pp_clstr_span = ahf.Plot_Parameters(x_vars=['pcs_press'], y_vars=['pca_press'], clr_map='cluster', extra_args={'plt_noise':False, 'extra_vars_to_keep':['SA']}, legend=True) 
    # Make the subplot groups
    group_clstrs_0 = ahf.Analysis_Group(ds_BGR04, pfs_0, pp_clstr_span)
    group_clstrs_1 = ahf.Analysis_Group(ds_BGR0506, pfs_0, pp_clstr_span)
    group_clstrs_2 = ahf.Analysis_Group(ds_BGR0607, pfs_0, pp_clstr_span)
    group_clstrs_3 = ahf.Analysis_Group(ds_BGR0708, pfs_0, pp_clstr_span)
    # # Make the figure
    ahf.make_figure([group_clstrs_0, group_clstrs_1, group_clstrs_2, group_clstrs_3])
# Spans in pressure, all time periods
if False:
    print('')
    print('- Creating plots of cluster span in pressure')
    # Make the Plot Parameters
    pp_clstr_span = ahf.Plot_Parameters(x_vars=['pcs_press'], y_vars=['pca_press'], clr_map='cluster', extra_args={'plt_noise':False, 'extra_vars_to_keep':['SA']}, legend=True) 
    # Make the subplot groups
    group_clstrs_0 = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_clstr_span)
    # # Make the figure
    ahf.make_figure([group_clstrs_0])

################################################################################
## Pre-clustered comparing clusters across time periods
################################################################################
# BGR ITP clustering, comparing specific clusters
if False:
    print('')
    print('- Creating SA-Time plot to look at a single cluster across different periods')
    # Define the profile filters
    pfs_these_clstrs = ahf.Profile_Filters(clstrs_to_plot=[4,5,6])
    # Make the Plot Parameters
    # pp_these_clstrs = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['lat'], clr_map='cluster', legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']})#, 'clstrs_to_plot':[4,5,6]}) 
    pp_these_clstrs = ahf.Plot_Parameters(x_vars=['BSA'], y_vars=['aCT'], clr_map='cluster', legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['cluster']})#, 'clstrs_to_plot':[4,5,6]}) 
    # Make the subplot groups
    group_these_clstrs = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_these_clstrs)
    # # Make the figure
    ahf.make_figure([group_these_clstrs])
################################################################################
## Pre-clustered comparing single clusters across time periods
################################################################################
# Define the profile filters
pfs_these_clstrs = ahf.Profile_Filters(clstrs_to_plot=[6])
# Just one cluster in SA vs. la_CT space
if False:
    print('')
    print('- Creating plot of pre-clustered BGR ITP data for just one cluster')
    pp_pre_clstrd = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'sort_clstrs':False, 'b_a_w_plt':True}, ax_lims={'x_lims':test_S_range})
    # Make the subplot groups
    group_pre_clstrd = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_pre_clstrd)
    # Plot the figure
    ahf.make_figure([group_pre_clstrd], use_same_y_axis=False)
# Just one cluster plotting SA, CT, lon, and lat vs. time
if False:
    print('')
    print('- Creating plots across time to look at a single cluster across different periods')
    # Make the Plot Parameters
    pp_SA_and_CT = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['SA','CT'], legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'extra_vars_to_keep':['SA','cluster']})#, 'clstrs_to_plot':[4,5,6]}) 
    pp_lat_and_lon = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['lon','lat'], legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'extra_vars_to_keep':['SA','cluster']})#, 'clstrs_to_plot':[4,5,6]}) 
    # Make the subplot groups
    group_SA_and_CT = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_SA_and_CT)
    group_lat_and_lon = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_lat_and_lon)
    # # Make the figure
    ahf.make_figure([group_SA_and_CT, group_lat_and_lon], row_col_list=[2,1, 0.45, 1.4])
# Just one cluster plotting SA, CT, and press vs. time with trend lines
if False:
    print('')
    print('- Creating plots across time to look at a single cluster across different periods')
    # Make the Plot Parameters
    pp_SA = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['SA'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']})#, 'clstrs_to_plot':[4,5,6]}) 
    pp_CT = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['CT'], legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']})
    pp_press = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['press'], legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']})
    # Make the subplot groups
    group_SA = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_SA)
    group_CT = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_CT)
    group_press = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_press)
    # Make the figure
    ahf.make_figure([group_SA, group_CT, group_press], row_col_list=[3,1, 0.45, 1.4])
# Just one cluster plotting SA, CT, and press vs. longitude with trend lines
if False:
    print('')
    print('- Creating plots across longitude to look at a single cluster across different periods')
    # Make the Plot Parameters
    pp_SA = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['SA'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR})#, 'clstrs_to_plot':[4,5,6]}) 
    pp_CT = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['CT'], legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR})
    pp_press = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['press'], legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR})
    # Make the subplot groups
    group_SA = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_SA)
    group_CT = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_CT)
    group_press = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_press)
    # Make the figure
    ahf.make_figure([group_SA, group_CT, group_press], row_col_list=[3,1, 0.45, 1.4])
# Just one cluster plotting SA, CT, and press vs. latitude with trend lines
if False:
    print('')
    print('- Creating plots across latitude to look at a single cluster across different periods')
    # Make the Plot Parameters
    pp_SA = ahf.Plot_Parameters(x_vars=['lat'], y_vars=['SA'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lat_BGR})#, 'clstrs_to_plot':[4,5,6]}) 
    pp_CT = ahf.Plot_Parameters(x_vars=['lat'], y_vars=['CT'], legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lat_BGR})
    pp_press = ahf.Plot_Parameters(x_vars=['lat'], y_vars=['press'], legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lat_BGR})
    # Make the subplot groups
    group_SA = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_SA)
    group_CT = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_CT)
    group_press = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_press)
    # Make the figure
    ahf.make_figure([group_SA, group_CT, group_press], row_col_list=[3,1, 0.45, 1.4])
# Maps and histograms of one cluster's profile averages in SA, CT, and press
if False:
    print('')
    print('- Creating maps of one cluster`s profile averages in SA, CT, and press')
    # Make the Plot Parameters
    pp_map_SA = ahf.Plot_Parameters(plot_type='map', clr_map='pca_SA', legend=False, extra_args={'map_extent':'Western_Arctic', 'extra_vars_to_keep':['SA','cluster']})
    pp_map_CT = ahf.Plot_Parameters(plot_type='map', clr_map='pca_CT', legend=False, extra_args={'map_extent':'Western_Arctic', 'extra_vars_to_keep':['SA','cluster']})
    pp_map_press = ahf.Plot_Parameters(plot_type='map', clr_map='pca_press', legend=False, extra_args={'map_extent':'Western_Arctic', 'extra_vars_to_keep':['SA','cluster']})
    pp_hist_SA = ahf.Plot_Parameters(plot_scale='by_pf', x_vars=['hist'], y_vars=['pca_SA'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']})
    pp_hist_CT = ahf.Plot_Parameters(plot_scale='by_pf', x_vars=['hist'], y_vars=['pca_CT'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']})
    pp_hist_press = ahf.Plot_Parameters(plot_scale='by_pf', x_vars=['hist'], y_vars=['pca_press'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']})
    # Make the subplot groups
    group_map_SA = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_map_SA)
    group_map_CT = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_map_CT)
    group_map_press = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_map_press)
    group_hist_SA = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_hist_SA)
    group_hist_CT = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_hist_CT)
    group_hist_press = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_hist_press)
    # # Make the figure
    ahf.make_figure([group_map_SA, group_map_CT, group_map_press, group_hist_SA, group_hist_CT, group_hist_press])#, row_col_list=[2,1, 0.45, 1.4])
# Maps and histograms of one cluster's profile spans in SA, CT, and press
if False:
    print('')
    print('- Creating maps of one cluster`s profile spans in SA, CT, and press')
    # Make the Plot Parameters
    pp_map_SA = ahf.Plot_Parameters(plot_type='map', clr_map='pcs_SA', legend=False, extra_args={'map_extent':'Western_Arctic', 'extra_vars_to_keep':['SA','cluster']})
    pp_map_CT = ahf.Plot_Parameters(plot_type='map', clr_map='pcs_CT', legend=False, extra_args={'map_extent':'Western_Arctic', 'extra_vars_to_keep':['SA','cluster']})
    pp_map_press = ahf.Plot_Parameters(plot_type='map', clr_map='pcs_press', legend=False, extra_args={'map_extent':'Western_Arctic', 'extra_vars_to_keep':['SA','cluster']})
    pp_hist_SA = ahf.Plot_Parameters(plot_scale='by_pf', x_vars=['hist'], y_vars=['pcs_SA'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']})
    pp_hist_CT = ahf.Plot_Parameters(plot_scale='by_pf', x_vars=['hist'], y_vars=['pcs_CT'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']})
    pp_hist_press = ahf.Plot_Parameters(plot_scale='by_pf', x_vars=['hist'], y_vars=['pcs_press'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']})
    # Make the subplot groups
    group_map_SA = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_map_SA)
    group_map_CT = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_map_CT)
    group_map_press = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_map_press)
    group_hist_SA = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_hist_SA)
    group_hist_CT = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_hist_CT)
    group_hist_press = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_hist_press)
    # # Make the figure
    ahf.make_figure([group_map_SA, group_map_CT, group_map_press, group_hist_SA, group_hist_CT, group_hist_press])#, row_col_list=[2,1, 0.45, 1.4])
# Maps and histograms of one cluster's profile averages and spans in sigma
if False:
    print('')
    print('- Creating maps of one cluster`s profile averages and spans in sigma')
    # Make the Plot Parameters
    pp_press_vs_sigma = ahf.Plot_Parameters(x_vars=['sigma'], y_vars=['press'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']})
    pp_map_sigma_pca = ahf.Plot_Parameters(plot_type='map', clr_map='pca_sigma', legend=False, extra_args={'map_extent':'Western_Arctic', 'extra_vars_to_keep':['SA','cluster']})
    pp_map_sigma_pcs = ahf.Plot_Parameters(plot_type='map', clr_map='pcs_sigma', legend=False, extra_args={'map_extent':'Western_Arctic', 'extra_vars_to_keep':['SA','cluster']})
    pp_hist_sigma = ahf.Plot_Parameters(plot_scale='by_pf', x_vars=['hist'], y_vars=['sigma'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']})
    pp_hist_sigma_pca = ahf.Plot_Parameters(plot_scale='by_pf', x_vars=['hist'], y_vars=['pca_sigma'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']})
    pp_hist_sigma_pcs = ahf.Plot_Parameters(plot_scale='by_pf', x_vars=['hist'], y_vars=['pcs_sigma'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']})
    # Make the subplot groups
    group_p_vs_sig = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_press_vs_sigma)
    group_map_sigma_pca = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_map_sigma_pca)
    group_map_sigma_pcs = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_map_sigma_pcs)
    group_hist_sigma = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_hist_sigma)
    group_hist_sigma_pca = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_hist_sigma_pca)
    group_hist_sigma_pcs = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_hist_sigma_pcs)
    # # Make the figure
    ahf.make_figure([group_p_vs_sig, group_map_sigma_pca, group_map_sigma_pcs, group_hist_sigma, group_hist_sigma_pca, group_hist_sigma_pcs])#, row_col_list=[2,1, 0.45, 1.4])

################################################################################
## Pre-clustered correcting single clusters with polyfit2d for salinity
################################################################################
# Plotting across lat-lon a cluster's salinity, polyfit2d, and residual
if False:
    print('')
    print('- Creating lat-lon plots for one cluster`s salinity, polyfit2d, and residual') 
    SA_lims = [34.4925,34.4700]
    res_lims = [-0.01125,0.01125]
    # Make the Plot Parameters
    pp_2d = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='SA', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':SA_lims})
    pp_2d_hist = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['SA'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'y_lims':SA_lims})

    pp_fit = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='SA', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':SA_lims})
    pp_og_vs_res = ahf.Plot_Parameters(x_vars=['fit_SA'], y_vars=['SA'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'fit_vars':['lon','lat'], 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':SA_lims, 'y_lims':SA_lims})

    pp_res = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='SA-fit', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'fit_vars':['lon','lat'], 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':res_lims})
    pp_res_hist = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['SA-fit'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'fit_vars':['lon','lat'], 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'y_lims':res_lims})
    # Make the subplot groups
    group_2d = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_2d, plot_title='Cluster 6')
    group_2d_hist = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_2d_hist, plot_title='Cluster 6')
    group_fit = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_fit, plot_title='Polynomial Fit')
    group_og_vs_res = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_og_vs_res, plot_title='Original vs. Fit')
    group_res = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_res, plot_title='Residuals')
    group_res_hist = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_res_hist, plot_title='Residuals')
    # # Make the figure
    ahf.make_figure([group_2d, group_fit, group_res, group_2d_hist, group_og_vs_res, group_res_hist])
    # ahf.make_figure([group_2d])
# Plotting a cluster across lat-lon with a polyfit2d in salinity in 2D / 3D
if False:
    print('')
    print('- Creating plot for one cluster with a lat-lon-SA polyfit2d') 
    # Make the Plot Parameters
    pp_2d = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='SA', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':[34.4925,34.4700]})
    pp_3d_fit = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], z_vars=['SA'], clr_map='SA', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'z_lims':[34.4925,34.4700]})
    # Make the subplot groups
    group_2d = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_2d)
    group_3d = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_3d_fit)
    # # Make the figure
    ahf.make_figure([group_2d, group_3d])
# Plotting a cluster's salinity-polyfit2d across lat-lon in 2D / 3D
if False:
    print('')
    print('- Creating plot for one cluster SA minus lat-lon-SA polyfit2d') 
    # Make the Plot Parameters
    pp_minus_fit = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='SA-fit', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'extra_vars_to_keep':['SA','cluster'], 'fit_vars':['lon','lat']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR})
    pp_minus_fit3d = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], z_vars=['SA-fit'], clr_map='SA-fit', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'extra_vars_to_keep':['SA','cluster'], 'fit_vars':['lon','lat']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'z_lims':[-0.011,0.011]})
    # Make the subplot groups
    group_minus_fit = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_minus_fit)
    group_minus_fit3d = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_minus_fit3d)
    # # Make the figure
    ahf.make_figure([group_minus_fit, group_minus_fit3d])
# Comparing plots along time for salinity and salinity-polyfit2d with trendlines
if False:
    print('')
    print('- Comparing plots along longitude for one cluster SA minus lat-lon-SA polyfit2d') 
    # Make the Plot Parameters
    pp_p_v_lat = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['SA'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']})
    pp_minus_fit = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['SA-fit'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster'], 'fit_vars':['lon','lat']})
    # Make the subplot groups
    group_p_v_lat = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_p_v_lat, plot_title='Uncorrected')
    group_minus_fit = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_minus_fit, plot_title='Corrected by lat-lon-SA polyfit2d')
    # # Make the figure
    # ahf.make_figure([group_p_v_lat])
    ahf.make_figure([group_p_v_lat, group_minus_fit], row_col_list=[2,1, 0.45, 1.4])
# Comparing plots along longitude for salinity and salinity-polyfit2d with trendlines
if False:
    print('')
    print('- Comparing plots along longitude for one cluster SA minus lat-lon-SA polyfit2d') 
    # Make the Plot Parameters
    pp_p_v_lat = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['SA'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR})
    pp_minus_fit = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['SA-fit'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster'], 'fit_vars':['lon','lat']}, ax_lims={'x_lims':lon_BGR})
    # Make the subplot groups
    group_p_v_lat = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_p_v_lat, plot_title='Uncorrected')
    group_minus_fit = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_minus_fit, plot_title='Corrected by lat-lon-SA polyfit2d')
    # # Make the figure
    # ahf.make_figure([group_p_v_lat])
    ahf.make_figure([group_p_v_lat, group_minus_fit], row_col_list=[2,1, 0.45, 1.4])
# Comparing plots along latitude for salinity and salinity-polyfit2d with trendlines
if False:
    print('')
    print('- Comparing plots along latitude for one cluster SA minus lat-lon-SA polyfit2d') 
    # Make the Plot Parameters
    pp_p_v_lat = ahf.Plot_Parameters(x_vars=['lat'], y_vars=['SA'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lat_BGR})
    pp_minus_fit = ahf.Plot_Parameters(x_vars=['lat'], y_vars=['SA-fit'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster'], 'fit_vars':['lon','lat']}, ax_lims={'x_lims':lat_BGR})
    # Make the subplot groups
    group_p_v_lat = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_p_v_lat, plot_title='Uncorrected')
    group_minus_fit = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_minus_fit, plot_title='Corrected by lat-lon-SA polyfit2d')
    # # Make the figure
    # ahf.make_figure([group_p_v_lat])
    ahf.make_figure([group_p_v_lat, group_minus_fit], row_col_list=[2,1, 0.45, 1.4])
################################################################################
## Pre-clustered correcting single clusters with polyfit2d for temperature
################################################################################
# Plotting across lat-lon a cluster's temperature, polyfit2d, and residual
if False:
    print('')
    print('- Creating lat-lon plots for one cluster`s temperature, polyfit2d, and residual') 
    CT_lims = [-0.83,-0.48]
    res_lims = [.175,-.175]
    # Make the Plot Parameters
    pp_2d = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='CT', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':CT_lims})
    pp_2d_hist = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['CT'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'y_lims':CT_lims})

    pp_fit = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='CT', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':CT_lims})
    pp_og_vs_res = ahf.Plot_Parameters(x_vars=['fit_CT'], y_vars=['CT'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'fit_vars':['lon','lat'], 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':CT_lims, 'y_lims':CT_lims})

    pp_res = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='CT-fit', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'fit_vars':['lon','lat'], 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':res_lims})
    pp_res_hist = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['CT-fit'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'fit_vars':['lon','lat'], 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'y_lims':res_lims})
    # Make the subplot groups
    group_2d = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_2d, plot_title='Cluster 6')
    group_2d_hist = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_2d_hist, plot_title='Cluster 6')
    group_fit = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_fit, plot_title='Polynomial Fit')
    group_og_vs_res = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_og_vs_res, plot_title='Original vs. Fit')
    group_res = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_res, plot_title='Residuals')
    group_res_hist = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_res_hist, plot_title='Residuals')
    # # Make the figure
    ahf.make_figure([group_2d, group_fit, group_res, group_2d_hist, group_og_vs_res, group_res_hist])
    # ahf.make_figure([group_2d])
# Plotting a cluster across lat-lon with a polyfit2d in temperature in 2D / 3D
if False:
    print('')
    print('- Creating plot for one cluster with a lat-lon-CT polyfit2d') 
    # Make the Plot Parameters
    pp_2d = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='CT', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'extra_vars_to_keep':['CT','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':[-1.1,-0.45]})
    pp_3d_fit = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], z_vars=['CT'], clr_map='CT', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'z_lims':[-1.1,-0.45]})
    # Make the subplot groups
    group_2d = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_2d)
    group_3d = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_3d_fit)
    # # Make the figure
    ahf.make_figure([group_2d, group_3d])
# Plotting a cluster's temperature-polyfit2d across lat-lon in 2D / 3D
if False:
    print('')
    print('- Creating plot for one cluster CT minus lat-lon-CT polyfit2d') 
    # Make the Plot Parameters
    pp_minus_fit = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='CT-fit', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'extra_vars_to_keep':['SA','cluster'], 'fit_vars':['lon','lat']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR})
    pp_minus_fit3d = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], z_vars=['CT-fit'], clr_map='CT-fit', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'extra_vars_to_keep':['SA','cluster'], 'fit_vars':['lon','lat']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'z_lims':[-0.42,0.12]})
    # Make the subplot groups
    group_minus_fit = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_minus_fit)
    group_minus_fit3d = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_minus_fit3d)
    # # Make the figure
    ahf.make_figure([group_minus_fit, group_minus_fit3d])
# Comparing plots along time for temperature and temperature-polyfit2d with trendlines
if False:
    print('')
    print('- Comparing plots along longitude for one cluster CT minus lat-lon-CT polyfit2d') 
    # Make the Plot Parameters
    pp_p_v_lat = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['CT'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']})
    pp_minus_fit = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['CT-fit'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster'], 'fit_vars':['lon','lat']})
    # Make the subplot groups
    group_p_v_lat = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_p_v_lat, plot_title='Uncorrected')
    group_minus_fit = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_minus_fit, plot_title='Corrected by lat-lon-CT polyfit2d')
    # # Make the figure
    # ahf.make_figure([group_p_v_lat])
    ahf.make_figure([group_p_v_lat, group_minus_fit], row_col_list=[2,1, 0.45, 1.4])
# Comparing plots along longitude for temperature and temperature-polyfit2d with trendlines
if False:
    print('')
    print('- Comparing plots along longitude for one cluster CT minus lat-lon-CT polyfit2d') 
    # Make the Plot Parameters
    pp_p_v_lat = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['CT'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR})
    pp_minus_fit = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['CT-fit'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster'], 'fit_vars':['lon','lat']}, ax_lims={'x_lims':lon_BGR})
    # Make the subplot groups
    group_p_v_lat = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_p_v_lat, plot_title='Uncorrected')
    group_minus_fit = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_minus_fit, plot_title='Corrected by lat-lon-CT polyfit2d')
    # # Make the figure
    # ahf.make_figure([group_p_v_lat])
    ahf.make_figure([group_p_v_lat, group_minus_fit], row_col_list=[2,1, 0.45, 1.4])
# Comparing plots along latitude for temperature and temperature-polyfit2d with trendlines
if False:
    print('')
    print('- Comparing plots along latitude for one cluster CT minus lat-lon-CT polyfit2d') 
    # Make the Plot Parameters
    pp_p_v_lat = ahf.Plot_Parameters(x_vars=['lat'], y_vars=['CT'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lat_BGR})
    pp_minus_fit = ahf.Plot_Parameters(x_vars=['lat'], y_vars=['CT-fit'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster'], 'fit_vars':['lon','lat']}, ax_lims={'x_lims':lat_BGR})
    # Make the subplot groups
    group_p_v_lat = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_p_v_lat, plot_title='Uncorrected')
    group_minus_fit = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_minus_fit, plot_title='Corrected by lat-lon-CT polyfit2d')
    # # Make the figure
    # ahf.make_figure([group_p_v_lat])
    ahf.make_figure([group_p_v_lat, group_minus_fit], row_col_list=[2,1, 0.45, 1.4])
################################################################################
## Pre-clustered correcting single clusters with polyfit2d for pressure
################################################################################
# Plotting across lat-lon a cluster's pressure, polyfit2d, and residual
if False:
    print('')
    print('- Creating lat-lon plots for one cluster`s pressure, polyfit2d, and residual') 
    press_lims = [300,170]
    res_lims = [65,-65]
    # Make the Plot Parameters
    pp_2d = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='press', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':press_lims})
    pp_2d_hist = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['press'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'y_lims':press_lims})

    pp_fit = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='press', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':press_lims})
    pp_og_vs_res = ahf.Plot_Parameters(x_vars=['fit_press'], y_vars=['press'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'fit_vars':['lon','lat'], 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':press_lims, 'y_lims':press_lims})

    pp_res = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='press-fit', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'fit_vars':['lon','lat'], 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':res_lims})
    pp_res_hist = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['press-fit'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'fit_vars':['lon','lat'], 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'y_lims':res_lims})
    # Make the subplot groups
    group_2d = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_2d, plot_title='Cluster 6')
    group_2d_hist = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_2d_hist, plot_title='Cluster 6')
    group_fit = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_fit, plot_title='Polynomial Fit')
    group_og_vs_res = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_og_vs_res, plot_title='Original vs. Fit')
    group_res = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_res, plot_title='Residuals')
    group_res_hist = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_res_hist, plot_title='Residuals')
    # # Make the figure
    ahf.make_figure([group_2d, group_fit, group_res, group_2d_hist, group_og_vs_res, group_res_hist])
    # ahf.make_figure([group_2d])
# Plotting a cluster across lat-lon with a polyfit2d in pressure in 2D / 3D
if False:
    print('')
    print('- Creating plot for one cluster with a lat-lon-press polyfit2d') 
    # Make the Plot Parameters
    pp_2d = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='press', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':[350,150]})
    pp_3d_fit = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], z_vars=['press'], clr_map='press', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'z_lims':[350,150]})
    # Make the subplot groups
    group_2d = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_2d)
    group_3d = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_3d_fit)
    # # Make the figure
    ahf.make_figure([group_2d, group_3d])
# Plotting a cluster's pressure-polyfit2d across lat-lon in 2D / 3D
if False:
    print('')
    print('- Creating plot for one cluster press minus lat-lon-press polyfit2d') 
    # Make the Plot Parameters
    pp_minus_fit = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='press-fit', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'extra_vars_to_keep':['SA','cluster'], 'fit_vars':['lon','lat']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR})
    pp_minus_fit3d = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], z_vars=['press-fit'], clr_map='press-fit', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'extra_vars_to_keep':['SA','cluster'], 'fit_vars':['lon','lat']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'z_lims':[-70,70]})
    # Make the subplot groups
    group_minus_fit = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_minus_fit)
    group_minus_fit3d = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_minus_fit3d)
    # # Make the figure
    ahf.make_figure([group_minus_fit, group_minus_fit3d])
# Comparing plots along time for pressure and pressure-polyfit2d with trendlines
if False:
    print('')
    print('- Comparing plots along time for one cluster press minus lat-lon-press polyfit2d') 
    # Make the Plot Parameters
    pp_p_v_lat = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['press'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']})
    pp_minus_fit = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['press-fit'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster'], 'fit_vars':['lon','lat']})
    # Make the subplot groups
    group_p_v_lat = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_p_v_lat, plot_title='Uncorrected')
    group_minus_fit = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_minus_fit, plot_title='Corrected by lat-lon-press polyfit2d')
    # # Make the figure
    # ahf.make_figure([group_p_v_lat])
    ahf.make_figure([group_p_v_lat, group_minus_fit], row_col_list=[2,1, 0.45, 1.4])
# Comparing plots along longitude for pressure and pressure-polyfit2d with trendlines
if False:
    print('')
    print('- Comparing plots along longitude for one cluster press minus lat-lon-press polyfit2d') 
    # Make the Plot Parameters
    pp_p_v_lat = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['press'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR})
    pp_minus_fit = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['press-fit'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster'], 'fit_vars':['lon','lat']}, ax_lims={'x_lims':lon_BGR})
    # Make the subplot groups
    group_p_v_lat = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_p_v_lat, plot_title='Uncorrected')
    group_minus_fit = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_minus_fit, plot_title='Corrected by lat-lon-press polyfit2d')
    # # Make the figure
    # ahf.make_figure([group_p_v_lat])
    ahf.make_figure([group_p_v_lat, group_minus_fit], row_col_list=[2,1, 0.45, 1.4])
# Comparing plots along latitude for pressure and pressure-polyfit2d with trendlines
if False:
    print('')
    print('- Comparing plots along latitude for one cluster press minus lat-lon-press polyfit2d') 
    # Make the Plot Parameters
    pp_p_v_lat = ahf.Plot_Parameters(x_vars=['lat'], y_vars=['press'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lat_BGR})
    pp_minus_fit = ahf.Plot_Parameters(x_vars=['lat'], y_vars=['press-fit'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster'], 'fit_vars':['lon','lat']}, ax_lims={'x_lims':lat_BGR})
    # Make the subplot groups
    group_p_v_lat = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_p_v_lat, plot_title='Uncorrected')
    group_minus_fit = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_minus_fit, plot_title='Corrected by lat-lon-press polyfit2d')
    # # Make the figure
    # ahf.make_figure([group_p_v_lat])
    ahf.make_figure([group_p_v_lat, group_minus_fit], row_col_list=[2,1, 0.45, 1.4])

################################################################################
## Pre-clustered correcting many clusters with polyfit2d for pressure
################################################################################
pfs_clstr_4 = ahf.Profile_Filters(clstrs_to_plot=[4])
pfs_clstr_5 = ahf.Profile_Filters(clstrs_to_plot=[5])
pfs_clstr_6 = ahf.Profile_Filters(clstrs_to_plot=[6])
pfs_clstr_7 = ahf.Profile_Filters(clstrs_to_plot=[7])
pfs_clstr_8 = ahf.Profile_Filters(clstrs_to_plot=[8])
pfs_clstr_9 = ahf.Profile_Filters(clstrs_to_plot=[9])
# Plotting across lat-lon many clusters' pressure and polyfit2d
if False:
    print('')
    print('- Creating lat-lon plots for many clusters` pressure and polyfit2d') 
    press_lims = [300,170]
    # Make the Plot Parameters
    pp_fit = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='press', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':press_lims})
    # Make the subplot groups
    group_clstr_4 = ahf.Analysis_Group(ds_this_BGR, pfs_clstr_4, pp_fit, plot_title='Cluster 4')
    group_clstr_5 = ahf.Analysis_Group(ds_this_BGR, pfs_clstr_5, pp_fit, plot_title='Cluster 5')
    group_clstr_6 = ahf.Analysis_Group(ds_this_BGR, pfs_clstr_6, pp_fit, plot_title='Cluster 6')
    group_clstr_7 = ahf.Analysis_Group(ds_this_BGR, pfs_clstr_7, pp_fit, plot_title='Cluster 7')
    group_clstr_8 = ahf.Analysis_Group(ds_this_BGR, pfs_clstr_8, pp_fit, plot_title='Cluster 8')
    group_clstr_9 = ahf.Analysis_Group(ds_this_BGR, pfs_clstr_9, pp_fit, plot_title='Cluster 9')
    # # Make the figure
    ahf.make_figure([group_clstr_4, group_clstr_5, group_clstr_6, group_clstr_7, group_clstr_8, group_clstr_9])
# Plotting histograms of many clusters' pressure and polyfit2d
if False:
    print('')
    print('- Creating histograms for many clusters` pressure and polyfit2d') 
    press_lims = [300,170]
    # Make the Plot Parameters
    pp_hist = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['press'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':False, 'extra_vars_to_keep':['SA','cluster']})#, ax_lims={'y_lims':press_lims})
    # Make the subplot groups
    group_clstr_4 = ahf.Analysis_Group(ds_this_BGR, pfs_clstr_4, pp_hist, plot_title='Cluster 4')
    group_clstr_5 = ahf.Analysis_Group(ds_this_BGR, pfs_clstr_5, pp_hist, plot_title='Cluster 5')
    group_clstr_6 = ahf.Analysis_Group(ds_this_BGR, pfs_clstr_6, pp_hist, plot_title='Cluster 6')
    group_clstr_7 = ahf.Analysis_Group(ds_this_BGR, pfs_clstr_7, pp_hist, plot_title='Cluster 7')
    group_clstr_8 = ahf.Analysis_Group(ds_this_BGR, pfs_clstr_8, pp_hist, plot_title='Cluster 8')
    group_clstr_9 = ahf.Analysis_Group(ds_this_BGR, pfs_clstr_9, pp_hist, plot_title='Cluster 9')
    # # Make the figure
    ahf.make_figure([group_clstr_4, group_clstr_5, group_clstr_6, group_clstr_7, group_clstr_8, group_clstr_9])
# Comparing plots along time for pressure and pressure-polyfit2d with trendlines
if False:
    print('')
    print('- Comparing plots along time for many clusters` press minus lat-lon-press polyfit2d') 
    # Make the Plot Parameters
    pp_p_v_lat = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['press'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']})
    pp_minus_fit = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['press-fit'], legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster'], 'fit_vars':['lon','lat']})
    pp_fit = pp_minus_fit
    # Make the subplot groups
    group_clstr_4 = ahf.Analysis_Group(ds_this_BGR, pfs_clstr_4, pp_fit, plot_title='Cluster 4')
    group_clstr_5 = ahf.Analysis_Group(ds_this_BGR, pfs_clstr_5, pp_fit, plot_title='Cluster 5')
    group_clstr_6 = ahf.Analysis_Group(ds_this_BGR, pfs_clstr_6, pp_fit, plot_title='Cluster 6')
    group_clstr_7 = ahf.Analysis_Group(ds_this_BGR, pfs_clstr_7, pp_fit, plot_title='Cluster 7')
    group_clstr_8 = ahf.Analysis_Group(ds_this_BGR, pfs_clstr_8, pp_fit, plot_title='Cluster 8')
    group_clstr_9 = ahf.Analysis_Group(ds_this_BGR, pfs_clstr_9, pp_fit, plot_title='Cluster 9')
    # # Make the figure
    ahf.make_figure([group_clstr_4, group_clstr_5, group_clstr_6, group_clstr_7, group_clstr_8, group_clstr_9])

################################################################################
## Pre-clustered comparing multiple clusters across time periods
################################################################################
# Define the profile filters
pfs_these_clstrs = ahf.Profile_Filters(clstrs_to_plot=[4,5,6,7,8,9])
# Multiple clusters in SA vs. la_CT space
if False:
    print('')
    print('- Creating plot of pre-clustered BGR ITP data for multiple cluster')
    pp_pre_clstrd = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'sort_clstrs':False, 'b_a_w_plt':True}, ax_lims={'x_lims':test_S_range})
    # Make the subplot groups
    group_pre_clstrd = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_pre_clstrd)
    # Plot the figure
    ahf.make_figure([group_pre_clstrd], use_same_y_axis=False)
# Multiple clusters plotting SA, CT, and press vs. time with trend lines
if False:
    print('')
    print('- Creating plots across time to look at multiple clusters across different periods')
    # Make the Plot Parameters
    pp_SA = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['SA'], clr_map='cluster', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']})#, 'clstrs_to_plot':[4,5,6]}) 
    pp_CT = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['CT'], clr_map='cluster', legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']})
    pp_press = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['press'], clr_map='cluster', legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']})
    # Make the subplot groups
    group_SA = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_SA)
    group_CT = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_CT)
    group_press = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_press)
    # Make the figure
    ahf.make_figure([group_SA, group_CT, group_press], row_col_list=[3,1, 0.45, 1.4])
# Multiple clusters plotting SA, CT, and press vs. longitude with trend lines
if False:
    print('')
    print('- Creating plots across longitude to look at multiple clusters across different periods')
    # Make the Plot Parameters
    pp_SA = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['SA'], clr_map='cluster', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR})#, 'clstrs_to_plot':[4,5,6]}) 
    pp_CT = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['CT'], clr_map='cluster', legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR})
    pp_press = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['press'], clr_map='cluster', legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR})
    # Make the subplot groups
    group_SA = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_SA)
    group_CT = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_CT)
    group_press = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_press)
    # Make the figure
    ahf.make_figure([group_SA, group_CT, group_press], row_col_list=[3,1, 0.45, 1.4])
# Multiple clusters plotting SA, CT, and press vs. latitude with trend lines
if False:
    print('')
    print('- Creating plots across latitude to look at multiple clusters across different periods')
    # Make the Plot Parameters
    pp_SA = ahf.Plot_Parameters(x_vars=['lat'], y_vars=['SA'], clr_map='cluster', legend=True, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lat_BGR})#, 'clstrs_to_plot':[4,5,6]}) 
    pp_CT = ahf.Plot_Parameters(x_vars=['lat'], y_vars=['CT'], clr_map='cluster', legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lat_BGR})
    pp_press = ahf.Plot_Parameters(x_vars=['lat'], y_vars=['press'], clr_map='cluster', legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':'OLS', 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lat_BGR})
    # Make the subplot groups
    group_SA = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_SA)
    group_CT = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_CT)
    group_press = ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_press)
    # Make the figure
    ahf.make_figure([group_SA, group_CT, group_press], row_col_list=[3,1, 0.45, 1.4])

################################################################################


## la_CT-SA vs. la_CT-SA-dt_start plots
if False:
    print('')
    print('- Creating TS and TS-time plots')
    # Make the Plot Parameters
    pp_CT_SA = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['SA'], clr_map='instrmt', legend=True)
    # pp_CT_SA = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['SA'], clr_map='density_hist', extra_args={'clr_min':0, 'clr_max':10, 'clr_ext':'max', 'xy_bins':1000, 'log_axes':[False,False,True]})
    # pp_CT_SA_3d = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['dt_start'], z_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'dt_start', 'cl_z_var':'la_CT', 'm_pts':280, 'b_a_w_plt':False})
    # Make the Analysis Group pfs_fltrd pfs_BGR1
    group_CT_SA_plot = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_CT_SA)
    # group_CT_SA_plot = ahf.Analysis_Group(ds_ITP_test, pfs_BGR1, pp_CT_SA)
    # group_CT_SA_3d_plot = ahf.Analysis_Group(ds_BGOS, pfs_fltrd, pp_CT_SA_3d)
    # Make the figure
    ahf.make_figure([group_CT_SA_plot], filename='Figure_2.pickle')
## lon-dt_start plots
if False:
    print('')
    print('- Creating lon-time plots')
    # Make the Plot Parameters
    pp_lon_dt = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['lon'], clr_map='instrmt')
    # Make the Analysis Group pfs_fltrd pfs_BGR1
    group_lon_dt_plot = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_lon_dt)
    # Make the figure
    ahf.make_figure([group_lon_dt_plot])

## Comparing multiple BGR ITP time period clusterings
# BGR ITP clustering
if False:
    print('')
    print('- Creating plots to compare pre-clustered BGR ITP data')
    # this_ds = ds_BGRa_m110
    # this_ds = ds_BGRm_m410
    # this_ds = ds_BGRn_m240
    this_ds = ds_BGRmn
    # this_ds = ds_BGRno
    # this_ds = ds_BGRmno
    # Make the Plot Parameters
    # pp_comp_clstrs = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['hist'], clr_map='clr_by_dataset', extra_args={'extra_vars_to_keep':['cluster']}) 
    pp_comp_clstrs = ahf.Plot_Parameters(x_vars=['ca_SA'], y_vars=['ca_CT'], clr_map='clr_by_dataset', extra_args={'extra_vars_to_keep':['cluster'], 'errorbars':True}) 
    # Make the subplot groups
    group_comp_clstrs = ahf.Analysis_Group(this_ds, pfs_0, pp_comp_clstrs)
    # print('done making analysis group')
    # # Make the figure
    ahf.make_figure([group_comp_clstrs])
if False:
    print('')
    print('- Creating plots to compare pre-clustered BGR ITP data')
    # this_ds = ds_BGRa_m110
    # this_ds = ds_BGRm_m410
    # this_ds = ds_BGRn_m240
    # this_ds = ds_BGRmn
    this_ds = ds_BGRno
    # this_ds = ds_BGRmno
    # Make the Plot Parameters
    # pp_comp_clstrs = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['hist'], clr_map='clr_by_dataset', extra_args={'extra_vars_to_keep':['cluster']}) 
    pp_comp_clstrs = ahf.Plot_Parameters(x_vars=['ca_SA'], y_vars=['ca_CT'], clr_map='clr_by_dataset', extra_args={'extra_vars_to_keep':['cluster'], 'errorbars':True}) 
    # Make the subplot groups
    group_comp_clstrs = ahf.Analysis_Group(this_ds, pfs_0, pp_comp_clstrs)
    # print('done making analysis group')
    # # Make the figure
    ahf.make_figure([group_comp_clstrs])
if False:
    print('')
    print('- Creating plots to compare pre-clustered BGR ITP periods')
    this_ds = ds_BGR05060708
    # Make the Plot Parameters
    # pp_comp_clstrs = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['hist'], clr_map='clr_by_dataset', extra_args={'extra_vars_to_keep':['cluster']}) 
    pp_comp_clstrs = ahf.Plot_Parameters(x_vars=['ca_SA'], y_vars=['ca_CT'], clr_map='clr_by_dataset', extra_args={'extra_vars_to_keep':['cluster'], 'errorbars':False}) 
    # Make the subplot groups
    group_comp_clstrs = ahf.Analysis_Group(this_ds, pfs_0, pp_comp_clstrs)
    # # Make the figure
    ahf.make_figure([group_comp_clstrs])
    #
    pp_comp_clstrs = ahf.Plot_Parameters(x_vars=['ca_SA'], y_vars=['ca_CT'], clr_map='clr_by_dataset', extra_args={'extra_vars_to_keep':['cluster'], 'errorbars':True}) 
    # Make the subplot groups
    group_comp_clstrs = ahf.Analysis_Group(this_ds, pfs_0, pp_comp_clstrs)
    # # Make the figure
    ahf.make_figure([group_comp_clstrs])

# BGR ITP clustering, histograms
if False:
    print('')
    print('- Creating histograms of BGR ITP data')
    # Make the Plot Parameters
    pp_comp_clstrs = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['hist'], clr_map='cluster', extra_args={'plt_noise':True}, legend=False, ax_lims={'x_lims':test_S_range})
    # Make the subplot groups
    group_clstr_hist = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_comp_clstrs)
    # # Make the figure
    ahf.make_figure([group_clstr_hist])
# BGR ITP clustering, comparing with histograms
if False:
    print('')
    print('- Creating plots to compare pre-clustered BGR ITP data')
    # Make the Plot Parameters
    pp_comp_clstrs = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['hist'], clr_map='cluster', extra_args={'plt_noise':False}, legend=False)#, ax_lims=test_S_range) 
    # Make the subplot groups
    group_comp_clstrs0 = ahf.Analysis_Group(ds_BGR0506, pfs_0, pp_comp_clstrs)
    group_comp_clstrs1 = ahf.Analysis_Group(ds_BGR0607, pfs_0, pp_comp_clstrs)
    group_comp_clstrs2 = ahf.Analysis_Group(ds_BGR0708, pfs_0, pp_comp_clstrs)
    # # Make the figure
    # ahf.make_figure([group_comp_clstrs0, group_comp_clstrs1], row_col_list=[2,1, 0.8, 1.25])
    ahf.make_figure([group_comp_clstrs0, group_comp_clstrs1, group_comp_clstrs2], row_col_list=[3,1, 0.4, 2.0])
# BGR ITP clustering, comparing with histograms, looking for valleys
if False:
    print('')
    print('- Creating plots to compare pre-clustered BGR ITP data')
    # Make the Plot Parameters
    h_bins = 1000
    pp_comp_clstrs = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['hist'], clr_map='clr_by_dataset', extra_args={'plt_noise':False, 'log_axes':[False,True,False], 'n_h_bins':h_bins}, legend=False, ax_lims=test_S_range) 
    pp_comp_clstrs1 = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['hist'], clr_map='clr_all_same', extra_args={'plt_noise':False, 'log_axes':[False,True,False], 'n_h_bins':h_bins}, legend=False, ax_lims=test_S_range) 
    # Make the subplot groups
    group_comp_clstrs0 = ahf.Analysis_Group(ds_BGR05060708, pfs_0, pp_comp_clstrs1, plot_title='BGR05060708 with noise')
    group_comp_clstrs1 = ahf.Analysis_Group(ds_BGR05060708_no_noise, pfs_0, pp_comp_clstrs1, plot_title='BGR05060708 without noise')
    # # Make the figure
    ahf.make_figure([group_comp_clstrs0, group_comp_clstrs1], row_col_list=[2,1, 0.8, 1.25])
# BGR ITP clustering, comparing with 2D histograms
if False:
    print('')
    print('- Creating plots to compare pre-clustered BGR ITP data')
    # Make the Plot Parameters
    pp_comp_clstrs = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['dt_start'], clr_map='density_hist', extra_args={'clr_min':0, 'clr_max':15, 'clr_ext':'max', 'xy_bins':1000, 'log_axes':[False,False,True]}, legend=False, ax_lims=test_S_range) 
    # Make the subplot groups
    group_comp_clstrs0 = ahf.Analysis_Group(ds_BGR05060708, pfs_0, pp_comp_clstrs, plot_title='BGR05060708 with noise')
    group_comp_clstrs1 = ahf.Analysis_Group(ds_BGR05060708_no_noise, pfs_0, pp_comp_clstrs, plot_title='BGR05060708 without noise')
    # # Make the figure
    ahf.make_figure([group_comp_clstrs0, group_comp_clstrs1], row_col_list=[2,1, 0.8, 1.25])
# BGR ITP clustering, comparing on salinity vs time
if False:
    print('')
    print('- Creating plots to compare pre-clustered BGR ITP data')
    # Make the Plot Parameters
    pp_comp_clstrs = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['dt_start'], clr_map='cluster', legend=False, ax_lims=test_S_range, extra_args={'extra_vars_to_keep':['CT']}) 
    # Make the subplot groups
    group_comp_clstrs0 = ahf.Analysis_Group(ds_BGR05060708, pfs_0, pp_comp_clstrs, plot_title='BGR05060708 with noise')
    group_comp_clstrs1 = ahf.Analysis_Group(ds_BGR05060708_no_noise, pfs_0, pp_comp_clstrs, plot_title='BGR05060708 without noise')
    # # Make the figure
    ahf.make_figure([group_comp_clstrs0, group_comp_clstrs1], row_col_list=[2,1, 0.8, 1.25], filename='test_SA_vs_time.pickle')
    # ahf.make_figure([group_comp_clstrs1])

# BGR ITP clustering, comparing across time
if False:
    print('')
    print('- Creating plots of pre-clustered BGR ITP data')
    pp_pre_clstrd = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['CT'], clr_map='cluster', extra_args={'b_a_w_plt':False}, legend=False)
    # Make the subplot groups
    group_pre_clstrd = ahf.Analysis_Group(ds_BGR0506, pfs_0, pp_pre_clstrd)
    group_pre_clstrd1 = ahf.Analysis_Group(ds_BGR0607, pfs_0, pp_pre_clstrd)
    group_pre_clstrd2 = ahf.Analysis_Group(ds_BGR0708, pfs_0, pp_pre_clstrd)
    # # Make the figure
    ahf.make_figure([group_pre_clstrd, group_pre_clstrd1, group_pre_clstrd2], use_same_x_axis=False)

    pp_pre_clstrd = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['press'], clr_map='cluster', extra_args={'b_a_w_plt':False}, legend=False)
    # Make the subplot groups
    group_pre_clstrd = ahf.Analysis_Group(ds_BGR0506, pfs_0, pp_pre_clstrd)
    group_pre_clstrd1 = ahf.Analysis_Group(ds_BGR0607, pfs_0, pp_pre_clstrd)
    group_pre_clstrd2 = ahf.Analysis_Group(ds_BGR0708, pfs_0, pp_pre_clstrd)
    # # Make the figure
    ahf.make_figure([group_pre_clstrd, group_pre_clstrd1, group_pre_clstrd2], use_same_x_axis=False)
# BGR ITP clustering, SA vs la_CT and SA vs time
exit(0)
for ds_this_BGR in [ds_BGR04, ds_BGR0506, ds_BGR0607, ds_BGR0708]:
    if True:
        print('')
        print('- Creating plots of pre-clustered BGR ITP data')
        pp_pre_clstrd_sali = ahf.Plot_Parameters(x_vars=['la_CT'], y_vars=['SA'], clr_map='cluster', extra_args={'b_a_w_plt':True}, ax_lims={'y_lims':test_S_range})
        pp_pre_clstrd_time = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['SA'], clr_map='cluster', extra_args={'b_a_w_plt':False}, legend=False)
        # Make the subplot groups
        group_pre_clstrd_time = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_pre_clstrd_sali)
        group_pre_clstrd_sali = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_pre_clstrd_time)
        # # Make the figure
        ahf.make_figure([group_pre_clstrd_time, group_pre_clstrd_sali], use_same_x_axis=False)
