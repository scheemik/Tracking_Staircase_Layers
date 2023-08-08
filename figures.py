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

# BGOS_S_range = [34.1, 34.76]
BGOS_S_range = [34.366, 34.9992]
LHW_S_range = [34.366, 35.5]
AIDJEX_S_range = [34.366, 35.0223]

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
start_pf = 1
import numpy as np
n_pfs_to_plot = 10
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

# All profiles from all sources in this study
# all_sources = {'AIDJEX_BigBear':'all','AIDJEX_BlueFox':'all','AIDJEX_Caribou':'all','AIDJEX_Snowbird':'all','ITP_2':'all','ITP_3':'all'}
all_sources = {'AIDJEX_BigBear':'all','AIDJEX_BlueFox':'all','AIDJEX_Caribou':'all','AIDJEX_Snowbird':'all','SHEBA_Seacat':'all','ITP_33':'all','ITP_34':'all','ITP_35':'all','ITP_41':'all','ITP_42':'all','ITP_43':'all'}

# All profiles from all ITPs in this study
all_ITPs = {'ITP_2':'all','ITP_3':'all','ITP_35':'all','ITP_41':'all','ITP_42':'all','ITP_43':'all'}
all_BGOS = {'ITP_33':'all','ITP_34':'all','ITP_35':'all','ITP_41':'all','ITP_42':'all','ITP_43':'all'}

# All profiles from all AIDJEX camps
all_AIDJEX = {'AIDJEX_BigBear':'all','AIDJEX_BlueFox':'all','AIDJEX_Caribou':'all','AIDJEX_Snowbird':'all'}
# Pre-clustered AIDJEX file
all_AIDJEX_clstrd = {'AIDJEX_all':'all'}

# All profiles from certain ITPs
ITP2_all  = {'ITP_2':'all'}
ITP3_all  = {'ITP_3':'all'}
ITP33_all = {'ITP_33':'all'}
ITP34_all = {'ITP_34':'all'}
ITP35_all = {'ITP_35':'all'}
ITP41_all = {'ITP_41':'all'}
ITP42_all = {'ITP_42':'all'}
ITP43_all = {'ITP_43':'all'}

# A list of many profiles to plot from ITP3
start_pf = 1331
import numpy as np
n_pfs_to_plot = 5
ITP3_some_pfs_1 = list(np.arange(start_pf, start_pf+(n_pfs_to_plot*2), 2))
ITP3_pfs1  = {'ITP_3':ITP3_some_pfs_1}
ITP35_pfs0 = {'ITP_35':ITP35_some_pfs0}
ITP35_pfs1 = {'ITP_35':ITP35_some_pfs1}
ITP35_pfs2 = {'ITP_35':ITP35_some_pfs2}

## AIDJEX

AIDJEX_BigBear_all = {'AIDJEX_BigBear':'all'}
AIDJEX_BlueFox_all = {'AIDJEX_BlueFox':'all'}
AIDJEX_Caribou_all = {'AIDJEX_Caribou':'all'}
AIDJEX_Snowbird_all = {'AIDJEX_Snowbird':'all'}

AIDJEX_BigBear_blacklist = {'AIDJEX_BigBear':[531, 535, 537, 539, 541, 543, 545, 547, 549]}
AIDJEX_BlueFox_blacklist = {'AIDJEX_BlueFox':[94, 308, 310]}
AIDJEX_Snowbird_blacklist = {'AIDJEX_Snowbird':[443]}

AIDJEX_missing_ll = {'AIDJEX_BigBear':[4, 5, 6, 7, 8, 9, 10, 13, 14, 22]}#, 28, 30, 32]}

## SHEBA

all_SHEBA = {'SHEBA_Seacat':'all'}
SHEBA_Seacat_blacklist = {'SHEBA_Seacat':['SH15200.UP', 'SH03200', 'SH30500', 'SH34100']}
SHEBA_example_profile = {'SHEBA_Seacat':['CT165654']}

################################################################################
# Create data filtering objects
print('- Creating data filtering objects')
################################################################################

dfs_all = ahf.Data_Filters(keep_black_list=True, cast_direction='any')
dfs0 = ahf.Data_Filters()
dfs1 = ahf.Data_Filters(min_press=400)
dfs2 = ahf.Data_Filters(min_press_CT_max=400)

################################################################################
# Create data sets by combining filters and the data to load in
print('- Creating data sets')
################################################################################

# ds_all_sources_all  = ahf.Data_Set(all_sources, dfs_all)
# ds_all_sources_up   = ahf.Data_Set(all_sources, dfs0)
# ds_all_sources_pmin = ahf.Data_Set(all_sources, dfs1)

## ITP

# ds_all_BGOS = ahf.Data_Set(all_BGOS, dfs_all)
ds_BGOS = ahf.Data_Set(all_BGOS, dfs1)
# ds_all_BGOS = ahf.Data_Set(all_BGOS, dfs0)

# ds_ITP2_all = ahf.Data_Set(ITP2_all, dfs0)

# Data Sets without filtering based on CT_max
# ds_ITP22_all  = ahf.Data_Set(ITP22_all, dfs_all)
# ds_ITP23_all  = ahf.Data_Set(ITP23_all, dfs_all)
# ds_ITP32_all  = ahf.Data_Set(ITP32_all, dfs_all)
# ds_ITP33_all  = ahf.Data_Set(ITP33_all, dfs_all)
# ds_ITP34_all  = ahf.Data_Set(ITP34_all, dfs_all)
# ds_ITP35_all  = ahf.Data_Set(ITP35_all, dfs_all)
# ds_ITP41_all  = ahf.Data_Set(ITP41_all, dfs0)
# ds_ITP42_all  = ahf.Data_Set(ITP42_all, dfs0)
# ds_ITP43_all  = ahf.Data_Set(ITP43_all, dfs0)

# Data Sets filtered based on CT_max
# ds_ITP35_dfs1 = ahf.Data_Set(ITP35_all, dfs1)
# ds_ITP41_dfs1 = ahf.Data_Set(ITP41_all, dfs1)
# ds_ITP42_dfs1 = ahf.Data_Set(ITP42_all, dfs1)
# ds_ITP43_dfs1 = ahf.Data_Set(ITP43_all, dfs1)
# 
# ds_ITP35_some_pfs0 = ahf.Data_Set(ITP35_pfs0, dfs0)
# ds_ITP35_some_pfs1 = ahf.Data_Set(ITP35_pfs1, dfs0)
# ds_ITP35_some_pfs2 = ahf.Data_Set(ITP35_pfs2, dfs0)

## AIDJEX

# ds_all_AIDJEX = ahf.Data_Set(all_AIDJEX, dfs_all)
ds_AIDJEX = ahf.Data_Set(all_AIDJEX, dfs1)
ds_AIDJEX_clstrd = ahf.Data_Set(all_AIDJEX_clstrd, dfs1)

# ds_AIDJEX_BigBear  = ahf.Data_Set(AIDJEX_BigBear_all, dfs1)
# ds_AIDJEX_BlueFox  = ahf.Data_Set(AIDJEX_BlueFox_all, dfs2)
# ds_AIDJEX_Caribou  = ahf.Data_Set(AIDJEX_Caribou_all, dfs2)
# ds_AIDJEX_Snowbird = ahf.Data_Set(AIDJEX_Snowbird_all, dfs2)

# ds_AIDJEX_BigBear_blacklist = ahf.Data_Set(AIDJEX_BigBear_blacklist, dfs_all)
# ds_AIDJEX_BlueFox_blacklist = ahf.Data_Set(AIDJEX_BlueFox_blacklist, dfs_all)
# ds_AIDJEX_Snowbird_blacklist = ahf.Data_Set(AIDJEX_Snowbird_blacklist, dfs_all)

# ds_AIDJEX_missing_ll = ahf.Data_Set(AIDJEX_missing_ll, dfs_all)

## SHEBA

# ds_all_SHEBA = ahf.Data_Set(all_SHEBA, dfs_all)
# ds_SHEBA = ahf.Data_Set(all_SHEBA, dfs1)
# ds_SHEBA_Seacat_blacklist = ahf.Data_Set(SHEBA_Seacat_blacklist, dfs_all)
# ds_SHEBA_example_profile = ahf.Data_Set(SHEBA_example_profile, dfs_all)

################################################################################
# Create profile filtering objects
print('- Creating profile filtering objects')
################################################################################

pfs_0 = ahf.Profile_Filters()
pfs_1 = ahf.Profile_Filters(SA_range=ITP2_S_range)
pfs_2 = ahf.Profile_Filters(p_range=[400,225])

# AIDJEX Operation Area
pfs_AOA = ahf.Profile_Filters(lon_range=[-152.9,-133.7],lat_range=[72.6,77.4])

pfs_ITP2  = ahf.Profile_Filters(SA_range=ITP2_S_range)
pfs_ITP3  = ahf.Profile_Filters(SA_range=Lu2022_S_range)
pfs_BGOS  = ahf.Profile_Filters(lon_range=[-152.9,-133.7],lat_range=[72.6,77.4],SA_range=BGOS_S_range)
pfs_fltrd = ahf.Profile_Filters(lon_range=[-152.9,-133.7], lat_range=[72.6,77.4], SA_range=LHW_S_range, lt_pCT_max=True)

pfs_subs = ahf.Profile_Filters(SA_range=ITP2_S_range, subsample=True)
pfs_regrid = ahf.Profile_Filters(SA_range=ITP2_S_range, regrid_TS=['CT',0.01,'SP',0.005])
pfs_ss_rg = ahf.Profile_Filters(SA_range=ITP2_S_range, subsample=True, regrid_TS=['CT',0.01,'SP',0.005])

pfs_AIDJEX = ahf.Profile_Filters(SA_range=AIDJEX_S_range)

################################################################################
# Create plotting parameter objects

# print('- Creating plotting parameter objects')
################################################################################

### Test plots

## Number of up-going profiles
# Make the Plot Parameters
# pp_test = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='prof_no', legend=True)
# pp_test = ahf.Plot_Parameters(x_vars=['SA','CT'], y_vars=['press'], plot_type='profiles')
# Make the Analysis Group
# group_AIDJEX_missing_ll = ahf.Analysis_Group(ds_AIDJEX_missing_ll, pfs_0, pp_test, plot_title='BigBear lat-lon errors')
# group_AIDJEX_missing_ll = ahf.Analysis_Group(ds_ITP35_some_pfs2, pfs_0, pp_test, plot_title='ITP35 test pfs')
# group_test = ahf.Analysis_Group(ds_all_AIDJEX, pfs_0, pp_test)
# Make the figure
# ahf.make_figure([group_AIDJEX_missing_ll])
# exit(0)

## Regrid
# Make the Profile Filters
pfs_regrid = ahf.Profile_Filters(SA_range=ITP2_S_range, regrid_TS=['CT',0.01,'SP',0.005])
# Make the Plot Parameters
pp_regrid = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['CT'], clr_map='clr_all_same', legend=True)
# Make the Analysis Group
# group_regrid = ahf.Analysis_Group(ds_ITP2_all, pfs_regrid, pp_regrid)
# Make the figure
# ahf.make_figure([group_regrid])

pp_clstr_test = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='cluster', extra_args={'b_a_w_plt':True}, legend=True, add_grid=True)

################################################################################
### Figures

## AIDJEX blacklisted profiles TS plots
if False:
    print('')
    print('- Creating figure for AIDJEX blacklisted profiles')
    # Make the Plot Parameters
    pp_all = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['iT'], clr_map='clr_by_instrmt', legend=True)
    pp_one = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['iT'], clr_map='prof_no', legend=True)
    # Make the Analysis Group
    group_AIDJEX__blacklist = ahf.Analysis_Group(ds_all_AIDJEX, pfs_0, pp_all, plot_title='All AIDJEX Profiles')
    group_BigBear_blacklist = ahf.Analysis_Group(ds_AIDJEX_BigBear_blacklist, pfs_0, pp_one, plot_title='BigBear Blacklist')
    group_BlueFox_blacklist = ahf.Analysis_Group(ds_AIDJEX_BlueFox_blacklist, pfs_0, pp_one, plot_title='BlueFox Blacklist')
    group_Snowbird_blacklist = ahf.Analysis_Group(ds_AIDJEX_Snowbird_blacklist, pfs_0, pp_one, plot_title='Snowbird Blacklist')
    # Make the figure
    ahf.make_figure([group_AIDJEX__blacklist, group_BigBear_blacklist, group_BlueFox_blacklist, group_Snowbird_blacklist], use_same_x_axis=False, use_same_y_axis=False)
## SHEBA blacklisted profiles plots
if False:
    print('')
    print('- Creating figure for SHEBA blacklisted profiles')
    # Make the Plot Parameters
    # pp_pfs = ahf.Plot_Parameters(x_vars=['SP','iT'], y_vars=['press'], plot_type='profiles')
    pp_pfs = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['iT'])
    # Make the Analysis Group
    group_SHEBA_blacklist = ahf.Analysis_Group(ds_SHEBA_Seacat_blacklist, pfs_0, pp_pfs, plot_title='SHEBA Blacklisted Profiles')
    # Make the figure
    ahf.make_figure([group_SHEBA_blacklist])
## AIDJEX BigBear profiles missing latitude and longitude values
if False:
    print('')
    print('- Creating figure for BigBear profiles missing latitude and longitude values')
    # Make the Plot Parameters
    pp_pfs = ahf.Plot_Parameters(x_vars=['SA','CT'], y_vars=['press'], plot_type='profiles')
    # Make the Analysis Group
    group_AIDJEX_missing_ll = ahf.Analysis_Group(ds_AIDJEX_missing_ll, pfs_0, pp_pfs, plot_title='BigBear lat-lon errors')
    # Make the figure
    ahf.make_figure([group_AIDJEX_missing_ll])
## SHEBA TS plot
if False:
    print('')
    print('- Creating TS plot for SHEBA profiles')
    # Make the Plot Parameters
    pp_TS = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='press')
    # Make the Analysis Group
    group_SHEBA_TS = ahf.Analysis_Group(ds_SHEBA, pfs_2, pp_TS)
    # Make the figure
    ahf.make_figure([group_SHEBA_TS])
## Example profiles from all sources
if False:
    print('')
    print('- Creating figure of example profiles')
    # Make the Plot Parameters
    pp_pfs = ahf.Plot_Parameters(x_vars=['SA','CT'], y_vars=['press'], plot_type='profiles')
    # Make the Analysis Group
    group_SHEBA_profile = ahf.Analysis_Group(ds_SHEBA_example_profile, pfs_0, pp_pfs)
    # Make the figure
    ahf.make_figure([group_SHEBA_profile])
## Histograms of pressure max
if False:
    print('')
    print('- Creating histograms of pressure max in each profile')
    # Make the Plot Parameters
    pp_max_press_hist = ahf.Plot_Parameters(plot_scale='by_pf', x_vars=['max_press'], y_vars=['hist'], clr_map='clr_all_same')
    # Make the subplot groups
    group_BGOS_max_press_hist = ahf.Analysis_Group(ds_all_BGOS, pfs_0, pp_max_press_hist, plot_title=r'BGOS ITPs')
    group_BGOS_mp_max_press_hist = ahf.Analysis_Group(ds_all_BGOS_mp, pfs_0, pp_max_press_hist, plot_title=r'BGOS ITPs, min press filtered')
    # group_AIDJEX_max_press_hist = ahf.Analysis_Group(ds_all_AIDJEX, pfs_0, pp_max_press_hist, plot_title=r'AIDJEX')
    # Make the figure
    ahf.make_figure([group_BGOS_max_press_hist, group_BGOS_mp_max_press_hist])
## Histograms of CT_max data
if False:
    print('')
    print('- Creating histograms of pressure at temperature max in each profile')
    # Make the Plot Parameters
    pp_press_CT_max_hist = ahf.Plot_Parameters(x_vars=['press_CT_max'], y_vars=['hist'], clr_map='clr_all_same')
    # Make the subplot groups
    group_BGOS_press_CT_max_hist = ahf.Analysis_Group(ds_all_BGOS, pfs_AOA, pp_press_CT_max_hist, plot_title=r'BGOS ITPs')
    group_AIDJEX_press_CT_max_hist = ahf.Analysis_Group(ds_all_AIDJEX, pfs_AOA, pp_press_CT_max_hist, plot_title=r'AIDJEX')
    # # Make the figure
    ahf.make_figure([group_BGOS_press_CT_max_hist, group_AIDJEX_press_CT_max_hist])

### Figure 1
## Maps
pp_map = ahf.Plot_Parameters(plot_type='map', clr_map='clr_by_instrmt', extra_args={'map_extent':'AIDJEX_focus'})
pp_map_full_Arctic = ahf.Plot_Parameters(plot_type='map', clr_map='clr_by_source', extra_args={'map_extent':'Full_Arctic'}, legend=False, add_grid=False)
pp_map_by_date = ahf.Plot_Parameters(plot_type='map', clr_map='dt_start', extra_args={'map_extent':'AIDJEX_focus'})
## Map of all profiles for all sources
if False:
    print('')
    print('- Creating a map of all profiles from all sources')
    # Make the subplot groups
    group_map_full_Arctic = ahf.Analysis_Group(ds_all_sources_all, pfs_0, pp_map_full_Arctic, plot_title='')
    group_map = ahf.Analysis_Group(ds_all_sources_all, pfs_0, pp_map, plot_title='')
    # Make the figure
    ahf.make_figure([group_map_full_Arctic, group_map], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_1.pickle')
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
## Map of select profiles from all sources within AOA
if False:
    print('')
    print('- Creating a map of select profiles from all sources within AOA')
    # Make the subplot groups
    group_map_full_Arctic = ahf.Analysis_Group(ds_all_sources_pmin, pfs_AOA, pp_map_full_Arctic, plot_title='')
    group_map = ahf.Analysis_Group(ds_all_sources_pmin, pfs_AOA, pp_map, plot_title='')
    # Make the figure
    ahf.make_figure([group_map_full_Arctic, group_map], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_1.pickle')
## Maps comparing available profiles to selected profiles
if False:
    print('')
    print('- Creating maps comparing available vs. selected profiles')
    # Make the subplot groups
    group_all      = ahf.Analysis_Group(ds_all_sources_all, pfs_0, pp_map, plot_title='Profiles available')
    group_selected = ahf.Analysis_Group(ds_all_sources_pmin, pfs_AOA, pp_map, plot_title='Profiles selected')
    # Make the figure
    ahf.make_figure([group_all, group_selected], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_1.pickle')
    # Find the maximum distance between any two profiles for each data set in the group
    # ahf.find_max_distance([group_all])
    # ahf.find_max_distance([group_selected])

## Subsampling ITP data
# Histograms of the first differences for data sets
if False:
    print('')
    print('- Creating histograms of first differences in pressures')
    # Make the Plot Parameters
    pp_first_diff_press = ahf.Plot_Parameters(x_vars=['press'], y_vars=['hist'], clr_map='clr_all_same', first_dfs=[True,False])
    # Make the subplot groups
    # group_AIDJEX_press_hist = ahf.Analysis_Group(ds_AIDJEX, pfs_fltrd, pp_first_diff_press, plot_title=r'AIDJEX')
    group_AIDJEX_press_hist = ahf.Analysis_Group(ds_AIDJEX, pfs_BGOS, pp_first_diff_press, plot_title=r'AIDJEX')
    # # Make the figure
    ahf.make_figure([group_AIDJEX_press_hist])#, use_same_x_axis=False, use_same_y_axis=False)

## TS diagrams
# ITP TS diagrams
if False:
    print('')
    print('- Creating TS plots to compare full profiles vs. filtered to p<pCT_max')
    # Make the Plot Parameters
    pp_TS = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='clr_all_same')
    # Make the subplot groups
    group_BGOS_full_pfs = ahf.Analysis_Group(ds_all_BGOS, pfs_AOA, pp_TS, plot_title=r'BGOS ITPs full')
    # group_BGOS_filtered = ahf.Analysis_Group(ds_all_BGOS, pfs_BGOS, pp_TS, plot_title=r'BGOS ITPs SA: 34.366 - 35.5')
    group_BGOS_filtered = ahf.Analysis_Group(ds_all_BGOS, pfs_BGOS_fltrd, pp_TS, plot_title=r'BGOS ITPs filtered')
    # # Make the figure
    ahf.make_figure([group_BGOS_full_pfs, group_BGOS_filtered], use_same_x_axis=False, use_same_y_axis=False)
# TS diagrams for AIDJEX, SHEBA, and BGOS datasets
if False:
    print('')
    print('- Creating TS plots of all datasets')
    # Make the Plot Parameters
    pp_TS = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='density_hist', extra_args={'clr_min':0, 'clr_max':100, 'clr_ext':'max', 'xy_bins':250})
    # Make the subplot groups
    group_AIDJEX_TS = ahf.Analysis_Group(ds_AIDJEX, pfs_fltrd, pp_TS, plot_title=r'AIDJEX')
    # group_SHEBA_TS  = ahf.Analysis_Group(ds_SHEBA, pfs_fltrd, pp_TS, plot_title=r'SHEBA')
    group_BGOS_TS   = ahf.Analysis_Group(ds_BGOS, pfs_fltrd, pp_TS, plot_title=r'BGOS ITPs')
    # # Make the figure
    # ahf.make_figure([group_AIDJEX_TS, group_SHEBA_TS, group_BGOS_TS])#, use_same_x_axis=False, use_same_y_axis=False)
    ahf.make_figure([group_AIDJEX_TS, group_BGOS_TS])#, use_same_x_axis=False, use_same_y_axis=False)

## Clustering parameter sweeps
## Parameter sweep for BGOS ITP data
if True:
    print('')
    print('- Creating clustering parameter sweep for BGOS ITP data')
    test_mpts = 670
    # Make the Plot Parameters
    pp_mpts_param_sweep = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'cl_ps_tuple':[300,721,20]}) #[50,711,20]
    pp_ell_param_sweep  = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'cl_ps_tuple':[10,271,10]}) 
    # Make the subplot groups
    group_mpts_param_sweep = ahf.Analysis_Group(ds_BGOS, pfs_fltrd, pp_mpts_param_sweep)
    group_ell_param_sweep  = ahf.Analysis_Group(ds_BGOS, pfs_fltrd, pp_ell_param_sweep)
    # # Make the figure
    ahf.make_figure([group_mpts_param_sweep, group_ell_param_sweep], filename='test_param_sweep_BGOS.pickle')
## Parameter sweep for AIDJEX data
if False:
    print('')
    print('- Creating clustering parameter sweep for AIDJEX data')
    test_mpts = 250
    # Make the Plot Parameters
    pp_mpts_param_sweep = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'cl_ps_tuple':[10,721,10]}) #[50,711,20]
    pp_ell_param_sweep  = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'cl_ps_tuple':[10,271,10]}) 
    # Make the subplot groups
    group_mpts_param_sweep = ahf.Analysis_Group(ds_AIDJEX, pfs_fltrd, pp_mpts_param_sweep)
    group_ell_param_sweep  = ahf.Analysis_Group(ds_AIDJEX, pfs_fltrd, pp_ell_param_sweep)
    # # Make the figure
    ahf.make_figure([group_mpts_param_sweep, group_ell_param_sweep], filename='test_param_sweep_AIDJEX.pickle')

## Clustering Results
# AIDJEX clustering
if False:
    print('')
    print('- Creating clustering plot of AIDJEX data')
    test_mpts = 250
    # Make the Plot Parameters
    pp_live_clstr = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'b_a_w_plt':True})
    pp_live_clstr2 = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':670, 'b_a_w_plt':True})
    # pp_pre_clstrd = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'b_a_w_plt':True})
    # Make the subplot groups
    # group_AIDJEX_BB_clstrs = ahf.Analysis_Group(ds_AIDJEX_BigBear, pfs_fltrd, pp_clstrs)
    group_AIDJEX_live_clstr = ahf.Analysis_Group(ds_AIDJEX, pfs_fltrd, pp_live_clstr, plot_title=r'AIDJEX m_pts = 250')
    group_AIDJEX_live_clstr2 = ahf.Analysis_Group(ds_AIDJEX, pfs_fltrd, pp_live_clstr2, plot_title=r'AIDJEX m_pts = 670')
    # group_AIDJEX_pre_clstrd = ahf.Analysis_Group(ds_AIDJEX_clstrd, pfs_fltrd, pp_pre_clstrd, plot_title=r'AIDJEX clusters from file')
    # # Make the figure
    # ahf.make_figure([group_AIDJEX_BB_clstrs, group_AIDJEX_clstrs])
    ahf.make_figure([group_AIDJEX_live_clstr, group_AIDJEX_live_clstr2])
# BGOS clustering
if False:
    print('')
    print('- Creating clustering plot of BGOS data')
    test_mpts = 250
    # Make the Plot Parameters
    pp_live_clstr = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'b_a_w_plt':True})
    pp_live_clstr2 = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':670, 'b_a_w_plt':True})
    # pp_pre_clstrd = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'b_a_w_plt':True})
    # Make the subplot groups
    group_BGOS_live_clstr = ahf.Analysis_Group(ds_BGOS, pfs_fltrd, pp_live_clstr, plot_title=r'BGOS m_pts = 250')
    group_BGOS_live_clstr2 = ahf.Analysis_Group(ds_BGOS, pfs_fltrd, pp_live_clstr2, plot_title=r'BGOS m_pts = 670')
    # group_BGOS_pre_clstrd = ahf.Analysis_Group(ds_BGOS_clstrd, pfs_fltrd, pp_pre_clstrd, plot_title=r'BGOS clusters from file')
    # # Make the figure
    ahf.make_figure([group_BGOS_live_clstr, group_BGOS_live_clstr2])


exit(0)
### Figure 2
## Plotting a TS diagram of both ITP and AIDJEX data
# TS diagram of all sources, colored by source
pp_TS_clr_all_same = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='clr_all_same')
pp_TS_by_source = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='clr_by_source')
pp_TS_by_instrmt = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='clr_by_instrmt')

pp_LA_TS = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='clr_all_same')

pp_CT_max = ahf.Plot_Parameters(x_vars=['CT_max'], y_vars=['press_CT_max'], clr_map='prof_no')
pp_SA_CT_max = ahf.Plot_Parameters(x_vars=['SA_CT_max'], y_vars=['press_CT_max'], clr_map='CT_max')
pp_SA_CT_max_hist = ahf.Plot_Parameters(x_vars=['SA_CT_max'], y_vars=['hist'], clr_map='clr_all_same')

test_mpts = 65
pp_ITP2_clstr = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'b_a_w_plt':True})
pp_ITP2_clstr_og = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'b_a_w_plt':True})

BGOS_mpts = 120
pp_BGOS_clstrd = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':BGOS_mpts, 'b_a_w_plt':True})
pp_BGOS_clstr_TS = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':BGOS_mpts, 'b_a_w_plt':True})

### Figure 3
## Parameter sweep across \ell and m_pts for ITP2 subsampled
ITP2_ss_mpts = 65
pp_ITP2_ss_mpts = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':ITP2_ss_mpts, 'cl_ps_tuple':[10,220,5]})
pp_ITP2_ss_ell  = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':ITP2_ss_mpts, 'cl_ps_tuple':[10,271,10]})
## Parameter sweep across \ell and m_pts for BGOS
BGOS_mpts = 170
pp_BGOS_mpts = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':BGOS_mpts, 'cl_ps_tuple':[10,440,10]})
pp_BGOS_ell  = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':BGOS_mpts, 'cl_ps_tuple':[10,271,10]})
## Parameter sweep across \ell and m_pts for AIDJEX
AIDJEX_mpts = 500
pp_AIDJEX_mpts = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':AIDJEX_mpts, 'cl_ps_tuple':[300,700,20]})
pp_AIDJEX_ell  = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':AIDJEX_mpts, 'cl_ps_tuple':[10,210,10]})

## Parameter sweep across \ell and m_pts for ITP3, Lu et al. 2022
pp_ITP3_ps_m_pts = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':Lu2022_m_pts, 'cl_ps_tuple':[800,1001,10]}, legend=False, add_grid=False)
pp_ITP3_ps_ell  = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':Lu2022_m_pts, 'cl_ps_tuple':[10,210,10]}, legend=False, add_grid=False)

### Figure 4
## Evaluating clusterings with the lateral density ratio and the normalized inter-cluster range
pp_salt_R_L = ahf.Plot_Parameters(x_vars=['cRL'], y_vars=['ca_press'], clr_map='cluster', extra_args={'b_a_w_plt':False, 'plt_noise':False, 'plot_slopes':True}, legend=False)
pp_salt_nir = ahf.Plot_Parameters(x_vars=['nir_SA'], y_vars=['ca_press'], clr_map='cluster', extra_args={'b_a_w_plt':False, 'plt_noise':False}, legend=False)

### Figure 5
## Tracking clusters across a subset of profiles
# For ITP2
pp_ITP2_some_pfs_0  = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['press'], plot_type='profiles', clr_map='cluster', extra_args={'pfs_to_plot':ITP2_some_pfs_0, 'plt_noise':True}, legend=False, ax_lims=ITP2_some_pfs_ax_lims_0)
pp_ITP2_some_pfs_1  = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['press'], plot_type='profiles', clr_map='cluster', extra_args={'pfs_to_plot':ITP2_some_pfs_1, 'plt_noise':True}, legend=False, ax_lims=ITP2_some_pfs_ax_lims_1)

### Figure 6
## Tracking clusters across profiles, reproducing Lu et al. 2022 Figure 3
pp_Lu2022_fig3a = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['pca_press'], clr_map='cluster', extra_args={'b_a_w_plt':False}, legend=False)
pp_Lu2022_fig3b = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['pca_CT'], clr_map='cluster', extra_args={'b_a_w_plt':False}, legend=False)
pp_Lu2022_fig3c = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['pca_SA'], clr_map='cluster', extra_args={'b_a_w_plt':False}, legend=False)

### Figure 8
## Tracking clusters across a subset of profiles
# For ITP35
pp_ITP35_some_pfs = ahf.Plot_Parameters(x_vars=['CT'], y_vars=['press'], plot_type='profiles')

################################################################################
# Create analysis group objects
print('- Creating analysis group objects')
################################################################################

## Test Analysis Groups
# my_group0 = ahf.Analysis_Group(ds_ITP2_all, pfs_1, pp_clstr_test)
# my_group1 = ahf.Analysis_Group(ds_AIDJEX_BigBear_all, pfs_1, pp_clstr_test)
# ahf.make_figure([my_group0, my_group1])#, filename='test.pickle')
## Test figures

################################################################################
### Figures for paper

### Figure 1
## Map of drifts for all sources
if False:
    print('')
    print('- Creating Figure 1')
    # Make the subplot groups pfs_AOA
    group_map_full_Arctic = ahf.Analysis_Group(ds_all_sources, pfs_AOA, pp_map_full_Arctic, plot_title='')
    group_map = ahf.Analysis_Group(ds_all_sources, pfs_AOA, pp_map, plot_title='')
    # Make the figure
    ahf.make_figure([group_map_full_Arctic, group_map], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_1.pickle')
    # Find the maximum distance between any two profiles for each data set in the group
    # ahf.find_max_distance([group_map])
## Map of drifts for all AIDJEX stations
if False:
    print('')
    print('- Creating Figure 1')
    # Make the subplot groups
    group_map = ahf.Analysis_Group(ds_all_AIDJEX, pfs_0, pp_map, plot_title='')
    group_map_full_Arctic = ahf.Analysis_Group(ds_all_AIDJEX, pfs_0, pp_map_full_Arctic, plot_title='')
    # Make the figure
    ahf.make_figure([group_map], use_same_x_axis=False, use_same_y_axis=False, filename='Figure_1.pickle')
    # Find the maximum distance between any two profiles for each data set in the group
    # ahf.find_max_distance([group_map])
## Map of drifts for selected ITPs
if False:
    print('')
    print('- Creating Figure 1')
    # Make the subplot groups
    group_map_full_Arctic = ahf.Analysis_Group(ds_all_BGOS, pfs_AOA, pp_map_full_Arctic, plot_title='')
    group_map = ahf.Analysis_Group(ds_all_BGOS, pfs_AOA, pp_map, plot_title='')
    # Make the figure
    ahf.make_figure([group_map_full_Arctic, group_map], use_same_x_axis=False, use_same_y_axis=False)
## Map of drifts colored by date
if False:
    print('')
    print('- Creating Figure 1')
    # Make the subplot groups
    group_AIDJEX_map = ahf.Analysis_Group(ds_all_AIDJEX, pfs_0, pp_map_by_date, plot_title='')
    group_ITPs_map = ahf.Analysis_Group(ds_all_BGOS, pfs_AOA, pp_map_by_date, plot_title='')
    # Make the figure
    ahf.make_figure([group_AIDJEX_map, group_ITPs_map], use_same_x_axis=False, use_same_y_axis=False)

### Figure 2
## Plotting a TS diagram of both ITP and AIDJEX data
if False:
    print('')
    print('- Creating Figure 2')
    # Make the subplot groups
    # group_TS_all_sources = ahf.Analysis_Group(ds_all_sources, pfs_AIDJEX, pp_TS_by_source, plot_title='')
    group_TS_all_BGOS = ahf.Analysis_Group(ds_all_BGOS, pfs_0, pp_TS_by_instrmt, plot_title='BGOS ITPs')
    group_TS_all_AIDJEX  = ahf.Analysis_Group(ds_all_AIDJEX, pfs_0, pp_TS_by_instrmt, plot_title='AIDJEX')
    # Make the figure
    ahf.make_figure([group_TS_all_AIDJEX, group_TS_all_BGOS])#, filename='Figure_2.png')
    # ahf.make_figure([group_TS_all_sources, group_TS_all_AIDJEX], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_2.png')
## Plotting a LA_TS diagram of both ITP and AIDJEX data
if False:
    print('')
    print('- Creating Figure 2')
    # Make the subplot groups
    group_TS_all_AIDJEX = ahf.Analysis_Group(ds_all_AIDJEX, pfs_AIDJEX, pp_LA_TS, plot_title='AIDJEX')
    # group_TS_all_BGOS = ahf.Analysis_Group(ds_all_BGOS, pfs_BGOS, pp_TS_by_instrmt, plot_title='')
    # # Make the figure
    ahf.make_figure([group_TS_all_AIDJEX])#, filename='Figure_2.png')
    # ahf.make_figure([group_TS_all_sources, group_TS_all_AIDJEX], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_2.png')
    # group_TS_all_BGOS = ahf.Analysis_Group(ds_all_BGOS, pfs_BGOS, pp_LA_TS, plot_title='BGOS')
    # Make the figure
    # ahf.make_figure([group_TS_all_BGOS])
## Plotting clusters of both ITP and AIDJEX data
if False:
    print('')
    print('- Creating Figure 2')
    # Make the subplot groups
    # group_TS_all_AIDJEX = ahf.Analysis_Group(ds_all_AIDJEX, pfs_ITP2, pp_TS_by_instrmt, plot_title='')
    # group_TS_all_BGOS = ahf.Analysis_Group(ds_all_BGOS, pfs_BGOS, pp_TS_by_instrmt, plot_title='')
    # # Make the figure
    # ahf.make_figure([group_TS_all_AIDJEX, group_TS_all_BGOS])#, filename='Figure_2.png')
    # ahf.make_figure([group_TS_all_sources, group_TS_all_AIDJEX], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_2.png')
    group_BGOS_clstrd = ahf.Analysis_Group(ds_all_BGOS, pfs_BGOS, pp_BGOS_clstrd, plot_title='')
    group_BGOS_clstr_TS = ahf.Analysis_Group(ds_all_BGOS, pfs_BGOS, pp_BGOS_clstr_TS, plot_title='')
    # Make the figure
    ahf.make_figure([group_BGOS_clstrd, group_BGOS_clstr_TS])
## Plotting CT_max data with min_press_CT_max limit
if False:
    print('')
    print('- Creating Figure 2')
    # # Make the subplot groups
    group_ITP35_CT_max = ahf.Analysis_Group(ds_ITP35_dfs1, pfs_AOA, pp_SA_CT_max)
    group_ITP41_CT_max = ahf.Analysis_Group(ds_ITP41_dfs1, pfs_AOA, pp_SA_CT_max)
    group_ITP42_CT_max = ahf.Analysis_Group(ds_ITP42_dfs1, pfs_AOA, pp_SA_CT_max)
    group_ITP43_CT_max = ahf.Analysis_Group(ds_ITP43_dfs1, pfs_AOA, pp_SA_CT_max)
    # # Make the figure
    ahf.make_figure([group_ITP35_CT_max, group_ITP41_CT_max, group_ITP42_CT_max, group_ITP43_CT_max])
    # Make the subplot groups
    # group_BB_CT_max = ahf.Analysis_Group(ds_AIDJEX_BigBear_all, pfs_0, pp_SA_CT_max)
    # group_BF_CT_max = ahf.Analysis_Group(ds_AIDJEX_BlueFox_all, pfs_0, pp_SA_CT_max)
    # group_Ca_CT_max = ahf.Analysis_Group(ds_AIDJEX_Caribou_all, pfs_0, pp_SA_CT_max)
    # group_Sn_CT_max = ahf.Analysis_Group(ds_AIDJEX_Snowbird_all, pfs_0, pp_SA_CT_max)
    # Make the figure
    # ahf.make_figure([group_BB_CT_max, group_BF_CT_max, group_Ca_CT_max, group_Sn_CT_max])

### Figure 3
## Parameter sweep across \ell and m_pts
# For ITP2 ss
if False:
    print('')
    print('- Creating Figure 3')
    # Make the subplot groups
    group_ITP2_ss_mpts = ahf.Analysis_Group(ds_ITP2_all, pfs_subs, pp_ITP2_ss_mpts, plot_title='ITP2 Subsampled')
    group_ITP2_ss_ell  = ahf.Analysis_Group(ds_ITP2_all, pfs_subs, pp_ITP2_ss_ell, plot_title='ITP2 Subsampled')
    # Make the figure
    ahf.make_figure([group_ITP2_ss_mpts, group_ITP2_ss_ell], filename='Figure_3.pickle')
# For BGOS
if False:
    print('')
    print('- Creating Figure 3')
    # Make the subplot groups
    group_BGOS_mpts = ahf.Analysis_Group(ds_all_BGOS, pfs_BGOS, pp_BGOS_mpts, plot_title='BGOS')
    group_BGOS_ell  = ahf.Analysis_Group(ds_all_BGOS, pfs_BGOS, pp_BGOS_ell, plot_title='BGOS')
    # Make the figure
    ahf.make_figure([group_BGOS_mpts, group_BGOS_ell], filename='Figure_3.pickle')
# For AIDJEX
if False:
    print('')
    print('- Creating Figure 3')
    # Make the subplot groups
    group_AIDJEX_mpts = ahf.Analysis_Group(ds_all_AIDJEX, pfs_ITP2, pp_AIDJEX_mpts, plot_title='AIDJEX')
    group_AIDJEX_ell  = ahf.Analysis_Group(ds_all_AIDJEX, pfs_ITP2, pp_AIDJEX_ell, plot_title='AIDJEX')
    # Make the figure
    ahf.make_figure([group_AIDJEX_mpts, group_AIDJEX_ell], filename='Figure_3.pickle')

### Figure 3-1
## Plotting test clusterings
if False:
    print('')
    print('- Creating Figure 3')
    # Make the subplot groups
    # group_ITP2_og = ahf.Analysis_Group(ds_ITP2_all, pfs_ITP2, pp_ITP2_clstr, plot_title='Original')
    group_ITP2_ss = ahf.Analysis_Group(ds_ITP2_all, pfs_subs, pp_ITP2_clstr, plot_title='Subsampled')
    group_ITP2_ss2 = ahf.Analysis_Group(ds_ITP2_all, pfs_subs, pp_ITP2_clstr_og, plot_title='Subsampled')
    # group_ITP2_rg = ahf.Analysis_Group(ds_ITP2_all, pfs_regrid, pp_ITP2_clstr, plot_title='Regridded')
    # group_ITP2_ss_rg = ahf.Analysis_Group(ds_ITP2_all, pfs_ss_rg, pp_ITP2_clstr, plot_title='Subsampled & Regridded')
    # Make the figure
    # ahf.make_figure([group_ITP2_og, group_ITP2_ss, group_ITP2_rg])#, group_ITP2_ss_rg])#, filename='Figure_3.png')
    ahf.make_figure([group_ITP2_ss, group_ITP2_ss2])#, group_ITP2_ss_rg])#, filename='Figure_3.png')

### Figure 4
## Evaluating clusterings with the overlap ratio and lateral density ratio
# For the reproduction of Timmermans et al. 2008
if False:
    print('')
    print('- Creating Figure 4')
    # Make the subplot groups
    group_salt_R_L = ahf.Analysis_Group(ds_ITP2_all, pfs_ITP2, pp_salt_R_L, plot_title='')
    group_salt_nir = ahf.Analysis_Group(ds_ITP2_all, pfs_ITP2, pp_salt_nir, plot_title='')
    # Make the figure
    ahf.make_figure([group_salt_R_L, group_salt_nir], filename='Figure_4.pickle')
# For the reproduction of Lu et al. 2022
if False:
    print('')
    print('- Creating Figure 4, for ITP3')
    group_salt_R_L = ahf.Analysis_Group(ds_ITP3_all, pfs_ITP3, pp_salt_R_L)
    group_salt_nir = ahf.Analysis_Group(ds_ITP3_all, pfs_ITP3, pp_salt_nir)
    # Make the figure
    #   Confirmed
    ahf.make_figure([group_salt_R_L, group_salt_nir])

### Figure 5
## Tracking clusters across a subset of profiles
# For ITP2
if False:
    print('')
    print('- Creating Figure 5')
    # Make the subplot groups
    group_ITP2_some_pfs_0 = ahf.Analysis_Group(ds_ITP2_all, pfs_ITP2, pp_ITP2_some_pfs_0, plot_title='')
    group_ITP2_some_pfs_1 = ahf.Analysis_Group(ds_ITP2_all, pfs_ITP2, pp_ITP2_some_pfs_1, plot_title='')
    # Make the figure
    ahf.make_figure([group_ITP2_some_pfs_0, group_ITP2_some_pfs_1], use_same_x_axis=False, use_same_y_axis=False, row_col_list=[2,1, 0.3, 1.70], filename='Figure_5.pickle')

### Figure 6
## Tracking clusters across profiles, reproducing Lu et al. 2022 Figure 3
if False:
    print('')
    print('- Creating Figure 6')
    # Make the subplot groups
    group_Lu2022_fig3a = ahf.Analysis_Group(ds_ITP3_all, pfs_ITP3, pp_Lu2022_fig3a, plot_title='')
    group_Lu2022_fig3b = ahf.Analysis_Group(ds_ITP3_all, pfs_ITP3, pp_Lu2022_fig3b, plot_title='')
    group_Lu2022_fig3c = ahf.Analysis_Group(ds_ITP3_all, pfs_ITP3, pp_Lu2022_fig3c, plot_title='')
    # Make the figure
    #   Confirmed that it will pull from netcdf. Takes a long time to run still
    ahf.make_figure([group_Lu2022_fig3a, group_Lu2022_fig3b, group_Lu2022_fig3c], filename='Figure_6.pickle')
# For ITP2
if False:
    print('')
    print('- Creating Figure 6, for ITP2')
    # Make the subplot groups
    group_Lu2022_fig3a = ahf.Analysis_Group(ds_ITP2_all, pfs_ITP2, pp_Lu2022_fig3a)
    group_Lu2022_fig3b = ahf.Analysis_Group(ds_ITP2_all, pfs_ITP2, pp_Lu2022_fig3b, plot_title='')
    group_Lu2022_fig3c = ahf.Analysis_Group(ds_ITP2_all, pfs_ITP2, pp_Lu2022_fig3c, plot_title='')
    # Make the figure
    ahf.make_figure([group_Lu2022_fig3a, group_Lu2022_fig3b, group_Lu2022_fig3c])

### Figure 8
## Tracking clusters across a subset of profiles
# For ITP3
if False:
    print('')
    print('- Creating Figure 8')
    # Make the subplot group
    group_BGOS_some_pfs0 = ahf.Analysis_Group(ds_ITP35_some_pfs0, pfs_AOA, pp_ITP35_some_pfs)
    group_BGOS_some_pfs1 = ahf.Analysis_Group(ds_ITP35_some_pfs1, pfs_AOA, pp_ITP35_some_pfs, plot_title='')
    group_BGOS_some_pfs2 = ahf.Analysis_Group(ds_ITP35_some_pfs2, pfs_AOA, pp_ITP35_some_pfs, plot_title='')
    # Make the figure
    ahf.make_figure([group_BGOS_some_pfs0, group_BGOS_some_pfs1, group_BGOS_some_pfs2], use_same_x_axis=False, use_same_y_axis=False, row_col_list=[3,1, 0.3, 1.70])
    # ahf.make_figure([group_ITP3_some_pfs_2], row_col_list=[1,1, 0.27, 0.90], filename='Figure_8.pickle')

################################################################################