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

JOIS_S_range = [34.1, 34.76]

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

# For showing how well the clustering does near the AW core
ITP35_some_pfs = [1,9,17,25] #[313, 315, 317, 319, 321]
ITP3_some_pfs_ax_lims_2 = {'y_lims':[370,300]}

### Filters for reproducing plots from Lu et al. 2022
Lu2022_p_range = [200,355]
Lu2022_T_range = [-1.0,0.9]
Lu2022_S_range = [34.21,34.82]
Lu2022_m_pts = 580

################################################################################
# Make dictionaries for what data to load in and analyze
################################################################################

# All profiles from all sources in this study
all_sources = {'AIDJEX_BigBear':'all','AIDJEX_BlueFox':'all','AIDJEX_Caribou':'all','AIDJEX_Snowbird':'all','ITP_2':'all','ITP_3':'all'}
all_sources = {'AIDJEX_BigBear':'all','AIDJEX_BlueFox':'all','AIDJEX_Caribou':'all','AIDJEX_Snowbird':'all','ITP_35':'all','ITP_41':'all','ITP_42':'all','ITP_43':'all'}

# All profiles from all ITPs in this study
all_ITPs = {'ITP_2':'all','ITP_3':'all'}
all_JOIS = {'ITP_35':'all','ITP_41':'all','ITP_42':'all','ITP_43':'all'}

# All profiles from all AIDJEX camps
all_AIDJEX = {'AIDJEX_BigBear':'all','AIDJEX_BlueFox':'all','AIDJEX_Caribou':'all','AIDJEX_Snowbird':'all'}

# All profiles from certain ITPs
ITP2_all  = {'ITP_2':'all'}
ITP3_all  = {'ITP_3':'all'}
ITP35_all = {'ITP_35':'all'}

# A list of many profiles to plot from ITP3
start_pf = 1331
import numpy as np
n_pfs_to_plot = 5
ITP3_some_pfs_1 = list(np.arange(start_pf, start_pf+(n_pfs_to_plot*2), 2))
ITP3_pfs1  = {'ITP_3':ITP3_some_pfs_1}
ITP35_pfs  = {'ITP_35':ITP35_some_pfs}

# AIDJEX camps
AIDJEX_BigBear_all = {'AIDJEX_BigBear':'all'}
AIDJEX_BlueFox_all = {'AIDJEX_BlueFox':'all'}
AIDJEX_Caribou_all = {'AIDJEX_Caribou':'all'}
AIDJEX_Snowbird_all = {'AIDJEX_Snowbird':'all'}

################################################################################
# Create data filtering objects
print('- Creating data filtering objects')
################################################################################

dfs0 = ahf.Data_Filters()

################################################################################
# Create data sets by combining filters and the data to load in
print('- Creating data sets')
################################################################################

# ds_all_sources = ahf.Data_Set(all_sources, dfs0)
# ds_all_ITPs = ahf.Data_Set(all_ITPs, dfs0)
# ds_all_JOIS = ahf.Data_Set(all_JOIS, dfs0)
# ds_all_AIDJEX = ahf.Data_Set(all_AIDJEX, dfs0)
# 
# ds_ITP2_all = ahf.Data_Set(ITP2_all, dfs0)
# 
# ds_AIDJEX_BigBear_all = ahf.Data_Set(AIDJEX_BigBear_all, dfs0)

ds_ITP35_all = ahf.Data_Set(ITP35_all, dfs0)
ds_ITP35_some_pfs = ahf.Data_Set(ITP35_pfs, dfs0)

################################################################################
# Create profile filtering objects
print('- Creating profile filtering objects')
################################################################################

pfs_0 = ahf.Profile_Filters()
pfs_1 = ahf.Profile_Filters(SA_range=ITP2_S_range)

# AIDJEX Operation Area
pfs_AOA = ahf.Profile_Filters(lon_range=[-152.8,-133.7],lat_range=[72.6,77.3])

pfs_ITP2  = ahf.Profile_Filters(SA_range=ITP2_S_range)
pfs_ITP3 = ahf.Profile_Filters(SA_range=Lu2022_S_range)
pfs_JOIS  = ahf.Profile_Filters(lon_range=[-152.8,-133.7],lat_range=[72.6,77.3],SA_range=JOIS_S_range)

pfs_subs = ahf.Profile_Filters(SA_range=ITP2_S_range, subsample=True)
pfs_regrid = ahf.Profile_Filters(SA_range=ITP2_S_range, regrid_TS=['CT',0.01,'SP',0.005])
pfs_ss_rg = ahf.Profile_Filters(SA_range=ITP2_S_range, subsample=True, regrid_TS=['CT',0.01,'SP',0.005])

################################################################################
# Create plotting parameter objects

print('- Creating plotting parameter objects')
################################################################################

### Test plots
## Regrid
# Make the Profile Filters
pfs_regrid = ahf.Profile_Filters(SA_range=ITP2_S_range, regrid_TS=['CT',0.01,'SP',0.005])
# Make the Plot Parameters
pp_regrid = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['CT'], clr_map='clr_all_same', legend=True)
# Make the Analysis Group
# group_regrid = ahf.Analysis_Group(ds_ITP2_all, pfs_regrid, pp_regrid)
# Make the figure
# ahf.make_figure([group_regrid])

pp_clstr_test = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['CT'], clr_map='cluster', extra_args={'b_a_w_plt':True}, legend=True, add_grid=True)

################################################################################
### Figures for paper

### Figure 1
## Map of ITP drifts for all sources
pp_map = ahf.Plot_Parameters(plot_type='map', clr_map='clr_by_instrmt', extra_args={'map_extent':'AIDJEX_focus'})
pp_map_full_Arctic = ahf.Plot_Parameters(plot_type='map', clr_map='clr_by_source', extra_args={'map_extent':'Full_Arctic'}, legend=False, add_grid=False)
pp_map_by_date = ahf.Plot_Parameters(plot_type='map', clr_map='dt_start', extra_args={'map_extent':'AIDJEX_focus'})

### Figure 2
## Plotting a TS diagram of both ITP and AIDJEX data
# TS diagram of all sources, colored by source
pp_TS_clr_all_same = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['CT'], clr_map='clr_all_same')
pp_TS_by_source = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['CT'], clr_map='clr_by_source')
pp_TS_by_instrmt = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['CT'], clr_map='clr_by_instrmt')

pp_LA_TS = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['la_CT'], clr_map='press')

pp_CT_max = ahf.Plot_Parameters(x_vars=['CT_max'], y_vars=['press_CT_max'], clr_map='prof_no')
pp_SA_CT_max = ahf.Plot_Parameters(x_vars=['SA_CT_max'], y_vars=['press_CT_max'], clr_map='prof_no')

test_mpts = 65
pp_ITP2_clstr = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['CT'], clr_map='cluster', extra_args={'cl_x_var':'SP', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'b_a_w_plt':True})
pp_ITP2_clstr_og = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SP', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'b_a_w_plt':True})

JOIS_mpts = 120
pp_JOIS_clstrd = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['CT'], clr_map='cluster', extra_args={'cl_x_var':'SP', 'cl_y_var':'la_CT', 'm_pts':JOIS_mpts, 'b_a_w_plt':True})
pp_JOIS_clstr_TS = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SP', 'cl_y_var':'la_CT', 'm_pts':JOIS_mpts, 'b_a_w_plt':True})

### Figure 3
## Parameter sweep across \ell and m_pts for ITP2 subsampled
ITP2_ss_mpts = 65
pp_ITP2_ss_mpts = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SP', 'cl_y_var':'la_CT', 'm_pts':ITP2_ss_mpts, 'cl_ps_tuple':[10,220,5]})
pp_ITP2_ss_ell  = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SP', 'cl_y_var':'la_CT', 'm_pts':ITP2_ss_mpts, 'cl_ps_tuple':[10,271,10]})
## Parameter sweep across \ell and m_pts for JOIS
JOIS_mpts = 170
pp_JOIS_mpts = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SP', 'cl_y_var':'la_CT', 'm_pts':JOIS_mpts, 'cl_ps_tuple':[10,440,10]})
pp_JOIS_ell  = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SP', 'cl_y_var':'la_CT', 'm_pts':JOIS_mpts, 'cl_ps_tuple':[10,271,10]})
## Parameter sweep across \ell and m_pts for AIDJEX
AIDJEX_mpts = 500
pp_AIDJEX_mpts = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SP', 'cl_y_var':'la_CT', 'm_pts':AIDJEX_mpts, 'cl_ps_tuple':[300,700,20]})
pp_AIDJEX_ell  = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SP', 'cl_y_var':'la_CT', 'm_pts':AIDJEX_mpts, 'cl_ps_tuple':[10,210,10]})

## Parameter sweep across \ell and m_pts for ITP3, Lu et al. 2022
pp_ITP3_ps_m_pts = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SP', 'cl_y_var':'la_CT', 'm_pts':Lu2022_m_pts, 'cl_ps_tuple':[800,1001,10]}, legend=False, add_grid=False)
pp_ITP3_ps_ell  = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SP', 'cl_y_var':'la_CT', 'm_pts':Lu2022_m_pts, 'cl_ps_tuple':[10,210,10]}, legend=False, add_grid=False)

### Figure 4
## Evaluating clusterings with the lateral density ratio and the normalized inter-cluster range
pp_salt_R_L = ahf.Plot_Parameters(x_vars=['cRL'], y_vars=['ca_press'], clr_map='cluster', extra_args={'b_a_w_plt':False, 'plt_noise':False, 'plot_slopes':True}, legend=False)
pp_salt_nir = ahf.Plot_Parameters(x_vars=['nir_SP'], y_vars=['ca_press'], clr_map='cluster', extra_args={'b_a_w_plt':False, 'plt_noise':False}, legend=False)

### Figure 5
## Tracking clusters across a subset of profiles
# For ITP2
pp_ITP2_some_pfs_0  = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['press'], plot_type='profiles', clr_map='cluster', extra_args={'pfs_to_plot':ITP2_some_pfs_0, 'plt_noise':True}, legend=False, ax_lims=ITP2_some_pfs_ax_lims_0)
pp_ITP2_some_pfs_1  = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['press'], plot_type='profiles', clr_map='cluster', extra_args={'pfs_to_plot':ITP2_some_pfs_1, 'plt_noise':True}, legend=False, ax_lims=ITP2_some_pfs_ax_lims_1)

### Figure 6
## Tracking clusters across profiles, reproducing Lu et al. 2022 Figure 3
pp_Lu2022_fig3a = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['pca_press'], clr_map='cluster', extra_args={'b_a_w_plt':False}, legend=False)
pp_Lu2022_fig3b = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['pca_CT'], clr_map='cluster', extra_args={'b_a_w_plt':False}, legend=False)
pp_Lu2022_fig3c = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['pca_SP'], clr_map='cluster', extra_args={'b_a_w_plt':False}, legend=False)

### Figure 8
## Tracking clusters across a subset of profiles
# For ITP35
pp_ITP35_some_pfs = ahf.Plot_Parameters(x_vars=['SA','CT'], y_vars=['press'], plot_type='profiles')

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
    # Make the subplot groups
    group_map = ahf.Analysis_Group(ds_all_sources, pfs_0, pp_map, plot_title='')
    group_map_full_Arctic = ahf.Analysis_Group(ds_all_sources, pfs_0, pp_map_full_Arctic, plot_title='')
    # Make the figure
    ahf.make_figure([group_map_full_Arctic, group_map], use_same_x_axis=False, use_same_y_axis=False, filename='Figure_1.pickle')
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
    group_map = ahf.Analysis_Group(ds_all_JOIS, pfs_AOA, pp_map, plot_title='')
    group_map_full_Arctic = ahf.Analysis_Group(ds_all_sources, pfs_0, pp_map_full_Arctic, plot_title='')
    # Make the figure
    ahf.make_figure([group_map, group_map_full_Arctic], use_same_x_axis=False, use_same_y_axis=False)
## Map of drifts colored by date
if False:
    print('')
    print('- Creating Figure 1')
    # Make the subplot groups
    group_AIDJEX_map = ahf.Analysis_Group(ds_all_AIDJEX, pfs_0, pp_map_by_date, plot_title='')
    group_ITPs_map = ahf.Analysis_Group(ds_all_JOIS, pfs_AOA, pp_map_by_date, plot_title='')
    # Make the figure
    ahf.make_figure([group_AIDJEX_map, group_ITPs_map], use_same_x_axis=False, use_same_y_axis=False)

### Figure 2
## Plotting a TS diagram of both ITP and AIDJEX data
if False:
    print('')
    print('- Creating Figure 2')
    # Make the subplot groups
    group_TS_all_sources = ahf.Analysis_Group(ds_all_sources, pfs_ITP2, pp_TS_by_source, plot_title='')
    group_TS_all_AIDJEX  = ahf.Analysis_Group(ds_all_AIDJEX, pfs_ITP2, pp_TS_by_instrmt, plot_title='')
    # Make the figure
    ahf.make_figure([group_TS_all_sources, group_TS_all_AIDJEX])#, filename='Figure_2.png')
    # ahf.make_figure([group_TS_all_sources, group_TS_all_AIDJEX], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_2.png')
## Plotting a TS diagram of both ITP and AIDJEX data
if False:
    print('')
    print('- Creating Figure 2')
    # Make the subplot groups
    # group_TS_all_AIDJEX = ahf.Analysis_Group(ds_all_AIDJEX, pfs_ITP2, pp_TS_by_instrmt, plot_title='')
    # group_TS_all_JOIS = ahf.Analysis_Group(ds_all_JOIS, pfs_JOIS, pp_TS_by_instrmt, plot_title='')
    # # Make the figure
    # ahf.make_figure([group_TS_all_AIDJEX, group_TS_all_JOIS])#, filename='Figure_2.png')
    # ahf.make_figure([group_TS_all_sources, group_TS_all_AIDJEX], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_2.png')
    group_TS_all_JOIS = ahf.Analysis_Group(ds_all_JOIS, pfs_JOIS, pp_LA_TS, plot_title='')
    # Make the figure
    ahf.make_figure([group_TS_all_JOIS])
## Plotting clusters of both ITP and AIDJEX data
if False:
    print('')
    print('- Creating Figure 2')
    # Make the subplot groups
    # group_TS_all_AIDJEX = ahf.Analysis_Group(ds_all_AIDJEX, pfs_ITP2, pp_TS_by_instrmt, plot_title='')
    # group_TS_all_JOIS = ahf.Analysis_Group(ds_all_JOIS, pfs_JOIS, pp_TS_by_instrmt, plot_title='')
    # # Make the figure
    # ahf.make_figure([group_TS_all_AIDJEX, group_TS_all_JOIS])#, filename='Figure_2.png')
    # ahf.make_figure([group_TS_all_sources, group_TS_all_AIDJEX], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_2.png')
    group_JOIS_clstrd = ahf.Analysis_Group(ds_all_JOIS, pfs_JOIS, pp_JOIS_clstrd, plot_title='')
    group_JOIS_clstr_TS = ahf.Analysis_Group(ds_all_JOIS, pfs_JOIS, pp_JOIS_clstr_TS, plot_title='')
    # Make the figure
    ahf.make_figure([group_JOIS_clstrd, group_JOIS_clstr_TS])
## Plotting CT_max data for profiles
if True:
    print('')
    print('- Creating Figure 2')
    # Make the subplot groups
    group_CT_max_scatter = ahf.Analysis_Group(ds_ITP35_all, pfs_JOIS, pp_CT_max)
    group_SA_CT_max_scatter = ahf.Analysis_Group(ds_ITP35_all, pfs_JOIS, pp_SA_CT_max)
    # Make the figure
    ahf.make_figure([group_CT_max_scatter, group_SA_CT_max_scatter])

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
# For JOIS
if False:
    print('')
    print('- Creating Figure 3')
    # Make the subplot groups
    group_JOIS_mpts = ahf.Analysis_Group(ds_all_JOIS, pfs_JOIS, pp_JOIS_mpts, plot_title='JOIS')
    group_JOIS_ell  = ahf.Analysis_Group(ds_all_JOIS, pfs_JOIS, pp_JOIS_ell, plot_title='JOIS')
    # Make the figure
    ahf.make_figure([group_JOIS_mpts, group_JOIS_ell], filename='Figure_3.pickle')
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
    group_JOIS_some_pfs = ahf.Analysis_Group(ds_ITP35_some_pfs, pfs_0, pp_ITP35_some_pfs)
    # Make the figure
    ahf.make_figure([group_JOIS_some_pfs])
    # ahf.make_figure([group_ITP3_some_pfs_2], row_col_list=[1,1, 0.27, 0.90], filename='Figure_8.pickle')

################################################################################