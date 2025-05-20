"""
Author: Mikhail Schee
Created: 2022-03-31

This script contains helper functions which filter data and create figures

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions in source code must retain the accompanying copyright notice, this list of conditions, and the following disclaimer.
    2. Redistributions in binary form must reproduce the accompanying copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Names of the copyright holders must not be used to endorse or promote products derived from this software without prior written permission from the copyright holders.
    4. If any files are modified, you must cause the modified files to carry prominent notices stating that you changed the files and the date of any change.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# For making insets in plots
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# For getting different marker styles
from matplotlib.markers import MarkerStyle as mplms
# For storing figure objects in files (can use `pickle` instead if need be)
import dill as pl
# For formatting data into dataframes
import pandas as pd
# For matching regular expressions
import re
# For formatting date objects
from datetime import datetime
# For reading netcdf files
import xarray as xr
# Import the Thermodynamic Equation of Seawater 2010 (TEOS-10) from GSW
# For finding alpha and beta values
import gsw
# For adding subplot labels a, b, c, ...
import string
# For taking moving averages
import scipy.ndimage as ndimage
# For interpolating
from scipy import interpolate
# For getting zscores to find outliers and for least squares
from scipy import stats
# For making clusters
import hdbscan
# For calculating the distance between pairs of (latitude, longitude)
from geopy.distance import geodesic
# For a style that matches scientific papers
import scienceplots
# For special colormaps
try:
    from cmcrameri import cm
except:
    foo = 2
# For importing custom functions from files
import importlib
# For common BGR parameters
bps = importlib.import_module('0_helper_files.BGR_params')
# For calculating Orthogonal Distance Regression for Total Least Squares
orthoregress = importlib.import_module('0_helper_files.orthoregress').orthoregress

"""
To install Cartopy and its dependencies, follow:
https://scitools.org.uk/cartopy/docs/latest/installing.html#installing

Relevent command:
$ conda install -c conda-forge cartopy
"""
try:
    import cartopy.crs as ccrs
    import cartopy.feature
except:
    print('Warning: could not import the cartopy package')

# science_data_file_path = '/Users/Grey/Documents/Research/Science_Data/'
science_data_file_path = '/Volumes/The_Abyss/Personal/Research/Science_Data/'

# This list gets filled in with the names of all the available columns for the
#   given data during the `apply_data_filters()` function
available_variables_list = []

################################################################################
# Declare variables for plotting
################################################################################
dark_mode = False
fixed_width_image = True
png_dpi = 100 # Setting the dpi higher causes lat and lon labels to disappear on maps. This is solved in more recent versions of cartopy and matplotlib
transparent_figures = False
plt.style.use('science')

# Colorblind-friendly palette by Krzywinski et al. (http://mkweb.bcgsc.ca/biovis2012/)
#   See the link below for a helpful color wheel:
#   https://jacksonlab.agronomy.wisc.edu/2016/05/23/15-level-colorblind-friendly-palette/
jackson_clr = np.array(["#000000",  #  0 black              #
                        "#004949",  #  1 dark olive         # light
                        "#009292",  #  2 teal               # light / dark
                        "#ff6db6",  #  3 hot pink           # 
                        "#ffb6db",  #  4 light pink         #         dark
                        "#490092",  #  5 dark purple        # light
                        "#006ddb",  #  6 royal blue         #         dark
                        "#b66dff",  #  7 violet             #         dark
                        "#6db6ff",  #  8 sky blue           #
                        "#b6dbff",  #  9 pale blue          #         dark
                        "#920000",  # 10 dark red           # light
                        "#924900",  # 11 brown              #
                        "#db6d00",  # 12 dark orange        # light
                        "#24ff24",  # 13 neon green         # light / dark
                        "#ffff6d"]) # 14 yellow             # light / dark

# Enable dark mode plotting
if dark_mode:
    plt.style.use('dark_background')
    std_clr = 'w'
    bg_clr = 'k'
    alt_std_clr = 'yellow'
    noise_clr = 'w'
    cnt_clr = '#db6d00'
    clr_ocean = 'k'
    clr_land  = 'grey'
    clr_lines = 'w'
    clstr_clrs = jackson_clr[[2,14,4,6,7,9,13]]
    bathy_clrs = ['#000040','k']
else:
    std_clr = 'k'
    bg_clr = 'w'
    alt_std_clr = 'olive'
    noise_clr = 'k'
    cnt_clr = "#ffff6d"
    clr_ocean = 'w'
    clr_land  = 'grey'
    clr_lines = 'k'
    clstr_clrs = jackson_clr[[12,1,10,14,5,2,13]] # old: jackson_clr[[12,1,10,6,5,2,13]]
    # clstr_clrs = jackson_clr[[12,1,10,6,5,2,13]] # For ITP vs time plot
    bathy_clrs = ['w', '#000080'] # ['w','#b6dbff']

# Define bathymetry colors
n_bathy = 7
cm0 = mpl.colors.LinearSegmentedColormap.from_list("Custom", bathy_clrs, N=n_bathy)
cm1 = mpl.colors.LinearSegmentedColormap.from_list("Custom", bathy_clrs[::-1], N=n_bathy)
# Initialize colormap to get ._lut attribute
cm0._init()
cm1._init()
# There are 3 extra colors at the end for some weird reason, so eliminate them
rgbas0 = cm0._lut[0:-3]
# Convert to a list of hex values in strings
bathy_clrs = [mpl.colors.rgb2hex(x) for x in rgbas0]
# Make an object for color bars
bathy_smap = plt.cm.ScalarMappable(cmap=cm1)
bathy_smap.set_array([])

# Set some plotting styles
if fixed_width_image:
    font_size_plt = 19
    font_size_labels = 21
    font_size_ticks = 15
    font_size_lgnd = 14
    big_map_mrkr  = 120
    cent_mrk_size = 55
else:
    font_size_plt = 12
    font_size_labels = 14
    font_size_ticks = 12
    font_size_lgnd = 10
    big_map_mrkr  = 80
    cent_mrk_size = 50
mpl.rcParams['font.size'] = font_size_plt
mpl.rcParams['axes.labelsize'] = font_size_labels
mpl.rcParams['xtick.labelsize'] = font_size_ticks
mpl.rcParams['ytick.labelsize'] = font_size_ticks
mpl.rcParams['legend.fontsize'] = font_size_lgnd
# plt.rcParams.update({'text.usetex':True})
mrk_size      = 10
mrk_size2     = 5
mrk_size3     = 0.5
mrk_alpha     = 0.5
mrk_alpha2    = 0.3
mrk_alpha3    = 0.1
noise_alpha   = 0.01
grid_alpha    = 0.3
plt.rcParams['grid.color'] = (0.5,0.5,0.5,grid_alpha)
hist_alpha    = 0.8
pf_alpha      = 0.9
map_alpha     = 0.7
pf_mrk_alpha  = 0.3
lgnd_mrk_size = 60
map_mrk_size  = 25
sml_map_mrkr  = 5
pf_mrk_size   = 5 #15
std_marker = '.'
map_marker = '.' #'x'
map_ln_wid = 0.5
l_cap_size = 3.0
anno_bbox = dict(boxstyle="round,pad=0.3", fc=bg_clr, ec=std_clr, lw=0.72, alpha=0.75)
map_bbox  = dict(clr='red', lw=2, ls='-', alpha=1.0)

# Plotting parameters for the LHW and AW
LHW_mrk = 'v'
LHW_clr = 'k'
LHW_facealtclr = jackson_clr[6]
LHW_edgeclr = jackson_clr[6]
AW_mrk = '^'
AW_clr = 'k'
AW_facealtclr = jackson_clr[3]
AW_edgeclr = jackson_clr[3]
# For SA divs
SA_divs_clr = alt_std_clr
SA_divs_line = '-'
# Colors to mark outliers
out_clr_RL = 'b'
out_clr_IR = 'r'
out_clr_end = jackson_clr[12]

# The magnitude limits for axis ticks before they use scientific notation
sci_lims = (-2,3)

#   Get list of standard colors
distinct_clrs = clstr_clrs
#   Make list of marker and fill styles
unit_star_3 = mpl.path.Path.unit_regular_star(3)
unit_star_4 = mpl.path.Path.unit_regular_star(4)
mpl_mrks = [mplms('o',fillstyle='left'), mplms('o',fillstyle='right'), 'x', unit_star_4, '*', unit_star_3, '1','+']
# Define array of linestyles to cycle through
l_styles = ['-', '--', '-.', ':']

# A list of variables for which the y-axis should be inverted so the surface is up
y_invert_vars = ['press', 'press_max', 'press_min', 'pca_press', 'ca_press', 'press-fit', 'cmm_mid', 'depth', 'pca_depth', 'ca_depth', 'sigma', 'ma_sigma', 'pca_sigma', 'ca_sigma', 'pca_iT', 'ca_iT', 'pca_CT', 'ca_CT', 'pca_PT', 'ca_PT', 'pca_SP', 'ca_SP', 'SA', 'pca_SA', 'ca_SA', 'SA-fit', 'CT', 'CT-fit', 'press_TC_max', 'press_TC_min', 'press_TC_max-fit', 'press_TC_min-fit', 'CT_TC_max', 'CT_TC_min', 'CT_TC_max-fit', 'CT_TC_min-fit', 'SA_TC_max', 'SA_TC_min', 'SA_TC_max-fit', 'SA_TC_min-fit', 'sig_TC_max', 'sig_TC_min', 'sig_TC_max-fit', 'sig_TC_min-fit']
# A list of the per profile variables
pf_vars = ['entry', 'prof_no', 'BL_yn', 'dt_start', 'dt_end', 'lon', 'lat', 'region', 'up_cast', 'CT_TC_max', 'press_TC_max', 'SA_TC_max', 'sig_TC_max', 'CT_TC_min', 'press_TC_min', 'SA_TC_min', 'sig_TC_min', 'R_rho']
# A list of the variables on the `Vertical` dimension
vertical_vars = ['press', 'depth', 'iT', 'CT', 'PT', 'SP', 'SA', 'sigma', 'alpha', 'beta', 'aiT', 'aCT', 'aPT', 'BSP', 'BSA', 'ss_mask', 'ma_iT', 'ma_CT', 'ma_PT', 'ma_SP', 'ma_SA', 'ma_sigma', 'la_iT', 'la_CT', 'la_PT', 'la_SP', 'la_SA', 'la_sigma']
# Add variable modifiers for profile variables
max_prefix = 'max_' # maximum
max_vars   = [ f'{max_prefix}{var}'  for var in vertical_vars]
min_prefix = 'min_' # maximum
min_vars   = [ f'{min_prefix}{var}'  for var in vertical_vars]
# Make lists of clustering variables
pca_prefix = 'pca_' # profile cluster average
pca_vars   = [ f'{pca_prefix}{var}' for var in vertical_vars]
pcs_prefix = 'pcs_' # profile cluster span
pcs_vars   = [ f'{pcs_prefix}{var}' for var in vertical_vars]
cmc_prefix = 'cmc_' # cluster mean-centered
cmc_vars   = [ f'{cmc_prefix}{var}' for var in vertical_vars]
ca_prefix  = 'ca_'  # cluster average
ca_vars    = [ f'{ca_prefix}{var}'  for var in vertical_vars]
cs_prefix  = 'cs_'  # cluster span
cs_vars    = [ f'{cs_prefix}{var}'  for var in vertical_vars]
csd_prefix = 'csd_'  # cluster standard deviation
csd_vars   = [ f'{csd_prefix}{var}'  for var in vertical_vars]
cmm_prefix = 'cmm_' # cluster min/max
cmm_vars   = [ f'{cmm_prefix}{var}'  for var in vertical_vars]
nir_prefix = 'nir_' # normalized inter-cluster range
nir_vars   = [ f'{nir_prefix}{var}'  for var in vertical_vars]
trd_prefix = 'trd_' # trend in a variable over dt_start
trd_vars   = [ f'{trd_prefix}{var}'  for var in vertical_vars]
# Make a complete list of cluster-related variables
clstr_vars = ['cluster', 'cRL', 'cRl', 'ca_dt_start'] + pca_vars + pcs_vars + cmc_vars + ca_vars + cs_vars + csd_vars + cmm_vars + nir_vars + trd_vars
# Make a complete list of per profile variables
pf_vars = pf_vars + max_vars + min_vars + ['cRL', 'cRl'] + pca_vars + pcs_vars + trd_vars
# For parameter sweeps of clustering
#   Independent variables
clstr_ps_ind_vars = ['m_pts', 'n_pfs', 'ell_size']
#   Dependent variables
clstr_ps_dep_vars = ['DBCV', 'n_clusters']
clstr_ps_vars = clstr_ps_ind_vars + clstr_ps_dep_vars

################################################################################
# Declare classes for custom objects
################################################################################

class Data_Set:
    """
    Loads data from netcdfs and returns a list of xarrays with filters applied
    Note: Only filters to a selection of profiles. Does not filter individual
        profiles in any manner

    sources_dict    A dictionary with the following format:
                    {'ITP_1':[13,22,32],'ITP_2':'all'}
                    where the keys are the netcdf filenames without the extension
                    and the values are lists of profiles to include or 'all'
    data_filters    A custom Data_Filters object that contains the filters to apply
    """
    def __init__(self, sources_dict, data_filters):
        # Load just the relevant profiles into the xarrays
        self.sources_dict = sources_dict
        self.data_filters = data_filters
        self.xarrs, self.var_attr_dicts = list_xarrays(sources_dict)
        self.arr_of_ds = apply_data_filters(self.xarrs, data_filters)

################################################################################

class Data_Filters:
    """
    A set of filters to select certain profiles
    Note: Only filters to a selection of profiles. Does not filter individual
        profiles in any manner

    keep_black_list     True/False whether to keep profiles marked on black list
    cast_direction      'up','down',or 'all' to keep certain directions of casts
    geo_extent          None to keep all profiles regardless of geographic region
                        or 'CS' for Chukchi Sea, 'SBS' for Southern Beaufort Sea,
                        'CB' for Canada Basin, 'MB' for Makarov Basin, 'EB' for
                        Eurasian Basin, or 'BS' for Barents Sea
    date_range          ['start_date','end_date'] where the dates are strings in
                        the format 'YYYY/MM/DD' or None to keep all profiles
    min_press           The minimum value of pressure to keep a profile
    press_TC_max_range  [p_min, p_max] where the values are floats in dbar
    press_TC_min_range  [p_min, p_max] where the values are floats in dbar
    clstr_labels        None to keep all cluster labels or a list of X lists, 
                        where X is the number of datasets, and each list contains
                        the cluster labels to keep from that dataset
    """
    def __init__(self, keep_black_list=False, cast_direction='up', geo_extent=None, lon_range=None, lat_range=None, date_range=None, min_press=None, press_TC_max_range=[100,700], press_TC_min_range=None, clstr_labels=None):
        self.keep_black_list = keep_black_list
        self.cast_direction = cast_direction
        self.geo_extent = geo_extent
        # self.lon_range = lon_range
        # self.lat_range = lat_range
        self.date_range = date_range
        self.min_press = min_press
        self.press_TC_max_range = press_TC_max_range
        self.press_TC_min_range = press_TC_min_range
        self.clstr_labels = clstr_labels

################################################################################

class Analysis_Group:
    """
    Takes in a list of xarray datasets and applies filters to individual profiles
    Returns a list of pandas dataframes, one for each source

    data_set            A Data_Set object
    profile_filters     A custom Profile_Filters object with filters to apply to
                        all individual profiles in the xarrays
    plt_params          A custom Plot_Parameters object
    plot_title          A string to use as the title for this subplot
    """
    def __init__(self, data_set, profile_filters, plt_params, plot_title=None):
        self.data_set = data_set
        self.vars_available = list(data_set.arr_of_ds[0].keys())
        self.profile_filters = profile_filters
        self.plt_params = get_axis_labels(plt_params, data_set.var_attr_dicts)
        self.plot_title = plot_title
        self.vars_to_keep = find_vars_to_keep(plt_params, profile_filters, self.vars_available)
        # Load just the relevant profiles into the xarrays
        self.data_frames = apply_profile_filters(data_set.arr_of_ds, self.vars_to_keep, profile_filters, plt_params)
    def __setitem__(self, key, value):
        setattr(self, key, value)
    def __getitem__(self, key):
        return getattr(self, key)

class Analysis_Group2:
    """
    Takes in a list of xarray datasets and applies filters to individual profiles
    Returns a list of pandas dataframes, one for each source

    data_frames            A list of pandas dataframes
    plt_params          A custom Plot_Parameters object
    plot_title          A string to use as the title for this subplot
    """
    def __init__(self, data_frames, plt_params, plot_title=None):
        self.data_frames = data_frames
        self.plt_params = get_axis_labels2(plt_params)
        self.plot_title = plot_title

################################################################################

class Profile_Filters:
    """
    A set of filters to apply to each profile. If None is passed for any filter,
    that filter will not be applied. All these are ignored when plotting 'by_pf'

    p_range             [p_min, p_max] where the values are floats in dbar
    d_range             [d_min, d_max] where the values are floats in m
    *T_range            [T_min, T_max] where the values are floats in degrees C
    S*_range            [S_min, S_max] where the values are floats in g/kg
    sig_range           [sig_min, sig_max] where the values are floats in kg/m^3
    lon_range           [lon_min,lon_max] where the values are floats in degrees
    lat_range           [lat_min,lat_max] where the values are floats in degrees
    lt_pTC_max          True/False whether to only keep points with p < p(TC_max)
    subsample           True/False whether to apply the subsample mask to the profiles
    every_nth_row       Integer n to keep every nth row in the dataframe
    regrid_TS           [1st_var_str, Delta_1st_var, 2nd_var_str, Delta_2nd_var], a pair of 
                            [var, Delta_var] where you specify the variable then the value
                            of the spacing to regrid that value to
    m_avg_win           The value in dbar of the moving average window to take for ma_ variables
    clstrs_to_plot      A list of the cluster id's to plot
    """
    def __init__(self, p_range=None, d_range=None, iT_range=None, CT_range=None, PT_range=None, SP_range=None, SA_range=None, sig_range=None, lon_range=None, lat_range=None, lt_pTC_max=False, subsample=False, every_nth_row=1, regrid_TS=None, m_avg_win=None, clstrs_to_plot=[]):
        self.p_range = p_range
        self.d_range = d_range
        self.iT_range = iT_range
        self.CT_range = CT_range
        self.PT_range = PT_range
        self.SP_range = SP_range
        self.SA_range = SA_range
        self.sig_range = sig_range
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.lt_pTC_max = lt_pTC_max
        self.subsample = subsample
        self.every_nth_row = every_nth_row
        self.regrid_TS = regrid_TS
        self.m_avg_win = m_avg_win
        self.clstrs_to_plot = clstrs_to_plot
    def __setitem__(self, key, value):
        setattr(self, key, value)
    def __getitem__(self, key):
        return getattr(self, key)

################################################################################

class Plot_Parameters:
    """
    A set of parameters to determine what kind of plot to make

    plot_type       A string of the kind of plot to make from these options:
                        'xy', 'map', 'profiles'
    plot_scale      A string specifying either 'by_pf' or 'by_vert'
                    which will determine whether to add one data point per
                    profile or per vertical observation, respectively
    x_vars          A list of strings of the variables to plot on the x axis
    y_vars          A list of strings of the variables to plot on the y axis
                    Note: 'xy' and 'profiles' expect both x_vars and y_vars and
                      'map' ignores this variable as it always plots lon vs. lat
                    Default plot is 'SP' vs. 'iT'
                    Accepted 'by_pf' vars: 'entry', 'prof_no', 'BL_yn', 'dt_start',
                      'dt_end', 'lon', 'lat', 'region', 'up_cast', 'R_rho', 'hist'
                    Accepted 'by_vert' vars: all 'by_pf' vars plus 'press',
                      'depth', 'iT', 'CT', 'PT', 'SP', 'SA', 'sigma', 'alpha', 'beta',
                      'aiT', 'aCT', 'aPT', 'BSP', 'BSA', 'ma_iT', 'ma_CT', 'ma_PT', 
                      'ma_SP', 'ma_SA', 'ma_sigma', 'ss_mask', 'la_iT', 'la_CT', 
                      'la_PT', 'la_SA','la_sigma'
    z_vars          A list of strings of the variables to plot on the z axis
    clr_map         A string to determine what color map to use in the plot
                    'xy' can use 'clr_all_same', 'source',
                      'instrmt', 'density_hist', 'cluster', or any
                      regular variable
                    'map' can use 'clr_all_same', 'source',
                      'instrmt', or any 'by_pf' regular variable
                    'profiles' can use 'clr_all_same' or any regular variable
    first_dfs       A list of booleans of whether to take the first differences
                      of the plot_vars before analysis. Defaults to all False
                    'xy' and 'profiles' expect a list of 2, ex: [True, False]
                    'map' and ignores this parameter
    finit_dfs       A list of booleans of whether to take the finite differences
                      of the plot_vars before analysis. Defaults to all False
                    'xy' and 'profiles' expect a list of 2, ex: [True, False]
                    'map' and ignores this parameter
    legend          True/False whether to add a legend on this plot. Default:True
    add_grid        True/False whether to add a grid on this plot. Default:True
    ax_lims         An optional dictionary of the limits on the axes for the final plot
                      Ex: {'x_lims':[x_min,x_max], 'y_lims':[y_min,y_max], 
                           'z_lims':[z_min,z_max], 'c_lims':[c_min,c_max], 'clr_ext':'max'}
    extra_args      A general use argument for passing extra info for the plot
                    'map' expects a dictionary following this format:
                        {'map_extent':'Canada_Basin'}
                    'density_hist' expects a dictionary following this format:
                        {'clr_min':0, 'clr_max':20, 'clr_ext':'max', 'xy_bins':250}
                        All 4 of those arguments must be provided for it to work
                        If no extra_args is given, those default values are used
                    If 'hist' is one of the plot variables, you can add the
                        number of histogram bins to use: {'n_h_bins':25}, if not
                        given, it will use that default value
                    'profiles' plots will accept optional arguments such as:
                        {'pfs_to_plot':[183,185,187], 'shift_pfs':True}
                        Note: the code narrows to these profiles just before
                        plotting, so use this argument instead of narrowing earlier
                        in the Data_Set object if you want to plot by 'cluster'
                    If plotting anything that has to do clustering, need these:
                        {'cl_x_var':var0, 'cl_y_var':var1, 'cl_z_var':var2} where the
                        variables can be any that work for an 'xy' plot
                        Optional 'b_a_w_plt':True/False for box and whisker plots,
                        'm_pts':90 / 'm_cls':90 to specify m_pts or m_cls,
                        'plot_slopes' as True, 'OLS' or 'TLS' to plot lines showing the 
                        ordinary or least squares slope of each cluster, 
                        'clstrs_to_plot':[0,1,2] to plot just specific clusters
                    If doing a parameter sweep of clustering, expects the following:
                        {'cl_x_var':var0, 'cl_y_var':var1, 'cl_ps_tuple':[100,410,50]}
                        where var0/var1/var2 are as specified above, 'cl_ps_tuple' is
                        the [start,stop,step] for the xvar can be 'm_pts', 'm_cls'
                        'n_pfs' or 'ell_size', and yvar(s) must be 'DBCV' or 'n_clusters'
                        Optional: {'z_var':'m_pts', 'z_list':[90,120,240]} where
                        z_var can be any variable that var0 can be
                    If running a parameter sweep in parallel, add the following:
                        {'mpi_run':True}. This will suppress any graphical output
                    To mark the top and bottom of the thermocline in profiles, add
                        {'mark_LHW_AW':True}. Can also have the value be 'max' or
                        'min' to mark just the top or bottom
                    To plot isopycnal contour lines add {'isopycnals':X} where X is the 
                        value in dbar to which the isopycnals are referenced or True 
                        which will set the reference to the median pressure
                    To manually place the inline labels for isopycnals, add this:
                        {'place_isos':'manual'}, otherwise they will be placed 
                        automatically
                    To add error bars to the plot, add {'errorbars':True}. This, for 
                        example, will add error bars equal to the standard deviation
                        if plotting a cluster average `ca_` variable
                    To change any axis to a log scale, add {'log_axes':[True,False,False]}
                        where the array should be 3 long with entries True/False for 
                        whether to put the x, y, or z axes (in that order) on a log scale
    """
    def __init__(self, plot_type='xy', plot_scale='by_vert', x_vars=['SA'], y_vars=['CT'], z_vars=[None], clr_map='clr_all_same', first_dfs=[False, False], finit_dfs=[False, False], legend=True, add_grid=True, ax_lims=None, extra_args=None):
        # Add all the input parameters to the object
        self.plot_type = plot_type
        self.plot_scale = plot_scale
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.z_vars = z_vars
        self.xlabels = [None,None]
        self.ylabels = [None,None]
        self.zlabels = [None,None]
        self.clr_map = clr_map
        self.clabel = None
        self.first_dfs = first_dfs
        self.finit_dfs = finit_dfs
        self.legend = legend
        self.add_grid = add_grid
        self.ax_lims = ax_lims
        # If trying to plot a map, make sure x and y vars are None
        if plot_type == 'map':
            x_vars = None
            y_vars = None
            z_vars = None
        # If trying to plot a map, make sure `map_extent` exists
        if plot_type == 'map' and isinstance(extra_args, type(None)):
            self.extra_args = {'map_extent':'Canada_Basin'}
        else:
            self.extra_args = extra_args

################################################################################
# Define class functions #######################################################
################################################################################

def list_xarrays(sources_dict):
    """
    Returns a list of xarrays, one for each data source as specified by the
    input dictionary

    sources_dict    A dictionary with the following format:
                    {'ITP_1':['1-13','1-22','1-32'],'ITP_2':'all'} or
                    {'HPC_BGR0506_clstrd_SA_divs':['3-313', '3-315', '3-317', '3-319', '3-321']}
                    where the keys are the netcdf filenames without the extension
                    and the values are lists of instrmt-profile_number to include or 'all'
    """
    # Make an empty list
    xarrays = []
    var_attr_dicts = []
    for source in sources_dict.keys():
        print('\t- Loading data from netcdfs/'+source+'.nc')
        ds = xr.load_dataset('netcdfs/'+source+'.nc')
        # Build the dictionary of netcdf attributes, variables, units, etc.
        var_attrs = {}
        for this_var in list(ds.keys()):
            var_attrs[ds[this_var].name] = ds[this_var].attrs
        var_attr_dicts.append(var_attrs)
        # Convert the datetime variables from dtype `object` to `datetime64`
        ds['Time'] = pd.DatetimeIndex(ds['Time'].values)
        # Check whether to only take certain profiles
        pf_list = sources_dict[source]
        if isinstance(pf_list, list):
            # print('Keeping just these profiles:',str(pf_list))
            # Start a blank list for all the profile-specific xarrays
            temp_list = []
            for inst_pf in pf_list:
                # Check to make sure inst_pf is a string
                if not isinstance(inst_pf, type('')):
                    print('Expecting list of profiles to contain string objects. Found '+str(inst_pf)+' '+str(type(inst_pf)))
                    print('Aborting script')
                    exit(0)
                # Split the instrmt number from the profile number (assumes a hyphen split)
                split_var = inst_pf.split('-', 1)
                this_instrmt = split_var[0] # do not convert type
                this_pf = int(split_var[1])
                # Remember to have parentheses around each condition
                temp_list.append(ds.where((ds.instrmt==this_instrmt) & (ds.prof_no==this_pf), drop=True).squeeze())
                # temp_list.append(ds.where((ds.prof_no==this_pf), drop=True).squeeze())
            # Concatonate all those xarrays together
            ds1 = xr.concat(temp_list, 'Time')
            xarrays.append(ds1)
        # Take all profiles
        elif pf_list=='all':
            xarrays.append(ds)
        else:
            print(pf_list,'is not a valid list of profiles')
            exit(0)
        #
    # print(var_attr_dicts)
    # print('xarrays',xarrays)
    return xarrays, var_attr_dicts

################################################################################

def apply_data_filters(xarrays, data_filters):
    """
    Returns a list of the same xarrays as provided, but with the filters applied

    xarrays         A list of xarray dataset objects
    data_filters    A custom Data_Filters object that contains the filters to apply
    """
    # Make an empty list
    output_arrs = []
    i = 0
    for ds in xarrays:
        ## Filters on a per-profile basis
        ##      I turned off squeezing because it drops the `Time` dimension if
        ##      you only pass in one profile per dataset
        #   Filter to just every 10th odd profile
        # ds = ds.where((ds.prof_no+1)%100==0, drop=True)
        #   Filter based on the black list
        if data_filters.keep_black_list == False:
            ds = ds.where(ds.BL_yn==False, drop=True)#.squeeze()
        #   Filter based on the cast direction
        if data_filters.cast_direction == 'up':
            ds = ds.where(ds.up_cast==True, drop=True)#.squeeze()
        elif data_filters.cast_direction == 'down':
            ds = ds.where(ds.up_cast==False, drop=True)#.squeeze()
        #   Filter based on the geographical region
        if data_filters.geo_extent == 'CB':
            ds = ds.where(ds.region=='CB', drop=True)#.squeeze()
        #   Filter based on the date range
        if not isinstance(data_filters.date_range, type(None)):
            # Allow for date ranges that do or don't specify the time
            try:
                start_date_range = datetime.strptime(data_filters.date_range[0], r'%Y/%m/%d %H:%M:%S')
            except:
                start_date_range = datetime.strptime(data_filters.date_range[0], r'%Y/%m/%d')
            try:
                end_date_range   = datetime.strptime(data_filters.date_range[1], r'%Y/%m/%d %H:%M:%S')
            except:
                end_date_range   = datetime.strptime(data_filters.date_range[1], r'%Y/%m/%d')
            try:
                ds = ds.sel(Time=slice(start_date_range, end_date_range))
                # print('\t- Filtering to the period',start_date_range,end_date_range)
            except:
                print('WARNING: cannot take time slice for')
                print('\t',ds.attrs['Source'],ds.attrs['Instrument'])
                # exit(0)
        #   Only keep profiles where press_max is deeper than min_press
        if not isinstance(data_filters.min_press, type(None)):
            if False:
                pf_list = list(set(ds.prof_no.values))
            ds = ds.where(ds.press_max>=data_filters.min_press, drop=True)#.squeeze()
            if False:
                pf_list2 = list(set(ds.prof_no.values))
                # Find the eliminated profiles
                elim_list = [x for x in pf_list if x not in pf_list2]
                print('p_max filter, # of profiles before:',len(pf_list),'after:',len(pf_list2),'difference:',len(pf_list)-len(pf_list2))#,'-',elim_list)
        #
        #   Only keep profiles with press_TC_min within the specified range
        if not isinstance(data_filters.press_TC_min_range, type(None)):
            ds = ds.where(ds.press_TC_min>=min(data_filters.press_TC_min_range), drop=True)
            ds = ds.where(ds.press_TC_min<=max(data_filters.press_TC_min_range), drop=True)
        #   Only keep profiles with press_TC_max within the specified range
        if not isinstance(data_filters.press_TC_max_range, type(None)):
            ds = ds.where(ds.press_TC_max>=min(data_filters.press_TC_max_range), drop=True)
            ds = ds.where(ds.press_TC_max<=max(data_filters.press_TC_max_range), drop=True)
        #
        #   Filter to just certain cluster labels
        if isinstance(data_filters.clstr_labels, type(None)):
            foo = 2
        elif data_filters.clstr_labels == 'no_noise':
            # print('ds variables:')
            # print(list(ds.keys()))
            # print('')
            ds = ds.where(ds.cluster != -1, drop=True)#.squeeze()
        elif len(data_filters.clstr_labels) > 0:
            list_of_ds_clstrs = []
            print('data_filters.clstr_labels:',data_filters.clstr_labels)
            these_clstr_labels = data_filters.clstr_labels[i]
            print('\t- Filtering to just these cluster labels:',these_clstr_labels)
            for this_clstr_label in these_clstr_labels:
                this_ds = ds.where(ds.cluster == this_clstr_label, drop=True)#.squeeze()
                list_of_ds_clstrs.append(this_ds)
            ds = xr.concat(list_of_ds_clstrs, dim='Time')
        #
        i += 1
        output_arrs.append(ds)
    return output_arrs

################################################################################

def find_vars_to_keep(pp, profile_filters, vars_available):
    """
    Returns a list the variables to keep for an analysis group based on the plot
    parameters and profile filters given

    pp                  A custom Plot_Parameters object
    profile_filters     A custom Profile_Filters object with filters to apply to
                        all individual profiles in the xarrays
    vars_available      A list of variables available for this data
    """
    # If plot type is 'summary', then keep all the variables
    if pp.plot_type == 'summary':
        # Check the plot scale
        #   This will include all the measurements within each profile
        if plt_params.plot_scale == 'by_vert':
            vars_to_keep = ['entry', 'prof_no', 'dt_start', 'dt_end', 'lon', 'lat', 'CT_TC_max', 'press_TC_max', 'SA_TC_max', 'sig_TC_max', 'CT_TC_min', 'press_TC_min', 'SA_TC_min', 'sig_TC_min', 'R_rho', 'press', 'depth', 'iT', 'CT', 'PT', 'SP', 'SA', 'sigma', 'alpha', 'beta', 'ss_mask', 'ma_iT', 'ma_CT', 'ma_PT', 'ma_SP', 'ma_SA', 'ma_sigma']
        #   This will only include variables with 1 value per profile
        elif plt_params.plot_scale == 'by_pf':
            vars_to_keep = ['entry', 'prof_no', 'dt_start', 'dt_end', 'lon', 'lat', 'CT_TC_max', 'press_TC_max', 'SA_TC_max', 'sig_TC_max', 'CT_TC_min', 'press_TC_min', 'SA_TC_min', 'sig_TC_min', 'R_rho']
        # print('vars_to_keep:')
        # print(vars_to_keep)
        return vars_to_keep
    else:
        # Always include the entry and profile numbers
        vars_to_keep = ['entry', 'prof_no', 'source', 'instrmt']
        # Add vars based on plot type
        if pp.plot_type == 'map':
            vars_to_keep.append('lon')
            vars_to_keep.append('lat')
        if pp.plot_type == 'profiles':
            vars_to_keep.append('dt_start')
        if pp.plot_type == 'waterfall':
            vars_to_keep.append('dt_start')
        # Add all the plotting variables
        re_run_clstr = False
        plot_vars = pp.x_vars+pp.y_vars+pp.z_vars+[pp.clr_map]
        # Remove None values
        plot_vars = [x for x in plot_vars if x is not None]
        # Check the extra arguments
        if not isinstance(pp.extra_args, type(None)):
            for key in pp.extra_args.keys():
                if key in ['cl_x_var', 'cl_y_var', 'cl_z_var']:
                    plot_vars.append(pp.extra_args[key])
                    re_run_clstr = True
                if key == 'z_var':
                    if pp.extra_args[key] == 'ell_size':
                        # Make sure the parameter sweeps run correctly 
                        vars_to_keep.append('press')
                # If marking the bounds of the thermocline
                if key == 'mark_LHW_AW':
                    if pp.extra_args[key] == True:
                        vars_to_keep.append('CT_TC_max')
                        vars_to_keep.append('press_TC_max')
                        vars_to_keep.append('SA_TC_max')
                        vars_to_keep.append('sig_TC_max')
                        vars_to_keep.append('CT_TC_min')
                        vars_to_keep.append('press_TC_min')
                        vars_to_keep.append('SA_TC_min')
                        vars_to_keep.append('sig_TC_min')
                    elif pp.extra_args[key] == 'max':
                        vars_to_keep.append('CT_TC_max')
                        vars_to_keep.append('press_TC_max')
                        vars_to_keep.append('SA_TC_max')
                        vars_to_keep.append('sig_TC_max')
                    elif pp.extra_args[key] == 'min':
                        vars_to_keep.append('CT_TC_min')
                        vars_to_keep.append('press_TC_min')
                        vars_to_keep.append('SA_TC_min')
                        vars_to_keep.append('sig_TC_min')
                # If adding isopycnals, make sure to keep press, SA, and iT to use with the 
                #   function gsw.pot_rho_t_exact(SA,t,p,p_ref)
                if key == 'isopycnals':
                    if not isinstance(pp.extra_args[key], type(None)):
                        if not 'press' in vars_to_keep:
                            vars_to_keep.append('press')
                        if not 'SA' in vars_to_keep:
                            vars_to_keep.append('SA')
                        if not 'iT' in vars_to_keep:
                            vars_to_keep.append('iT')
                # If adding variables over which to run polyfit2d
                if key == 'fit_vars':
                    if not isinstance(pp.extra_args[key], type(False)):
                        for var in pp.extra_args[key]:
                            vars_to_keep.append(var)
                    #
                # If adding extra variables to keep
                if key == 'extra_vars_to_keep':
                    for var in pp.extra_args[key]:
                        if var in ['cRL', 'nir_SA']:
                            plot_vars.append(var)
                        else:
                            vars_to_keep.append(var)
                    #
                #
            #
        # print('plot_vars:',plot_vars)
        # print('vars_available:',vars_available)
        # print('vars_to_keep:',vars_to_keep)
        for var in plot_vars:
            if var in vars_available:
                vars_to_keep.append(var)
            if var == 'cluster':
                vars_to_keep.append('clst_prob')
            if var == 'R_rho':
                vars_to_keep.append(var)
                vars_to_keep.append('aCT')
                vars_to_keep.append('alpha')
                vars_to_keep.append('CT')
                vars_to_keep.append('BSA')
                vars_to_keep.append('beta')
                vars_to_keep.append('SA')
            if var == 'aiT':
                vars_to_keep.append(var)
                vars_to_keep.append('alpha_iT')
                vars_to_keep.append('iT')
            elif var == 'aCT':
                vars_to_keep.append(var)
                vars_to_keep.append('alpha')
                vars_to_keep.append('CT')
            elif var == 'aPT':
                vars_to_keep.append(var)
                vars_to_keep.append('alpha_PT')
                vars_to_keep.append('PT')
            elif var == 'BSP':
                vars_to_keep.append(var)
                vars_to_keep.append('beta')
                vars_to_keep.append('SP')
            elif var == 'BSt':
                vars_to_keep.append(var)
                vars_to_keep.append('beta_PT')
                vars_to_keep.append('SP')
            elif var == 'BSA':
                vars_to_keep.append(var)
                vars_to_keep.append('beta')
                vars_to_keep.append('SA')
            elif var == 'cRL':
                vars_to_keep.append('alpha')
                vars_to_keep.append('CT')
                vars_to_keep.append('beta')
                vars_to_keep.append('SA')
            elif var == 'cRl':
                vars_to_keep.append('alpha_PT')
                vars_to_keep.append('PT')
                vars_to_keep.append('beta_PT')
                vars_to_keep.append('SP')
            elif var == 'ell_size':
                vars_to_keep.append('press')
            elif var == 'distance':
                vars_to_keep.append('lon')
                vars_to_keep.append('lat')
                vars_to_keep.append('dt_start')
            # Check for variables that have underscores, ie. profile cluster
            #   average variables and local anomalies
            if '_' in var:
                # Split the prefix from the original variable (assumes an underscore split)
                split_var = var.split('_', 1)
                prefix = split_var[0]
                var_str = split_var[1]
                # print('this var:',var,'prefix:',prefix,'var_str:',var_str)
                # If plotting a cluster variable, add the original variable (without
                #   the prefix) to the list of variables to keep
                if var in clstr_vars:
                    vars_to_keep.append(var_str)
                # Add very specific variables to the list, without prefixes
                if prefix in ['max', 'min', 'ca']:
                    vars_to_keep.append(var_str)
                # Add very specific variables directly to the list, including prefixes
                if prefix in ['la']:
                    vars_to_keep.append(var)
                    vars_to_keep.append(var_str)
                # Add dt_start if prefix indicates a trend in a variable
                if prefix in ['trd']:
                    vars_to_keep.append('dt_start')
            # Check for variables for which to normalize by subtracting a polyfit2d
            #   ex: `press-fit`
            if '-' in var:
                # print('this - var:',var)
                # Split the prefix from the original variable (assumes a hyphen split)
                split_var = var.split('-', 1)
                var_str = split_var[0]
                suffix = split_var[1]
                if '-fit' in var and 'trd' in var:
                    # Skip this variable, it's already been added
                    foo = 2
                else:
                    # Add var to the list
                    vars_to_keep.append(var_str)
            #
        # If re-running the clustering, remove 'cluster' from vars_to_keep
        if re_run_clstr:
            try:
                vars_to_keep.remove('cluster')
            except:
                foo = 2
            try:
                vars_to_keep.remove('clst_prob')
            except:
                foo = 2
        # Add vars for the profile filters, if applicable
        scale = pp.plot_scale
        if True:#scale == 'by_vert':
            if not isinstance(profile_filters.lon_range, type(None)):
                vars_to_keep.append('lon')
            if not isinstance(profile_filters.lat_range, type(None)):
                vars_to_keep.append('lat')
            if not isinstance(profile_filters.p_range, type(None)):
                vars_to_keep.append('press')
            if not isinstance(profile_filters.d_range, type(None)):
                vars_to_keep.append('depth')
            if not isinstance(profile_filters.iT_range, type(None)):
                vars_to_keep.append('iT')
            if not isinstance(profile_filters.CT_range, type(None)):
                vars_to_keep.append('CT')
            if not isinstance(profile_filters.PT_range, type(None)):
                vars_to_keep.append('PT')
            if not isinstance(profile_filters.SP_range, type(None)):
                vars_to_keep.append('SP')
            if not isinstance(profile_filters.SA_range, type(None)):
                vars_to_keep.append('SA')
            if profile_filters.lt_pTC_max:
                vars_to_keep.append('press')
                vars_to_keep.append('press_TC_max')
            if profile_filters.subsample:
                vars_to_keep.append('ss_mask')
            #
        #
        # Remove duplicates
        vars_to_keep = list(set(vars_to_keep))
        # print('vars_to_keep:')
        # print(vars_to_keep)
        # exit(0)
        return vars_to_keep

################################################################################

def apply_profile_filters(arr_of_ds, vars_to_keep, profile_filters, pp):
    """
    Returns a list of pandas dataframes, one for each array in arr_of_ds with
    the filters applied to all individual profiles

    arr_of_ds           An array of datasets from a custom Data_Set object
    vars_to_keep        A list of variables to keep for the analysis
    profile_filters     A custom Profile_Filters object that contains the filters to apply
    pp                  A custom Plot_Parameters object that contains at least:
    """
    print('- Applying profile filters')
    plot_scale = pp.plot_scale
    # print('plot_scale:',plot_scale)
    # Add all the plotting variables
    plot_vars = pp.x_vars+pp.y_vars+[pp.clr_map]
    if not isinstance(pp.extra_args, type(None)):
        for key in pp.extra_args.keys():
            if key in ['cl_x_var', 'cl_y_var', 'cl_z_var']:
                plot_vars.append(pp.extra_args[key])
        if 'sort_clstrs' in pp.extra_args.keys():
            sort_clstrs = pp.extra_args['sort_clstrs']
        else:
            sort_clstrs = False
    else:
        sort_clstrs = False
    # Make an empty list
    output_dfs = []
    # What's the plot scale?
    if plot_scale == 'by_vert':
        for ds in arr_of_ds:
            # print('\t- Applying filters to',ds.Source,ds.Instrument)
            # Find extra variables, if applicable
            ds = calc_extra_vars(ds, vars_to_keep)
            # Convert to a pandas data frame
            df = ds[vars_to_keep].to_dataframe()
            # Add a notes column
            df['notes'] = ''
            #   If the m_avg_win is not None, take the moving average of the data
            if not isinstance(profile_filters.m_avg_win, type(None)):
                # print('\t-Applying m_avg_win filter')
                df = take_m_avg(df, profile_filters.m_avg_win, vars_to_keep)
                ds.attrs['Moving average window'] = str(profile_filters.m_avg_win)+' dbar'
            #   True/False, apply the subsample mask to the profiles
            if profile_filters.subsample:
                # print('\t-Applying subsample filter')
                # `ss_mask` is null for the points that should be masked out
                # so apply mask to all the variables that were kept
                df.loc[df['ss_mask'].isnull(), vars_to_keep] = None
                # Set a new column so the ss_scheme can be found later for the title
                df['ss_scheme'] = ds.attrs['Sub-sample scheme']
            # Keep every nth row
            if profile_filters.every_nth_row > 1:
                n = int(profile_filters.every_nth_row)
                print('\t- Keeping every',n,'th row')
                print('before:',len(df))
                df = df.iloc[::n, :]
                print('after:',len(df))
            # Remove rows where the plot variables are null
            for var in plot_vars:
                # print('\t- Removing null values',var)
                if var in vars_to_keep and var not in ['distance']:
                    # print('\t\t- for:',var)
                    df = df[df[var].notnull()]
                if var == 'depth':
                    # Multiply all depth values by -1 to make them positive
                    df[var] = df[var] * -1
                    # print('\t- Multiplying all depth values by -1')
            # Add source and instrument columns if applicable
            if not 'source' in list(df):
                df['source'] = ds.Source
            if not 'instrmt' in list(df):
                df['instrmt'] = ds.Instrument
            # print(ds.Source,ds.Instrument)
            ## Filter each profile to certain ranges
            df = filter_profile_ranges(df, profile_filters, 'press', 'depth', 'sigma', iT_key='iT', CT_key='CT',PT_key='PT', SP_key='SP', SA_key='SA')
            # print('\t- Actually done filtering profile ranges')
            # Filter to just pressures above p(TC_max)
            if profile_filters.lt_pTC_max:
                # print('\t- Applying lt_pTC_max filter')
                # Get list of profiles
                pfs = list(set(df['prof_no'].values))
                # print(ds.Source,ds.Instrument,pfs)
                # Loop across each profile
                new_dfs = []
                for pf in pfs:
                    # Find the data for just that profile
                    data_pf = df[df['prof_no'] == pf]
                    # Filter the data frame to p < p(TC_max)
                    data_pf = data_pf[data_pf['press'] < data_pf['press_TC_max'].values[0]]
                    new_dfs.append(data_pf)
                # Re-form the dataframe
                try:
                    df = pd.concat(new_dfs)
                except:
                    print('\t- Keeping no profiles from',ds.Source,ds.Instrument)
                    continue
            ## Filter to just some cluster id's 
            if len(profile_filters.clstrs_to_plot) > 0:
                plt_these_clstrs = profile_filters.clstrs_to_plot
                # Find the cluster labels that exist in the dataframe
                cluster_numbers = np.unique(np.array(df['cluster'].values, dtype=int))
                #   Delete the noise point label "-1"
                cluster_numbers = np.delete(cluster_numbers, np.where(cluster_numbers == -1))
                n_clusters = int(len(cluster_numbers))
                # Sort clusters first?
                if sort_clstrs:
                    df = sort_clusters(df, cluster_numbers, ax=None)
                # Filter to just some clusters
                # print('\t- Before filtering to',plt_these_clstrs,'len(df):',len(df))
                df = filter_to_these_clstrs(df, cluster_numbers, plt_these_clstrs)
                # print('\t- After filtering to',plt_these_clstrs,'len(df):',len(df))
            ## Re-grid temperature and salinity data
            if not isinstance(profile_filters.regrid_TS, type(None)):
                # print('\t-Applying regrid_TS filter')
                # Figure out which salinity and temperature variable to re-grid
                T_var = profile_filters.regrid_TS[0]
                S_var = profile_filters.regrid_TS[2]
                # Make coarse grids for temp and salt based on given arguments
                d_temp = profile_filters.regrid_TS[1]
                d_salt = profile_filters.regrid_TS[3]
                t_arr  = np.array(df[T_var])
                s_arr  = np.array(df[S_var])
                t_min, t_max = min(t_arr), max(t_arr)
                s_min, s_max = min(s_arr), max(s_arr)
                t_grid = np.arange(t_min-d_temp, t_max+d_temp, d_temp)
                s_grid = np.arange(s_min-d_salt, s_max+d_salt, d_salt)
                # Make re-gridded arrays of temp and salt
                temp_rg = t_grid[abs(t_arr[None, :] - t_grid[:, None]).argmin(axis=0)]
                salt_rg = s_grid[abs(s_arr[None, :] - s_grid[:, None]).argmin(axis=0)]
                # Overwrite original temp and salt values with those regridded arrays
                df[T_var] = temp_rg
                df[S_var] = salt_rg
                # Note the regridding in the notes column
                df['notes'] = df['notes'] + r'$\Delta t_{rg}=$'+str(d_temp)+r', $\Delta s_{rg}=$'+str(d_salt)
            #
            # Calculate along-path distance, if applicable
            if 'distance' in plot_vars:
                print('\t- Calculating along-path distance')
                # Add blank column, will add distance values later
                df['distance'] = ''
                # Get list of profile times
                pf_times = list(set(df['dt_start'].values))
                # Convert them to a datetime series and sort
                pf_times = pd.to_datetime(pd.Series(pf_times)).sort_values()
                # Start with the first profile
                #   Get lat and lon values
                last_lon = df[df['dt_start']==str(pf_times[0])].lon[0]
                last_lat = df[df['dt_start']==str(pf_times[0])].lat[0]
                # Set distance for first profile
                df.loc[df['dt_start']==str(pf_times[0]), 'distance'] = 0
                # print('last lon and lat:',last_lon, last_lat)
                total_distance = 0
                # Loop over each profile
                for i in range(len(pf_times)):
                    # Get lat and lon values
                    this_lon = df[df['dt_start']==str(pf_times[i])].lon[0]
                    this_lat = df[df['dt_start']==str(pf_times[i])].lat[0]
                    # Calculate distance from the last profile
                    last_lat_lon = (last_lat, last_lon)
                    this_lat_lon = (this_lat, this_lon)
                    this_span = geodesic(last_lat_lon, this_lat_lon).km
                    # Add this to the along-path distance so far
                    total_distance += this_span
                    # print('this pf_time:', pf_times[i])
                    # print('\thas this many rows:',len(df[df['dt_start']==str(pf_times[i])]))
                    # Assign the rows of this time stamp new distance values
                    df.loc[df['dt_start']==str(pf_times[i]), 'distance'] = total_distance
                    # Assign new values to last values for next loop
                    last_lon = this_lon
                    last_lat = this_lat
                #
            #
            # Check whether or not to take first differences
            first_dfs = pp.first_dfs
            if any(first_dfs):
                print('\t-Applying first_dfs filter')
                # Get list of profiles
                pfs = list(set(df['prof_no'].values))
                # Take first differences in x variables
                if first_dfs[0]:
                    # Loop across each profile
                    new_dfs = []
                    for pf in pfs:
                        # Find the data for just that profile
                        data_pf = df[df['prof_no'] == pf]
                        # Loop across all x variables
                        for var in pp.x_vars:
                            # Replace the plot variable names
                            dvar = 'd_'+var
                            # Make the given variable into first differences
                            data_pf[dvar] = data_pf[var].diff()
                        new_dfs.append(data_pf)
                    # Re-form the dataframe
                    df = pd.concat(new_dfs)
                    # Replace the plot variable names
                    for i in range(len(pp.x_vars)):
                        dvar = 'd_'+pp.x_vars[i]
                        pp.x_vars[i] = dvar
                        # Remove rows with null values (one per profile because of diff())
                        df = df[df[dvar].notnull()]
                    #
                # Take first differences in y variables
                if len(first_dfs)==2 and first_dfs[1]:
                    # Loop across each profile
                    new_dfs = []
                    for pf in pfs:
                        # Find the data for just that profile
                        data_pf = df[df['prof_no'] == pf]
                        # Loop across all y variables
                        for var in pp.y_vars:
                            # Replace the plot variable names
                            dvar = 'd_'+var
                            # Make the given variable into first differences
                            data_pf[dvar] = data_pf[var].diff()
                        new_dfs.append(data_pf)
                    # Re-form the dataframe
                    df = pd.concat(new_dfs)
                    # Repalce the plot varaible names
                    for i in range(len(pp.y_vars)):
                        dvar = 'd_'+pp.y_vars[i]
                        pp.y_vars[i] = dvar
                        # Remove rows with null values (one per profile because of diff())
                        df = df[df[dvar].notnull()]
                    #
                # 
            #
            # Check whether or not to take finite differences
            finit_dfs = pp.finit_dfs
            if any(finit_dfs):
                print('\t-Applying finit_dfs filter')
                # Get list of profiles
                pfs = list(set(df['prof_no'].values))
                # Take finite differences in x variables
                if finit_dfs[0]:
                    # Find y variable
                    y_var = pp.y_vars[0]
                    # Loop across each profile
                    new_dfs = []
                    for pf in pfs:
                        # Find the data for just that profile
                        data_pf = df[df['prof_no'] == pf]
                        # Loop across all x variables
                        for var in pp.x_vars:
                            # Replace the plot variable names
                            dvar = 'd_'+var
                            # Make the given variable into finite difference
                            data_pf[dvar] = data_pf[var].diff() / data_pf[y_var].diff()
                        new_dfs.append(data_pf)
                    # Re-form the dataframe
                    df = pd.concat(new_dfs)
                    # Replace the plot variable names
                    for i in range(len(pp.x_vars)):
                        dvar = 'd_'+pp.x_vars[i]
                        pp.x_vars[i] = dvar
                        # Remove rows with null values (one per profile because of diff())
                        df = df[df[dvar].notnull()]
                    #
                # Take finite differences in y variables
                if len(finit_dfs)==2 and finit_dfs[1]:
                    # Find x variable
                    x_var = pp.x_vars[0]
                    # Loop across each profile
                    new_dfs = []
                    for pf in pfs:
                        # Find the data for just that profile
                        data_pf = df[df['prof_no'] == pf]
                        # Loop across all y variables
                        for var in pp.y_vars:
                            # Replace the plot variable names
                            dvar = 'd_'+var
                            # Make the given variable into finite difference
                            data_pf[dvar] = data_pf[var].diff() / data_pf[x_var].diff()
                        new_dfs.append(data_pf)
                    # Re-form the dataframe
                    df = pd.concat(new_dfs)
                    # Repalce the plot varaible names
                    for i in range(len(pp.y_vars)):
                        dvar = 'd_'+pp.y_vars[i]
                        pp.y_vars[i] = dvar
                        # Remove rows with null values (one per profile because of diff())
                        df = df[df[dvar].notnull()]
                    #
                # 
            #
            # Append the filtered dataframe to the list, but only if it has data
            if len(df) > 0:
                output_dfs.append(df)
        #
    elif plot_scale == 'by_pf':
        for ds in arr_of_ds:
            # Find extra variables, if applicable
            ds = calc_extra_vars(ds, vars_to_keep)
            # Convert to a pandas data frame
            df = ds[vars_to_keep].to_dataframe()
            # Add source and instrument columns if applicable
            if not 'source' in list(df):
                df['source'] = ds.Source
            if not 'instrmt' in list(df):
                df['instrmt'] = ds.Instrument
            # Calculate extra variables, as needed
            for var in plot_vars:
                if var == 'depth':
                    # Multiply all depth values by -1 to make them positive
                    df[var] = df[var] * -1
                    # print('\t- Multiplying all depth values by -1')
                if 'max' in var and not 'TC_max' in var:
                    # Split the prefix from the original variable (assumes an underscore split)
                    split_var = var.split('_', 1)
                    prefix = split_var[0]
                    var_str = split_var[1]
                    # Get list of profiles
                    pfs = list(set(df['prof_no'].values))
                    # Loop across each profile
                    for pf in pfs:
                        # Find the data for just that profile
                        data_pf = df[df['prof_no'] == pf]
                        # Find the maximum for that profile
                        this_pf_max = max(data_pf[var_str])
                        # Fill this profile's max_var with that value
                        df.loc[df['prof_no']==pf, var] = this_pf_max
                    # 
                    # Calculate vertical density ratio R_rho, if applicable
                if var == 'R_rho':
                    print('\t- Calculating vertical density ratio')
                    # Make a new column with the combination of instrument and profile number
                    #   This avoids accidentally lumping two profiles with the same number from
                    #   different instruments together, which would give the incorrect result
                    df['instrmt-prof_no'] = df.instrmt.map(str) + ' ' + df.prof_no.map(str)
                    instrmt_pf_nos = np.unique(np.array(df['instrmt-prof_no'].values))
                    n_profiles = len(instrmt_pf_nos)
                    if n_profiles > 0:
                        digits = int(np.log10(n_profiles))+1
                    else:
                        digits = 1
                    # Loop over each profile
                    i = 0
                    for pf in instrmt_pf_nos:
                        # Print a progress counter, padding the number with spaces to digits
                        ## Note, the `end="\r"` argument makes the console output dynamically update
                        i += 1
                        print('\t\t- Calculating R_rho for profile '+str(i).rjust(digits)+' of '+str(n_profiles), end="\r")
                        # Find the data from just this profile
                        df_this_pf = df[df['instrmt-prof_no']==pf]
                        # Get aCT and BSA values for this profile
                        these_aCT = df_this_pf.aCT.values
                        these_BSA = df_this_pf.BSA.values
                        # Remove null values
                        these_aCT = these_aCT[~np.isnan(these_aCT)]
                        these_BSA = these_BSA[~np.isnan(these_BSA)]
                        if len(these_aCT) > 0 and len(these_BSA) > 0:
                            # Find the slope of this profile in aT-BS space
                            # m, c = np.linalg.lstsq(np.array([BSs, np.ones(len(BSs))]).T, aTs, rcond=None)[0]
                            # Find the slope of the total least-squares of the points for this cluster
                            m, c, sd_m, sd_c = orthoregress(these_BSA, these_aCT)
                            # The lateral density ratio is the inverse of the slope
                            this_R_rho = 1/m
                            # Assign the rows of this entry to be this vertical density ratio
                            df.loc[df['instrmt-prof_no']==pf, 'R_rho'] = this_R_rho
                        else:
                            # Either these_aCT and/or these_BSA were empty
                            print('\t\t- Unable to calculate R_rho for',pf)
                            # Assign the rows of this entry to be this vertical density ratio
                            df.loc[df['instrmt-prof_no']==pf, 'R_rho'] = None
                    #
                #
            ## Filter each profile to certain ranges
            df = filter_profile_ranges(df, profile_filters, 'press', 'depth', 'sigma', iT_key='iT', CT_key='CT', PT_key='PT', SP_key='SP', SA_key='SA')
            # Add a notes column
            df['notes'] = ''
            ## Filter to just some cluster id's 
            if len(profile_filters.clstrs_to_plot) > 0:
                plt_these_clstrs = profile_filters.clstrs_to_plot
                # Find the cluster labels that exist in the dataframe
                cluster_numbers = np.unique(np.array(df['cluster'].values, dtype=int))
                #   Delete the noise point label "-1"
                cluster_numbers = np.delete(cluster_numbers, np.where(cluster_numbers == -1))
                n_clusters = int(len(cluster_numbers))
                # Sort clusters first?
                if sort_clstrs:
                    df = sort_clusters(df, cluster_numbers, ax=None)
                # Filter to just some clusters
                df = filter_to_these_clstrs(df, cluster_numbers, plt_these_clstrs)
            # Filter to just one entry per profile
            #   Turns out, if `vars_to_keep` doesn't have any multi-dimensional
            #       vars like 'temp' or 'salt', the resulting dataframe only has
            #       one row per profile automatically. Neat
            #   Actually, need to wait until later to do this anyway to allow for 
            #       variables that are per profile but involve clustering, like `pca_`
            # Remove missing data (actually, don't because it will get rid of all
            #   data if one var isn't filled, like dt_end often isn't)
            # for var in vars_to_keep:
            #     df = df[df[var].notnull()]
            # print(df)
            output_dfs.append(df)
        #
    #
    # print('\t- Almost done applying profile filters')
    if False:
        # Output a list of all the profiles this dataset contains
        for df in output_dfs:
            if True:
                list_of_pfs = list(set((df['prof_no'].values)))
            else:
                list_of_pfs = []
                for instrmt in list(set(df['instrmt'].values)):
                    print('this instrmt:',instrmt)
                    these_pfs = list(set(df[df['instrmt']==instrmt]['prof_no'].values))
                    print('these pfs:',these_pfs)
                    list_of_pfs.append(these_pfs)
            # print('\tThere are',len(list_of_pfs),'profiles in this dataframe:',list_of_pfs)
            # print(df)
        # exit(0)
    # print('\t- Done applying profile filters')
    return output_dfs

################################################################################

def take_m_avg(df, m_avg_win, vars_available):
    """
    Returns the same pandas dataframe, but with the filters provided applied to
    the data within

    df                  A pandas dataframe
    m_avg_win           The value of the moving average window in dbar
    vars_available      A list of variables available in the dataframe
    """
    print('\tIn take_m_avg(), m_avg_win:',m_avg_win)
    # Vertical resolution of the data
    res = 0.25
    # The number of data points across which to average
    c3 = int(m_avg_win/res)
    # Use the pandas `rolling` function to get the moving average
    #   center=True makes the first and last window/2 of the profiles are masked
    #   win_type='boxcar' uses a rectangular window shape
    #   on='press' means it will take `press` as the index column
    #   .mean() takes the average of the rolling
    df1 = df.rolling(window=int(c3), center=True, win_type='boxcar', on='press').mean()
    # Put the moving average profiles for temperature, salinity, and density into the dataset
    for var in vars_available:
        if var in ['iT','ma_iT','la_iT']:
            df['ma_iT'] = df1['iT']
            if 'la_iT' in vars_available:
                df['la_iT'] = df['iT'] - df['ma_iT']
        if var in ['CT','ma_CT','la_CT']:
            df['ma_CT'] = df1['CT']
            if 'la_CT' in vars_available:
                df['la_CT'] = df['CT'] - df['ma_CT']
        if var in ['PT','ma_PT','la_PT']:
            df['ma_PT'] = df1['PT']
            if 'la_PT' in vars_available:
                df['la_PT'] = df['PT'] - df['ma_PT']
        if var in ['SP','ma_SP','la_SP']:
            df['ma_SP'] = df1['SP']
            if 'la_SP' in vars_available:
                df['la_SP'] = df['SP'] - df['ma_SP']
        if var in ['SA','ma_SA','la_SA']:
            df['ma_SA'] = df1['SA']
            if 'la_SA' in vars_available:
                df['la_SA'] = df['SA'] - df['ma_SA']
        if var in ['sigma','ma_sigma','la_sigma']:
            df['ma_sigma'] = df1['sigma']
            if 'la_sigma' in vars_available:
                df['la_sigma'] = df['sigma'] - df['ma_sigma']
            #
        #
    #
    return df

################################################################################

def filter_profile_ranges(df, profile_filters, p_key, d_key, sig_key, iT_key=None, CT_key=None, PT_key=None, SP_key=None, SA_key=None):
    """
    Returns the same pandas dataframe, but with the filters provided applied to
    the data within

    df                  A pandas dataframe
    profile_filters     A custom Profile_Filters object that contains the filters to apply
    p_key               A string of the pressure variable to filter
    d_key               A string of the depth variable to filter
    sig_key             A string of the density anomaly variable to filter
    iT_key              A string of the in-situ temperature variable to filter
    CT_key              A string of the conservative temperature variable to filter
    PT_key              A string of the potential temperature variable to filter
    SP_key              A string of the practical salinity variable to filter
    SA_key              A string of the absolute salinity variable to filter
    """
    # print('\t- Filtering profile ranges')
    #   Filter to a certain longitude range
    if not isinstance(profile_filters.lon_range, type(None)):
        # print('\t\t-Applying lon_range filter')
        # Get endpoints of longitude range
        l_max = max(profile_filters.lon_range)
        l_min = min(profile_filters.lon_range)
        # Filter the data frame to the specified longitude range
        df = df[(df['lon'] < l_max) & (df['lon'] > l_min)]
    #   Filter to a certain latitude range
    if not isinstance(profile_filters.lat_range, type(None)):
        # print('\t\t-Applying lat_range filter')
        # Get endpoints of latitude range
        l_max = max(profile_filters.lat_range)
        l_min = min(profile_filters.lat_range)
        # Filter the data frame to the specified latitude range
        df = df[(df['lat'] < l_max) & (df['lat'] > l_min)]
    # Find number of profiles before
    # pf_list0 = list(set(df['prof_no'].values))
    #   Filter to a certain pressure range
    if not isinstance(profile_filters.p_range, type(None)):
        # print('\t\t-Applying p_range filter')
        # Get endpoints of pressure range
        p_max = max(profile_filters.p_range)
        p_min = min(profile_filters.p_range)
        # Filter the data frame to the specified pressure range
        df = df[(df[p_key] < p_max) & (df[p_key] > p_min)]
    #   Filter to a certain depth range
    if not isinstance(profile_filters.d_range, type(None)):
        # print('\t\t-Applying d_range filter')
        # Get endpoints of depth range
        d_max = max(profile_filters.d_range)
        d_min = min(profile_filters.d_range)
        # Filter the data frame to the specified depth range
        df = df[(df[d_key] < d_max) & (df[d_key] > d_min)]
    #   Filter to a certain density anomaly range
    if not isinstance(profile_filters.sig_range, type(None)):
        # print('\t\t-Applying d_range filter')
        # Get endpoints of depth range
        sig_max = max(profile_filters.sig_range)
        sig_min = min(profile_filters.sig_range)
        # Filter the data frame to the specified density anomaly range
        df = df[(df[sig_key] < sig_max) & (df[sig_key] > sig_min)]
    #   Filter to a certain in-situ temperature range
    if not isinstance(profile_filters.iT_range, type(None)):
        # print('\t\t-Applying iT_range filter')
        # Get endpoints of in-situ temperature range
        t_max = max(profile_filters.iT_range)
        t_min = min(profile_filters.iT_range)
        # Filter the data frame to the specified in-situ temperature range
        df = df[(df[iT_key] < t_max) & (df[iT_key] > t_min)]
    #   Filter to a certain conservative temperature range
    if not isinstance(profile_filters.CT_range, type(None)):
        # print('\t\t-Applying CT_range filter')
        # Get endpoints of conservative temperature range
        t_max = max(profile_filters.CT_range)
        t_min = min(profile_filters.CT_range)
        # Filter the data frame to the specified conservative temperature range
        df = df[(df[CT_key] < t_max) & (df[CT_key] > t_min)]
    #   Filter to a certain potential temperature range
    if not isinstance(profile_filters.PT_range, type(None)):
        # print('\t\t-Applying PT_range filter')
        # Get endpoints of potential temperature range
        t_max = max(profile_filters.PT_range)
        t_min = min(profile_filters.PT_range)
        # Filter the data frame to the specified potential temperature range
        df = df[(df[PT_key] < t_max) & (df[PT_key] > t_min)]
    #   Filter to a certain practical salinity range
    if not isinstance(profile_filters.SP_range, type(None)):
        # print('\t\t-Applying SP_range filter')
        # Get endpoints of practical salinity range
        s_max = max(profile_filters.SP_range)
        s_min = min(profile_filters.SP_range)
        # Filter the data frame to the specified practical salinity range
        df = df[(df[SP_key] < s_max) & (df[SP_key] > s_min)]
    #   Filter to a certain absolute salinity range
    if not isinstance(profile_filters.SA_range, type(None)):
        # print('\t\t-Applying SA_range filter')
        # Get endpoints of absolute salinity range
        s_max = max(profile_filters.SA_range)
        s_min = min(profile_filters.SA_range)
        # Filter the data frame to the specified absolute salinity range
        df = df[(df[SA_key] < s_max) & (df[SA_key] > s_min)]
    # Find number of profiles after
    # pf_list1 = list(set(df['prof_no'].values))
    # if len(pf_list0) != len(pf_list1):
    #     # Find the eliminated profiles
    #     elim_list = [x for x in pf_list0 if x not in pf_list1]
    #     # print('profile ranges filter, # of profiles before:',len(pf_list0),'after:',len(pf_list1),'difference:',len(pf_list0)-len(pf_list1),'-',elim_list)
    # print('\t\t-Done filtering profile ranges')
    return df

################################################################################

def calc_extra_vars(ds, vars_to_keep):
    """
    Takes in an xarray object and a list of variables and, if there are extra
    variables to calculate, it will add those to the xarray

    ds                  An xarray from the arr_of_ds of a custom Data_Set object
    vars_to_keep        A list of variables to keep for the analysis
    """
    # print('\t- Calculating extra variables')
    # Check for variables to calculate
    if 'aiT' in vars_to_keep:
        ds['aiT'] = ds['alpha'] * ds['iT']
    if 'aCT' in vars_to_keep:
        ds['aCT'] = ds['alpha'] * ds['CT']
    if 'aPT' in vars_to_keep:
        ds['aPT'] = ds['alpha_PT'] * ds['PT']
    if 'BSP' in vars_to_keep:
        ds['BSP'] = ds['beta'] * ds['SP']
    if 'BSt' in vars_to_keep:
        ds['BSt'] = ds['beta_PT'] * ds['SP']
    if 'BSA' in vars_to_keep:
        ds['BSA'] = ds['beta'] * ds['SA']
    # Check for variables with an underscore
    for this_var in vars_to_keep:
        if '_' in this_var and this_var not in ['prof_no','clst_prob']:
            # Split the prefix from the original variable (assumes an underscore split)
            split_var = this_var.split('_', 1)
            prefix = split_var[0]
            var = split_var[1]
            # print('prefix:',prefix,'- var:',var)
            if prefix == 'la':
                # Calculate the local anomaly of this variable
                ds[this_var] = ds[var] - ds['ma_'+var]
            #
        #
    return ds

################################################################################

def get_axis_labels(pp, var_attr_dicts):
    """
    Using the dictionaries of variable attributes, this adds the xlabel and ylabel
    attributes to the plot parameters object passed in (this replaced the
    `get_axis_label` function)

    pp                  A custom Plot_Parameters object
    var_attr_dicts      A list of dictionaries containing the attribute info
                            about all the variables in the netcdf file
    """
    # Get x axis labels
    if not isinstance(pp.x_vars, type(None)):
        if len(pp.x_vars) > 0:
            try:
                pp.xlabels[0] = var_attr_dicts[0][pp.x_vars[0]]['label']
            except:
                pp.xlabels[0] = get_axis_label(pp.x_vars[0], var_attr_dicts)
        else:
            pp.xlabels[0] = None
        if len(pp.x_vars) == 2:
            try:
                pp.xlabels[1] = var_attr_dicts[0][pp.x_vars[1]]['label']
            except:
                pp.xlabels[1] = get_axis_label(pp.x_vars[1], var_attr_dicts)
        else:
            pp.xlabels[1] = None
    else:
        pp.xlabels[0] = None
        pp.xlabels[1] = None
    # Get y axis labels
    if not isinstance(pp.y_vars, type(None)):
        if len(pp.y_vars) > 0:
            try:
                pp.ylabels[0] = var_attr_dicts[0][pp.y_vars[0]]['label']
            except:
                pp.ylabels[0] = get_axis_label(pp.y_vars[0], var_attr_dicts)
        else:
            pp.ylabels[0] = None
        if len(pp.y_vars) == 2:
            try:
                pp.ylabels[1] = var_attr_dicts[0][pp.y_vars[1]]['label']
            except:
                pp.ylabels[1] = get_axis_label(pp.y_vars[1], var_attr_dicts)
        else:
            pp.ylabels[1] = None
    else:
        pp.ylabels[0] = None
        pp.ylabels[1] = None
    # Get z axis labels
    if not isinstance(pp.z_vars, type(None)) and not isinstance(pp.z_vars[0], type(None)):
        if len(pp.z_vars) > 0:
            try:
                pp.zlabels[0] = var_attr_dicts[0][pp.z_vars[0]]['label']
            except:
                pp.zlabels[0] = get_axis_label(pp.z_vars[0], var_attr_dicts)
        else:
            pp.zlabels[0] = None
        if len(pp.z_vars) == 2:
            try:
                pp.zlabels[1] = var_attr_dicts[0][pp.z_vars[1]]['label']
            except:
                pp.zlabels[1] = get_axis_label(pp.z_vars[1], var_attr_dicts)
        else:
            pp.zlabels[1] = None
    else:
        pp.zlabels[0] = None
        pp.zlabels[1] = None
    # Get colormap label
    if not isinstance(pp.clr_map, type(None)):
        try:
            pp.clabel = var_attr_dicts[0][pp.clr_map]['label']
        except:
            pp.clabel = get_axis_label(pp.clr_map, var_attr_dicts)

    # Check for first differences of variables
    if any(pp.first_dfs):
        if pp.first_dfs[0]:
            if not isinstance(pp.xlabels[0], type(None)):
                pp.xlabels[0] = 'First difference in '+pp.xlabels[0]
            if not isinstance(pp.xlabels[1], type(None)):
                pp.xlabels[1] = 'First difference in '+pp.xlabels[1]
        if len(pp.first_dfs)==2 and pp.first_dfs[1]:
            if not isinstance(pp.ylabels[0], type(None)):
                pp.ylabels[0] = 'First difference in '+pp.ylabels[0]
            if not isinstance(pp.ylabels[1], type(None)):
                pp.ylabels[1] = 'First difference in '+pp.ylabels[1]
    if any(pp.finit_dfs):
        if pp.finit_dfs[0]:
            if not isinstance(pp.xlabels[0], type(None)):
                pp.xlabels[0] = r'Finite difference $\Delta$'+pp.xlabels[0]+'('+pp.ylabels[0]+')'
            if not isinstance(pp.xlabels[1], type(None)):
                pp.xlabels[1] = r'Finite difference $\Delta$'+pp.xlabels[1]+'('+pp.ylabels[0]+')'
        if len(pp.finit_dfs)==2 and pp.finit_dfs[1]:
            if not isinstance(pp.ylabels[0], type(None)):
                pp.ylabels[0] = r'First difference $\Delta$'+pp.ylabels[0]+'('+pp.xlabels[0]+')'
            if not isinstance(pp.ylabels[1], type(None)):
                pp.ylabels[1] = r'First difference $\Delta$'+pp.ylabels[1]+'('+pp.xlabels[0]+')'
    if pp.plot_type == 'map':
        pp.plot_scale = 'by_pf'
        pp.x_vars  = ['lon']
        pp.y_vars  = ['lat']
        pp.xlabels = [None, None]
        pp.ylabels = [None, None]
    #
    return pp

################################################################################

def get_axis_labels2(pp):
    """
    Adds in the x and y labels based on the plot variables in the 
    Plot_Parameters object

    pp              A custom Plot_Parameters object
    """
    # Build dictionary of axis labels
    ax_labels = {
                 'instrmt':r'Instrument',
                 'dt_start':r'Datetime of profile start',
                 'dt_end':r'Datetime of profile end',
                 'lat':r'Latitude ($^\circ$N)',
                 'lon':r'Longitude ($^\circ$E+)',
                 'hist':r'Occurrences',
                 'depth':r'Depth (m)',
                 'press':r'Pressure (dbar)',
                 'press_TC_max':r'$p(\Theta_{max})$ (dbar)',
                 'press_TC_min':r'$p(S_A\approx34.1)$ (dbar)',
                 'SA_TC_max':r'$S_A(\Theta_{max})$ (g/kg)',
                 'SA_TC_min':r'$S_A\approx34.1$ (g/kg)',
                 'CT_TC_max':r'$\Theta_{max}$ ($^\circ$C)',
                 'CT_TC_min':r'$\Theta(S_A\approx34.1)$ ($^\circ$C)',
                 'SA':r'$S_A$ (g/kg)',
                 'CT':r'$\Theta$ ($^\circ$C)',
                 'sigma':r'$\sigma_1$ (kg/m$^3$)',
                 'rho':r'$\rho$ (kg/m$^3$)',
                 'cp':r'$C_p$ (J kg$^{-1}$ K$^{-1}$)',
                 'FH':r'Heat Flux (W/m$^2$)',
                 'FH_cumul':r'Cumulative Heat Flux (W/m$^2$)',
                 'alpha':r'$\alpha$ (1/$^\circ$C)',
                 'beta':r'$\beta$ (kg/g)',
                 'aiT':r'$\alpha T$',
                 'aCT':r'$\alpha \Theta$',
                 'aPT':r'$\alpha \theta$',
                 'BSP':r'$\beta S_P$',
                 'BSt':r'$\beta_{PT} S_P$',
                 'BSA':r'$\beta S_A$',
                 'distance':r'Along-path distance (km)',
                 'm_pts':r'Minimum density threshold $m_{pts}$',
                 'DBCV':'Relative validity measure (DBCV)',
                 'n_clusters':'Number of clusters',
                 'cRL':r'Lateral density ratio $R_L$',
                 'cRl':r'Lateral density ratio $R_L$ with $\theta$',
                 'n_points':'Number of points in cluster',
                }
    # Build dictionary of axis labels without units
    ax_labels_no_units = {
                 'instrmt':r'Instrument',
                 'dt_start':r'Datetime of profile start',
                 'dt_end':r'Datetime of profile end',
                 'lat':r'Latitude',
                 'lon':r'Longitude',
                 'hist':r'Occurrences',
                 'depth':r'Depth',
                 'press':r'Pressure',
                 'press_TC_max':r'$p(\Theta_{max})$',
                 'press_TC_min':r'$p(S_A\approx34.1)$',
                 'SA_TC_max':r'$S_A(\Theta_{max})$',
                 'SA_TC_min':r'$S_A\approx34.1$',
                 'CT_TC_max':r'$\Theta_{max}$',
                 'CT_TC_min':r'$\Theta(S_A\approx34.1)$',
                 'SA':r'$S_A$',
                 'CT':r'$\Theta$',
                 'sigma':r'$\sigma_1$',
                 'rho':r'$\rho$',
                 'cp':r'$C_p$',
                 'FH':r'Heat Flux',
                 'FH_cumul':r'Cumulative Heat Flux',
                 'alpha':r'$\alpha$',
                 'beta':r'$\beta$',
                 'aiT':r'$\alpha T$',
                 'aCT':r'$\alpha \Theta$',
                 'aPT':r'$\alpha \theta$',
                 'BSP':r'$\beta S_P$',
                 'BSt':r'$\beta_{PT} S_P$',
                 'BSA':r'$\beta S_A$',
                 'distance':r'Along-path distance',
                 'm_pts':r'Minimum density threshold $m_{pts}$',
                 'DBCV':'Relative validity measure (DBCV)',
                 'n_clusters':'Number of clusters',
                 'cRL':r'Lateral density ratio $R_L$',
                 'cRl':r'Lateral density ratio $R_L$ with $\theta$',
                 'n_points':'Number of points in cluster',
                }
    # Build dictionary of axis labels
    ax_labels_units = {
                 'instrmt':r'',
                 'dt_start':r'',
                 'dt_end':r'',
                 'lat':r'$^\circ$N',
                 'lon':r'$^\circ$E+',
                 'hist':r'',
                 'depth':r'm',
                 'press':r'dbar',
                 'press_TC_max':r'dbar',
                 'press_TC_min':r'dbar',
                 'SA_TC_max':r'g/kg',
                 'SA_TC_min':r'g/kg',
                 'CT_TC_max':r'$^\circ$C',
                 'CT_TC_min':r'$^\circ$C',
                 'SA':r'g/kg',
                 'CT':r'$^\circ$C',
                 'sigma':r'kg/m$^3$',
                 'rho':r'kg/m$^3$',
                 'cp':r'J kg$^{-1}$ K$^{-1}$',
                 'FH':r'W/m$^2$',
                 'FH_cumul':r'W/m$^2$',
                 'alpha':r'1/$^\circ$C',
                 'beta':r'kg/g',
                 'aiT':r'',
                 'aCT':r'',
                 'aPT':r'',
                 'BSP':r'',
                 'BSt':r'',
                 'BSA':r'',
                 'distance':r'km',
                 'm_pts':r'',
                 'DBCV':'',
                 'n_clusters':'',
                 'cRL':r'',
                 'cRl':r'',
                 'n_points':'',
                }
    # Get x axis labels
    if not isinstance(pp.x_vars, type(None)):
        for i in range(len(pp.x_vars)):
            pp.xlabels[i] = make_var_label(pp.x_vars[i], ax_labels_units, ax_labels_no_units)
    else:
        pp.xlabels[0] = None
        pp.xlabels[1] = None
    # Get y axis labels
    if not isinstance(pp.y_vars, type(None)):
        for i in range(len(pp.y_vars)):
            pp.ylabels[i] = make_var_label(pp.y_vars[i], ax_labels_units, ax_labels_no_units)
    else:
        pp.ylabels[0] = None
        pp.ylabels[1] = None
    # Get colormap label
    if not isinstance(pp.clr_map, type(None)):
        try:
            pp.clabel = var_attr_dicts[0][pp.clr_map]['label']
        except:
            pp.clabel = make_var_label(pp.clr_map, ax_labels_units, ax_labels_no_units)
    #
    return pp

def str_lowercase(this_str):
    """
    Takes in a string and returns a version that is lower case only when the string 
    contains only letters a-z, which avoids changing the case of greek letters

    this_str        The string to be converted
    """
    # print('CHECKING:',this_str)
    # Check to make sure this_str is a string
    if isinstance(this_str, type('')) and len(this_str) > 0:
        # print('YES THIS IS A STRING')
        if all(x.isalpha() or x.isspace() for x in this_str):
            return this_str.lower()
        return this_str
    else:
        print('Warning: attempted use of str_lowercase() on datatype',type(this_str))
        return this_str

def make_var_label(var_key, ax_labels_units, ax_labels_no_units):
    """
    Takes in a variable key and returns the corresponding label

    var_key             A string of the variable key
    ax_labels_units     A dictionary of variable keys and their units
    ax_labels_no_units  A dictionary of variable keys and their labels
    """
    def get_units_strs(var_str):
        if len(ax_labels_units[var_str]) > 0:
            units_str = ' (' + ax_labels_units[var_str] + ')'
            units_str_per_year = ' (' + ax_labels_units[var_str] + '/year)'
        else:
            units_str = ''
            units_str_per_year = ' / year'
        return [units_str, units_str_per_year]
    polyfit2d_str = 'fit' # 'polyfit2d'
    # Check for certain modifications to variables,
    #   check longer strings first to avoid mismatching percnztrd_pcs_press
    # Check for percent non-zero trend in profile cluster span variables
    if 'percnztrd_pcs_' in var_key:
        # Take out the first 13 characters of the string to leave the original variable name
        var_str = var_key[13:]
        # return 'Percent non-zero trend in '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
        return 'Layer thickness trend (\%/year)'
    # Check for non-zero cluster average of profile cluster span variables
    if 'nzca_pcs_' in var_key:
        # Take out the first 9 characters of the string to leave the original variable name
        var_str = var_key[9:]
        # return 'NZCA/CS/P of '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
        # return 'Cluster average thickness in '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
        return 'Layer average thickness' + get_units_strs(var_str)[0]
    # Check for cluster average of profile cluster span variables
    if 'ca_pcs_' in var_key:
        # Take out the first 7 characters of the string to leave the original variable name
        var_str = var_key[7:]
        # return 'Profile cluster span of '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
        return 'CA/CS/P of '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
    # Check for non-zero trend in profile cluster span variables
    if 'nztrd_pcs_' in var_key:
        # Take out the first 10 characters of the string to leave the original variable name
        var_str = var_key[10:]
        # return 'NZ Trend in CS/P '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[1]
        return 'Trend in layer thickness in '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[1]
    # Check for trend in profile cluster span variables
    if 'trd_pcs_' in var_key:
        # Take out the first 8 characters of the string to leave the original variable name
        var_str = var_key[8:]
        return 'Trend in CS/P '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[1]
    # Check for profile cluster average variables
    if 'pca_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'CA/P of '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
        # return 'Profile cluster average of '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
    # Check for non-zero profile cluster span variables
    if 'nzpcs_' in var_key:
        # Take out the first 6 characters of the string to leave the original variable name
        var_str = var_key[6:]
        # Check for variables normalized by subtracting a polyfit2d
        if '-fit' in var_str:
            # Take out the first 3 characters of the string to leave the original variable name
            var_str = var_str[:-4]
            return 'NZCS/P of '+ ax_labels_no_units[var_str] + ' $-$ ' + polyfit2d_str
        elif 'mean' in var_str:
            # Take out the last 5 characters of the string to leave the original variable name
            var_str = var_str[:-5]
            return 'Mean layer thickness in '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
        else:
            # return 'NZCS/P of '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
            return 'Layer thickness in '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
        # return 'Non-zero profile cluster span of '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
    # Check for profile cluster span variables
    if 'pcs_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        # Check for variables normalized by subtracting a polyfit2d
        if '-fit' in var_str:
            # Take out the first 3 characters of the string to leave the original variable name
            var_str = var_str[:-4]
            return 'CS/P of '+ ax_labels_no_units[var_str] + ' $-$ ' + polyfit2d_str
        else:
            # return 'Profile cluster span of '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
            return 'CS/P of '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
    # Check for cluster mean-centered variables
    elif 'cmc_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'Cluster mean-centered '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
    # Check for local anomaly variables
    elif 'la_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[3:]
        return r"$\Theta'$ ($^\circ$C)"
        # return 'Local anomaly of '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
    # Check for local anomaly variables
    elif 'max_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'Maximum '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
    # Check for local anomaly variables
    elif 'min_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'Minimum '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
    # Check for cluster average variables
    elif 'ca_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[3:]
        if var_str == 'FH_cumul':
            return ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
        else:
            return 'Layer average '+ str_lowercase(ax_labels_no_units[var_str]) + get_units_strs(var_str)[0]
    # Check for cluster span variables
    elif 'cs_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[3:]
        return 'Cluster span of '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
    # Check for cluster standard deviation variables
    elif 'csd_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'Cluster standard deviation of '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
    # Check for cluster min/max variables
    elif 'cmm_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'Cluster min/max of '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
    # Check for normalized inter-cluster range variables
    elif 'nir_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[4:]
        if 'press' in var_str:
            return r'Normalized inter-cluster range $IR_{press}$'
        elif 'SA' in var_str:
            return r'Normalized inter-cluster range $IR_{S_A}$'
        elif 'SP' in var_str:
            return r'Normalized inter-cluster range $IR_{S_P}$'
        elif 'iT' in var_str:
            return r'Normalized inter-cluster range $IR_{T}$'
        elif 'CT' in var_str:
            return r'Normalized inter-cluster range $IR_{\Theta}$'
        if 'sigma' in var_str:
            return r'Normalized inter-cluster range $IR_{\sigma}$'
        # return r'Normalized inter-cluster range $IR$ of '+ ax_labels_no_units[var_str] + get_units_strs(var_str)[0]
    # Check for trend variables
    elif 'trd_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[4:]
        # Check whether also normalizing by subtracting a polyfit2d
        if '-fit' in var_str:
            # Take out the first 3 characters of the string to leave the original variable name
            var_str = var_str[:-4]
            return ax_labels_no_units[var_str] + ' $-$ ' + polyfit2d_str + ' trend' + get_units_strs(var_str)[1]
        else:  
            return ''+ ax_labels[var_str] + get_units_strs(var_str)[1]
    # Check for the polyfit2d of a variable
    elif 'fit_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return polyfit2d_str + ' of ' + ax_labels[var_str]
    # Check for variables normalized by subtracting a polyfit2d
    elif '-fit' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[:-4]
        return ax_labels_no_units[var_str] + ' $-$ ' + polyfit2d_str
    else:
        var_str = var_key
    if var_key in ax_labels_no_units.keys():
        return ax_labels_no_units[var_key] + get_units_strs(var_str)[0]
    else:
        return 'None'

################################################################################

def get_axis_label(var_key, var_attr_dicts):
    """
    Takes in a variable name and returns a nicely formatted string for an axis label

    var_key             A string of the variable name
    var_attr_dicts      A list of dictionaries of variable attributes
    """
    # Check for certain modifications to variables,
    #   check longer strings first to avoid mismatching
    # Check for profile cluster average variables
    if 'pca_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'CA/P of '+ var_attr_dicts[0][var_str]['label']
        # return 'Profile cluster average of '+ var_attr_dicts[0][var_str]['label']
    # Check for profile cluster span variables
    if 'pcs_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        # return 'Profile cluster span of '+ var_attr_dicts[0][var_str]['label']
        return 'CS/P of '+ var_attr_dicts[0][var_str]['label']
    # Check for cluster mean-centered variables
    elif 'cmc_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'Cluster mean-centered '+ var_attr_dicts[0][var_str]['label']
    # Check for local anomaly variables
    elif 'la_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[3:]
        return r"$\Theta'$ ($^\circ$C)"
        # return 'Local anomaly of '+ var_attr_dicts[0][var_str]['label']
    # Check for local anomaly variables
    elif 'max_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'Maximum '+ var_attr_dicts[0][var_str]['label']
    # Check for local anomaly variables
    elif 'min_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'Minimum '+ var_attr_dicts[0][var_str]['label']
    # Check for cluster average variables
    elif 'ca_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[3:]
        return 'Cluster average '+ var_attr_dicts[0][var_str]['label']
    # Check for cluster span variables
    elif 'cs_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[3:]
        return 'Cluster span of '+ var_attr_dicts[0][var_str]['label']
    # Check for cluster standard deviation variables
    elif 'csd_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'Cluster standard deviation of '+ var_attr_dicts[0][var_str]['label']
    # Check for cluster min/max variables
    elif 'cmm_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'Cluster min/max of '+ var_attr_dicts[0][var_str]['label']
    # Check for normalized inter-cluster range variables
    elif 'nir_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[4:]
        if 'press' in var_str:
            return r'Normalized inter-cluster range $IR_{press}$'
        elif 'SA' in var_str:
            return r'Normalized inter-cluster range $IR_{S_A}$'
        elif 'SP' in var_str:
            return r'Normalized inter-cluster range $IR_{S_P}$'
        elif 'iT' in var_str:
            return r'Normalized inter-cluster range $IR_{T}$'
        elif 'CT' in var_str:
            return r'Normalized inter-cluster range $IR_{\Theta}$'
        # return r'Normalized inter-cluster range $IR$ of '+ var_attr_dicts[0][var_str]['label']
    # Check for trend variables
    elif 'atrd_' in var_key:
        # Take out the first 5 characters of the string to leave the original variable name
        var_str = var_key[5:]
        # Check whether also normalizing by subtracting a polyfit2d
        if '-fit' in var_str:
            # Take out the last 4 characters of the string to leave the original variable name
            var_str = var_str[:-4]
            return 'Trend in '+ var_attr_dicts[0][var_str]['label'] + ' $-$ polyfit2d'
        else:  
            return 'Trend in '+ var_attr_dicts[0][var_str]['label'] + '/year'
    # Check for trend variables
    elif 'trd_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        # Check whether also normalizing by subtracting a polyfit2d
        if '-fit' in var_str:
            # Take out the last 4 characters of the string to leave the original variable name
            var_str = var_str[:-4]
            return 'Trend in '+ var_attr_dicts[0][var_str]['label'] + ' $-$ polyfit2d'
        else:  
            return 'Trend in '+ var_attr_dicts[0][var_str]['label'] + '/year'
    # Check for the polyfit2d of a variable
    elif 'fit_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'polyfit2d of ' + var_attr_dicts[0][var_str]['label']
    # Check for variables normalized by subtracting a polyfit2d
    elif '-fit' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[:-4]
        return var_attr_dicts[0][var_str]['label'] + ' $-$ polyfit2d'
    # Build dictionary of axis labels
    ax_labels = {
                 'instrmt':r'Instrument',
                 'hist':r'Occurrences',
                 'aiT':r'$\alpha T$',
                 'aCT':r'$\alpha \Theta$',
                 'aPT':r'$\alpha \theta$',
                 'BSP':r'$\beta S_P$',
                 'BSt':r'$\beta_{PT} S_P$',
                 'BSA':r'$\beta S_A$',
                 'distance':r'Along-path distance (km)',
                 'm_pts':r'Minimum density threshold $m_{pts}$',
                 'DBCV':'Relative validity measure (DBCV)',
                 'n_clusters':'Number of clusters',
                 'cRL':r'Lateral density ratio $R_L$',
                 'cRl':r'Lateral density ratio $R_L$ with $\theta$'
                }
    if var_key in ax_labels.keys():
        return ax_labels[var_key]
    else:
        return 'None'

################################################################################
# Admin summary functions ######################################################
################################################################################

def txt_summary(groups_to_summarize, filename=None):
    """
    Takes in a list of Analysis_Group objects. Analyzes and outputs summary info
    for each one.

    groups_to_plot      A list of Analysis_Group objects
    filename            The file to write the output to. If none, it will print
                            the output to the console
    """
    # Loop over all Analysis_Group objects
    i = 0
    lines = []
    for a_group in groups_to_summarize:
        i += 1
        # Concatonate all the pandas data frames together
        df = pd.concat(a_group.data_frames)
        # Find the total number of profiles
        df['prof_no-dt_start'] = df['instrmt'].astype(str)+' '+df['prof_no'].astype(str)
        n_profs2 = len(np.unique(np.array(df['prof_no-dt_start'], dtype=type(''))))
        n_profs = len(np.unique(np.array(df['dt_start'], dtype=type(''))))
        # Find the unique ITP numbers
        itp_nos = np.unique(np.array(df['instrmt'].values))
        # Put all of the lines together
        lines.append("Group "+str(i)+":")
        # lines.append("\t"+add_std_title(a_group))
        lines.append(print_global_variables(a_group.data_set))
        lines.append("\t\tVariables to keep:")
        lines.append("\t\t"+str(a_group.vars_to_keep))
        lines.append("\t\tNumber of ITPs: ")
        lines.append("\t\t\t"+str(len(itp_nos))+': '+str(itp_nos))
        lines.append("\t\tNumber of profiles: ")
        lines.append("\t\t\t"+str(n_profs)+' / '+str(n_profs2))
        lines.append("\t\tNumber of data points: ")
        lines.append("\t\t\t"+str(len(df)))
        # Find the ranges of the variables available
        # print('working on ranges')
        # var_attr_dict = a_group.data_set.var_attr_dicts[0]
        # for var in a_group.vars_to_keep:
        #     if var in var_attr_dict.keys():
        #         lines.append(report_range(df, var, var_attr_dict[var]))
            #
        #
        # Check for SDA variables
        if 'SDA_n_ml_dc' in a_group.vars_to_keep and 'SDA_n_ml_sf' in a_group.vars_to_keep and 'SDA_n_gl_dc' in a_group.vars_to_keep and 'SDA_n_gl_sf' in a_group.vars_to_keep:# and 'SDA_ml_press' in a_group.vars_to_keep and 'SDA_gl_h' in a_group.vars_to_keep:
            print('working on SDA part')
            lines.append("\n\t\t-- SDA output summary --")
            lines.append(summarize_SDA_output(df, n_profs))
        lines.append("\n")
    #
    if filename != None:
        print('Writing summary file to',filename)
        with open(filename, 'w') as f:
            f.write('\n'.join(lines))
    else:
        for line in lines:
            print(line)

################################################################################

def print_profile_filters(pfs):
    """
    Returns a human readable string of all the profile filters in the given object

    pfs         A Profile_Filters object
    """
    return_string = ''
    if not isinstance(pfs.p_range, type(None)): 
        return_string += ('Pressure range: ['+str(pfs.p_range[0])+', '+str(pfs.p_range[1])+'] ')
    if not isinstance(pfs.d_range, type(None)): 
        return_string += ('Depth range: ['+str(pfs.d_range[0])+', '+str(pfs.d_range[1])+'] ')
    if not isinstance(pfs.iT_range, type(None)): 
        return_string += ('iT range: ['+str(pfs.iT_range[0])+', '+str(pfs.iT_range[1])+'] ')
    if not isinstance(pfs.CT_range, type(None)): 
        return_string += ('CT range: ['+str(pfs.CT_range[0])+', '+str(pfs.CT_range[1])+'] ')
    if not isinstance(pfs.PT_range, type(None)): 
        return_string += ('PT range: ['+str(pfs.PT_range[0])+', '+str(pfs.PT_range[1])+'] ')
    if not isinstance(pfs.SP_range, type(None)): 
        return_string += ('SP range: ['+str(pfs.SP_range[0])+', '+str(pfs.SP_range[1])+'] ')
    if not isinstance(pfs.SA_range, type(None)): 
        return_string += ('SA range: ['+str(pfs.SA_range[0])+', '+str(pfs.SA_range[1])+'] ')
    if not isinstance(pfs.lon_range, type(None)): 
        return_string += ('Longitude range: ['+str(pfs.lon_range[0])+', '+str(pfs.lon_range[1])+'] ')
    if not isinstance(pfs.lat_range, type(None)): 
        return_string += ('Latitude range: ['+str(pfs.lat_range[0])+', '+str(pfs.lat_range[1])+'] ')
    if pfs.lt_pTC_max: 
        return_string += ('Pressures less than p(TC_max) ')
    if pfs.subsample: 
        return_string += ('Subsampled ')
    if not isinstance(pfs.regrid_TS, type(None)): 
        return_string += ('Regrid: d'+str(pfs.regrid_TS[0])+'='+str(pfs.regrid_TS[1])+', d'+str(pfs.regrid_TS[2])+'='+str(pfs.regrid_TS[3])+' ')
    if not isinstance(pfs.m_avg_win, type(None)): 
        return_string += ('Moving average window: ['+str(pfs.m_avg_win)+' ')
    return return_string

################################################################################

def report_range(df, key, name_and_units):
    """
    Reports the range of the key value given for the given dataframe

    df          A pandas data frame
    key         A string of the column to search in the data frame
    name_and_units      A dictionary containing:
        name_str    A string of the long version of the column data type
        units       A string of the units of the data type
    """
    name_str  = name_and_units['long_name']
    try:
        units = name_and_units['units']
    except:
        # The above will fail if no units are given, which currently happens for
        #   the time variables in the GDS netcdfs because I needed to sort them
        print('Missing units for:',key)
        units = 'YYYY-MM-DD HH:MM:SS'
    # Find this range, making sure to skip null (nan) values
    this_arr = df[df[key].notnull()][key]
    if len(this_arr) > 1:
        this_min = min(this_arr)
        this_max = max(this_arr)
        if key in ['source', 'instrmt', 'dt_start', 'dt_end']:
            return "\t\t"+key+': '+name_str+" \n\t\t\trange: \n\t\t\t"+str(this_min)+" to "+str(this_max)+" "+units
        else:#if key != 'dt_start' and key != 'dt_end':
            # This line formats all the numbers to have just 2 digits after the decimal point
            return "\t\t"+key+': '+name_str+" \n\t\t\trange: %.2f"%(this_max-this_min)+" "+units+"\n\t\t\t%.2f"%this_min+" to %.2f"%this_max+" "+units
            # This line outputs the numbers with all available digits after the decimal point
            # return "\t\t"+name_str+" range: "+str(this_max-this_min)+" "+units+"\n\t\t\t"+str(this_min)+" to "+str(this_max)+" "+units
    else:
        return "\t\t"+key+': '+name_str+" \n\t\t\trange: N/A \n\t\t\tN/A"

################################################################################

def print_global_variables(Dataset):
    """
    Prints out the global variables of the data sources in a Data_Set object

    Dataset         A custom Data_Set object
    """
    lines = ''
    for ds in Dataset.arr_of_ds:
        for attr in ds.attrs:
            # print('\t'+str(attr)+': '+str(ds.attrs[attr]))
            if attr in ['Source', 'Instrument']:
                lines = lines+'\t'+str(attr)+': '+str(ds.attrs[attr])+'\n'
            elif attr in ['Creation date', 'Last modified', 'Last modification', 'Sub-sample scheme']:
                print('\t'+attr+': '+ds.attrs[attr])
            else:
                lines = lines+'\t'+attr+': '+ds.attrs[attr]+'\n'
    return lines

################################################################################

def find_max_distance(groups_to_analyze):
    """
    Reports the maximum distance between any two profiles in the given dataframe

    groups_to_analyze   A list of Analysis_Group objects
                        Each Analysis_Group contains a list of dataframes
    """
    for ag in groups_to_analyze:
        df = pd.concat(ag.data_frames)
        # Drop duplicates in lat/lon to have just one row per profile
        df.drop_duplicates(subset=['lon','lat','instrmt'], keep='first', inplace=True)
        # Get source and instrument for this df
        this_source = df['source'].values[0]
        this_instrmt = df['instrmt'].values[0]
        this_title = add_std_title(ag)
        # Get arrays of values
        prof_nos = df['prof_no'].values
        lon_vals = df['lon'].values
        lat_vals = df['lat'].values
        try:
            dt_starts = [datetime.strptime(date, r'%Y-%m-%d %H:%M:%S').date() for date in df['dt_start'].values]
        except:
            dt_starts = [datetime.strptime(date, r'%Y/%m/%d').date() for date in df['dt_start'].values]
        # Find the maximum and minimum datetimes
        dt_max_i = np.argmax(dt_starts)
        dt_max = dt_starts[dt_max_i] # max(dt_starts)
        dt_min_i = np.argmin(dt_starts)
        dt_min = dt_starts[dt_min_i] # min(dt_starts)
        dt_span = dt_max - dt_min
        print('For ' + this_title + ', dt_span: ' + str(dt_span) + ' between profiles ' + str(prof_nos[dt_min_i]) + ' on ' + str(dt_min) + ' and ' + str(prof_nos[dt_max_i]) + ' on ' + str(dt_max))
        # Find the number of profiles
        n_pfs = len(prof_nos)
        # Placeholders to store the largest span between any two profiles
        max_span = 0
        ms_i = None
        ms_i_latlon = None
        ms_j = None
        ms_j_latlon = None
        # Nested loop over the profiles, only the upper triangle
        #   Don't need to double-count, comparing profiles twice
        for i in range(n_pfs):
            for j in range(i+1,n_pfs):
                i_lat_lon = (lat_vals[i], lon_vals[i])
                j_lat_lon = (lat_vals[j], lon_vals[j])
                this_span = geodesic(i_lat_lon, j_lat_lon).km
                if this_span > max_span:
                    max_span = this_span
                    ms_i = prof_nos[i] 
                    ms_i_latlon = i_lat_lon
                    ms_j = prof_nos[j]
                    ms_j_latlon = j_lat_lon
                # print('i:',i,'j:',j,'i_pf:',prof_nos[i],'j_pf:',prof_nos[j],'this_span:',this_span,'max_span:',max_span)
            #
        #
        print_string = r'For %s, max_span: %.3f km between profiles %s at %.3f N, %.3f E and %s at %.3f N, %.3f E'%(this_title, max_span, ms_i, ms_i_latlon[0], ms_i_latlon[1], ms_j, ms_j_latlon[0], ms_j_latlon[1])
        print(print_string)

################################################################################
# Admin stats functions ########################################################
################################################################################

def cluster_stats(group_to_summarize, stat_vars=['SA'], filename=None):
    """
    Takes in a list of Analysis_Group objects. Analyzes and outputs summary info
    for each one.

    group_to_summarize      An Analysis_Group object
    stat_vars               A list of variables for which to compute the stats
    filename                The file to write the output to. If none, it will print
                                the output to the console
    """
    # Import statistics
    from scipy import stats
    import csv
    # Remove `cluster` from stat_vars
    stat_vars.remove('cluster')
    lines = []
    # Find the dataframes
    df = pd.concat(group_to_summarize.data_frames)
    # Check for cluster based variables
    new_cl_vars = []
    for var in stat_vars:
        if var in clstr_vars:
            new_cl_vars.append(var)
            stat_vars.remove(var)
    # Check for fit variables
    new_fit_vars = []
    for var in stat_vars:
        if 'fit' in var:
            new_fit_vars.append(var)
            stat_vars.remove(var)
    #
    print('stat_vars:',stat_vars)
    print('new_cl_vars:',new_cl_vars)
    print('new_fit_vars:',new_fit_vars)
    exit(0)
    # Create the header row
    header_str = 'cluster,len'
    for var in stat_vars:
        # header_str += ','+var+'_mean,'+var+'_std'
        header_str += ','+var+'_mean'
    if len(new_cl_vars) > 0:
        for var in new_cl_vars:
            header_str += ','+var
    lines.append(header_str)
    # Find the cluster labels that exist in the dataframe
    cluster_numbers = np.unique(np.array(df['cluster'].values, dtype=int))
    #   Delete the noise point label "-1"
    cluster_numbers = np.delete(cluster_numbers, np.where(cluster_numbers == -1))
    # Calculate new cluster variables
    if len(new_cl_vars) > 0:
        df = calc_extra_cl_vars(df, new_cl_vars)
    # Loop across all clusters
    for i in cluster_numbers:
        # Add the cluster ID
        this_line = str(i)
        # Find the data from this cluster
        df_this_cluster = df[df['cluster']==i]
        # Add the number of points in this cluster
        this_line += ','+str(len(df_this_cluster))
        # Check whether to calculate fit variables
        if len(new_fit_vars) > 0:
            df_this_cluster = ahf.calc_fit_vars(df_this_cluster, new_fit_vars, ['lon','lat'])
        # Add mean and standard deviation of each var
        for var in stat_vars:
            this_line += ','+str(df_this_cluster[var].mean())#+','+str(df_this_cluster[var].std())
            #
        lines.append(this_line)
    #
    if filename != None:
        filename = 'outputs/' + filename
        print('Writing summary file to',filename)
        with open(filename, 'w') as f:
            f.write('\n'.join(lines))
    else:
        for line in lines:
            print(line)

################################################################################
# Admin plotting functions #####################################################
################################################################################

def make_figure(groups_to_plot, filename=None, use_same_x_axis=None, use_same_y_axis=None, row_col_list=None):
    """
    Takes in a list of Analysis_Group objects, one for each subplot. Determines
    the needed arrangement of subplots, then passes one Analysis_Group object to
    each axis for plotting

    groups_to_plot  A list of Analysis_Group objects, one for each subplot
                    Each Analysis_Group contains the info to create each subplot
    filename        The filename in which to save the figure
                        only accepts .png or .pickle filenames
    use_same_x_axis     True/False whether to force subplots to share x axis
                        ranges if they have the same variable
    use_same_y_axis     True/False whether to force subplots to share y axis
                        ranges if they have the same variable
    row_col_list        [rows, cols, f_ratio, f_size], if none given, will use 
                        the defaults specified below in n_row_col_dict
    """
    print('- Making the figure')
    # Define number of rows and columns based on number of subplots
    #   key: number of subplots, value: (rows, cols, f_ratio, f_size)
    n_row_col_dict = {'1':[1,1, 0.8, 1.25], '2':[1,2, 0.5, 1.25], '2.5':[2,1, 0.8, 1.25],
                      '3':[1,3, 0.2, 1.50], '4':[2,2, 0.7, 2.00],
                      '5':[2,3, 0.5, 2.00], '6':[2,3, 0.6, 2.00],
                      '7':[2,4, 0.4, 2.00], '8':[2,4, 0.4, 2.00],
                      '9':[3,3, 0.8, 2.00]}
    # Figure out what layout of subplots to make
    n_subplots = len(groups_to_plot)
    x_keys = []
    y_keys = []
    z_keys = []
    for group in groups_to_plot:
        pp = group.plt_params
        u_x_vars = np.unique(pp.x_vars)
        u_x_vars = np.delete(u_x_vars, np.where(u_x_vars == None))
        u_y_vars = np.unique(pp.y_vars)
        u_y_vars = np.delete(u_y_vars, np.where(u_y_vars == None))
        u_z_vars = np.unique(pp.z_vars)
        u_z_vars = np.delete(u_z_vars, np.where(u_z_vars == None))
        if len(u_x_vars) > 0:
            for var in u_x_vars:
                x_keys.append(var)
        if len(u_y_vars) > 0:
            for var in u_y_vars:
                y_keys.append(var)
        if len(u_z_vars) > 0:
            for var in u_z_vars:
                z_keys.append(var)
            #
        #
    #
    if len(x_keys) > 1 and isinstance(use_same_x_axis, type(None)):
        # If all the x_vars are the same, share the x axis between subplots
        if all(x == x_keys[0] for x in x_keys):
            use_same_x_axis = True
            print('\t- Set share_x_axis to True')
        else:
            use_same_x_axis = False
        #
    else:
        use_same_x_axis = False
    if len(y_keys) > 1 and isinstance(use_same_y_axis, type(None)):
        # If all the y_vars are the same, share the y axis between subplots
        if all(y == y_keys[0] for y in y_keys):
            use_same_y_axis = True
            print('\t- Set share_y_axis to True')
        else:
            use_same_y_axis = False
        #
    else:
        use_same_y_axis = False
    if len(z_keys) > 1 and isinstance(use_same_z_axis, type(None)):
        # If all the z_vars are the same, share the z axis between subplots
        if all(z == z_keys[0] for z in z_keys):
            use_same_z_axis = True
            print('\t- Set share_z_axis to True')
        else:
            use_same_z_axis = False
        #
    else:
        use_same_z_axis = False
    tight_layout_h_pad = 1.0 #None
    tight_layout_w_pad = 0 #None
    subplot_label_x = -0.13
    subplot_label_y = -0.18
    if n_subplots == 1:
        if isinstance(row_col_list, type(None)):
            rows, cols, f_ratio, f_size = n_row_col_dict[str(n_subplots)]
        else:
            rows, cols, f_ratio, f_size = row_col_list
        n_subplots = int(np.floor(n_subplots))
        fig, ax = set_fig_axes([1], [1], fig_ratio=f_ratio, fig_size=f_size)
        print('- Subplot 1/1')
        xlabel, ylabel, zlabel, plt_title, ax, invert_y_axis = make_subplot(ax, groups_to_plot[0], fig, 111)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if not isinstance(zlabel, type(None)):
            ax.set_zlabel(zlabel)
        # Invert y-axis if specified
        if invert_y_axis:
            ax.invert_yaxis()
            print('\t- Inverting y-axis')
        # If axes limits given, limit axes
        if not isinstance(pp.ax_lims, type(None)):
            try:
                ax.set_xlim(pp.ax_lims['x_lims'])
                print('\t- Set x_lims to',pp.ax_lims['x_lims'])
            except:
                foo = 2
            try:
                ax.set_ylim(pp.ax_lims['y_lims'])
                print('\t- Set y_lims to',pp.ax_lims['y_lims'])
            except:
                foo = 2
            try:
                ax.set_zlim(pp.ax_lims['z_lims'])
                print('\t- Set z_lims to',pp.ax_lims['z_lims'])
            except:
                foo = 2
            # 
        # If called for, add a grid to the plot by default
        print('\t- Adding grid lines:',pp.add_grid)
        if pp.add_grid:
            ax.grid(color=std_clr, linestyle='--', alpha=grid_alpha)
        ax.set_title(plt_title)
    elif n_subplots > 1 and n_subplots < 10:
        if isinstance(row_col_list, type(None)):
            rows, cols, f_ratio, f_size = n_row_col_dict[str(n_subplots)]
        else:
            rows, cols, f_ratio, f_size = row_col_list
        n_subplots = int(np.floor(n_subplots))
        fig, axes = set_fig_axes([1]*rows, [1]*cols, fig_ratio=f_ratio, fig_size=f_size, share_x_axis=use_same_x_axis, share_y_axis=use_same_y_axis, share_z_axis=use_same_z_axis)
        # Create a blank array for the axis labels
        xlabels = [[None for x in range(cols)] for y in range(rows)]
        ylabels = [[None for x in range(cols)] for y in range(rows)]
        for i in range(n_subplots):
            print('- Subplot '+string.ascii_lowercase[i])
            if rows > 1 and cols > 1:
                i_ax = (i//cols,i%cols)
            else:
                i_ax = i
            ax_pos = int(str(rows)+str(cols)+str(i+1))
            xlabel, ylabel, zlabel, plt_title, ax, invert_y_axis = make_subplot(axes[i_ax], groups_to_plot[i], fig, ax_pos)
            xlabels[i//cols][i%cols] = xlabel
            ylabels[i//cols][i%cols] = ylabel
            # Find the plot type
            plot_type = groups_to_plot[i].plt_params.plot_type
            if use_same_x_axis:
                # If more than one row and on the top row
                if rows > 1 and i < cols:# == 0:
                    subplot_label_y = -0.1
                    tight_layout_h_pad = -0.5
                    ax.set_title(plt_title)
                # If in the bottom row
                elif i >= n_subplots-cols:
                    subplot_label_y = -0.18
                    ax.set_xlabel(xlabel)
            else:
                ax.set_title(plt_title)
                # If on the top row
                if i < cols:
                    ax.set_xlabel(xlabel)
                else:
                    # If this has the same label as the plot above
                    if xlabel == xlabels[i//cols-1][i%cols]:
                        # Delete the xlabel from the plot above
                        axes[(i//cols-1,i%cols)].set_xlabel(None)
                    # Add in xlabel to this plot
                    ax.set_xlabel(xlabel)
                if plot_type == 'map':
                    subplot_label_y = 0
                    # If not on the top row
                    if i >= cols:
                        tight_layout_h_pad = 2.0
            ax.set_title(plt_title)
            if use_same_y_axis:
                # If in the far left column or just one column
                if i%cols == 0 or cols == 1:
                    tight_layout_w_pad = -1
                    ax.set_ylabel(ylabel)
                # Invert y-axis if specified
                if i == 0 and invert_y_axis:
                    ax.invert_yaxis()
                    print('\t- Inverting y-axis')
            else:
                # If in the far left column
                if i%cols == 0:
                    ax.set_ylabel(ylabel)
                else:
                    # If this has the same label as the plot to the left
                    if ylabel == ylabels[i//cols][i%cols-1]:
                        # Don't add ylabel
                        foo = 2
                    else:
                        ax.set_ylabel(ylabel)
                if invert_y_axis:
                    ax.invert_yaxis()
                    print('\t- Inverting y-axis-')
            if plot_type == 'map':
                subplot_label_y = 0
            # Label subplots a, b, c, ...
            print('\t- Adding subplot label with offsets:',subplot_label_x, subplot_label_y)
            ax.text(subplot_label_x, subplot_label_y, r'\textbf{('+string.ascii_lowercase[i]+')}', transform=ax.transAxes, size=font_size_lgnd, fontweight='bold')
            # If axes limits given, limit axes
            this_ax_pp = groups_to_plot[i].plt_params
            if not isinstance(this_ax_pp.ax_lims, type(None)):
                try:
                    ax.set_xlim(this_ax_pp.ax_lims['x_lims'])
                    print('\t- Set x_lims to',this_ax_pp.ax_lims['x_lims'])
                except:
                    foo = 2
                try:
                    ax.set_ylim(this_ax_pp.ax_lims['y_lims'])
                    print('\t- Set y_lims to',this_ax_pp.ax_lims['y_lims'])
                except:
                    foo = 2
                try:
                    ax.set_zlim(this_ax_pp.ax_lims['z_lims'])
                    print('\t- Set z_lims to',this_ax_pp.ax_lims['z_lims'])
                except:
                    foo = 2
                #
            # If called for, add a grid to the plot by default
            print('\t- Adding grid lines:',this_ax_pp.add_grid)
            if this_ax_pp.add_grid:
                ax.grid(color=std_clr, linestyle='--', alpha=grid_alpha)
        # Turn off unused axes
        if n_subplots < (rows*cols):
            for i in range(rows*cols-1, n_subplots-1, -1):
                axes[i//cols,i%cols].set_axis_off()
    else:
        print('Too many subplots')
        exit(0)
    #
    plt.tight_layout(pad=0.4, h_pad=tight_layout_h_pad, w_pad=tight_layout_w_pad)
    #
    if filename != None:
        print('- Saving figure to outputs/'+filename)
        if '.png' in filename or '.pdf' in filename:
            plt.savefig('outputs/'+filename, dpi=png_dpi, transparent=transparent_figures)
        elif '.pickle' in filename:
            pl.dump(fig, open('outputs/'+filename, 'wb'))
        else:
            print('File extension not recognized in',filename)
    else:
        print('- Displaying figure')
        plt.show()

################################################################################
# Formatting plotting functions ################################################
################################################################################

def set_fig_axes(heights, widths, fig_ratio=0.5, fig_size=1, share_x_axis=None, share_y_axis=None, share_z_axis=None, prjctn=None):
    """
    Creates fig and axes objects based on desired heights and widths of subplots
    Ex: if widths=[1,5], there will be 2 columns, the 1st 1/5 the width of the 2nd

    heights      array of integers for subplot height ratios, len=rows
    widths       array of integers for subplot width  ratios, len=cols
    fig_ratio    ratio of height to width of overall figure
    fig_size     size scale factor, 1 changes nothing, 2 makes it very big
    share_x_axis bool whether the subplots should share their x axes
    share_y_axis bool whether the subplots should share their y axes
    share_z_axis bool whether the subplots should share their z axes
    prjctn       projection type for the subplots
    """
    # Set aspect ratio of overall figure
    w, h = mpl.figure.figaspect(fig_ratio)
    # Find rows and columns of subplots
    rows = len(heights)
    cols = len(widths)
    # This dictionary makes each subplot have the desired ratios
    # The length of heights will be nrows and likewise len(widths)=ncols
    plot_ratios = {'height_ratios': heights,
                   'width_ratios': widths}
    # Determine whether to share x or y axes
    if share_x_axis == None and share_y_axis == None:
        if rows == 1 and cols != 1: # if only one row, share y axis
            share_x_axis = False
            share_y_axis = True
        elif rows != 1 and cols == 1: # if only one column, share x axis
            share_x_axis = True
            share_y_axis = False
        else:                       # otherwise, forget about it
            share_x_axis = False
            share_y_axis = False
    elif share_y_axis == None:
        share_y_axis = False
        print('\tSet share_y_axis to', share_y_axis)
    elif share_x_axis == None:
        share_x_axis = False
        print('\tSet share_x_axis to', share_x_axis)
    # Set ratios by passing dictionary as 'gridspec_kw', and share y axis
    fig, axes = plt.subplots(figsize=(w*fig_size,h*fig_size), nrows=rows, ncols=cols, gridspec_kw=plot_ratios, sharex=share_x_axis, sharey=share_y_axis, subplot_kw=dict(projection=prjctn))
    if rows != 1 or cols != 1:
        fixed_width = 16
        fig.set_size_inches(fixed_width, fixed_width*h/w)
    # Set ticklabel format for all axes
    if (rows+cols)>2:
        for ax in axes.flatten():
            ax.ticklabel_format(style='sci', scilimits=sci_lims, useMathText=True)
    else:
        axes.ticklabel_format(style='sci', scilimits=sci_lims, useMathText=True)
    # Turn off excess ticks if the axes are shared
    if share_x_axis and cols == 1:
        print('\t- Turning off excess x axes')
        for ax in axes.flatten()[0:-2]:
            plt.setp(ax.get_xticklabels(), visible=False)
            # plt.setp(ax.get_xlabel(), visible=False)
    return fig, axes

################################################################################

def add_std_title(a_group):
    """
    Adds in standard information to this subplot's title, as appropriate

    a_group         A Analysis_Group object containing the info to create a subplot
    """
    # Check for an override title
    if not isinstance(a_group.plot_title, type(None)):
        return a_group.plot_title
    else:
        plt_title = ''
    # Determine a prefix, if any, for the plot title
    pfs = a_group.profile_filters
    if pfs.subsample:
        # Get a string of the subsampling scheme
        ss_scheme = np.unique(np.array(a_group.data_frames[0]['ss_scheme']))[0]
        plt_title = 'Subsampled w/ ' + ss_scheme + ':'
    sources_dict = a_group.data_set.sources_dict
    # Get a list of the sources in the plot
    sources = list(sources_dict.keys())
    # Get just the unique sources
    sources = np.unique(sources)
    if len(sources) > 3:
        plt_title += 'Many sources'
    else:
        # Note all the data sources in the title
        plt_title += sources[0].replace('_',' ')
        for i in range(1,len(sources)):
            plt_title += ', '
            plt_title += sources[i].replace('_',' ')
    #
    return plt_title

################################################################################

def add_std_legend(ax, data, x_key, lgd_str=' points', mark_LHW_AW=False, lgd_loc=None):
    """
    Adds in standard information to this subplot's legend, as appropriate

    ax          The axis on which to add the legend
    data        A pandas data frame which is plotted on this axis
    x_key       The string of the column name for the x data from the dataframe
    lgd_str     A string to append to the legend label
    mark_LHW_AW A boolean to indicate whether to add LHW and AW markers
    lgd_loc     The location of the legend, taking the value of `legend` from the Plot_Parameters
    """
    lgnd_hndls = []
    # Add legend to report the total number of points and notes on the data
    n_pts_patch  = mpl.patches.Patch(color='none', label=str(len(data[x_key]))+lgd_str)
    lgnd_hndls.append(n_pts_patch)
    # notes_string = ''.join(data.notes.unique())
    notes_string = ''
    # Only add the notes_string if it contains something
    if len(notes_string) > 1:
        notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
        lgnd_hndls.append(notes_patch)
    if mark_LHW_AW:
        # Get existing legend labels
        old_hndls, foo = ax.get_legend_handles_labels()
        # Add legend entries for LHW and AW
        lgnd_hndls = []
        lgnd_hndls.append(mpl.lines.Line2D([],[],color=LHW_clr, label='LHW', marker=LHW_mrk, markersize=10, fillstyle='bottom', markerfacecoloralt=LHW_facealtclr, markeredgecolor=LHW_edgeclr, markeredgewidth=0.5, linewidth=0))
        lgnd_hndls.append(mpl.lines.Line2D([],[],color=AW_clr, label='AW', marker=AW_mrk, markersize=10, fillstyle='top', markerfacecoloralt=AW_facealtclr, markeredgecolor=AW_edgeclr, markeredgewidth=0.5, linewidth=0))
        # Add in old labels
        for hndl in old_hndls:
            lgnd_hndls.append(hndl)
    if isinstance(lgd_loc, type('string')):
        print('\t- Adding standard legend in:',lgd_loc)
        lgnd = ax.legend(handles=lgnd_hndls, loc=lgd_loc)
    else:
        lgnd = ax.legend(handles=lgnd_hndls)

################################################################################

def get_var_color(var):
    """
    Takes in the variable name and for the variable to colormap

    var         A string of the variable name to get the color for
    """
    # Check whether taking first or finite differences
    if 'd_' in var:
        var = var[2:]
    # Build dictionary of available colors
    var_clrs = {
            'aiT':'salmon',
            'aCT':'salmon',
            'aPT':'salmon',
            'BSP':'darkturquoise',
            'BSt':'darkturquoise',
            'BSA':'darkturquoise'
            }
    if var in ['entry', 'prof_no', 'dt_start', 'dt_end', 'lon', 'lat', 'press', 'depth', 'og_press', 'og_depth', 'press_TC_max', 'press_TC_min']:
        return std_clr
    elif var in ['iT', 'CT', 'PT', 'alpha']:
        return 'indianred' # 'lightcoral'
    elif var in ['SP', 'SA', 'beta']:
        return 'cornflowerblue'
    elif var in ['sigma']:
        return 'mediumpurple'
    elif var in ['CT_TC_max', 'CT_TC_min', 'ma_iT', 'ma_CT', 'ma_PT', 'la_iT', 'la_CT', 'la_PT']:
        return 'tab:red'
    elif var in ['SA_TC_max', 'SA_TC_min', 'ma_SP', 'ma_SA', 'la_SP', 'la_SA']:
        return 'tab:blue'
    elif var in ['ma_sigma', 'la_sigma', 'sigma_TC_max', 'sigma_TC_min']:
        return 'tab:purple'
    if var in var_clrs.keys():
        return var_clrs[var]
    else:
        return std_clr

################################################################################

def get_color_map(cmap_var):
    """
    Takes in the variable name and for the variable to colormap

    cmap_var    A string of the variable name for the colormap
    """
    # print('in get_color_map()')
    # print('cmap_var:',cmap_var)
    # For density histogram
    if cmap_var == 'density_hist':
        if dark_mode:
            return cm.batlowK
        else:
            return cm.batlowW.reversed()
    # For diverging colormaps
    else:
        return cm.roma.reversed()
    # if not isinstance(cmap_var,type(None)) and cmap_var[-4:]=='-fit':
    #     return cm.roma
    # else:
    #     return cm.batlow.reversed()
    # return cm.davos.reversed()
    # Build dictionary of axis labels
    # cmaps = {'entry':'plasma',
    #          'prof_no':'plasma',
    #          'dt_start':'viridis',
    #          'dt_end':'viridis',
    #          'lon':'YlOrRd',
    #          'lat':'PuBuGn',
    #          'R_rho':'ocean',
    #          'density_hist':'inferno'
    #          }
    # # if cmap_var in ['press', 'depth', 'press_TC_max']:
    # if 'press' in cmap_var or 'depth' in cmap_var:
    #     return 'cividis'
    # # elif cmap_var in ['SP', 'SA', 'beta', 'BSP', 'BSt', 'BSA', 'ma_SP', 'ma_SA', 'la_SP', 'la_SA', 'SA_TC_max']:
    # elif 'SP' in cmap_var or 'SA' in cmap_var or 'BS' in cmap_var or cmap_var=='beta':
    #     return 'Blues'
    # # elif cmap_var in ['iT', 'CT', 'PT', 'alpha', 'aiT', 'aCT', 'aPT', 'ma_iT', 'ma_CT', 'ma_PT', 'la_iT', 'la_CT', 'la_PT', 'CT_TC_max']:
    # elif 'iT' in cmap_var or 'CT' in cmap_var or 'PT' in cmap_var or cmap_var=='alpha':
    #     return 'Reds'
    # # elif cmap_var in ['sigma', 'ma_sigma', 'la_sigma']:
    # elif 'sigma' in cmap_var:
    #     return 'Purples'
    # elif cmap_var in ['BL_yn', 'up_cast', 'ss_mask']:
    #     cmap = mpl.colors.ListedColormap(['green'])
    #     cmap.set_bad(color='red')
    #     return cmap
    # if cmap_var in cmaps.keys():
    #     return cmaps[cmap_var]
    # else:
    #     return 'viridis'

################################################################################

def get_marker_size_and_alpha(n_points):
    """
    Determines the size and alpha of the markers based on the number of points

    n_points    The number of points in the dataset
    """
    # Set the marker size and alpha based on the number of points
    if n_points < 200:
        m_size = map_mrk_size*2
        m_alpha = mrk_alpha
    elif n_points < 1000:
        m_size = map_mrk_size
        m_alpha = mrk_alpha
    elif n_points < 10000:
        m_size = mrk_size
        m_alpha = mrk_alpha
    elif n_points < 100000:#00:
        m_size = mrk_size2
        m_alpha = mrk_alpha2
    else:
        m_size = mrk_size3
        m_alpha = mrk_alpha3
    return m_size, m_alpha

################################################################################

def format_datetime_axes(add_grid, x_key, y_key, ax, tw_x_key=None, tw_ax_y=None, tw_y_key=None, tw_ax_x=None, z_key=None, aug_zorder=10):
    """
    Formats any datetime axes to show actual dates, as appropriate. Also adds
    vertical lines at every August 15th, if applicable

    add_grid    True/False whether to add grid lines to the plot
    x_key       The string of the name for the x data on the main axis
    y_key       The string of the name for the y data on the main axis
    ax          The main axis on which to format
    tw_x_key    The string of the name for the x data on the twin axis
    tw_ax_y     The twin y axis on which to format
    tw_y_key    The string of the name for the y data on the twin axis
    tw_ax_x     The twin x axis on which to format
    z_key       The string of the name for the z data on the main axis
    aug_zorder  The zorder to use for the August 15th vertical lines
    """
    # Make an array of all August 15ths
    all_08_15s = [  '2005/08/15 00:00:00', 
                    '2006/08/15 00:00:00', 
                    '2007/08/15 00:00:00', 
                    '2008/08/15 00:00:00', 
                    '2009/08/15 00:00:00',
                    '2010/08/15 00:00:00', 
                    '2011/08/15 00:00:00', 
                    '2012/08/15 00:00:00', 
                    '2013/08/15 00:00:00', 
                    '2014/08/15 00:00:00',
                    '2015/08/15 00:00:00',
                    '2016/08/15 00:00:00', 
                    '2017/08/15 00:00:00', 
                    '2018/08/15 00:00:00', 
                    '2019/08/15 00:00:00', 
                    '2020/08/15 00:00:00',
                    '2021/08/15 00:00:00', 
                    '2022/08/15 00:00:00'
                 ]
    all_08_15s = [datetime.strptime(date, r'%Y/%m/%d %H:%M:%S').date() for date in all_08_15s]
    # Make the date locator
    loc = mpl.dates.AutoDateLocator()
    if x_key in ['dt_start', 'dt_end']:
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(loc))
        # Add vertical lines
        if add_grid:
            for aug_15 in all_08_15s:
                ax.axvline(aug_15, color=bg_clr, linestyle='-', alpha=1, zorder=aug_zorder)
                ax.axvline(aug_15, color=std_clr, linestyle='--', alpha=0.7, zorder=aug_zorder+1)
    if y_key in ['dt_start', 'dt_end']:
        ax.yaxis.set_major_locator(loc)
        ax.yaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(loc))
    if z_key in ['dt_start', 'dt_end']:
        ax.zaxis.set_major_locator(loc)
        ax.zaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(loc))
    if tw_x_key in ['dt_start', 'dt_end']:
        tw_ax_y.xaxis.set_major_locator(loc)
        tw_ax_y.xaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(loc))
    if tw_y_key in ['dt_start', 'dt_end']:
        tw_ax_x.yaxis.set_major_locator(loc)
        tw_ax_x.yaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(loc))

################################################################################

def format_sci_notation(x, ndp=2, sci_lims_f=sci_lims, pm_val=None, condense=False):
    """
    Formats a number into scientific notation

    x           The number to format
    ndp         The number of decimal places
    sci_lims    The limits on powers of 10 between which scientific notation will not be used
    pm_val      The uncertainty value to put after a $\pm$ symbol
    condense    True/False whether to remove spaces around `\pm` and `\times` operators
    """
    # Check to make sure x is not None or nan
    if isinstance(x, type(None)) or np.isnan(x):
        print('Warning: could not format',x,'into scientific notation')
        return r'N/A'
    if isinstance(pm_val, type(None)):
        try:
            s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
            m, e = s.split('e')
        except:
            print('Warning: could not format',x,'with',ndp,'decimal places')
            return r'{x:0.{ndp:d}f}'.format(x=x, ndp=ndp)
        # Check to see whether it's outside the scientific notation exponent limits
        if int(e) < min(sci_lims_f) or int(e) > max(sci_lims_f):
            if condense:
                return r'{m:s}{{\times}}10^{{{e:d}}}'.format(m=m, e=int(e))
            else:
                return r'{m:s}\times10^{{{e:d}}}'.format(m=m, e=int(e))
        else:
            return r'{x:0.{ndp:d}f}'.format(x=x, ndp=ndp)
    else:
        try:
            # Find magnitude and base 10 exponent for x
            s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
            m, e = s.split('e')
        except:
            print('Warning: could not format',x,'with',ndp,'decimal places and pm_val:',pm_val)
            if condense:
                return r'{x:0.{ndp:d}f}{{\pm}}{pm_val:0.{ndp:d}f}'.format(x=x, ndp=ndp, pm_val=pm_val)
            else:
                return r'{x:0.{ndp:d}f}{{\pm}}{pm_val:0.{ndp:d}f}'.format(x=x, ndp=ndp, pm_val=pm_val)
        # Find magnitude and base 10 exponent for pm_val
        pm_s = '{pm_val:0.{ndp:d}e}'.format(pm_val=pm_val, ndp=ndp)
        pm_m, pm_e = pm_s.split('e')
        # Find difference between exponents to use as new number of decimal places
        new_ndp = max(2, int(e)-int(pm_e))
        # Reformat x
        s = '{x:0.{ndp:d}e}'.format(x=x, ndp=new_ndp)
        m, e = s.split('e')
        # Shift value of pm_val to correct exponent
        pm_m = '{pm_val:0.{ndp:d}f}'.format(pm_val=pm_val/(10**int(e)), ndp=new_ndp)
        # Check to see whether it's outside the scientific notation exponent limits
        if int(e) < min(sci_lims_f) or int(e) > max(sci_lims_f):
            if condense:
                return r'({m:s}{{\pm}}{pm_m:s}){{\times}} 10^{{{e:d}}}'.format(m=m, pm_m=pm_m, e=int(e))
            else:
                return r'({m:s}\pm{pm_m:s})\times 10^{{{e:d}}}'.format(m=m, pm_m=pm_m, e=int(e))
        else:
            if condense:
                return r'{x:0.{ndp:d}f}{{\pm}}{pm_val:0.{ndp:d}f}'.format(x=x, ndp=new_ndp, pm_val=pm_val)
            else:
                return r'{x:0.{ndp:d}f}\pm{pm_val:0.{ndp:d}f}'.format(x=x, ndp=new_ndp, pm_val=pm_val)

################################################################################

def add_h_scale_bar(ax, ax_lims, unit="", clr=std_clr, tw_clr=False):
    """
    Adds a horizontal scale bar to a plot

    ax          The axis on which to add the legend
    ax_lims     The limits on the horizontal axis range
    unit        A string of the units of the horizontal axis
    tw_clr      The color of the twin axis, if necessary
    """
    h_bar_clr = clr
    # Find distance between ticks
    ax_ticks = np.array(ax.get_xticks())
    # Use the ticks to decide on the bar length
    bar_len = min(np.diff(ax_ticks))/2
    # Find the bounds of the plot axis
    if isinstance(ax_lims, type(None)):
        x_bnds = ax.get_xbound()
        y_bnds = ax.get_ybound()
    else:
        if 'x_lims' in ax_lims.keys():
            x_bnds = ax_lims['x_lims']
        else:
            x_bnds = ax.get_xbound()
        if 'y_lims' in ax_lims.keys():
            y_bnds = ax_lims['y_lims']
        else:
            y_bnds = ax.get_ybound()
    # print('x_bnds:',x_bnds,'y_bnds:',y_bnds)
    x_span = abs(max(x_bnds)-min(x_bnds))
    y_span = abs(max(y_bnds)-min(y_bnds))
    # Adjust bar length if too small
    if x_span/bar_len > 15:
        bar_len = bar_len*2
    # Define end points for the scale bar
    x_pad = bar_len/3.8
    sb_x_min = min(x_bnds) + x_pad 
    sb_x_max = sb_x_min + bar_len
    if tw_clr==False:
        sb_y_val = max(y_bnds) - y_span/8
    else:
        sb_y_val = max(y_bnds) - y_span/4
        h_bar_clr = tw_clr
    # Plot scale bar
    ax.plot([sb_x_min,sb_x_max], [sb_y_val,sb_y_val], color=h_bar_clr)
    # Plot bar ends
    ax.scatter([sb_x_min,sb_x_max], [sb_y_val,sb_y_val], color=h_bar_clr, marker='|')
    # Annotate the bar length
    ax.annotate(r'%.3f'%(bar_len)+unit, xy=(sb_x_min+(sb_x_max-sb_x_min)/2, sb_y_val+y_span/16), ha='center', xycoords='data', color=h_bar_clr, bbox=anno_bbox, zorder=12)
    # Turn off the x tick labels
    ax.set_xticklabels([])

################################################################################
# Main plotting function #######################################################
################################################################################

def make_subplot(ax, a_group, fig, ax_pos):
    """
    Takes in an Analysis_Group object which has the data and plotting parameters
    to produce a subplot. Returns the x and y labels and the subplot title

    ax              The axis on which to make the plot
    a_group         A Analysis_Group object containing the info to create this subplot
    fig             The figure in which ax is contained
    ax_pos          A tuple of the ax (rows, cols, linear number of this subplot)
    """
    # Get relevant parameters for the plot
    pp = a_group.plt_params
    plot_type = pp.plot_type
    clr_map   = pp.clr_map
    # Check the extra arguments
    add_isos = False
    errorbars = False
    fit_vars = False
    log_axes = 'None'
    mpi_run = False
    # print('\t- Checking extra arguments')
    if not isinstance(pp.extra_args, type(None)):
        extra_args = pp.extra_args
        if 'plot_slopes' in extra_args.keys():
            plot_slopes = extra_args['plot_slopes']
            # print('\t- Plot slopes:',plot_slopes)
        else:
            plot_slopes = False
        if 'mark_outliers' in extra_args.keys():
            mrk_outliers = extra_args['mark_outliers']
        else:
            mrk_outliers = False
        if 'isopycnals' in extra_args.keys():
            isopycnals = extra_args['isopycnals']
            if not isinstance(isopycnals, type(None)) and not isopycnals is False:
                add_isos = True
        if 'place_isos' in extra_args.keys():
            place_isos = extra_args['place_isos']
        if 'errorbars' in extra_args.keys():
            errorbars = extra_args['errorbars']
        else:
            place_isos = False
        if 'fit_vars' in extra_args.keys():
            fit_vars = extra_args['fit_vars']
        else:
            fit_vars = False
        if 'log_axes' in extra_args.keys():
            log_axes = extra_args['log_axes']
        if 'mv_avg' in extra_args.keys():
            mv_avg = extra_args['mv_avg']
        else:
            mv_avg = False
        if 'mpi_run' in extra_args.keys():
            mpi_run = extra_args['mpi_run']
        if 're_run_clstr' in extra_args.keys():
            re_run_clstr = extra_args['re_run_clstr']
        else:
            re_run_clstr = True
        if 'mark_LHW_AW' in extra_args.keys():
            mark_LHW_AW = extra_args['mark_LHW_AW']
        else:
            mark_LHW_AW = False
        try:
            plt_noise = extra_args['plt_noise']
        except:
            plt_noise = True
        try:
            invert_y_axis = extra_args['invert_y_axis']
        except:
            invert_y_axis = None
    else:
        extra_args = False
        plot_slopes = False
        mv_avg = False
        mark_LHW_AW = False
        plt_noise = True
    # Concatonate all the pandas data frames together
    # print('\t- Concatenating data frames')
    df = pd.concat(a_group.data_frames)
    # print('plot_type:',plot_type)
    # Decide on a plot type
    if plot_type == 'xy':
        ## Make a standard x vs. y scatter plot
        # Set the main x and y data keys
        x_key = pp.x_vars[0]
        y_key = pp.y_vars[0]
        z_key = pp.z_vars[0]
        # print('test print of plot vars 1:',x_key, y_key, z_key, clr_map)
        # Check whether to invert y-axis
        if isinstance(invert_y_axis, type(None)):
            if y_key in y_invert_vars:
                invert_y_axis = True
            else:
                invert_y_axis = False
        # Check for histogram
        if x_key == 'hist' or y_key == 'hist':
            plot_hist = True
        else:
            plot_hist = False
        # Check for twin x and y data keys
        try:
            tw_x_key = pp.x_vars[1]
            tw_ax_y  = ax.twiny()
            # Set ticklabel format for twin axis
            tw_ax_y.ticklabel_format(style='sci', scilimits=sci_lims, useMathText=True)
        except:
            tw_x_key = None
            tw_ax_y  = None
        try:
            tw_y_key = pp.y_vars[1]
            tw_ax_x  = ax.twinx()
            # Set ticklabel format for twin axis
            tw_ax_x.ticklabel_format(style='sci', scilimits=sci_lims, useMathText=True)
        except:
            tw_y_key = None
            tw_ax_x  = None
        # Check whether plotting a parameter sweep of clustering
        for var in [x_key, y_key, tw_x_key, tw_y_key]:
            if var in clstr_ps_vars:
                # Add a standard title
                plt_title = add_std_title(a_group)
                # Plot the parameter sweep
                if mpi_run:
                    return df
                else:
                    xlabel, ylabel = plot_clstr_param_sweep(ax, tw_ax_x, a_group, plt_title)
                return xlabel, ylabel, pp.zlabels[0], plt_title, ax, invert_y_axis
        # Concatonate all the pandas data frames together
        # print('\t- Concatenating data frames again')
        df = pd.concat(a_group.data_frames)
        # Format the dates if necessary
        if x_key in ['dt_start', 'dt_end']:
            print('\t- Formatting datetime axis (this takes a while)')
            df[x_key] = mpl.dates.date2num(df[x_key])
        if y_key in ['dt_start', 'dt_end']:
            print('\t- Formatting datetime axis (this takes a while)')
            df[y_key] = mpl.dates.date2num(df[y_key])
        if z_key in ['dt_start', 'dt_end']:
            print('\t- Formatting datetime axis (this takes a while)')
            df[z_key] = mpl.dates.date2num(df[z_key])
        # print('\t- Formatting ITP numbers')
        # Format the ITP numbers if necessary
        if x_key == 'instrmt':
            df[x_key] = df[x_key].str.zfill(3)
        if y_key == 'instrmt':
            df[y_key] = df[y_key].str.zfill(3)
        if y_key == 'prof_no':
            df[y_key] = df[y_key].astype(str)
        if z_key == 'instrmt':
            df[z_key] = df[z_key].str.zfill(3)
        # Check for cluster-based variables
        if x_key in clstr_vars or y_key in clstr_vars or z_key in clstr_vars or tw_x_key in clstr_vars or tw_y_key in clstr_vars or clr_map in clstr_vars:
            print('\t- Checking for cluster-based variables')
            m_pts, m_cls, cl_x_var, cl_y_var, cl_z_var, plot_slopes, b_a_w_plt = get_cluster_args(pp)
            df, rel_val, m_pts, m_cls, ell = HDBSCAN_(a_group.data_set.arr_of_ds, df, cl_x_var, cl_y_var, cl_z_var, m_pts, m_cls=m_cls, extra_cl_vars=[x_key,y_key,z_key,tw_x_key,tw_y_key,clr_map], re_run_clstr=re_run_clstr)
        print('\t- Plot slopes:',plot_slopes)
        # Check whether to normalize by subtracting a polyfit2d
        if fit_vars:# and clr_map != 'cluster':
            df = calc_fit_vars(df, (x_key, y_key, z_key, clr_map), fit_vars)
        # If plotting per profile, collapse the vertical dimension
        if pp.plot_scale == 'by_pf':
            # Drop dimensions, if needed
            if 'Vertical' in df.index.names:
                # Drop duplicates along the `Time` dimension
                #   (need to make the index `Time` a column first)
                print('\t- Dropping `Vertical` dimension')
                df = df.droplevel('Vertical')
                df.reset_index(level=['Time'], inplace=True)
                df.drop_duplicates(inplace=True, subset=['Time'])
        # Check whether to add LHW and AW
        if mark_LHW_AW:
            mark_the_LHW_AW(ax, x_key, y_key)
        # Check whether to remove noise points
        if plt_noise == 'both':
            # Make a deep copy of the data frame
            df_with_noise = df.copy()
        if plt_noise == False or plt_noise == 'both':
            print('\t- Removing noise points')
            print('before removing noise points:',len(df))
            df = df[df.cluster!=-1]
            print('after removing noise points:',len(df))
        # print('x_key:',x_key)
        # print('y_key:',y_key)
        # Determine the color mapping to be used
        if clr_map == 'clr_all_same':
            # Check for histogram
            if plot_hist:
                return plot_histogram(a_group, ax, pp, df, x_key, y_key, clr_map, legend=pp.legend, txk=tw_x_key, tay=tw_ax_y, tyk=tw_y_key, tax=tw_ax_x, df_w_noise=df_with_noise)
            # Set whether to plot in 3D
            if isinstance(z_key, type(None)):
                df_z_key = 0
                # Drop duplicates
                # print('df columns:',df.columns)
                df.drop_duplicates(subset=[x_key, y_key], keep='first', inplace=True)
                plot_3d = False
            else:
                df_z_key = df[z_key]
                # Drop duplicates
                df.drop_duplicates(subset=[x_key, y_key, z_key], keep='first', inplace=True)
                plot_3d = True
            if plot_3d:
                # Remove the current axis
                ax.remove()
                # Replace axis with one that has 3 dimensions
                ax = fig.add_subplot(ax_pos, projection='3d')
                # Turn the panes transparent
                ax.w_xaxis.pane.fill = False
                ax.w_yaxis.pane.fill = False
                ax.w_zaxis.pane.fill = False
            # If there aren't that many points, make the markers bigger
            m_size, m_alpha = get_marker_size_and_alpha(len(df[x_key]))
            # if len(df[x_key]) < 1000:
            #     m_size = map_mrk_size
            # elif len(df[x_key]) < 100000:
            #     m_size = mrk_size2
            # else: 
            #     m_size = mrk_size
            # Plot every point the same color, size, and marker
            if plot_3d == False:
                # Plot in 2D
                ax.scatter(df[x_key], df[y_key], color=std_clr, s=m_size, marker=std_marker, alpha=m_alpha, zorder=5)
                # Take moving average of the data if the x-axis is time
                if x_key in ['dt_start', 'dt_end'] and mv_avg:
                    print('\t- Taking moving average of the data')
                    # Make a copy of the dataframe
                    time_df = df.copy()
                    # Need to convert dt_start back to datetime to take rolling average
                    time_df[x_key] = mpl.dates.num2date(time_df[x_key])
                    # Set dt_start as the index
                    time_df = time_df.set_index(x_key)
                    # Sort by the dt_start index
                    time_df.sort_index(inplace=True)
                    # Take the moving average of the data
                    time_df[y_key+'_mv_avg'] = time_df[y_key].rolling(mv_avg, min_periods=1).mean()
                    # Convert the original dataframe's dt_start back to numbers for plotting
                    time_df.reset_index(level=x_key, inplace=True)
                    time_df[x_key] = mpl.dates.date2num(time_df[x_key])
                    ax.plot(time_df[x_key], time_df[y_key+'_mv_avg'], color='b', zorder=6)
            else:
                # Plot in 3D
                ax.scatter(df[x_key], df[y_key], zs=df_z_key, color=std_clr, s=m_size, marker=std_marker, alpha=m_alpha, zorder=5)
            if plot_slopes:
                # Mark outliers
                print('\t- Marking outliers:',mrk_outliers)
                if mrk_outliers:
                    if 'trd_' in x_key:
                        other_out_vars = ['cRL', 'nir_SA']
                        other_out_vars = []
                    else:
                        other_out_vars = []
                    mark_outliers(ax, df, x_key, y_key, clr_map, mrk_outliers, mrk_for_other_vars=other_out_vars)
                    # Get data without outliers
                    x_data = np.array(df[df['out_'+x_key]==False][x_key].values, dtype=np.float64)
                    y_data = np.array(df[df['out_'+x_key]==False][y_key].values, dtype=np.float64)
                else:
                    # Get data
                    x_data = np.array(df[x_key].values, dtype=np.float64)
                    y_data = np.array(df[y_key].values, dtype=np.float64)
                if plot_3d == False:
                    add_linear_slope(ax, pp, df, x_data, y_data, x_key, y_key, alt_std_clr, plot_slopes)
                else:
                    # Fit a 2d polynomial to the z data
                    plot_polyfit2d(ax, pp, x_data, y_data, df_z_key)
            # Invert y-axis if specified
            if y_key in y_invert_vars:
                invert_y_axis = True
            # Plot on twin axes, if specified
            if not isinstance(tw_x_key, type(None)):
                tw_clr = get_var_color(tw_x_key)
                if tw_clr == std_clr:
                    tw_clr = alt_std_clr
                tw_ax_y.scatter(df[tw_x_key], df[y_key], color=tw_clr, s=m_size, marker=std_marker, alpha=m_alpha)
                tw_ax_y.set_xlabel(pp.xlabels[1])
                # Change color of the ticks on the twin axis
                tw_ax_y.tick_params(axis='x', colors=tw_clr)
                # Check whether to adjust the twin axes limits
                if not isinstance(pp.ax_lims, type(None)):
                    if 'tw_x_lims' in pp.ax_lims.keys():
                        tw_ax_y.set_ylim(pp.ax_lims['tw_x_lims'])
                        print('\t- Set tw_x_lims to',pp.ax_lims['tw_x_lims'])
            elif not isinstance(tw_y_key, type(None)):
                tw_clr = get_var_color(tw_y_key)
                if tw_clr == std_clr:
                    tw_clr = alt_std_clr
                tw_ax_x.scatter(df[x_key], df[tw_y_key], color=tw_clr, s=m_size, marker=std_marker, alpha=m_alpha)
                # Invert y-axis if specified
                if tw_y_key in y_invert_vars:
                    tw_ax_x.invert_yaxis()
                tw_ax_x.set_ylabel(pp.ylabels[1])
                # Change color of the ticks on the twin axis
                tw_ax_x.tick_params(axis='y', colors=tw_clr)
                # Invert twin y-axis if specified
                if tw_y_key in y_invert_vars:
                    tw_ax_x.invert_yaxis()
                    print('\t- Inverting twin y-axis')
                # Check whether to adjust the twin axes limits
                if not isinstance(pp.ax_lims, type(None)):
                    if 'tw_y_lims' in pp.ax_lims.keys():
                        tw_ax_x.set_ylim(pp.ax_lims['tw_y_lims'])
                        print('\t- Set tw_y_lims to',pp.ax_lims['tw_y_lims'])
            # Add a standard legend
            if pp.legend:
                add_std_legend(ax, df, x_key, mark_LHW_AW=mark_LHW_AW, lgd_loc=pp.legend)
            # Format the axes for datetimes, if necessary
            format_datetime_axes(pp.add_grid, x_key, y_key, ax, tw_x_key, tw_ax_y, tw_y_key, tw_ax_x, aug_zorder=4)
            # Check whether to plot isopycnals
            if add_isos:
                add_isopycnals(ax, df, x_key, y_key, p_ref=isopycnals, place_isos=place_isos, tw_x_key=tw_x_key, tw_ax_y=tw_ax_y, tw_y_key=tw_y_key, tw_ax_x=tw_ax_x)
            # Check whether to change any axes to log scale
            if len(log_axes) == 3:
                if log_axes[0]:
                    ax.set_xscale('log')
                if log_axes[1]:
                    ax.set_yscale('log')
                if log_axes[2]:
                    ax.set_zscale('log')
            # Add a standard title
            plt_title = add_std_title(a_group)
            return pp.xlabels[0], pp.ylabels[0], pp.zlabels[0], plt_title, ax, invert_y_axis
        elif clr_map == 'source':
            if plot_hist:
                return plot_histogram(a_group, ax, pp, df, x_key, y_key,clr_map, legend=pp.legend)
            # Get unique sources
            these_sources = np.unique(df['source'])
            # If there aren't that many points, make the markers bigger
            m_size, m_alpha = get_marker_size_and_alpha(len(df[x_key]))
            # if len(df[x_key]) < 1000:
            #     m_size = map_mrk_size
            # elif len(df[x_key]) < 100000:
            #     m_size = mrk_size2
            # else: 
            #     m_size = mrk_size
            i = 0
            lgnd_hndls = []
            for source in these_sources:
                # Decide on the color, don't go off the end of the array
                my_clr = distinct_clrs[i%len(distinct_clrs)]
                # Get the data for just this source
                this_df = df[df['source'] == source] 
                # Plot every point from this df the same color, size, and marker
                ax.scatter(this_df[x_key], this_df[y_key], color=my_clr, s=m_size, marker=std_marker, alpha=m_alpha, zorder=5)
                i += 1
                # Add legend to report the total number of points for this instrmt
                lgnd_label = source+': '+str(len(this_df[x_key]))+' points'
                lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label))
            # notes_string = ''.join(this_df.notes.unique())
            notes_string = ''
            # Only add the notes_string if it contains something
            if len(notes_string) > 1:
                notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
                lgnd_hndls.append(notes_patch)
            # Format the axes for datetimes, if necessary
            format_datetime_axes(pp.add_grid, x_key, y_key, ax)
            # Add legend with custom handles
            if pp.legend:
                lgnd = ax.legend(handles=lgnd_hndls)
            # Check whether to plot isopycnals
            if add_isos:
                add_isopycnals(ax, df, x_key, y_key, p_ref=isopycnals, place_isos=place_isos, tw_x_key=tw_x_key, tw_ax_y=tw_ax_y, tw_y_key=tw_y_key, tw_ax_x=tw_ax_x)
            # Check whether to change any axes to log scale
            if len(log_axes) == 3:
                if log_axes[0]:
                    ax.set_xscale('log')
                if log_axes[1]:
                    ax.set_yscale('log')
                if log_axes[2]:
                    ax.set_zscale('log')
            # Add a standard title
            plt_title = add_std_title(a_group)
            return pp.xlabels[0], pp.ylabels[0], pp.zlabels[0], plt_title, ax, invert_y_axis
        elif clr_map == 'plt_by_dataset' or clr_map == 'stacked':
            if plot_hist:
                return plot_histogram(a_group, ax, pp, df, x_key, y_key, clr_map, legend=pp.legend)
        elif clr_map == 'clr_by_dataset':
            if plot_hist:
                return plot_histogram(a_group, ax, pp, df, x_key, y_key, clr_map, legend=pp.legend)
            # # Check for cluster-based variables
            new_cl_vars = []
            for key in [x_key, y_key, z_key]:
                if key in clstr_vars:
                    new_cl_vars.append(key)
            if errorbars:
                if isinstance(x_key, type('str')) and '_' in x_key:
                    # Split the prefix from the original variable (assumes an underscore split)
                    split_var = x_key.split('_', 1)
                    prefix = split_var[0]
                    var_str = split_var[1]
                    if prefix == 'ca':
                        x_err_key = 'csd_'+var_str
                        new_cl_vars.append(x_err_key)
                    else:
                        x_err_key = None
                if isinstance(y_key, type('str')) and '_' in y_key:
                    # Split the prefix from the original variable (assumes an underscore split)
                    split_var = y_key.split('_', 1)
                    prefix = split_var[0]
                    var_str = split_var[1]
                    if prefix == 'ca':
                        y_err_key = 'csd_'+var_str
                        new_cl_vars.append(y_err_key)
                    else:
                        y_err_key = None
            print('new_cl_vars:',new_cl_vars)
            i = 0
            lgnd_hndls = []
            df_labels = [*a_group.data_set.sources_dict.keys()]
            print('\tdf_labels:',df_labels)
            # Loop through each dataframe 
            #   which correspond to the datasets input to the Data_Set object's sources_dict
            for this_df in a_group.data_frames:
                if len(new_cl_vars) > 0:
                    this_df = calc_extra_cl_vars(this_df, new_cl_vars)
                    # Drop duplicates to have just one row per cluster
                    this_df.drop_duplicates(subset=new_cl_vars, keep='first', inplace=True)
                # If there aren't that many points, make the markers bigger
                m_size, m_alpha = get_marker_size_and_alpha(len(this_df[x_key]))
                # if len(this_df[x_key]) < 1000:
                #     m_size = map_mrk_size
                # else: 
                #     m_size = mrk_size
                # Decide on the color, don't go off the end of the array
                my_clr = distinct_clrs[i%len(distinct_clrs)]
                # Format the dates if necessary
                if x_key in ['dt_start', 'dt_end']:
                    this_df[x_key] = mpl.dates.date2num(this_df[x_key])
                if y_key in ['dt_start', 'dt_end']:
                    this_df[y_key] = mpl.dates.date2num(this_df[y_key])
                # Plot every point from this df the same color, size, and marker
                ax.scatter(this_df[x_key], this_df[y_key], color=my_clr, s=m_size, marker=std_marker, alpha=m_alpha, zorder=5)
                # Add error bars if applicable
                if errorbars:
                    if not isinstance(x_err_key, type(None)):
                        ax.errorbar(this_df[x_key], this_df[y_key], xerr=this_df[x_err_key], color=my_clr, capsize=l_cap_size, fmt='none', alpha=m_alpha, zorder=4)
                    if not isinstance(y_err_key, type(None)):
                        ax.errorbar(this_df[x_key], this_df[y_key], yerr=this_df[y_err_key], color=my_clr, capsize=l_cap_size, fmt='none', alpha=m_alpha, zorder=4)
                # Add legend to report the total number of points for this instrmt
                lgnd_label = df_labels[i]+': '+str(len(this_df[x_key]))+' points'
                lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label))
                i += 1
            # notes_string = ''.join(this_df.notes.unique())
            notes_string = ''
            # Only add the notes_string if it contains something
            if len(notes_string) > 1:
                notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
                lgnd_hndls.append(notes_patch)
            # Format the axes for datetimes, if necessary
            format_datetime_axes(pp.add_grid, x_key, y_key, ax)
            # Add legend with custom handles
            if pp.legend:
                lgnd = ax.legend(handles=lgnd_hndls)
            # Check whether to plot isopycnals
            if add_isos:
                add_isopycnals(ax, df, x_key, y_key, p_ref=isopycnals, place_isos=place_isos, tw_x_key=tw_x_key, tw_ax_y=tw_ax_y, tw_y_key=tw_y_key, tw_ax_x=tw_ax_x)
            # Check whether to change any axes to log scale
            if len(log_axes) == 3:
                if log_axes[0]:
                    ax.set_xscale('log')
                if log_axes[1]:
                    ax.set_yscale('log')
                if log_axes[2]:
                    ax.set_zscale('log')
            # Add a standard title
            plt_title = add_std_title(a_group)
            return pp.xlabels[0], pp.ylabels[0], pp.zlabels[0], plt_title, ax, invert_y_axis
        elif clr_map == 'instrmt':
            if plot_hist:
                return plot_histogram(a_group, ax, pp, df, x_key, y_key, clr_map, legend=pp.legend)
            # Add column where the source-instrmt combination ensures uniqueness
            df['source-instrmt'] = df['source']+' '+df['instrmt'].astype("string")
            # Get unique instrmts
            these_instrmts = np.unique(df['source-instrmt'])
            if len(these_instrmts) < 1:
                print('No instruments included, aborting script')
                exit(0)
            # If there aren't that many points, make the markers bigger
            m_size, m_alpha = get_marker_size_and_alpha(len(df[x_key]))
            # if len(df[x_key]) < 1000:
            #     m_size = map_mrk_size
            # elif len(df[x_key]) < 100000:
            #     m_size = mrk_size2
            # else: 
            #     m_size = mrk_size
            i = 0
            lgnd_hndls = []
            # Loop through each instrument
            for instrmt in these_instrmts:
                # Get the data for just this instrmt
                this_df = df[df['source-instrmt'] == instrmt]
                # Decide on the color, don't go off the end of the array
                my_clr = distinct_clrs[i%len(distinct_clrs)]
                # Plot every point from this df the same color, size, and marker
                ax.scatter(this_df[x_key], this_df[y_key], color=my_clr, s=m_size, marker=std_marker, alpha=m_alpha, zorder=5)
                i += 1
                # Add legend to report the total number of points for this instrmt
                lgnd_label = instrmt+': '+str(len(this_df[x_key]))+' points'
                lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label))
                notes_string = ''.join(this_df.notes.unique())
            # Only add the notes_string if it contains something
            if len(notes_string) > 1:
                notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
                lgnd_hndls.append(notes_patch)
            # Add legend with custom handles
            if pp.legend:
                lgnd = ax.legend(handles=lgnd_hndls)
            # Format the axes for datetimes, if necessary
            format_datetime_axes(pp.add_grid, x_key, y_key, ax)
            # Check whether to plot isopycnals
            if add_isos:
                add_isopycnals(ax, df, x_key, y_key, p_ref=isopycnals, place_isos=place_isos, tw_x_key=tw_x_key, tw_ax_y=tw_ax_y, tw_y_key=tw_y_key, tw_ax_x=tw_ax_x)
            # Check whether to change any axes to log scale
            if len(log_axes) == 3:
                if log_axes[0]:
                    ax.set_xscale('log')
                if log_axes[1]:
                    ax.set_yscale('log')
                if log_axes[2]:
                    ax.set_zscale('log')
            # Color the axis tick labels if axis is 'instrmt'
            if x_key == 'instrmt':
                i = 0
                for tick in ax.get_xticklabels():
                    # Decide on the color, don't go off the end of the array
                    my_clr = distinct_clrs[i%len(distinct_clrs)]
                    # Color just the tick label for this instrmt
                    tick.set_color(my_clr)
            if y_key == 'instrmt':
                i = 0
                for tick in ax.get_yticklabels():
                    # Decide on the color, don't go off the end of the array
                    my_clr = distinct_clrs[i%len(distinct_clrs)]
                    # Color just the tick label for this instrmt
                    tick.set_color(my_clr)
                    i += 1
            # Add a standard title
            plt_title = add_std_title(a_group)
            return pp.xlabels[0], pp.ylabels[0], pp.zlabels[0], plt_title, ax, invert_y_axis
        elif clr_map == 'density_hist':
            if plot_hist:
                print('Cannot use density_hist colormap with the `hist` variable')
                exit(0)
            # Plot a density histogram where each grid box is colored to show how
            #   many points fall within it
            # Get the density histogram plotting parameters from extra args
            d_hist_dict = pp.extra_args
            if not isinstance(d_hist_dict, type(None)) and 'clr_min' in d_hist_dict.keys():
                # Pull out the values from the dictionary provided
                clr_min = d_hist_dict['clr_min']
                clr_max = d_hist_dict['clr_max']
                clr_ext = d_hist_dict['clr_ext']
                xy_bins = d_hist_dict['xy_bins']
            else:
                # If none are given, use the defaults here
                clr_min = 0
                clr_max = 20
                clr_ext = 'max'        # adds arrow indicating values go past the bounds
                #                       #   use 'min', 'max', or 'both'
                xy_bins = 250
            # Use either `hist2d` or `hexbin` to make the density histogram
            # dhist_type = 'hist2d'
            dhist_type = 'hexbin'
            # Check whether to change the colorbar axis to log scale
            cbar_scale = None
            if len(log_axes) == 3:
                if log_axes[2]:
                    if dhist_type == 'hist2d':
                        cbar_scale = mpl.colors.LogNorm()
                    elif dhist_type == 'hexbin':
                        cbar_scale = 'log'
                    if clr_min <= 0:
                        clr_min = 1
                    # clr_min = None
                    # clr_max = 100
            # Make the 2D histogram, the number of bins really changes the outcome
            if dhist_type == 'hist2d':
                heatmap = ax.hist2d(df[x_key], df[y_key], bins=xy_bins, cmap=get_color_map(clr_map), cmin=clr_min, cmax=clr_max, norm=cbar_scale)
                # `hist2d` returns a tuple, the index 3 of which is the mappable for a colorbar
                cbar = plt.colorbar(heatmap[3], ax=ax, extend=clr_ext)
                cbar.set_label('Points per pixel')
            elif dhist_type == 'hexbin':
                # Note: extent expects (xmin, xmax, ymin, ymax)
                hb_extent = None
                # Check whether to adjust the twin axes limits
                if not isinstance(pp.ax_lims, type(None)):
                    if 'x_lims' in pp.ax_lims.keys() and 'y_lims' in pp.ax_lims.keys():
                        hb_extent = (pp.ax_lims['x_lims'][0], pp.ax_lims['x_lims'][1], pp.ax_lims['y_lims'][0], pp.ax_lims['y_lims'][1])
                        print('\t- Set hexbin extent to',hb_extent)
                heatmap = ax.hexbin(df[x_key], df[y_key], gridsize=xy_bins*3, cmap=get_color_map(clr_map), bins=cbar_scale, extent=hb_extent)
                heatmap.set_clim(clr_min, clr_max)
                cbar = plt.colorbar(heatmap, ax=ax, extend=clr_ext)
                cbar.set_label('Points per pixel')
            # Add legend to report the total number of points and pixels on the plot
            n_pts_patch  = mpl.patches.Patch(color='none', label=str(len(df[x_key]))+' points')
            pixel_patch  = mpl.patches.Patch(color='none', label=str(xy_bins)+'x'+str(xy_bins)+' pixels')
            # notes_string = ''.join(df.notes.unique())
            notes_string = ''
            # Only add the notes_string if it contains something
            if pp.legend:
                if len(notes_string) > 1:
                    notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
                    ax.legend(handles=[n_pts_patch, pixel_patch, notes_patch])
                else:
                    ax.legend(handles=[n_pts_patch, pixel_patch])
            # Format the axes for datetimes, if necessary
            format_datetime_axes(pp.add_grid, x_key, y_key, ax)
            # Check whether to plot isopycnals
            if add_isos:
                add_isopycnals(ax, df, x_key, y_key, p_ref=isopycnals, place_isos=place_isos, tw_x_key=tw_x_key, tw_ax_y=tw_ax_y, tw_y_key=tw_y_key, tw_ax_x=tw_ax_x)
            # Check whether to change any axes to log scale
            if len(log_axes) == 3:
                if log_axes[0]:
                    ax.set_xscale('log')
                if log_axes[1]:
                    ax.set_yscale('log')
            # Add a standard title
            plt_title = add_std_title(a_group)
            return pp.xlabels[0], pp.ylabels[0], pp.zlabels[0], plt_title, ax, invert_y_axis
        elif clr_map == 'cluster':
            # Add a standard title
            plt_title = add_std_title(a_group)
            #   Legend is handled inside plot_clusters()
            # Check for histogram
            if plot_hist:
                return plot_histogram(a_group, ax, pp, df, x_key, y_key, clr_map, legend=pp.legend, txk=tw_x_key, tay=tw_ax_y, tyk=tw_y_key, tax=tw_ax_x, clstr_dict={'m_pts':m_pts, 'rel_val':rel_val})
            # Format the dates if necessary (don't need this because it's done above)
            # if x_key in ['dt_start', 'dt_end']:
            #     print(df[x_key])
            #     df[x_key] = mpl.dates.date2num(df[x_key])
            # if y_key in ['dt_start', 'dt_end']:
            #     df[y_key] = mpl.dates.date2num(df[y_key])
            if isinstance(z_key, type(None)):
                df_z_key = 0
                # Drop duplicates
                # df.drop_duplicates(subset=[x_key, y_key], keep='first', inplace=True)
                plot_3d = False
            elif z_key in ['dt_start', 'dt_end']:
                df_z_key = mpl.dates.date2num(df[z_key])
                # Drop duplicates
                df.drop_duplicates(subset=[x_key, y_key, z_key], keep='first', inplace=True)
                plot_3d = True
            else:
                df_z_key = df[z_key]
                # Drop duplicates
                df.drop_duplicates(subset=[x_key, y_key, z_key], keep='first', inplace=True)
                plot_3d = True
            if plot_3d:
                # Remove the current axis
                ax.remove()
                # Replace axis with one that has 3 dimensions
                ax = fig.add_subplot(ax_pos, projection='3d')
                # Turn the panes transparent
                ax.w_xaxis.pane.fill = False
                ax.w_yaxis.pane.fill = False
                ax.w_zaxis.pane.fill = False
            # Make sure to assign m_pts and DBCV to the analysis group to enable writing out to netcdf
            invert_y_axis, a_group.data_frames, a_group.data_set.arr_of_ds[0].attrs['Clustering m_pts'], a_group.data_set.arr_of_ds[0].attrs['Clustering m_cls'], a_group.data_set.arr_of_ds[0].attrs['Clustering DBCV'] = plot_clusters(a_group, ax, pp, df, x_key, y_key, z_key, cl_x_var, cl_y_var, cl_z_var, clr_map, m_pts, m_cls, rel_val, ell, box_and_whisker=b_a_w_plt, plot_slopes=plot_slopes)
            # Format the axes for datetimes, if necessary
            format_datetime_axes(pp.add_grid, x_key, y_key, ax)
            # Check whether to plot isopycnals
            if add_isos:
                add_isopycnals(ax, df, x_key, y_key, p_ref=isopycnals, place_isos=place_isos, tw_x_key=tw_x_key, tw_ax_y=tw_ax_y, tw_y_key=tw_y_key, tw_ax_x=tw_ax_x)
            # Check whether to change any axes to log scale
            if len(log_axes) == 3:
                if log_axes[0]:
                    ax.set_xscale('log')
                if log_axes[1]:
                    ax.set_yscale('log')
                if log_axes[2]:
                    ax.set_zscale('log')
            return pp.xlabels[0], pp.ylabels[0], pp.zlabels[0], plt_title, ax, invert_y_axis
            #
        elif clr_map in np.array(df.columns):
            if pp.plot_scale == 'by_pf':
                if clr_map not in pf_vars:
                    print('Cannot use',clr_map,'with plot scale by_pf')
                    exit(0)
            # Set whether to plot in 3D
            if isinstance(z_key, type(None)):
                plot_3d = False
            else:
                df_z_key = df[z_key]
                plot_3d = True
            # Format color map dates if necessary
            if clr_map in ['dt_start', 'dt_end']:
                cmap_data = mpl.dates.date2num(df[clr_map])
            else:
                cmap_data = df[clr_map]
            # Get the colormap
            this_cmap = get_color_map(clr_map)
            #
            if plot_hist:
                return plot_histogram(a_group, ax, pp, df, x_key, y_key, clr_map=clr_map, legend=pp.legend)
            # If there aren't that many points, make the markers bigger
            m_size, m_alpha = get_marker_size_and_alpha(len(df[x_key]))
            # if len(df[x_key]) < 1000:
            #     m_size = map_mrk_size
            # elif len(df[x_key]) < 100000:
            #     m_size = mrk_size2
            # else: 
            #     m_size = mrk_size
            if plot_3d:
                # Remove the current axis
                ax.remove()
                # Replace axis with one that has 3 dimensions
                ax = fig.add_subplot(ax_pos, projection='3d')
                # Turn the panes transparent
                ax.w_xaxis.pane.fill = False
                ax.w_yaxis.pane.fill = False
                ax.w_zaxis.pane.fill = False
                # Plot
                heatmap = ax.scatter(df[x_key], df[y_key], zs=df_z_key, c=cmap_data, cmap=this_cmap, s=m_size, marker=std_marker, zorder=5)
                if plot_slopes:
                    # Fit a 2d polynomial to the z data
                    plot_polyfit2d(ax, pp, df[x_key], df[y_key], df_z_key)
            else:
                if plot_slopes and x_key=='lon' and y_key=='lat':
                    # Fit a 2d polynomial to the z data
                    plot_polyfit2d(ax, pp, df[x_key], df[y_key], df[clr_map], in_3D=False)
                    # Plot the scatter
                    heatmap = ax.scatter(df[x_key], df[y_key], c=cmap_data, cmap=this_cmap, s=m_size, marker=std_marker, zorder=5, alpha=map_alpha)
                elif plot_slopes:
                    # Plot the scatter
                    heatmap = ax.scatter(df[x_key], df[y_key], c=cmap_data, cmap=this_cmap, s=m_size, marker=std_marker, zorder=5, alpha=map_alpha)
                    # Plot a linear slope
                    add_linear_slope(ax, pp, df, df[x_key], df[y_key], x_key, y_key, alt_std_clr, plot_slopes)
                else:
                    # Plot the scatter
                    heatmap = ax.scatter(df[x_key], df[y_key], c=cmap_data, cmap=this_cmap, s=m_size, marker=std_marker, zorder=5)
            # Plot on twin axes, if specified
            if not isinstance(tw_x_key, type(None)):
                tw_clr = get_var_color(tw_x_key)
                if tw_clr == std_clr:
                    tw_clr = alt_std_clr
                # Add backing x to distinguish from main axis
                tw_ax_y.scatter(df[tw_x_key], df[y_key], color=tw_clr, s=m_size*10, marker='x', zorder=3)
                tw_ax_y.scatter(df[tw_x_key], df[y_key], c=cmap_data, cmap=this_cmap, s=m_size, marker=std_marker, zorder=4)
                tw_ax_y.set_xlabel(pp.xlabels[1])
                # Change color of the axis label on the twin axis
                tw_ax_y.xaxis.label.set_color(tw_clr)
                # Change color of the ticks on the twin axis
                tw_ax_y.tick_params(axis='x', colors=tw_clr)
                # Create the colorbar
                cbar = plt.colorbar(heatmap, ax=tw_ax_y)
                # Check whether to adjust the twin axes limits
                if not isinstance(pp.ax_lims, type(None)):
                    if 'tw_x_lims' in pp.ax_lims.keys():
                        tw_ax_y.set_ylim(pp.ax_lims['tw_x_lims'])
                        print('\t- Set tw_x_lims to',pp.ax_lims['tw_x_lims'])
            elif not isinstance(tw_y_key, type(None)):
                tw_clr = get_var_color(tw_y_key)
                if tw_clr == std_clr:
                    tw_clr = alt_std_clr
                # Add backing x to distinguish from main axis
                tw_ax_x.scatter(df[x_key], df[tw_y_key], color=tw_clr, s=m_size*10, marker='x', zorder=3)
                tw_ax_x.scatter(df[x_key], df[tw_y_key], c=cmap_data, cmap=this_cmap, s=m_size, marker=std_marker, zorder=4)
                # Invert y-axis if specified
                if tw_y_key in y_invert_vars:
                    tw_ax_x.invert_yaxis()
                tw_ax_x.set_ylabel(pp.ylabels[1])
                # Change color of the axis label on the twin axis
                tw_ax_x.yaxis.label.set_color(tw_clr)
                # Change color of the ticks on the twin axis
                tw_ax_x.tick_params(axis='y', colors=tw_clr)
                # Create the colorbar
                cbar = plt.colorbar(heatmap, ax=tw_ax_x)
                # Check whether to adjust the twin axes limits
                if not isinstance(pp.ax_lims, type(None)):
                    if 'tw_y_lims' in pp.ax_lims.keys():
                        tw_ax_x.set_ylim(pp.ax_lims['tw_y_lims'])
                        print('\t- Set tw_y_lims to',pp.ax_lims['tw_y_lims'])
            else:
                # Create the colorbar
                cbar = plt.colorbar(heatmap, ax=ax)
            # Format the colorbar ticks, if necessary
            if clr_map in ['dt_start', 'dt_end']:
                loc = mpl.dates.AutoDateLocator()
                cbar.ax.yaxis.set_major_locator(loc)
                cbar.ax.yaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(loc))
            # Change the colorbar limits, if necessary
            try:
                ax_lims_keys = list(pp.ax_lims.keys())
                if 'c_lims' in ax_lims_keys:
                    cbar.mappable.set_clim(vmin=min(pp.ax_lims['c_lims']), vmax=max(pp.ax_lims['c_lims']))
                    print('\t- Set c_lims to',pp.ax_lims['c_lims'])
            except:
                foo = 2
            # Invert colorbar if necessary
            if clr_map in y_invert_vars:
                cbar.ax.invert_yaxis()
            cbar.set_label(pp.clabel)
            # Add a standard legend
            if pp.legend:
                add_std_legend(ax, df, x_key, lgd_loc=pp.legend)
            # Format the axes for datetimes, if necessary
            format_datetime_axes(pp.add_grid, x_key, y_key, ax, tw_x_key, tw_ax_y, tw_y_key, tw_ax_x)
            # Check whether to plot isopycnals
            if add_isos:
                add_isopycnals(ax, df, x_key, y_key, p_ref=isopycnals, place_isos=place_isos, tw_x_key=tw_x_key, tw_ax_y=tw_ax_y, tw_y_key=tw_y_key, tw_ax_x=tw_ax_x)
            # Check whether to change any axes to log scale
            if len(log_axes) == 3:
                if log_axes[0]:
                    ax.set_xscale('log')
                if log_axes[1]:
                    ax.set_yscale('log')
                if log_axes[2]:
                    ax.set_zscale('log')
            # Add a standard title
            plt_title = add_std_title(a_group)
            return pp.xlabels[0], pp.ylabels[0], pp.zlabels[0], plt_title, ax, invert_y_axis
        else:
            # Did not provide a valid clr_map
            print('Colormap',clr_map,'not valid')
            print('df.columns:',np.array(df.columns))
            exit(0)
        #
    #
    elif plot_type == 'map':
        ## Plot profile locations on a map of the Arctic Ocean
        print('\t- Making a map')
        # Get the map plotting parameters from extra args
        map_dict = pp.extra_args
        # See what was in that dictionary
        try:
            map_extent = map_dict['map_extent']
        except:
            map_extent = None
        print('\t- Map extent:',map_extent)
        try:
            bbox = map_dict['bbox']
        except:
            bbox = None
        bb_zorder = 11
        print('\t- Map extent:',map_extent)
        #   Set latitude and longitude extents
        #   I found these values by guess-and-check, there really isn't a good way
        #       to know beforehand what you'll actually get
        if map_extent == 'Canada_Basin':
            cent_lon = -135 #-140
            ex_N = 78 #80
            ex_S = 73.5 #69
            ex_E = -145 #-156
            ex_W = -125 #-124
        elif map_extent == 'Western_Arctic':
            # print('\tWestern Arctic')
            # cent_lon = -140
            # ex_N = 84
            # ex_S = 69
            # ex_E = -162
            # ex_W = -124
            cent_lon = np.average(bps.lon_BGR)
            ex_N = 82
            ex_S = 71
            ex_E = -161
            ex_W = -129
        elif map_extent == 'AIDJEX_focus':
            cent_lon = -137
            ex_N = 78
            ex_S = 69
            ex_E = -155
            ex_W = -124
        else:
            cent_lon = -120
            ex_N = 90
            ex_S = 70
            ex_E = -180
            ex_W = 180
        # Create projection
        arctic_proj = ccrs.NorthPolarStereo(central_longitude=cent_lon)
        # Set higher resolution
        arctic_proj._threshold = 100
        # Remove the current axis
        ax.remove()
        # Replace axis with one that can be made into a map
        ax = fig.add_subplot(ax_pos, projection=arctic_proj)
        ax.set_extent([ex_E, ex_W, ex_S, ex_N], ccrs.PlateCarree())
        #   Add ocean first, then land. Otherwise the ocean covers the land shapes
        # ax.add_feature(cartopy.feature.OCEAN, color=bathy_clrs[0])
        ax.add_feature(cartopy.feature.LAND, color=clr_land, alpha=0.5)
        # Make bathymetry features
        def add_bathy_features(ax, add_colors=False, add_lines=False):
            bathy_0200 = cartopy.feature.NaturalEarthFeature(category='physical',name='bathymetry_K_200',scale='10m')
            bathy_1000 = cartopy.feature.NaturalEarthFeature(category='physical',name='bathymetry_J_1000',scale='10m')
            bathy_2000 = cartopy.feature.NaturalEarthFeature(category='physical',name='bathymetry_I_2000',scale='10m')
            bathy_3000 = cartopy.feature.NaturalEarthFeature(category='physical',name='bathymetry_H_3000',scale='10m')
            bathy_4000 = cartopy.feature.NaturalEarthFeature(category='physical',name='bathymetry_G_4000',scale='10m')
            bathy_5000 = cartopy.feature.NaturalEarthFeature(category='physical',name='bathymetry_F_5000',scale='10m')
            # Add bathymetry colors
            if add_colors:
                ax.add_feature(bathy_0200, facecolor=bathy_clrs[1], zorder=1)
                ax.add_feature(bathy_1000, facecolor=bathy_clrs[2], zorder=2)
                ax.add_feature(bathy_2000, facecolor=bathy_clrs[3], zorder=3)
                ax.add_feature(bathy_3000, facecolor=bathy_clrs[4], zorder=4)
                ax.add_feature(bathy_4000, facecolor=bathy_clrs[5], zorder=5)
                ax.add_feature(bathy_5000, facecolor=bathy_clrs[6], zorder=6)
            # Add bathymetry lines
            if add_lines:
                ax.add_feature(bathy_0200, facecolor='none', edgecolor=std_clr, linestyle='-.', alpha=0.3, zorder=1)
                ax.add_feature(bathy_1000, facecolor='none', edgecolor=std_clr, linestyle='-.', alpha=0.3, zorder=2)
                ax.add_feature(bathy_2000, facecolor='none', edgecolor=std_clr, linestyle='-.', alpha=0.3, zorder=3)
                ax.add_feature(bathy_3000, facecolor='none', edgecolor=std_clr, linestyle='-.', alpha=0.3, zorder=4)
                ax.add_feature(bathy_4000, facecolor='none', edgecolor=std_clr, linestyle='-.', alpha=0.3, zorder=5)
                ax.add_feature(bathy_5000, facecolor='none', edgecolor=std_clr, linestyle='-.', alpha=0.3, zorder=6)
        #   Add gridlines to show longitude and latitude
        gl = ax.gridlines(draw_labels=True, color=clr_lines, alpha=grid_alpha, linestyle='--', zorder=7)
        #       x is actually all labels around the edge
        gl.xlabel_style = {'size':font_size_lgnd, 'color':clr_lines, 'zorder':7}
        #       y is actually all labels within map
        gl.ylabel_style = {'size':font_size_lgnd, 'color':clr_lines, 'zorder':7}
        #   Plotting the coastlines takes a really long time
        # ax.coastlines()
        #   Don't plot outside the extent chosen
        if map_extent == 'Canada_Basin':
            # Add bathymetry features
            add_bathy_features(ax, add_colors=True, add_lines=True)
            if bbox == 'CB':
                # Only the Eastern boundary appears in this extent
                ax.plot([-130,-130], [73.7, 78.15], color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Eastern boundary
        elif map_extent == 'Western_Arctic':
            # Add bathymetry features
            add_bathy_features(ax, add_colors=False, add_lines=True)
            if bbox == 'BGR': # Beaufort Gyre Region
                CB_lons = np.linspace(-130, -160, 50)
                ax.plot(CB_lons, 81.5*np.ones(len(CB_lons)), color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Northern boundary
                ax.plot(CB_lons, 73*np.ones(len(CB_lons)), color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Southern boundary
                ax.plot([-130,-130], [73, 81.5], color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Eastern boundary
                ax.plot([-160,-160], [73, 81.5], color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Western boundary
            elif bbox == 'CB': # Canada Basin
                CB_lons = np.linspace(-130, -155, 50)
                ax.plot(CB_lons, 84*np.ones(len(CB_lons)), color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Northern boundary
                ax.plot(CB_lons, 72*np.ones(len(CB_lons)), color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Southern boundary
                ax.plot([-130,-130], [72, 84], color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Eastern boundary
                ax.plot([-155,-155], [72, 84], color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Western boundary
            elif bbox == 'AOA': # AIDJEX Operation Area
                AJ_lons = np.linspace(-133.7, -152.9, 50)
                ax.plot(AJ_lons, 77.4*np.ones(len(AJ_lons)), color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Northern boundary
                ax.plot(AJ_lons, 72.6*np.ones(len(AJ_lons)), color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Southern boundary
                ax.plot([-133.7,-133.7], [72.6, 77.4], color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Eastern boundary
                ax.plot([-152.9,-152.9], [72.6, 77.4], color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Western boundary
        elif map_extent == 'AIDJEX_focus':
            # Add bathymetry features
            add_bathy_features(ax, add_colors=True, add_lines=False)
            if bbox == 'BGR': # Beaufort Gyre Region
                CB_lons = np.linspace(-130, -160, 50)
                ax.plot(CB_lons, 81.5*np.ones(len(CB_lons)), color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Northern boundary
                ax.plot(CB_lons, 73*np.ones(len(CB_lons)), color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Southern boundary
                ax.plot([-130,-130], [73, 81.5], color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Eastern boundary
                ax.plot([-160,-160], [73, 81.5], color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Western boundary
            elif bbox == 'CB': # Canada Basin
                CB_lons = np.linspace(-130, -155, 50)
                # Northern boundary does not appear
                ax.plot(CB_lons, 72*np.ones(len(CB_lons)), color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Southern boundary
                ax.plot([-130,-130], [72, 80.5], color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Eastern boundary
                ax.plot([-155,-155], [72, 80.5], color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Western boundary
            elif bbox == 'AOA': # AIDJEX Operation Area
                AJ_lons = np.linspace(-133.7, -152.9, 50)
                ax.plot(AJ_lons, 77.4*np.ones(len(AJ_lons)), color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Southern boundary
                ax.plot(AJ_lons, 72.6*np.ones(len(AJ_lons)), color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Southern boundary
                ax.plot([-133.7,-133.7], [72.6, 77.4], color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Eastern boundary
                ax.plot([-152.9,-152.9], [72.6, 77.4], color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Western boundary
        else:
            # Add bathymetry features
            add_bathy_features(ax, add_colors=True, add_lines=False)
            bb_res = 50
            if bbox == 'BGR': # Beaufort Gyre Region
                CB_lons = np.linspace(-130, -160, 50)
                # map_bbox  = dict(clr='red', lw=2, ls='-', alpha=1.0)
                ax.plot(CB_lons, 81.5*np.ones(len(CB_lons)), color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Northern boundary
                ax.plot(CB_lons, 73*np.ones(len(CB_lons)), color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Southern boundary
                ax.plot([-130,-130], [73, 81.5], color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Eastern boundary
                ax.plot([-160,-160], [73, 81.5], color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Western boundary
                # # Add a polygon patch
                # # The coordinates need to be listed in counter-clockwise order
                # lat_corners = np.array([73, 73, 81.5, 81.5])
                # lon_corners = np.array([-160, -130, -130, -160])
                # # The above makes a box, below curves the top and bottom along latitude lines
                # top_lons = np.linspace(-130, -160, bb_res)
                # bot_lons = np.linspace(-160, -130, bb_res)
                # top_lats = np.zeros((len(top_lons)), np.float64) + 81.5
                # bot_lats = np.zeros((len(bot_lons)), np.float64) + 73.0
                # lon_corners = np.concatenate([top_lons, bot_lons])
                # lat_corners = np.concatenate([top_lats, bot_lats])
                # poly_corners = np.zeros((len(lat_corners), 2), np.float64)
                # poly_corners[:,0] = lon_corners
                # poly_corners[:,1] = lat_corners
                # poly_patch = plt.Polygon(poly_corners, closed=True, ec='r', fill=True, lw=0.5, fc='r', transform=ccrs.Geodetic(), zorder=bb_zorder)
                # ax.add_patch(poly_patch)
            elif bbox == 'CB': # Canada Basin
                CB_lons = np.linspace(-130, -155, 50)
                ax.plot(CB_lons, 84*np.ones(len(CB_lons)), color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Northern boundary
                ax.plot(CB_lons, 72*np.ones(len(CB_lons)), color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Southern boundary
                ax.plot([-130,-130], [72, 84], color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Eastern boundary
                ax.plot([-155,-155], [72, 84], color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Western boundary
            elif bbox == 'AOA': # AIDJEX Operation Area
                AJ_lons = np.linspace(-133.7, -152.9, 50)
                ax.plot(AJ_lons, 77.4*np.ones(len(AJ_lons)), color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Southern boundary
                ax.plot(AJ_lons, 72.6*np.ones(len(AJ_lons)), color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Southern boundary
                ax.plot([-133.7,-133.7], [72.6, 77.4], color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Eastern boundary
                ax.plot([-152.9,-152.9], [72.6, 77.4], color=map_bbox['clr'], linewidth=map_bbox['lw'], linestyle=map_bbox['ls'], transform=ccrs.Geodetic(), zorder=bb_zorder) # Western boundary
        # Concatonate all the pandas data frames together
        df = pd.concat(a_group.data_frames)
        # Check for cluster-based variables
        if clr_map in clstr_vars:
            m_pts, m_cls, cl_x_var, cl_y_var, cl_z_var, plot_slopes, b_a_w_plt = get_cluster_args(pp)
            df, rel_val, m_pts, m_cls, ell = HDBSCAN_(a_group.data_set.arr_of_ds, df, cl_x_var, cl_y_var, cl_z_var, m_pts, m_cls=m_cls, extra_cl_vars=[clr_map], re_run_clstr=re_run_clstr)
        print('\t- Plot slopes:',plot_slopes)
        # Drop dimensions, if needed
        if 'Vertical' in df.index.names and clr_map not in vertical_vars:
            # Drop duplicates along the `Time` dimension
            #   (need to make the index `Time` a column first)
            print('\t- Dropping `Vertical` dimension')
            df = df.droplevel('Vertical')
            df.reset_index(level=['Time'], inplace=True)
            df.drop_duplicates(inplace=True, subset=['Time'])
        # print('df:')
        # print(df)
        # exit(0)
        # Determine the color mapping to be used
        if clr_map == 'clr_all_same':
            # If not plotting very many points, increase the marker size
            # print('len(df):',len(df))
            if len(df) < 10:
                mrk_s = big_map_mrkr
                mrk_a = mrk_alpha
            elif len(df) > 10000:
                mrk_s = map_mrk_size*(2/3)
                mrk_a = 0.05
            else:
                mrk_s = map_mrk_size
                mrk_a = mrk_alpha3
            # print('mrk_s:',mrk_s)
            # print('mrk_a:',mrk_a)
            # Plot every point the same color, size, and marker
            ax.scatter(df['lon'], df['lat'], color=std_clr, s=mrk_s, marker=map_marker, alpha=0, linewidths=map_ln_wid, transform=ccrs.PlateCarree(), zorder=10)
            # Add a standard legend
            if pp.legend:
                add_std_legend(ax, df, 'lon', lgd_str=' profiles', lgd_loc=pp.legend)
            # Create the colorbar for the bathymetry
            if map_extent == 'Full_Arctic':
                cbar = plt.colorbar(bathy_smap, ax=ax, extend='min', shrink=0.7, pad=0.15)# fraction=0.05)
                # Fixing ticks
                # ticks_loc = cbar.ax.get_yticks().tolist()
                # Not sure why the above doesn't work, but this does, setting it kinda manually
                ticks_loc = np.linspace(0, 1, n_bathy+1)
                cbar.ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks_loc))
                # cbar.ax.set_yticklabels(['','3000','2000','1000','200','0'])
                cbar.ax.set_yticklabels(['','5000','4000','3000','2000','1000','200','0'])
                cbar.ax.tick_params(labelsize=font_size_ticks)
                cbar.set_label('Bathymetry (m)', size=font_size_labels)
            # Add a standard title
            plt_title = add_std_title(a_group)
            return pp.xlabels[0], pp.ylabels[0], None, plt_title, ax, False
        elif clr_map == 'source':
            # Get unique sources
            these_sources = np.unique(df['source'])
            # If not plotting very many points, increase the marker size
            if len(df) < 10:
                mrk_s = big_map_mrkr
            else:
                mrk_s = map_mrk_size
            i = 0
            lgnd_hndls = []
            notes_string = ''
            for source in these_sources:
                # Decide on the color, don't go off the end of the array
                my_clr = distinct_clrs[i%len(distinct_clrs)]
                # Get the data for just this source
                this_df = df[df['source'] == source]
                # Plot every point from this df the same color, size, and marker
                ax.scatter(this_df['lon'], this_df['lat'], color=my_clr, s=mrk_s, marker=map_marker, alpha=mrk_alpha, linewidths=map_ln_wid, transform=ccrs.PlateCarree(), zorder=10)
                i += 1
                # Add legend to report the total number of points for this instrmt
                lgnd_label = source+': '+str(len(this_df['lon']))+' profiles'
                lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label))
                notes_string = ''.join(this_df.notes.unique())
            # Only add the notes_string if it contains something
            if len(notes_string) > 1:
                notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
                lgnd_hndls.append(notes_patch)
            # Add legend with custom handles
            if pp.legend:
                if isinstance(pp.legend, type('string')):
                    lgnd = ax.legend(handles=lgnd_hndls, loc=pp.legend)
                else:   
                    lgnd = ax.legend(handles=lgnd_hndls, loc='lower left')
            # Add a standard title
            plt_title = add_std_title(a_group)
            return pp.xlabels[0], pp.ylabels[0], None, plt_title, ax, False
        elif clr_map == 'instrmt':
            # Add column where the source-instrmt combination ensures uniqueness
            df['source-instrmt'] = df['source']+' '+df['instrmt'].astype("string")
            # Get unique instrmts
            these_instrmts = np.unique(df['source-instrmt'])
            # If not plotting very many points, increase the marker size
            if len(df) < 10:
                if map_extent == 'Canada_Basin':
                    mrk_s = big_map_mrkr
                else:
                    mrk_s = map_mrk_size
            else:
                if map_extent == 'Canada_Basin':
                    mrk_s = map_mrk_size
                else:
                    mrk_s = sml_map_mrkr
            i = 0
            lgnd_hndls = []
            # Loop through each instrument
            for instrmt in these_instrmts:
                # Decide on the color, don't go off the end of the array
                my_clr = distinct_clrs[i%len(distinct_clrs)]
                # Get the data for just this instrmt
                this_df = df[df['source-instrmt'] == instrmt]
                # Plot every point from this df the same color, size, and marker
                ax.scatter(this_df['lon'], this_df['lat'], color=my_clr, s=mrk_s, marker=map_marker, alpha=map_alpha, linewidths=map_ln_wid, transform=ccrs.PlateCarree(), zorder=10)
                # Find first and last profile, based on entry
                entries = this_df['entry'].values
                first_pf_df = this_df[this_df['entry']==min(entries)]
                last_pf_df  = this_df[this_df['entry']==max(entries)]
                # Plot first and last profiles on map
                ax.scatter(first_pf_df['lon'], first_pf_df['lat'], color=my_clr, s=mrk_s*5, marker='>', alpha=map_alpha, linewidths=map_ln_wid, transform=ccrs.PlateCarree(), zorder=11)
                ax.scatter(last_pf_df['lon'], last_pf_df['lat'], color=my_clr, s=mrk_s*5, marker='s', alpha=map_alpha, linewidths=map_ln_wid, transform=ccrs.PlateCarree(), zorder=11)
                # Increase i to get next color in the array
                i += 1
                # Add legend to report the total number of points for this instrmt
                lgnd_label = instrmt+': '+str(len(this_df['lon']))+' profiles'
                lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label))
                notes_string = ''.join(this_df.notes.unique())
            # Only add the notes_string if it contains something
            if len(notes_string) > 1:
                notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
                lgnd_hndls.append(notes_patch)
            # Add legend entries for stopping and starting locations
            lgnd_hndls.append(mpl.lines.Line2D([],[],color=std_clr, label='Start', marker='>', linewidth=0))
            lgnd_hndls.append(mpl.lines.Line2D([],[],color=std_clr, label='Stop', marker='s', linewidth=0))
            # Add legend with custom handles
            if pp.legend:
                if isinstance(pp.legend, type('string')):
                    lgnd = ax.legend(handles=lgnd_hndls, loc=pp.legend)
                else:   
                    lgnd = ax.legend(handles=lgnd_hndls, loc='lower left')
            # Create the colorbar for the bathymetry
            if map_extent == 'Full_Arctic':
                cbar = plt.colorbar(bathy_smap, ax=ax, extend='min', shrink=0.7, pad=0.15)# fraction=0.05)
                # Fixing ticks
                # ticks_loc = cbar.ax.get_yticks().tolist()
                # Not sure why the above doesn't work, but this does, setting it kinda manually
                ticks_loc = np.linspace(0, 1, n_bathy+1)
                cbar.ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks_loc))
                # cbar.ax.set_yticklabels(['','3000','2000','1000','200','0'])
                cbar.ax.set_yticklabels(['','5000','4000','3000','2000','1000','200','0'])
                cbar.ax.tick_params(labelsize=font_size_ticks)
                cbar.set_label('Bathymetry (m)', size=font_size_labels)
            # Add a standard title
            plt_title = add_std_title(a_group)
            return pp.xlabels[0], pp.ylabels[0], None, plt_title, ax, False
        elif clr_map == 'clr_by_dataset':
            i = 0
            lgnd_hndls = []
            df_labels = [*a_group.data_set.sources_dict.keys()]
            # print('\tdf_labels:',df_labels)
            # Loop through each dataframe 
            #   which correspond to the datasets input to the Data_Set object's sources_dict
            for this_df in a_group.data_frames:
                # Make a new column for the instrmt-prof_no combination
                this_df['instrmt-prof_no'] = this_df['instrmt']+' '+this_df['prof_no'].astype("string")
                # Remove duplicates to get one row per profile
                this_df.drop_duplicates(subset=['instrmt-prof_no'], inplace=True)
                # print(this_df)
                # If not plotting very many points, increase the marker size
                if len(this_df) < 100:
                    if map_extent == 'Canada_Basin':
                        mrk_s = big_map_mrkr
                    else:
                        mrk_s = map_mrk_size
                    mrk_s = big_map_mrkr
                else:
                    if map_extent == 'Canada_Basin':
                        mrk_s = map_mrk_size
                    else:
                        mrk_s = sml_map_mrkr
                # Decide on the color, don't go off the end of the array
                my_clr = distinct_clrs[i%len(distinct_clrs)]
                # Plot every point from this df the same color, size, and marker
                ax.scatter(this_df['lon'], this_df['lat'], color=my_clr, s=mrk_s, marker=map_marker, alpha=1.0, linewidths=map_ln_wid, transform=ccrs.PlateCarree(), zorder=10)
                # Add legend to report the total number of points for this instrmt
                lgnd_label = str(df_labels[i][4:-7])+': '+str(len(this_df['lon']))+' profiles'
                if len(this_df['lon']) > 0 and len(this_df['lon']) < 5:
                    lgnd_label += ', '+str(list(np.unique(this_df['prof_no'].values)))
                lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label))
                notes_string = ''.join(this_df.notes.unique())
                # Increase i to get next color in the array
                i += 1
            # Only add the notes_string if it contains something
            if len(notes_string) > 1:
                notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
                lgnd_hndls.append(notes_patch)
            # Add legend with custom handles
            if pp.legend:
                if isinstance(pp.legend, type('string')):
                    lgnd = ax.legend(handles=lgnd_hndls, loc=pp.legend)
                else:   
                    lgnd = ax.legend(handles=lgnd_hndls, loc='lower left')
            # Add a standard title
            plt_title = add_std_title(a_group)
            return pp.xlabels[0], pp.ylabels[0], None, plt_title, ax, False
        else:
            # Make sure it isn't a vertical variable
            # if clr_map in vertical_vars:
            #     print('Cannot use',clr_map,'as colormap for a map plot')
            #     exit(0)
            # print('df.columns:',df.columns)
            # print(df)
            if clr_map not in np.array(df.columns):
                print('Error: Could not find variable',clr_map,'to use as colormap. Aborting script')
                exit(0)
            # If not plotting very many points, increase the marker size
            # print('len(df):',len(df))
            # exit(0)
            if len(df) < 10:
                mrk_s = big_map_mrkr
                mrk_a = mrk_alpha
            elif len(df) > 20000:
                mrk_s = map_mrk_size
                mrk_a = 0.05
            else:
                mrk_s = big_map_mrkr/2
                mrk_a = mrk_alpha
                # mrk_s = map_mrk_size
                # mrk_a = mrk_alpha3
            # mrk_s, mrk_a = get_marker_size_and_alpha(len(df['lon']))
            # Format the dates if necessary
            if clr_map == 'dt_start' or clr_map == 'dt_end':
                cmap_data = mpl.dates.date2num(df[clr_map])
            else:
                cmap_data = df[clr_map]
            # The color of each point corresponds to the number of the profile it came from
            heatmap = ax.scatter(df['lon'], df['lat'], c=cmap_data, cmap=get_color_map(clr_map), s=mrk_s, marker=map_marker, linewidths=map_ln_wid, transform=ccrs.PlateCarree(), zorder=10)
            # Check whether to fit a 2d polynomial to the data
            if plot_slopes:
                plot_polyfit2d(ax, pp, df['lon'], df['lat'], df[clr_map], in_3D=False, plot_type=plot_type)
            # Create the colorbar
            if True:
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                # create an axes on the right side of ax. The width of cax will be 5%
                # of ax and the padding between cax and ax will be fixed at 0.05 inch.
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.5, axes_class=mpl.axes.Axes)
                cbar = plt.colorbar(heatmap, cax=cax)
            else:
                cbar = plt.colorbar(heatmap, ax=ax)
            # Format the colorbar ticks, if necessary
            if clr_map == 'dt_start' or clr_map == 'dt_end':
                loc = mpl.dates.AutoDateLocator()
                cbar.ax.yaxis.set_major_locator(loc)
                cbar.ax.yaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(loc))
            # Change the colorbar limits, if necessary
            try:
                ax_lims_keys = list(pp.ax_lims.keys())
                if 'c_lims' in ax_lims_keys:
                    cbar.mappable.set_clim(vmin=min(pp.ax_lims['c_lims']), vmax=max(pp.ax_lims['c_lims']))
                    print('\t- Set c_lims to',pp.ax_lims['c_lims'])
            except:
                foo = 2
            # Invert colorbar if necessary
            if clr_map in y_invert_vars:
                cbar.ax.invert_yaxis()
            # cbar.set_label(a_group.data_set.var_attr_dicts[0][clr_map]['label'])
            cbar.set_label(pp.clabel)
            # Add a standard legend
            if pp.legend:
                add_std_legend(ax, df, 'lon', lgd_str=' profiles', lgd_loc=pp.legend)
            # Add a standard title
            plt_title = add_std_title(a_group)
            return pp.xlabels[0], pp.ylabels[0], None, plt_title, ax, False
    elif plot_type == 'profiles':
        # Call profile plotting function
        return plot_profiles(ax, a_group, pp)
    elif plot_type == 'waterfall':
        # Call waterfall plotting function
        return plot_waterfall(ax, a_group, fig, ax_pos, pp)
    else:
        # Did not provide a valid plot type
        print('Plot type',plot_type,'not valid')
        exit(0)
    #
    print('ERROR: Should not have gotten here. Aborting script')
    exit(0)

################################################################################
# Auxiliary plotting functions #################################################
################################################################################

def parse_LHW_AW_keys(x_key, y_key):
    """
    Takes in the x and y keys and returns the corresponding LHW and AW keys

    x_key       A string of the x key
    y_key       A string of the y key
    """
    return_keys = []
    # Check for the LHW and AW keys
    for ext in ['_TC_min', '_TC_max']:
        for key in [x_key, y_key]:
            affix = ''
            suffix = ''
            if 'ca_' in key:
                # Take out first 3 characters of the string to leave the original variable name
                key_str = key[3:]
                affix = 'av_'
            elif 'trd_' in key:
                # Take out first 4 characters of the string to leave the original variable name
                key_str = key[4:]
                affix = 'trd_'
            else:
                key_str = key
            if '-fit' in key:
                # Take out last 4 characters of the string to leave the original variable name
                key_str = key_str[:-4]
                suffix = '-fit'
            if key_str == 'sigma':
                key_str = 'sig'
            if key_str in ['dt_start', 'dt_end']:
                return_keys.append(key_str)
            else:
                return_keys.append(affix + key_str + ext + suffix)
    return return_keys[0], return_keys[1], return_keys[2], return_keys[3]

def mark_the_LHW_AW(ax, x_key, y_key):
    """
    Marks the location of the LHW and AW boundaries on the plot

    ax          The axis on which to mark the boundaries
    x_key       The string of the name for the x data
    y_key       The string of the name for the y data
    """
    print('\t- Marking the LHW and AW boundaries')
    # Parse the bounds keys
    x_LHW_key, y_LHW_key, x_AW_key, y_AW_key = parse_LHW_AW_keys(x_key, y_key)
    # Load the bounds data
    this_BGR = 'BGR_all'
    # this_BGR = 'BGR1516'
    bnds_df = pl.load(open('outputs/'+this_BGR+'_LHW_AW_properties.pickle', 'rb'))
    # Find the x and y values for the LHW and AW
    var_arrs = []
    for var_key in [x_LHW_key, y_LHW_key, x_AW_key, y_AW_key]:
        if var_key in ['dt_start', 'dt_end']:
            var_arrs.append(mpl.dates.date2num(bnds_df[var_key].values))
        else:
            var_arrs.append(bnds_df[var_key].values)
    # Get marker size and alpha
    mrk_s, mrk_a = get_marker_size_and_alpha(len(var_arrs[0]))
    mrk_a = mrk_a*2
    # Plot the LHW and AW
    # ax.scatter(var_arrs[0], var_arrs[1], color=jackson_clr[6], s=mrk_s, marker='v', zorder=4, alpha=mrk_a)
    # ax.scatter(var_arrs[2], var_arrs[3], color=jackson_clr[3], s=mrk_s, marker='^', zorder=4, alpha=mrk_a)
    s_to_markersize = np.sqrt(mrk_s)
    ax.plot(var_arrs[0], var_arrs[1], color=LHW_clr, markersize=s_to_markersize, marker=LHW_mrk, fillstyle='bottom', markerfacecoloralt=LHW_facealtclr, markeredgecolor=LHW_edgeclr, linewidth=0, alpha=mrk_a, zorder=4)
    ax.plot(var_arrs[2], var_arrs[3], color=AW_clr, markersize=s_to_markersize, marker=AW_mrk, fillstyle='top', markerfacecoloralt=AW_facealtclr, markeredgecolor=AW_edgeclr, linewidth=0, alpha=mrk_a, zorder=4)

################################################################################

def get_display_unit_string(var_key):
    """
    Returns a string of the unit associated with var_key in a display format

    var_key     The string of the name for the data, a column header in the dataframe
    """
    if var_key in ['ca_press', 'nzca_pcs_press']:
        return r'dbar'
    elif var_key == 'ca_SA':
        return r'(g/kg)'
    elif var_key == 'ca_CT':
        return r' ^\circ C'
    elif var_key == 'ca_FH':
        return r'W/m^2'
    elif var_key == 'trd_press-fit':
        return r'dbar/yr'
    elif var_key == 'trd_SA-fit':
        return r'(g/kg)/yr'
    elif var_key == 'trd_CT-fit':
        return r' ^\circ C/yr'
    elif var_key == 'percnztrd_pcs_press':
        return r' \%/yr'
    else:
        return ''

def add_linear_slope(ax, pp, df, x_data, y_data, x_key, y_key, linear_clr, plot_slopes, anno_prefix=''):
    """
    Adds a line of best fit to the data

    ax          The axis on which to add the line
    pp          The Plot_Params object
    df          Pandas dataframe of the data
    x_data      The x data to use for the line of best fit
    y_data      The y data to use for the line of best fit
    x_key       The string of the name for the x data
    y_key       The string of the name for the y data
    linear_clr  The color to make the line
    plot_slopes The type of line to plot, either 'OLS' or 'TLS'
    """
    unit_str = '\ '+get_display_unit_string(x_key)
    # print('x_key:',x_key,'unit_str:',unit_str)
    if x_key in ['dt_start', 'dt_end']:
        per_unit = '/yr'
        adjustment_factor = 365.25 # units for dt_start and dt_end are in days, adjust to years
    else:
        per_unit = get_display_unit_string(y_key)
        if not per_unit == '':
            per_unit = '/'+per_unit
        adjustment_factor = 1
    # Check whether to switch x and y for finding the slope
    if x_key in ['BSP', 'BSA'] or 'trd_' in x_key or 'ca_' in x_key or 'pcs_' in x_key:
        switch_xy = True
        # Switch x and y data
        x_data_ = y_data
        y_data_ = x_data
    else:
        switch_xy = False
        x_data_ = x_data
        y_data_ = y_data
    if plot_slopes == 'OLS':
        # Find the slope of the ordinary least-squares of the points for this cluster
        # m, c = np.linalg.lstsq(np.array([x_data, np.ones(len(x_data))]).T, y_data, rcond=None)[0]
        # sd_m, sd_c = 0, 0
        m, c, rvalue, pvalue, sd_m = stats.linregress(x_data_, y_data_)
        # m, c, sd_m, sd_c = reg.slope, reg.intercept, reg.stderr, reg.intercept_stderr
        print('\t\t- Slope is',m*adjustment_factor,'+/-',sd_m*adjustment_factor,unit_str+per_unit,'for',len(x_data_),'points')
        print('\t\t- Intercept is',c)
        print('\t\t- R^2 value is',rvalue)
    else:
        # Find the slope of the total least-squares of the points for this cluster
        m, c, sd_m, sd_c = orthoregress(x_data_, y_data_)
        print('\t\t- Slope is',m*adjustment_factor,'+/-',sd_m*adjustment_factor,unit_str+per_unit,'for',len(x_data_),'points')
        print('\t\t- Intercept is',c,'+/-',sd_c)
    # Find mean and standard deviation of x and y data
    # x_mean = df.loc[:,x_key].mean()
    # y_mean = df.loc[:,y_key].mean()
    # x_stdv = df.loc[:,x_key].std()
    # y_stdv = df.loc[:,y_key].std()
    x_mean = np.mean(x_data)
    y_mean = np.mean(y_data)
    x_stdv = np.std(x_data)
    y_stdv = np.std(y_data)
    # Get bounds of axes
    #   If there are x_lims and y_lims provided, use those
    try:
        x_bnds = pp.ax_lims['x_lims']
        if isinstance(x_bnds, type(None)):
            x_bnds = ax.get_xbound()
    except:
        x_bnds = ax.get_xbound()
    try:
        y_bnds = pp.ax_lims['y_lims']
        if isinstance(y_bnds, type(None)):
            y_bnds = ax.get_ybound()
    except:
        y_bnds = ax.get_ybound()
    # Check whether to convert the x_bnds from strings to numbers
    if x_key in ['dt_start', 'dt_end'] and isinstance(x_bnds[0], str):
        # Replace the '/'s with '-'s in x_bnds
        x_bnds = [x_bnds[0].replace('/','-'), x_bnds[1].replace('/','-')]
        x_bnds = mpl.dates.date2num(x_bnds)
    print('\t\t- x_bnds:',x_bnds)
    x_span = abs(x_bnds[1] - x_bnds[0])
    y_span = abs(y_bnds[1] - y_bnds[0])
    # Add annotation to say what the slope is
    annotation_string = format_sci_notation(m*adjustment_factor, pm_val=sd_m*adjustment_factor, sci_lims_f=(0,3))
    if switch_xy:
        # Annotate the inverse of the slope
        # annotation_string = r'1/ '+annotation_string
        # Plot the least-squares fit line for this cluster through the centroid
        ax.axline((x_mean, y_mean), slope=1/m, color=linear_clr, linewidth=2, zorder=6)
    else:
        # Plot the least-squares fit line for this cluster through the centroid
        ax.axline((x_mean, y_mean), slope=m, color=linear_clr, linewidth=2, zorder=6)
    
    # Add an auxiliary legend to note the slope
    #   Note: because of a really annoying limitation on not being able to use add_artist
    #   programatically with multiple subplots, this only works if you aren't otherwise
    #   adding a legend. If you try to use add_artist, you'll get this error:
    #   ValueError: Can not reset the axes. You are probably trying to re-use an artist in more than one Axes which is not supported
    temp_lgnd = ax.legend(handles=[mpl.lines.Line2D([],[],color=linear_clr, label=r'$\mathbf{'+anno_prefix+annotation_string+'}$', marker=None, linewidth=0),
                                   mpl.lines.Line2D([],[],color=linear_clr, label=r'$\mathbf{'+unit_str+per_unit+'}$', marker=None, linewidth=0)], fontsize="12")
    # Set color of legend text
    for lgnd_line, lgnd_text in zip(temp_lgnd.get_lines(), temp_lgnd.get_texts()):
        lgnd_text.set_color(linear_clr)
    
    # Below code still works, but the above code automatically puts the label in a place where it likely doesn't overlap
    # Put annotation on the plot
    # anno_x = x_mean+x_stdv/4
    # anno_x = max(x_bnds) - 5*x_span/12
    # anno_y = y_mean+y_stdv/0.5
    # if dark_mode:
    #     ax.annotate(anno_prefix+annotation_string+per_unit, xy=(anno_x, anno_y), xycoords='data', color=linear_clr, fontsize=font_size_lgnd, fontweight='bold', bbox=anno_bbox, zorder=12)
    # else:
    #     ax.annotate(r'$\mathbf{'+anno_prefix+annotation_string+per_unit+'}$', xy=(anno_x, anno_y), xycoords='data', color=linear_clr, fontsize=font_size_lgnd, fontweight='bold', bbox=anno_bbox, zorder=12)

################################################################################

def add_isopycnals(ax, df, x_key, y_key, p_ref=None, place_isos=False, tw_x_key=None, tw_ax_y=None, tw_y_key=None, tw_ax_x=None):
    """
    Adds lines of constant density anomaly, if applicable

    ax          The main axis on which to format
    df          Pandas dataframe of the data these isopycnals will be plotted under
    x_key       The string of the name for the x data on the main axis
    y_key       The string of the name for the y data on the main axis
    p_ref       Value of pressure to reference the isopycnals to. Default is surface (0 dbar)
    place_isos  How to place the isopycnals, either 'auto' or 'manual'
    tw_x_key    The string of the name for the x data on the twin axis
    tw_ax_y     The twin y axis on which to format
    tw_y_key    The string of the name for the y data on the twin axis
    tw_ax_x     The twin x axis on which to format
    """
    # Check whether to make isopycnals or not
    intersect_arr = set([x_key, y_key, tw_x_key, tw_y_key]).intersection(['iT', 'CT', 'PT', 'SP', 'SA'])
    if len(intersect_arr) < 2:
        intersect_arr = set([x_key, y_key, tw_x_key, tw_y_key]).intersection(['aiT','aCT','aPT','BSP','BSt','BSA'])
        if len(intersect_arr) < 2:
            return
        else:
            ## Plot lines of slope -1 all across the domain
            # Get bounds of axes
            x_bnds = ax.get_xbound()
            y_bnds = ax.get_ybound()
            # print('x_bnds:',x_bnds)
            # print('y_bnds:',y_bnds)
            x_arr = np.linspace(x_bnds[0], x_bnds[1], 50)
            y_arr = [(y_bnds[1]-y_bnds[0])/2]*len(x_arr)
            # Plot the least-squares fit line for this cluster through the centroid
            for i in range(len(x_arr)):
                ax.axline((x_arr[i], y_arr[i]), slope=1, color=std_clr, alpha=0.25, linestyle='--', zorder=1)
            # Set offset on axis, incredibly specific for making a nice plot for the paper
            ax.ticklabel_format(axis='x', style='sci', scilimits=sci_lims, useMathText=True, useOffset=0.0268)
            return
    # Zoom in on data so there's no border space
    x_var_s = df.loc[:,x_key]
    x_var_min = x_var_s.min()
    x_var_max = x_var_s.max()
    ax.set_xlim([x_var_min, x_var_max])
    y_var_s = df.loc[:,y_key]
    y_var_min = y_var_s.min()
    y_var_max = y_var_s.max()
    ax.set_ylim([y_var_min, y_var_max])
    # Get bounds of axes
    x_bnds = ax.get_xbound()
    y_bnds = ax.get_ybound()
    # print('x_bnds:',x_bnds)
    # print('y_bnds:',y_bnds)
    # Figure out which orientation to make the grid
    if x_key in ['iT','CT','PT']:
        iso_x_key = 'iT'
    elif x_key in ['SP','SA']:
        iso_x_key = 'SA'
    if y_key in ['iT','CT','PT']:
        iso_y_key = 'iT'
    elif y_key in ['SP','SA']:
        iso_y_key = 'SA'
    # If no reference pressure is given, take p_ref to be the median pressure
    if p_ref==True:
        press_s = df.loc[:, 'press']
        p_ref = press_s.median()
    print('\t- Adding isopycnals referenced to:',p_ref,'dbar')
    # Get bounds of iso keys that fit within axes bounds
    iso_x_min = df[(df[x_key] == x_var_min)].loc[:,iso_x_key].min()
    iso_x_max = df[(df[x_key] == x_var_max)].loc[:,iso_x_key].max()
    iso_y_min = df[(df[y_key] == y_var_min)].loc[:,iso_y_key].min()
    iso_y_max = df[(df[y_key] == y_var_max)].loc[:,iso_y_key].max()
    # Number of points for the mesh grid in both directions
    n_grid_pts = 100
    # Make meshgrid for calculating
    iso_x_arr = np.arange(iso_x_min, iso_x_max, abs(iso_x_max-iso_x_min)/n_grid_pts)
    iso_y_arr = np.arange(iso_y_min, iso_y_max, abs(iso_y_max-iso_y_min)/n_grid_pts)
    iso_X, iso_Y = np.meshgrid(iso_x_arr, iso_y_arr)
    # Make grid of pressure values
    P = iso_X*0+p_ref
    # Calculate potential density referenced to p_ref
    if iso_x_key == 'SA' and iso_y_key == 'iT':
        Z = gsw.pot_rho_t_exact(iso_X, iso_Y, P, p_ref)-1000
    # Make meshgrid for plotting
    x_arr = np.arange(x_var_min, x_var_max, abs(x_var_max-x_var_min)/n_grid_pts)
    y_arr = np.arange(y_var_min, y_var_max, abs(y_var_max-y_var_min)/n_grid_pts)
    X, Y = np.meshgrid(x_arr, y_arr)
    # Plot the contours
    CS = ax.contour(X, Y, Z, colors=std_clr, alpha=0.4, zorder=1, linestyles='dashed')
    # Place contour inline labels
    if place_isos == 'manual':
        # Place contour labels manualy, interactively
        ax.clabel(CS, manual=True, fontsize=10, fmt='%.1f')
    else:
        # Place contour labels automatically
        ax.clabel(CS, inline=True, fontsize=10, fmt='%.1f')

################################################################################

def plot_histogram(a_group, ax, pp, df, x_key, y_key, clr_map, legend=True, txk=None, tay=None, tyk=None, tax=None, clstr_dict=None, df_w_noise=None):
    """
    Takes in an Analysis_Group object which has the data and plotting parameters
    to produce a subplot of individual profiles. Returns the x and y labels and
    the subplot title

    a_group     A Analysis_Group object containing the info to create this subplot
    ax          The axis on which to make the plot
    pp          The Plot_Parameters object for a_group
    df          A pandas dataframe
    x_key       The string of the name for the x data on the main axis
    y_key       The string of the name for the y data on the main axis
    clr_map     A string to determine what color map to use in the plot
    legend      True/False whether to add a legend
    txk         Twin x variable key
    tay         Twin y axis
    tyk         Twin y variable key
    tax         Twin x axis
    clstr_dict  A dictionary of clustering values: m_pts, rel_val
    df_w_noise  A pandas dataframe that includes the noise points
    """
    print('\t- Plotting a histogram')
    # print(df)
    # Find the histogram parameters, if given
    if not isinstance(pp.extra_args, type(None)):
        try:
            n_h_bins = pp.extra_args['n_h_bins']
        except:
            n_h_bins = None
        try:
            bin_size = pp.extra_args['bin_size']
        except:
            bin_size = None
        try:
            plt_hist_lines = pp.extra_args['plt_hist_lines']
        except:
            plt_hist_lines = False
        try:
            pdf_hist = pp.extra_args['pdf_hist']
        except:
            pdf_hist = False
        try:
            plt_noise = pp.extra_args['plt_noise']
        except:
            plt_noise = True
        try:
            sort_clstrs = pp.extra_args['sort_clstrs']
        except:
            sort_clstrs = True
        try:
            log_axes = pp.extra_args['log_axes']
        except:
            log_axes = 'None'
        try:
            clstrs_to_plot = pp.extra_args['clstrs_to_plot']
        except:
            clstrs_to_plot = []
        try:
            mv_avg = pp.extra_args['mv_avg']
        except:
            mv_avg = False
    else:
        n_h_bins = None
        plt_hist_lines = False
        pdf_hist = False
        plt_noise = True
        sort_clstrs = True
        log_axes = 'None'
        clstrs_to_plot = []
        mv_avg = False
    # Load in variables
    if x_key == 'hist':
        var_key = y_key
        orientation = 'horizontal'
        x_label = 'Number of points'
        y_label = pp.ylabels[0]
        tw_var_key = tyk
        tw_ax = tax
        tw_label = pp.ylabels[1]
    elif y_key == 'hist':
        var_key = x_key
        orientation = 'vertical'
        x_label = pp.xlabels[0]
        y_label = 'Number of points'
        tw_var_key = txk
        tw_ax = tay
        tw_label = pp.xlabels[1]
    # Histograms don't work with datetime data
    if var_key in ['dt_start', 'dt_end']:
        print('Cannot make histogram with',var_key,'data')
        exit(0)
    # Set variable as to whether to invert the y axis
    invert_y_axis = False
    # Determine the color mapping to be used
    if clr_map in ['clr_all_same', 'fnn', 'bnn']:
        # Check whether to remove noise points
        if plt_noise == False or plt_noise == 'both':
            df = df[df.cluster!=-1]
        if plt_noise == 'both':
            # Get histogram parameters for the dataframe that includes noise points
            wn_h_var, wn_res_bins, wn_median, wn_mean, wn_std_dev = get_hist_params(df_w_noise, var_key, n_h_bins, bin_size)
            # Set label of the line where noise points have been removed
            no_noise_label = 'Stage 3'
        else:
            no_noise_label = None
        # Get histogram parameters
        h_var, res_bins, median, mean, std_dev = get_hist_params(df, var_key, n_h_bins, bin_size)
        # Plot the histogram
        if pdf_hist:
            # Get the histogram
            this_hist, these_bin_edges = np.histogram(h_var, bins=res_bins)
            # Estimate the (non-normalized) PDF from the histogram by 
            #   plotting line through the bin centers
            bin_centers = 0.5*(these_bin_edges[1:]+these_bin_edges[:-1])
            if orientation == 'vertical':
                pdf_x_arr = bin_centers
                pdf_y_arr = this_hist
                # Add vertical lines at all the salinity divisions and the moving average line
                if var_key == 'SA':
                    for SA_div in bps.BGR_HPC_SA_divs:
                        ax.axvline(SA_div, color=SA_divs_clr, linestyle=SA_divs_line)
                # Add moving average line from the csv file
                if plt_noise == False or plt_noise == 'both':
                    if not isinstance(mv_avg, type(None)):
                        # Load the data from the csv
                        hist_df = pd.read_csv('outputs/SA_divs_mv_avg_line.csv')
                        # Remove rows where 'this_hist_mv_avg' is null
                        hist_df = hist_df[hist_df['this_hist_mv_avg'].notnull()]
                        # Narrow range to just what is within this histograms bin centers
                        hist_df = hist_df[hist_df['bin_centers'] > min(pdf_x_arr)]
                        hist_df = hist_df[hist_df['bin_centers'] < max(pdf_x_arr)]
                        # Plot the line
                        ax.plot(hist_df['bin_centers'], hist_df['this_hist_mv_avg'], color=SA_divs_clr, linestyle=SA_divs_line, label='Moving average')
            else:
                pdf_x_arr = this_hist
                pdf_y_arr = bin_centers
            ax.plot(pdf_x_arr, pdf_y_arr, color=std_clr, linestyle='-', label=no_noise_label)
            if plt_noise == 'both':
                # Get the histogram
                wn_this_hist, wn_these_bin_edges = np.histogram(wn_h_var, bins=wn_res_bins)
                # Estimate the (non-normalized) PDF from the histogram by
                #   plotting line through the bin centers
                wn_bin_centers = 0.5*(wn_these_bin_edges[1:]+wn_these_bin_edges[:-1])
                if orientation == 'vertical':
                    wn_pdf_x_arr = wn_bin_centers
                    wn_pdf_y_arr = wn_this_hist
                else:
                    wn_pdf_x_arr = wn_this_hist
                    wn_pdf_y_arr = wn_bin_centers
                # Plot the line of the histogram
                ax.plot(wn_pdf_x_arr, wn_pdf_y_arr, color=std_clr, alpha=0.5, linestyle='--', label='Stage 2')
        else:
            ax.hist(h_var, bins=res_bins, color=std_clr, orientation=orientation)
        # Invert y-axis if specified
        if y_key in y_invert_vars:
            invert_y_axis = True
        # Check whether to plot lines for mean and standard deviation
        if plt_hist_lines:
            if orientation == 'vertical':
                ax.axvline(mean, color='r')
                ax.axvline(mean-2*std_dev, color='r', linestyle='--')
                ax.axvline(mean+2*std_dev, color='r', linestyle='--')
            elif orientation == 'horizontal':
                ax.axhline(mean, color='r')
                ax.axhline(mean-2*std_dev, color='r', linestyle='--')
                ax.axhline(mean+2*std_dev, color='r', linestyle='--')
        # Take the moving average across a histogram
        if False:# mv_avg and pdf_hist:
            # Make a new pandas dataframe with columns for bin_centers and this_hist
            hist_df = pd.DataFrame({'bin_centers':bin_centers, 'this_hist':this_hist})
            # Find the spacing between the bin centers (assuming np.histogram does evenly spaced bins)
            bin_spacing = abs(bin_centers[1]-bin_centers[0])
            # Find number of bin_spacings that most closely matches the window size
            window_size = int(mv_avg/bin_spacing)
            # Take the moving average across the histogram
            print('\t- Taking moving average across the histogram, window size:',window_size,'rows')
            hist_df['this_hist_mv_avg'] = hist_df['this_hist'].rolling(window=window_size, center=True, win_type='boxcar').mean()
            if orientation == 'vertical':
                mv_avg_x_key = 'bin_centers'
                mv_avg_y_key = 'this_hist_mv_avg'
            else:
                mv_avg_x_key = 'this_hist_mv_avg'
                mv_avg_y_key = 'bin_centers'
            ax.plot(hist_df[mv_avg_x_key], hist_df[mv_avg_y_key], color=SA_divs_clr, linestyle=SA_divs_line)
            # Save this line to a csv
            hist_df.to_csv('outputs/SA_divs_mv_avg_line.csv')
            # Find the intersection points of the moving average with the histogram
            if True:
                # Make a new array to store values of min_x
                min_x_arr = []
                # Find the difference between the moving average and the histogram
                hist_df['diff'] = hist_df['this_hist'] - hist_df['this_hist_mv_avg']
                # Find the points where the difference changes sign
                hist_df['sign_change'] = np.sign(hist_df['diff'].shift(1)) != np.sign(hist_df['diff'])
                # Get just the rows where diff is not nan
                hist_df = hist_df[~np.isnan(hist_df['diff'])]
                # Eliminate window_size//2 points at the beginning and end
                # hist_df = hist_df[window_size//2:len(hist_df)-window_size//2]
                # Find just the rows where the sign changes
                hist_df_intsec = hist_df[hist_df['sign_change']]
                # For each pair of intersection points, 
                for i in range(0, len(hist_df_intsec)-1):
                    # Get the x values of the intersection points
                    x1 = hist_df_intsec['bin_centers'].iloc[i]
                    x2 = hist_df_intsec['bin_centers'].iloc[i+1]
                    if x1 == x2:
                        print('x1 and x2 are the same')
                        exit(0)
                    elif np.isnan(x1) or np.isnan(x2):
                        print('x1 or x2 is NaN')
                        continue
                    # print('x1:',x1)
                    # print('x2:',x2)
                    # Get just the rows between these points
                    hist_df_between = hist_df[(hist_df['bin_centers'] > x1) & (hist_df['bin_centers'] < x2)]
                    # Check to make sure there was at least one row between those points
                    if len(hist_df_between) > 1:
                        # print('\thist_df_between[diff]:')
                        # print(hist_df_between['diff'])
                        # If the 'diff' values between these points are negative,
                        #   then find the minimum of 'this_hist' between these points
                        if hist_df_between['diff'].max() < 0:
                            # Find the minimum of 'this_hist' between these points
                            min_this_hist = hist_df_between['this_hist'].min()
                            # print('\tmin_this_hist:',min_this_hist)
                            min_x = hist_df_between['bin_centers'][hist_df_between['this_hist'] == min_this_hist].values[0]
                            # print('\tmin_x:',min_x)
                            min_x_arr.append(min_x)
                            # Plot vertical lines at the minimums between intersection points
                            ax.axvline(min_x, color=alt_std_clr, linestyle='--')
                        #
                    #
                print('\t- Found',len(min_x_arr),'minimums between intersection points')
                print(min_x_arr)
            #
        # Add legend to report overall statistics
        n_pts_patch   = mpl.patches.Patch(color=std_clr, label=str(len(h_var))+' points')
        median_patch  = mpl.patches.Patch(color=std_clr, label='Median:  '+'%.4f'%median)
        mean_patch    = mpl.patches.Patch(color=std_clr, label='Mean:    ' + '%.4f'%mean)
        std_dev_patch = mpl.patches.Patch(color=std_clr, label='Std dev: '+'%.4f'%std_dev)
        hndls = [n_pts_patch, median_patch, mean_patch, std_dev_patch]
        # notes_string = ''.join(df.notes.unique())
        notes_string = ''
        # Plot twin axis if specified
        if not isinstance(tw_var_key, type(None)):
            print('\t- Plotting histogram on twin axis')
            tw_clr = get_var_color(tw_var_key)
            if tw_clr == std_clr:
                tw_clr = alt_std_clr
            # Get histogram parameters
            h_var, res_bins, median, mean, std_dev = get_hist_params(df, tw_var_key, n_h_bins, bin_size)
            # Plot the histogram
            tw_ax.hist(h_var, bins=res_bins, color=tw_clr, alpha=hist_alpha, orientation=orientation)
            if orientation == 'vertical':
                tw_ax.set_xlabel(tw_label)
                tw_ax.xaxis.label.set_color(tw_clr)
                tw_ax.tick_params(axis='x', colors=tw_clr)
            elif orientation == 'horizontal':
                tw_ax.set_ylabel(tw_label)
                tw_ax.yaxis.label.set_color(tw_clr)
                tw_ax.tick_params(axis='y', colors=tw_clr)
                # Invert twin y-axis if specified
                if tw_var_key in y_invert_vars:
                    tw_ax.invert_yaxis()
                    print('\t- Inverting twin y-axis')
            # Check whether to plot lines for mean and standard deviation
            if plt_hist_lines:
                if orientation == 'vertical':
                    tw_ax.axvline(mean, color='r', alpha=0.5)
                    tw_ax.axvline(mean-2*std_dev, color='r', linestyle='--', alpha=0.5)
                    tw_ax.axvline(mean+2*std_dev, color='r', linestyle='--', alpha=0.5)
                elif orientation == 'horizontal':
                    tw_ax.axhline(mean, color='r', alpha=0.5)
                    tw_ax.axhline(mean-2*std_dev, color='r', linestyle='--', alpha=0.5)
                    tw_ax.axhline(mean+2*std_dev, color='r', linestyle='--', alpha=0.5)
            # Check whether to adjust the twin axes limits
            if not isinstance(pp.ax_lims, type(None)):
                if 'tw_y_lims' in pp.ax_lims.keys():
                    tw_ax.set_ylim(pp.ax_lims['tw_y_lims'])
                if 'tw_x_lims' in pp.ax_lims.keys():
                    tw_ax.set_xlim(pp.ax_lims['tw_x_lims'])
            # Add legend to report overall statistics
            n_pts_patch   = mpl.patches.Patch(color=tw_clr, label=str(len(h_var))+' points', alpha=hist_alpha)
            median_patch  = mpl.patches.Patch(color=tw_clr, label='Median:  '+'%.4f'%median, alpha=hist_alpha)
            mean_patch    = mpl.patches.Patch(color=tw_clr, label='Mean:    ' + '%.4f'%mean, alpha=hist_alpha)
            std_dev_patch = mpl.patches.Patch(color=tw_clr, label='Std dev: '+'%.4f'%std_dev, alpha=hist_alpha)
            tw_notes_string = ''.join(df.notes.unique())
            # Only add the notes_string if it contains something
            if len(tw_notes_string) > 1:
                tw_notes_patch  = mpl.patches.Patch(color='none', label=notes_string, alpha=hist_alpha)
                tw_hndls=[n_pts_patch, median_patch, mean_patch, std_dev_patch, tw_notes_patch]
            else:
                tw_hndls=[n_pts_patch, median_patch, mean_patch, std_dev_patch]
        else:
            tw_hndls = []
        # Only add the notes_string if it contains something
        if legend:
            if plt_noise == 'both':
                ax.legend()
            elif len(notes_string) > 1:
                notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
                ax.legend(handles=hndls+notes_patch+tw_hndls)
            else:
                ax.legend(handles=hndls+tw_hndls)
        # Check whether to change any axes to log scale
        if len(log_axes) == 3:
            print('\t- Changing axes to log scale: ',log_axes)
            if log_axes[0]:
                ax.set_xscale('log')
            if log_axes[1]:
                ax.set_yscale('log')
            if log_axes[2]:
                ax.set_zscale('log')
        # Add a standard title
        plt_title = add_std_title(a_group)
        return x_label, y_label, None, plt_title, ax, invert_y_axis
    elif clr_map == 'source':
        # Get unique sources
        these_sources = np.unique(df['source'])
        i = 0
        lgnd_hndls = []
        for source in these_sources:
            # Decide on the color, don't go off the end of the array
            my_clr = distinct_clrs[i%len(distinct_clrs)]
            # Get the data for just this source
            this_df = df[df['source'] == source]
            # Get histogram parameters
            h_var, res_bins, median, mean, std_dev = get_hist_params(this_df, var_key, n_h_bins, bin_size)
            # Plot the histogram
            ax.hist(h_var, bins=res_bins, color=my_clr, alpha=hist_alpha, orientation=orientation)
            # Check whether to plot lines for mean and standard deviation
            if plt_hist_lines:
                if orientation == 'vertical':
                    ax.axvline(mean, color='r')
                    ax.axvline(mean-2*std_dev, color='r', linestyle='--')
                    ax.axvline(mean+2*std_dev, color='r', linestyle='--')
                elif orientation == 'horizontal':
                    ax.axhline(mean, color='r')
                    ax.axhline(mean-2*std_dev, color='r', linestyle='--')
                    ax.axhline(mean+2*std_dev, color='r', linestyle='--')
            i += 1
            # Add legend handle to report the total number of points for this source
            lgnd_label = source+': '+str(len(this_df[var_key]))+' points, Median:'+'%.4f'%median
            lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label, alpha=hist_alpha))
            # Add legend handle to report overall statistics
            lgnd_label = 'Mean:'+ '%.4f'%mean+', Std dev:'+'%.4f'%std_dev
            lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label, alpha=hist_alpha))
            notes_string = ''.join(this_df.notes.unique())
        # Only add the notes_string if it contains something
        if len(notes_string) > 1:
            notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
            lgnd_hndls.append(notes_patch)
        # Add legend with custom handles
        if legend:
            lgnd = ax.legend(handles=lgnd_hndls)
        # Invert y-axis if specified
        if y_key in y_invert_vars:
            invert_y_axis = True
        # Check whether to change any axes to log scale
        if len(log_axes) == 3:
            if log_axes[0]:
                ax.set_xscale('log')
            if log_axes[1]:
                ax.set_yscale('log')
            if log_axes[2]:
                ax.set_zscale('log')
        # Add a standard title
        plt_title = add_std_title(a_group)
        return x_label, y_label, None, plt_title, ax, invert_y_axis
    elif clr_map == 'plt_by_dataset':
        i = 0
        lgnd_hndls = []
        df_labels = [*a_group.data_set.sources_dict.keys()]
        print('df_labels:',df_labels)
        # Prepare to split axis by making a divider
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        # Loop through each dataframe 
        #   which correspond to the datasets input to the Data_Set object's sources_dict
        for this_df in a_group.data_frames:
            # If a subsequent dataset, split the axis
            if i > 0:
                if orientation == 'vertical':
                    sub_ax = divider.append_axes("bottom", size="100%", pad=0, axes_class=mpl.axes.Axes)
                else:
                    sub_ax = divider.append_axes("right", size="100%", pad=0, axes_class=mpl.axes.Axes)
            else:
                sub_ax = ax
            # Check whether to remove noise points
            if plt_noise == False:
                print('before removing noise points:',len(df))
                df = df[df.cluster!=-1]
                print('after removing noise points:',len(df))
                this_df = this_df[this_df.cluster!=-1]
            # Decide on the color, don't go off the end of the array
            my_clr = distinct_clrs[i%len(distinct_clrs)]
            # Get histogram parameters
            h_var, res_bins, median, mean, std_dev = get_hist_params(this_df, var_key, n_h_bins, bin_size)
            # Plot the histogram
            if pdf_hist:
                # Get the histogram
                this_hist, these_bin_edges = np.histogram(h_var, bins=res_bins)
                # Estimate the (non-normalized) PDF from the histogram by 
                #   plotting line through the bin centers
                bin_centers = 0.5*(these_bin_edges[1:]+these_bin_edges[:-1])
                if orientation == 'vertical':
                    pdf_x_arr = bin_centers
                    pdf_y_arr = this_hist
                else:
                    pdf_x_arr = this_hist
                    pdf_y_arr = bin_centers
                sub_ax.plot(pdf_x_arr, pdf_y_arr, color=std_clr, linestyle='-')
            else:
                sub_ax.hist(h_var, bins=res_bins, color=my_clr, alpha=hist_alpha, orientation=orientation)
            # Add legend handle to report the total number of points for this dataset
            lgnd_label = df_labels[i]+': '+str(len(this_df[var_key]))+' points, Median:'+'%.4f'%median
            lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label, alpha=hist_alpha))
            # Add legend handle to report overall statistics
            lgnd_label = 'Mean:'+ '%.4f'%mean+', Std dev:'+'%.4f'%std_dev
            lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label, alpha=hist_alpha))
            notes_string = ''.join(this_df.notes.unique())
            # If subsequent dataset, adjust axis
            if i > 0:
                # Invert y-axis if specified
                if y_key in y_invert_vars:
                    sub_ax.invert_yaxis()
                # Apply the same axis limits to the sub_ax
                if not isinstance(pp.ax_lims, type(None)):
                    try:
                        sub_ax.set_xlim(pp.ax_lims['x_lims'])
                        # print('\t- Applying x-axis limits to sub_ax')
                    except:
                        foo = 2
                    try:
                        sub_ax.set_ylim(pp.ax_lims['y_lims'])
                        # print('\t- Applying y-axis limits to sub_ax')
                    except:
                        foo = 2
                # Apply a grid, if applicable
                if pp.add_grid:
                    sub_ax.grid(color=std_clr, linestyle='--', alpha=grid_alpha)
                # Turn of the shared axis
                if orientation == 'vertical':
                    sub_ax.get_xaxis().set_visible(False)
                else:
                    sub_ax.get_yaxis().set_visible(False)
                # Add axis label to the sub_ax to denote each dataset
                if orientation == 'vertical':
                    sub_ax.set_ylabel(df_labels[i][7:11])
                else:
                    sub_ax.set_xlabel(df_labels[i][7:11])
            # Check whether to change any axes to log scale
            if len(log_axes) == 3:
                if log_axes[0]:
                    sub_ax.set_xscale('log')
                if log_axes[1]:
                    sub_ax.set_yscale('log')
                if log_axes[2]:
                    sub_ax.set_zscale('log')
            i += 1
        # Only add the notes_string if it contains something
        if len(notes_string) > 1:
            notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
            lgnd_hndls.append(notes_patch)
        # Invert y-axis if specified
        if y_key in y_invert_vars:
            invert_y_axis = True
        # Add legend with custom handles
        if legend:
            lgnd = ax.legend(handles=lgnd_hndls)
        # Add a standard title
        plt_title = add_std_title(a_group)
        return x_label, y_label, None, plt_title, ax, invert_y_axis
    elif clr_map == 'stacked':
        # Only works if plotting a pdf_hist
        if not pdf_hist:
            print('Cannot plot stacked histograms without pdf_hist=True')
            exit(0)
        i = 0
        tick_vals = []
        df_labels = [*a_group.data_set.sources_dict.keys()]
        y_label = ''
        # Get the number of data_frames in a_group
        n_dfs = len(a_group.data_frames)
        # Loop through each dataframe 
        #   which correspond to the datasets input to the Data_Set object's sources_dict
        for this_df in a_group.data_frames:
            # Trim df_label
            df_labels[i] = df_labels[i][7:9] # use 7:11 for four digit period labels (ex: 0506)
            tick_vals.append(int(i))
            # Check whether to remove noise points
            if plt_noise == False:
                # print('before removing noise points:',len(df))
                df = df[df.cluster!=-1]
                # print('after removing noise points:',len(df))
                this_df = this_df[this_df.cluster!=-1]
            # Decide on the color, don't go off the end of the array
            my_clr = distinct_clrs[i%len(distinct_clrs)]
            # Get histogram parameters
            h_var, res_bins, median, mean, std_dev = get_hist_params(this_df, var_key, n_h_bins, bin_size)
            ## Plot the histogram
            # Get the histogram
            this_hist, these_bin_edges = np.histogram(h_var, bins=res_bins)
            # Normalize the heights of the histogram bars and add offset
            norm_factor = 2.0 # the bigger the norm_factor, the more the lines overlap
            this_hist = this_hist/np.max(this_hist)*norm_factor - norm_factor/5 + i
            # Estimate the PDF from the histogram by 
            #   plotting line through the bin centers
            bin_centers = 0.5*(these_bin_edges[1:]+these_bin_edges[:-1])
            if orientation == 'vertical':
                pdf_x_arr = bin_centers
                pdf_y_arr = this_hist
            else:
                pdf_x_arr = this_hist
                pdf_y_arr = bin_centers
            ax.plot(pdf_x_arr, pdf_y_arr, color=my_clr, zorder=n_dfs+10-i)
            i += 1
        # Add vertical lines at all the salinity divisions
        if True:
            for SA_div in bps.BGR_HPC_SA_divs:
                ax.axvline(SA_div, color=SA_divs_clr, linestyle=SA_divs_line)
        # Invert y-axis if specified
        if y_key in y_invert_vars:
            invert_y_axis = True
        # Set ticks and color them to match the stack
        if orientation == 'horizontal':
            # Set ticks
            ax.xaxis.set_ticks(tick_vals, df_labels)
            i = 0
            for tick in ax.get_xticklabels():
                # Decide on the color, don't go off the end of the array
                my_clr = distinct_clrs[i%len(distinct_clrs)]
                # Color just the tick label for this instrmt
                tick.set_color(my_clr)
        if orientation == 'vertical':
            # Set ticks
            ax.yaxis.set_ticks(tick_vals, df_labels)
            i = 0
            for tick in ax.get_yticklabels():
                # Decide on the color, don't go off the end of the array
                my_clr = distinct_clrs[i%len(distinct_clrs)]
                # Color just the tick label for this instrmt
                tick.set_color(my_clr)
                i += 1
        # Add a standard title
        plt_title = add_std_title(a_group)
        return x_label, y_label, None, plt_title, ax, invert_y_axis
    elif clr_map == 'clr_by_dataset':
        i = 0
        lgnd_hndls = []
        df_labels = [*a_group.data_set.sources_dict.keys()]
        print('df_labels:',df_labels)
        # Loop through each dataframe 
        #   which correspond to the datasets input to the Data_Set object's sources_dict
        for this_df in a_group.data_frames:
            # Decide on the color, don't go off the end of the array
            my_clr = distinct_clrs[i%len(distinct_clrs)]
            # Get histogram parameters
            h_var, res_bins, median, mean, std_dev = get_hist_params(this_df, var_key, n_h_bins, bin_size)
            # Plot the histogram
            ax.hist(h_var, bins=res_bins, color=my_clr, alpha=hist_alpha, orientation=orientation)
            # Add legend handle to report the total number of points for this dataset
            lgnd_label = df_labels[i]+': '+str(len(this_df[var_key]))+' points, Median:'+'%.4f'%median
            lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label, alpha=hist_alpha))
            # Add legend handle to report overall statistics
            lgnd_label = 'Mean:'+ '%.4f'%mean+', Std dev:'+'%.4f'%std_dev
            lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label, alpha=hist_alpha))
            notes_string = ''.join(this_df.notes.unique())
            i += 1
        # Only add the notes_string if it contains something
        if len(notes_string) > 1:
            notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
            lgnd_hndls.append(notes_patch)
        # Add legend with custom handles
        if legend:
            lgnd = ax.legend(handles=lgnd_hndls)
        # Invert y-axis if specified
        if y_key in y_invert_vars:
            invert_y_axis = True
        # Check whether to change any axes to log scale
        if len(log_axes) == 3:
            if log_axes[0]:
                ax.set_xscale('log')
            if log_axes[1]:
                ax.set_yscale('log')
            if log_axes[2]:
                ax.set_zscale('log')
        # Add a standard title
        plt_title = add_std_title(a_group)
        return x_label, y_label, None, plt_title, ax, invert_y_axis
    elif clr_map == 'instrmt':
        # Add column where the source-instrmt combination ensures uniqueness
        df['source-instrmt'] = df['source']+' '+df['instrmt'].astype("string")
        # Get unique instrmts
        these_instrmts = np.unique(df['source-instrmt'])
        i = 0
        lgnd_hndls = []
        # Loop through each instrument
        for instrmt in these_instrmts:
            # Decide on the color, don't go off the end of the array
            my_clr = distinct_clrs[i%len(distinct_clrs)]
            # Get the data for just this instrmt
            this_df = df[df['source-instrmt'] == instrmt]
            # Get histogram parameters
            h_var, res_bins, median, mean, std_dev = get_hist_params(df, var_key, n_h_bins, bin_size)
            # Plot the histogram
            ax.hist(h_var, bins=res_bins, color=my_clr, alpha=hist_alpha, orientation=orientation)
            # Check whether to plot lines for mean and standard deviation
            if plt_hist_lines:
                if orientation == 'vertical':
                    ax.axvline(mean, color='r')
                    ax.axvline(mean-2*std_dev, color='r', linestyle='--')
                    ax.axvline(mean+2*std_dev, color='r', linestyle='--')
                elif orientation == 'horizontal':
                    ax.axhline(mean, color='r')
                    ax.axhline(mean-2*std_dev, color='r', linestyle='--')
                    ax.axhline(mean+2*std_dev, color='r', linestyle='--')
            i += 1
            # Add legend to report the total number of points for this instrmt
            lgnd_label = instrmt+': '+str(len(df[var_key]))+' points, Median:'+'%.4f'%median
            lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label, alpha=hist_alpha))
            # Add legend handle to report overall statistics
            lgnd_label = 'Mean:'+ '%.4f'%mean+', Std dev:'+'%.4f'%std_dev
            lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label, alpha=hist_alpha))
            # notes_string = ''.join(df.notes.unique())
            notes_string = ''
        # Only add the notes_string if it contains something
        if len(notes_string) > 1:
            notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
            lgnd_hndls.append(notes_patch)
        # Add legend with custom handles
        if legend:
            lgnd = ax.legend(handles=lgnd_hndls)
        # Invert y-axis if specified
        if y_key in y_invert_vars:
            invert_y_axis = True
        # Check whether to change any axes to log scale
        if len(log_axes) == 3:
            if log_axes[0]:
                ax.set_xscale('log')
            if log_axes[1]:
                ax.set_yscale('log')
            if log_axes[2]:
                ax.set_zscale('log')
        # Add a standard title
        plt_title = add_std_title(a_group)
        return x_label, y_label, None, plt_title, ax, invert_y_axis
    elif clr_map == 'cluster':
        # Find the list of cluster ids 
        clstr_ids  = np.unique(np.array(df['cluster'].values, dtype=int))
        if -1 in clstr_ids: 
            clstr_ids.remove(-1)
        n_clusters = int(len(clstr_ids))
        # Get cluster parameters from the dictionary
        m_pts = clstr_dict['m_pts']
        rel_val = clstr_dict['rel_val']
        # Re-order the cluster labels, if specified
        if sort_clstrs:
            df = sort_clusters(df, clstr_ids, ax)
        if len(clstrs_to_plot) > 0:
            df = filter_to_these_clstrs(df, clstr_ids, clstrs_to_plot)
        # Make blank lists to record values
        pts_per_cluster = []
        clstr_means = []
        clstr_stdvs = []
        # Loop through each cluster
        for i in clstr_ids:
            # Decide on the color and symbol, don't go off the end of the arrays
            my_clr = distinct_clrs[i%len(distinct_clrs)]
            my_mkr = mpl_mrks[i%len(mpl_mrks)]
            # Get relevant data
            this_clstr_df = df[df.cluster == i]
            if len(this_clstr_df) < 1:
                continue
            h_data = this_clstr_df[var_key]
            h_mean = np.mean(h_data)
            # Get histogram parameters
            h_var, res_bins, median, mean, std_dev = get_hist_params(this_clstr_df, var_key, n_h_bins, bin_size)
            # Plot the histogram
            n, bins, patches = ax.hist(h_var, bins=res_bins, color=my_clr, alpha=hist_alpha, orientation=orientation, zorder=5)
            # Find the maximum value for this histogram
            h_max = n.max()
            # Find where that max value occured
            bin_h_max = bins[np.where(n == h_max)][0]
            # Plot a symbol to indicate which cluster is which histogram
            if orientation == 'vertical':
                # This will plot the symbol for a cluster just above the max
                # ax.scatter(bin_h_max, h_max, color=my_clr, s=cent_mrk_size, marker=my_mkr, zorder=1)
                # This will plot the number for a cluster just above the max
                ax.scatter(bin_h_max, h_max, color=my_clr, s=cent_mrk_size, marker=r"${}$".format(str(i)), zorder=1)
            elif orientation == 'horizontal':
                ax.scatter(h_max, bin_h_max, color=my_clr, s=cent_mrk_size, marker=my_mkr, zorder=1)
                # ax.scatter(h_max, bin_h_max, color=my_clr, s=cent_mrk_size, marker=r"${}$".format(str(i)), zorder=10)
            #
        # Noise points are labeled as -1
        df_noise = df[df.cluster==-1]
        # Plot noise points
        if plt_noise and len(df_noise) > 0:
            # Get histogram parameters
            # if isinstance(n_h_bins, type(None)):
            #     n_h_bins = 25
            h_var, res_bins, median, mean, std_dev = get_hist_params(df_noise, var_key, n_h_bins*n_clusters, bin_size)
            # Plot the noise histogram on a twin axis
            if orientation == 'vertical':
                tw_ax = ax.twinx()
                # Set ticklabel format for twin axis
                tw_ax.ticklabel_format(style='sci', scilimits=sci_lims, useMathText=True)
                tw_ax.set_ylabel('Number of noise points')
                tw_ax.set_yscale('log')
            elif orientation == 'horizontal':
                tw_ax = ax.twiny()
                # Set ticklabel format for twin axis
                tw_ax.ticklabel_format(style='sci', scilimits=sci_lims, useMathText=True)
                tw_ax.set_xlabel('Number of noise points')
                tw_ax.set_xscale('log')
            n, bins, patches = tw_ax.hist(h_var, bins=res_bins, color=std_clr, alpha=noise_alpha, orientation=orientation, zorder=1)
        n_noise_pts = len(df_noise)
        # Add legend to report the total number of points and notes on the data
        n_pts_patch   = mpl.patches.Patch(color='none', label=str(len(df[var_key]))+' points')
        m_pts_patch = mpl.patches.Patch(color='none', label=r'$m_{pts}$: '+str(m_pts))
        n_clstr_patch = mpl.lines.Line2D([],[],color=cnt_clr, label=r'$n_{clusters}$: '+str(n_clusters), marker='*', linewidth=0)
        n_noise_patch = mpl.patches.Patch(color=std_clr, label=r'$n_{noise pts}$: '+str(n_noise_pts), alpha=noise_alpha, edgecolor=None)
        rel_val_patch = mpl.patches.Patch(color='none', label='DBCV: %.4f'%(rel_val))
        if legend:
            ax.legend(handles=[n_pts_patch, m_pts_patch, n_clstr_patch, n_noise_patch, rel_val_patch])
        # Invert y-axis if specified
        if y_key in y_invert_vars:
            invert_y_axis = True
        # Check whether to change any axes to log scale
        if len(log_axes) == 3:
            if log_axes[0]:
                ax.set_xscale('log')
            if log_axes[1]:
                ax.set_yscale('log')
            if log_axes[2]:
                ax.set_zscale('log')
        # Add a standard title
        plt_title = add_std_title(a_group)
        return x_label, y_label, None, plt_title, ax, invert_y_axis
    else:
        # Did not provide a valid clr_map
        print('Colormap',clr_map,'not valid')
        exit(0)

################################################################################

def get_hist_params(df, h_key, n_h_bins=None, bin_size=None):
    """
    Returns the needed information to make a histogram

    df          A pandas data frame of the data to plot
    h_key       A string of the column header to plot
    n_h_bins    The number of histogram bins to use
    bin_size    The size of bin to use
                    Note: can only specify either n_h_bins or bin_size, not both
    """
    # print('\t\t- in get_hist_params')
    # print('\t\t\t- n_h_bins:',n_h_bins)
    # print('\t\t\t- bin_size:',bin_size)
    # Check whether n_h_bins or bin_size was specified
    if isinstance(n_h_bins, type(None)):
        if isinstance(bin_size, type(None)):
            n_h_bins = 50
            use_n_h_bins = True
        else:
            use_n_h_bins = False
    else:
        if isinstance(bin_size, type(None)):
            use_n_h_bins = True
        else:
            print('Error: cannot specify both `n_h_bins` and `bin_size`')
            exit(0)
    # Pull out the variable to plot, removing null values
    try:
        h_var = np.array(df[df[h_key].notnull()][h_key])
    except:
        print('Cannot find',h_key,'in the dataframe. Aborting script')
        exit(0)
    # Find overall statistics
    median  = np.median(h_var)
    mean    = np.mean(h_var)
    std_dev = np.std(h_var)
    # Define the bins to use in the histogram, np.arange(start, stop, bin_size)
    start = min(h_var)
    stop  = max(h_var)
    # start = mean - 3*std_dev
    # stop  = mean + 3*std_dev
    if use_n_h_bins:
        bin_size  = (stop - start) / n_h_bins
        # bin_size  = std_dev / n_h_bins
    else:
        n_h_bins = int((stop - start) / bin_size)
    res_bins = np.arange(start, stop, bin_size)
    print('\t\t- Using '+str(n_h_bins)+' bins with start: '+str(start)+', stop: '+str(stop)+', bin_size: '+str(bin_size))
    return h_var, res_bins, median, mean, std_dev

################################################################################

def plot_profiles(ax, a_group, pp, clr_map=None):
    """
    Takes in an Analysis_Group object which has the data and plotting parameters
    to produce a subplot of individual profiles. Returns the x and y labels and
    the subplot title

    ax              The axis on which to make the plot
    a_group         An Analysis_Group object containing the info to create this subplot
    pp              The Plot_Parameters object for a_group
    clr_map         A string to determine what color map to use in the plot
    """
    legend = pp.legend
    scale = pp.plot_scale
    ax_lims = pp.ax_lims
    if scale == 'by_pf':
        print('Cannot use',pp.plot_type,'with plot scale by_pf')
        exit(0)
    clr_map = pp.clr_map
    ## Make a plot of the given profiles vs. the vertical
    # Set the main x and y data keys
    x_key = pp.x_vars[0]
    y_key = pp.y_vars[0]
    var_clr = get_var_color(x_key)
    # Check for histogram
    if x_key == 'hist' or y_key == 'hist':
        print('Cannot plot histograms with profiles plot type')
        exit(0)
    # Check for twin x and y data keys
    tw_clr = std_clr
    try:
        tw_x_key = pp.x_vars[1]
        tw_ax_y  = ax.twiny()
        # Set ticklabel format for twin axis
        tw_ax_y.ticklabel_format(style='sci', scilimits=sci_lims, useMathText=True)
        tw_clr = get_var_color(tw_x_key)
        if tw_clr == std_clr:
            tw_clr = alt_std_clr
    except:
        tw_x_key = None
        tw_ax_y  = None
    try:
        tw_y_key = pp.y_vars[1]
        tw_ax_x  = ax.twinx()
        # Set ticklabel format for twin axis
        tw_ax_x.ticklabel_format(style='sci', scilimits=sci_lims, useMathText=True)
        tw_clr = get_var_color(tw_y_key)
        if tw_clr == std_clr:
            tw_clr = alt_std_clr
    except:
        tw_y_key = None
        tw_ax_x  = None
    # Invert y-axis if specified
    if y_key in y_invert_vars:
        invert_y_axis = True
    else:
        invert_y_axis = False
    if not isinstance(tw_y_key, type(None)):
        if tw_y_key in y_invert_vars:
            invert_tw_y_axis = True
        else:
            invert_tw_y_axis = False
    else:
        invert_tw_y_axis = False
    # Whether to add a scale bar or not
    add_scale_bar = False
    # Get extra args dictionary, if it exists
    try:
        extra_args = pp.extra_args
        # print('extra_args:',extra_args)
        if 're_run_clstr' in extra_args:
            re_run_clstr = extra_args['re_run_clstr']
        else:
            re_run_clstr = True
    except:
        extra_args = False
    # Make a blank list for dataframes of each profile
    profile_dfs = []
    # Concatonate all the pandas data frames together
    df = pd.concat(a_group.data_frames)
    # Check whether to run the clustering algorithm
    #   Need to run the clustering algorithm BEFORE filtering to specified range
    cluster_this = False
    #   Find all the plotting variables
    plot_vars = [x_key,y_key,tw_x_key,tw_y_key,clr_map]
    for var in plot_vars:
        if var in clstr_vars:
            cluster_this = True
        #
    if cluster_this:
        m_pts, m_cls, cl_x_var, cl_y_var, cl_z_var, plot_slopes, b_a_w_plt = get_cluster_args(pp)
        df, rel_val, m_pts, m_cls, ell = HDBSCAN_(a_group.data_set.arr_of_ds, df, cl_x_var, cl_y_var, cl_z_var, m_pts, m_cls=m_cls, extra_cl_vars=plot_vars, re_run_clstr=re_run_clstr)
        print('\t- Plot slopes:',plot_slopes)
    # Filter to specified range if applicable
    if not isinstance(a_group.plt_params.ax_lims, type(None)):
        try:
            y_lims = a_group.plt_params.ax_lims['y_lims']
            # Get endpoints of vertical range, make sure they are positive
            y_lims = [abs(ele) for ele in y_lims]
            y_max = max(y_lims)
            y_min = min(y_lims)
            # Filter the data frame to the specified vertical range
            df = df[(df[y_key] < y_max) & (df[y_key] > y_min)]
        except:
            foo = 2
        #
    # Clean out the null values
    df = df[df[x_key].notnull() & df[y_key].notnull()]
    if tw_x_key:
        df = df[df[tw_x_key].notnull()]
    if tw_y_key:
        df = df[df[tw_y_key].notnull()]
    # Get notes
    # notes_string = ''.join(df.notes.unique())
    notes_string = ''
    # Check for extra arguments
    if extra_args:
        # Check whether to narrow down the profiles to plot
        try:
            pfs_to_plot = extra_args['pfs_to_plot']
            # Make a temporary list of dataframes for the profiles to plot
            tmp_df_list = []
            for pf_no in pfs_to_plot:
                tmp_df_list.append(df[df['prof_no'] == pf_no])
            df = pd.concat(tmp_df_list)
        except:
            foo = 2
        # Check whether to narrow down the datetimes to plot
        try:
            dts_to_plot = extra_args['dts_to_plot']
            # Make a temporary list of dataframes for the profiles to plot
            tmp_df_list = []
            for dt in dts_to_plot:
                tmp_df_list.append(df[df['dt_start'] == dt])
            df = pd.concat(tmp_df_list)
        except:
            foo = 2
        try:
            plt_noise = extra_args['plt_noise']
        except:
            plt_noise = True
        try:
            sort_clstrs = pp.extra_args['sort_clstrs']
        except:
            sort_clstrs = True
        try:
            shift_pfs = extra_args['shift_pfs']
        except:
            shift_pfs = True
        try:
            plot_pts = extra_args['plot_pts']
        except:
            plot_pts = True
        try:
            clstrs_to_plot = extra_args['clstrs_to_plot']
        except:
            clstrs_to_plot = []
        try:
            mark_LHW_AW = extra_args['mark_LHW_AW']
        except:
            mark_LHW_AW = False
        try:
            separate_periods = extra_args['separate_periods']
        except:
            separate_periods = False
        try:
            add_inset = extra_args['add_inset']
        except:
            add_inset = None
    else:
        plt_noise = True
        shift_pfs = True
        sort_clstrs = True
        plot_pts = True
        clstrs_to_plot = []
        mark_LHW_AW = False
        separate_periods = False
    # Re-order the cluster labels, if specified
    if sort_clstrs:
        try:
            # Find the cluster labels that exist in the dataframe
            cluster_numbers = np.unique(np.array(df['cluster'].values, dtype=int))
            #   Delete the noise point label "-1"
            cluster_numbers = np.delete(cluster_numbers, np.where(cluster_numbers == -1))
            n_clusters = int(len(cluster_numbers))
            df = sort_clusters(df, cluster_numbers, ax)
        except:
            foo = 2
    if len(clstrs_to_plot) > 0:
        df = filter_to_these_clstrs(df, cluster_numbers, clstrs_to_plot)
    # Decide whether to shift the profiles over so they don't overlap or not
    if shift_pfs:
        shift_pfs = 1
        print('\t- Shifting profiles for clarity')
    else:
        shift_pfs = 0
    # Find the unique profiles for this instrmt
    #   Need to search by datetime because different instruments can have 
    #   profiles with the same profile number
    pfs_in_this_df = []
    # If I'm separating by period, it makes sense to do that here, while the whole dataframe is still together
    if separate_periods:
        # Sort the dataframe by the time
        df.sort_index(level='Time', inplace=True)
        # Make a dictionary to hold the dataframes for each period
        sep_periods_dfs_dict = {}
        # Loop through the different time periods
        for this_period in bps.date_range_dict.keys():
            # Skip BGR_all
            if this_period == 'BGR_all':
                continue
            # Find the date range for this period
            date_0 = bps.date_range_dict[this_period][0]
            date_1 = bps.date_range_dict[this_period][1]
            # Get just the slice of the dataframe for this period
            df_this_period = df.xs(slice(date_0,date_1), level='Time', drop_level=False)
            # Check to make sure there was data left over
            if len(df_this_period) < 1:
                # If not, move on to the next period
                continue
            # Find the number of unique instrmt-prof_no combinations for this period
            df_this_period['instrmt-prof_no'] = df_this_period['instrmt']+' '+df_this_period['prof_no'].astype("string")
            # Find the datetimes for this period, converted to numbers
            these_pfs = mpl.dates.date2num(df_this_period['dt_start'].unique())
            n_pfs = len(these_pfs)
            print('\t- Found',n_pfs,'profiles for',this_period)
            print('\t\t- Profiles:',these_pfs)
            # Add an entry to the dictionary
            sep_periods_dfs_dict[this_period] = []
            # Get the unique profiles for this period
            for pf_dt in df_this_period['dt_start'].values:
                if pf_dt not in pfs_in_this_df:
                    pfs_in_this_df.append(pf_dt)
                    # Get just the part of the dataframe for this profile
                    pf_df = df_this_period[df_this_period['dt_start']==pf_dt]
                    profile_dfs.append(pf_df)
                    sep_periods_dfs_dict[this_period].append(pf_df)
                #
            #
        #
    else:
        for pf_dt in df['dt_start'].values:
            if pf_dt not in pfs_in_this_df:
                pfs_in_this_df.append(pf_dt)
                # Get just the part of the dataframe for this profile
                pf_df = df[df['dt_start']==pf_dt]
                profile_dfs.append(pf_df)
            #
        #
    # Make sure you're not trying to plot too many profiles
    if len(pfs_in_this_df) > 15 and separate_periods==False:
        print('You are trying to plot',len(pfs_in_this_df),'profiles')
        print('That is too many. Try to plot less than 15')
        exit(0)
    elif len(pfs_in_this_df) < 1:
        print('No profiles to plot')
    else:
        print('\t- Plotting',len(pfs_in_this_df),'profiles')
        if separate_periods==False:
            print('\t- Profiles to plot:',pfs_in_this_df)
        # 
    # Check to see whether any profiles were actually loaded
    n_pfs = len(profile_dfs)
    if n_pfs < 1:
        print('No profiles loaded')
        # Add a standard title
        plt_title = add_std_title(a_group)
        return pp.xlabels[0], pp.ylabels[0], None, plt_title, ax, invert_y_axis
    # 
    # Set the keys for CT max markers
    TC_max_key = False
    TC_min_key = False
    tw_TC_max_key = False
    tw_TC_min_key = False
    if mark_LHW_AW:
        print('\t- Marking thermocline')
        if mark_LHW_AW in [True, 'max']:
            if x_key == 'CT':
                TC_max_key = 'CT_TC_max'
            elif x_key == 'SA':
                TC_max_key = 'SA_TC_max'
            elif x_key == 'sigma':
                TC_max_key = 'sigma_TC_max'
            if tw_x_key == 'CT':
                tw_TC_max_key = 'TC_max'
            elif tw_x_key == 'SA':
                tw_TC_max_key = 'SA_TC_max'
            elif tw_x_key == 'sigma':
                tw_TC_max_key = 'sigma_TC_max'
        if mark_LHW_AW in [True, 'min']:
            if x_key == 'CT':
                TC_min_key = 'CT_TC_min'
            elif x_key == 'SA':
                TC_min_key = 'SA_TC_min'
            elif x_key == 'sigma':
                TC_min_key = 'sigma_TC_min'
            if tw_x_key == 'CT':
                tw_TC_min_key = 'TC_min'
            elif tw_x_key == 'SA':
                tw_TC_min_key = 'SA_TC_min'
            elif tw_x_key == 'sigma':
                tw_TC_min_key = 'sigma_TC_min'
            #
        #
    #
    # Plot each profile
    if separate_periods:
        print('\t- Plotting profiles separately by period')
        i = 0
        # Keep track of the average x span for the profiles to shift each period over
        x_span_avg = 0.0
        # Keep track of the x means for each period to add ticks later
        period_x_ticks = {}
        # Loop through the periods in the dictionary
        for this_period in sep_periods_dfs_dict.keys():
            period_x_ticks[this_period] = x_span_avg
            n_pfs = len(sep_periods_dfs_dict[this_period])
            print('\t- Plotting',n_pfs,'profiles for',this_period,'with x_span_avg =',x_span_avg)
            if n_pfs > 0:
                ret_dict = add_profiles(ax, a_group, pp, n_pfs, sep_periods_dfs_dict[this_period], x_key, y_key, clr_map, std_clr, distinct_clrs, mpl_mrks, l_styles, plot_pts, 0, TC_max_key, TC_min_key, tw_ax_y, tw_clr, tw_x_key, tw_TC_max_key, tw_TC_min_key, plt_noise, separate_periods, x_span_avg)
                x_span_avg += ret_dict['xv_span_avg']
            i += 1
        print('\t- Done plotting profiles separately by period, i =',i)
    else:
        ret_dict = add_profiles(ax, a_group, pp, n_pfs, profile_dfs, x_key, y_key, clr_map, var_clr, distinct_clrs, mpl_mrks, l_styles, plot_pts, shift_pfs, TC_max_key, TC_min_key, tw_ax_y, tw_clr, tw_x_key, tw_TC_max_key, tw_TC_min_key, plt_noise, separate_periods=False)
        if not isinstance(add_inset, type(None)):
            print('\t- Adding inset to axis in the range '+str(add_inset[0])+' to '+str(add_inset[1]))
            # Highlight the inset range on the main axis
            ax.axhspan(add_inset[0], add_inset[1], color=std_clr, alpha=0.1)
            # Import needed packages
            import matplotlib.ticker as ticker
            import mpl_toolkits.axes_grid.inset_locator as plt_inset
            # Set inset position, fractions of x and y spans: 
            #   [left=x_ax_min+frac*x_span, bottom=y_ax_min+frac*y_span, right=left+frac*x_span, top=bottom+frac*y_span]
            inset_pos = [[0.1, 0.1, 0.3, 0.4]]
            # Create inset axes using provided position
            ax_in = ax.inset_axes(inset_pos[0], zorder=10)
            # If needed, create a twin axis for the inset
            try:
                tw_x_key = pp.x_vars[1]
                in_tw_ax_y  = ax_in.twiny()
                # Set ticklabel format for twin axis
                in_tw_ax_y.ticklabel_format(style='sci', scilimits=sci_lims, useMathText=True)
                # Change color of the axis label on the twin axis
                in_tw_ax_y.xaxis.label.set_color(tw_clr)
                # Change color of the ticks on the twin axis
                in_tw_ax_y.tick_params(axis='x', colors=tw_clr)
            except:
                tw_x_key = None
                in_tw_ax_y  = None
            # Go through the profile_dfs and narrow to just the specified inset range
            for pf_df in profile_dfs:
                pf_df.drop(pf_df[pf_df[y_key] < min(add_inset)].index, inplace=True)
                pf_df.drop(pf_df[pf_df[y_key] > max(add_inset)].index, inplace=True)
            # Make sure it doesn't try to mark the thermocline
            TC_max_key = False
            TC_min_key = False
            tw_TC_max_key = False
            tw_TC_min_key = False
            # Plot on the inset axis
            add_profiles(ax_in, a_group, pp, n_pfs, profile_dfs, x_key, y_key, clr_map, var_clr, distinct_clrs, mpl_mrks, l_styles, plot_pts, shift_pfs, TC_max_key, TC_min_key, in_tw_ax_y, tw_clr, tw_x_key, tw_TC_max_key, tw_TC_min_key, plt_noise, separate_periods=False)
            # Adjust the vertical axis limits
            ax_in.set_ylim(add_inset)
            # Something seems a bit weird with trying to get the inset axis on top of the original one
            # ax_in.set_zorder(10)
    # Adjust twin axes, if specified
    if not isinstance(tw_x_key, type(None)):
        # Adjust bounds on axes
        # tw_ax_y.set_xlim([tw_left_bound-tw_x_pad, twin_high+tw_x_pad])
        tw_ax_y.set_xlim([ret_dict['tw_left_bound']-ret_dict['tw_x_pad'], ret_dict['twin_high']+ret_dict['tw_x_pad']])
        # ax.set_xlim([left_bound-x_pad, right_bound+x_pad])
        ax.set_xlim([ret_dict['left_bound']-ret_dict['x_pad'], ret_dict['right_bound']+ret_dict['x_pad']])
        # Add label to twin axis
        tw_ax_y.set_xlabel(pp.xlabels[1])
        # Change color of the axis label on the twin axis
        tw_ax_y.xaxis.label.set_color(tw_clr)
        # Change color of the ticks on the twin axis
        tw_ax_y.tick_params(axis='x', colors=tw_clr)
        # Add a grid
        tw_ax_y.grid(color=tw_clr, linestyle='--', alpha=grid_alpha+0.2, axis='x')
        if invert_tw_y_axis:
            tw_ax_y.invert_yaxis()
            print('\t- Inverting twin y axis')
        # Check whether to adjust the twin axes limits
        if not isinstance(pp.ax_lims, type(None)):
            if 'tw_x_lims' in pp.ax_lims.keys():
                tw_ax_y.set_ylim(pp.ax_lims['tw_x_lims'])
                print('\t- Set tw_x_lims to',pp.ax_lims['tw_x_lims'])
    # if True:
    #     # Change color of the axis label
    #     ax.xaxis.label.set_color(var_clr)
    #     # Change color of the ticks
    #     ax.tick_params(axis='x', colors=var_clr)
    # else:
    #     ax.set_xlim([left_bound-x_pad, right_bound+x_pad])
    # Check whether to add a scale bar
    # if add_scale_bar and (shift_pfs == 1 or separate_periods == True):
    if (shift_pfs == 1 or separate_periods == True):
        if tw_x_key:
            add_h_scale_bar(ax, ax_lims, unit=' g/kg', clr=var_clr)
            add_h_scale_bar(tw_ax_y, ax_lims, unit=r' $^\circ$C', tw_clr=tw_clr)
        else:
            add_h_scale_bar(ax, ax_lims, unit=' g/kg')
            if separate_periods:
                ax.set_xticks(list(period_x_ticks.values()))
                ax.set_xticklabels(list(period_x_ticks.keys()))#, rotation=45)
    # Add legend
    if legend:
        lgnd = ax.legend()
        # Only add the notes_string if it contains something
        if len(notes_string) > 1:
            handles, labels = ax.get_legend_handles_labels()
            handles.append(mpl.patches.Patch(color='none'))
            labels.append(notes_string)
            lgnd = ax.legend(handles=handles, labels=labels)
        # Need to change the marker size for each label in the legend individually
        for hndl in lgnd.legendHandles:
            hndl._sizes = [lgnd_mrk_size]
        #
    # Add a standard title
    plt_title = add_std_title(a_group)
    return pp.xlabels[0], pp.ylabels[0], None, plt_title, ax, invert_y_axis

################################################################################

# Plot each profile
def add_profiles(ax, a_group, pp, n_pfs, profile_dfs, x_key, y_key, clr_map, var_clr, distinct_clrs, mpl_mrks, l_styles, plot_pts, shift_pfs, TC_max_key, TC_min_key, tw_ax_y, tw_clr, tw_x_key, tw_TC_max_key, tw_TC_min_key, plt_noise, separate_periods=False, x_span_avg=0.0):
    """
    Adds the profiles to the plot

    ax              The axis on which to make the plot
    a_group         An Analysis_Group object containing the info to create this subplot
    pp              The Plot Parameters
    n_pfs           The number of profiles to plot
    profile_dfs     A list of pandas dataframes, each containing the data for a profile
    x_key           A string of the column header to plot on the x-axis
    y_key           A string of the column header to plot on the y-axis
    clr_map         A string to determine what color map to use in the plot
    var_clr         A string of the color to use for the variable
    distinct_clrs   A list of distinct colors to use in the plot
    mpl_mrks        A list of distinct markers to use in the plot
    l_styles        A list of distinct line styles to use in the plot
    plot_pts        A boolean to determine whether to plot the individual points
    shift_pfs       A boolean to determine whether to shift profiles over so they don't overlap
    TC_max_key      A string of the column header to plot the CT max on the x-axis
    TC_min_key      A string of the column header to plot the CT min on the x-axis
    tw_ax_y         The twin axis to plot on
    tw_clr          A string of the color to use for the twin axis
    tw_x_key        A string of the column header to plot on the twin x-axis
    tw_TC_max_key   A string of the column header to plot the CT max on the twin x-axis
    tw_TC_min_key   A string of the column header to plot the CT min on the twin x-axis
    """
    # Check to make sure there are profiles to plot
    if n_pfs < 1:
        print('No profiles loaded, aborting script')
        exit(0)
    # Keep track of the largest span in x
    xv_span_max = 0
    xv_span = 0
    tw_span_max = 0
    tw_span = 0
    tw_left_bound = 0
    tw_x_pad = 0
    twin_high = 0
    x_span_arr = []
    if separate_periods == False:
        period_shift = 0
        x_span_avg = 0
        pf_line_alpha = pf_alpha
    else:
        period_shift = 1
        pf_line_alpha = 0.3
    # Make a dictionary of variables to return
    ret_dict = {}
    for i in range(n_pfs):
        # Pull the dataframe for this profile
        pf_df = profile_dfs[i]
        # Make a label for this profile
        if len(pf_df) > 1:
            try:
                pf_label = pf_df['source'][0]+pf_df['instrmt'][0]+'-'+str(int(pf_df['prof_no'][0]))
            except:
                pf_label = pf_df['source'][0]+pf_df['instrmt'][0]+'-'+str(pf_df['prof_no'][0])
        elif len(pf_df) == 1:
            try:
                pf_label = pf_df['source']+pf_df['instrmt']+'-'+str(int(pf_df['prof_no']))
            except:
                pf_label = pf_df['source']+pf_df['instrmt']+'-'+str(pf_df['prof_no'])
        # Decide on marker and line styles, don't go off the end of the array
        mkr     = mpl_mrks[i%len(mpl_mrks)]
        l_style = l_styles[i%len(l_styles)]
        # if shift_pfs == 0:
        #     var_clr = distinct_clrs[i%len(distinct_clrs)]
        # Get array to plot and find bounds
        if i == 0:
            # Pull data to plot
            xvar = pf_df[x_key] - np.mean(pf_df[x_key])*period_shift + x_span_avg
            # Find CT max if applicable
            if TC_max_key:
                TC_max = np.unique(np.array(pf_df[TC_max_key].values))
            # Find CT min if applicable
            if TC_min_key:
                TC_min = np.unique(np.array(pf_df[TC_min_key].values))
            # Find upper and lower bounds of first profile, for reference points
            xvar_low  = min(xvar)
            xvar_high = max(xvar)
            # Find span of the profile
            xv_span = abs(xvar_high - xvar_low)
            x_span_arr.append(xv_span)
            # Define far left bound for plot
            left_bound = xvar_low
            # Define pads for x bound (applied at the end)
            x_pad = xv_span/15
            # Find upper and lower bounds of first profile for twin axis
            if tw_x_key:
                tvar = pf_df[tw_x_key]
                if tw_TC_max_key:
                    tw_TC_max = np.unique(np.array(pf_df[tw_TC_max_key].values))
                if tw_TC_min_key:
                    tw_TC_min = np.unique(np.array(pf_df[tw_TC_min_key].values))
                twin_low  = min(tvar)
                twin_high = max(tvar)
                tw_span   = abs(twin_high - twin_low)
                # Find normalized profile difference for spacing
                norm_xv = (xvar-xvar_low)/xv_span
                norm_tw = (tvar-twin_low)/tw_span
                norm_pf_diff = min(norm_tw-norm_xv)
                # Shift things over
                right_bound = xvar_high - norm_pf_diff*xv_span
                tw_left_bound = twin_low + norm_pf_diff*tw_span
                # Find index of largest x value
                # TC_max_idx = np.argmax(xvar)
                # Find value of twin profile at that index
                # tw_xvar_max = tvar[xvar_max_idx]
                tw_x_pad = tw_span/15
            else:
                tvar = None
                tw_TC_max = None
                right_bound = xvar_high
            #
        # Adjust the starting points of each subsequent profile
        elif i > 0:
            add_scale_bar = True
            # Keep the old profile for reference
            old_xvar = xvar
            old_xvar_low = xvar_low
            old_xvar_high = xvar_high
            old_xv_span = xv_span
            x_span_arr.append(xv_span)
            # Find array of data for this profile
            xvar = pf_df[x_key] - min(pf_df[x_key])*shift_pfs + old_xvar_low*shift_pfs - np.mean(pf_df[x_key])*period_shift + x_span_avg
            if TC_max_key:
                TC_max = np.unique(np.array(pf_df[TC_max_key].values)) - min(pf_df[x_key])*shift_pfs + old_xvar_low*shift_pfs
            if TC_min_key:
                TC_min = np.unique(np.array(pf_df[TC_min_key].values)) - min(pf_df[x_key])*shift_pfs + old_xvar_low*shift_pfs
            # Find new upper and lower bounds of profile
            xvar_high = max(xvar)
            xvar_low  = min(xvar)
            # Find span of this profile
            xv_span = abs(xvar_high - xvar_low)
            if tw_x_key:
                tw_shift = 1#0.8
                xvar = xvar + old_xv_span*tw_shift
                if TC_max_key:
                    TC_max = TC_max + old_xv_span*tw_shift
                if TC_min_key:
                    TC_min = TC_min + old_xv_span*tw_shift
                xvar_high = max(xvar)
                xvar_low = min(xvar)
                tvar = pf_df[tw_x_key] - min(pf_df[tw_x_key])*shift_pfs + twin_low*shift_pfs + tw_span*tw_shift
                if tw_TC_max_key:
                    tw_TC_max = np.unique(np.array(pf_df[tw_TC_max_key].values)) - min(pf_df[tw_x_key])*shift_pfs + twin_low*shift_pfs + tw_span*tw_shift
                if tw_TC_min_key:
                    tw_TC_min = np.unique(np.array(pf_df[tw_TC_min_key].values)) - min(pf_df[tw_x_key])*shift_pfs + twin_low*shift_pfs + tw_span*tw_shift
                # 
                twin_low  = min(tvar)
                twin_high = max(tvar)
                tw_span   = abs(twin_high - twin_low)
                # Find normalized profile difference for spacing
                norm_xv = (xvar-xvar_low)/xv_span
                norm_tw = (tvar-twin_low)/tw_span
                norm_pf_diff = min(norm_tw-norm_xv)
                # Shift things over
                right_bound = max(xvar_high - norm_pf_diff*xv_span, right_bound)
                twin_high = max(tvar)
                # Find index of largest x value
                # xvar_max_idx = np.argmax(xvar)
                # Find value of twin profile at that index
                # tw_xvar_max = tvar[xvar_max_idx]
            else:
                tvar = None
                tw_TC_max = None
                # Shift things over
                xvar = xvar + old_xv_span*0.45*shift_pfs
                if TC_max_key:
                    TC_max = TC_max + old_xv_span*0.45*shift_pfs
                if TC_min_key:
                    TC_min = TC_min + old_xv_span*0.45*shift_pfs
                xvar_low = min(xvar)
                xvar_high = max(xvar)
                right_bound = max(xvar_high, right_bound)
            #
        # Determine the color mapping to be used
        if clr_map in a_group.vars_to_keep and clr_map != 'cluster':
            # Format the dates if necessary
            if clr_map == 'dt_start' or clr_map == 'dt_end':
                cmap_data = mpl.dates.date2num(pf_df[clr_map])
            else:
                cmap_data = pf_df[clr_map]
            # Plot a background line for each profile
            ax.plot(xvar, pf_df[y_key], color=var_clr, linestyle=l_style, label=pf_label, zorder=1)
            # Get the colormap
            this_cmap = get_color_map(clr_map)
            # Plot the points as a heatmap
            heatmap = ax.scatter(xvar, pf_df[y_key], c=cmap_data, cmap=this_cmap, s=pf_mrk_size, marker=mkr)
            # Plot on twin axes, if specified
            if not isinstance(tw_x_key, type(None)):
                tw_ax_y.plot(tvar, pf_df[y_key], color=tw_clr, linestyle=l_style, zorder=1)
                tw_ax_y.scatter(tvar, pf_df[y_key], c=cmap_data, cmap=this_cmap, s=pf_mrk_size, marker=mkr)
                # Check whether to adjust the twin axes limits
                if not isinstance(pp.ax_lims, type(None)):
                    if 'tw_x_lims' in pp.ax_lims.keys():
                        tw_ax_y.set_ylim(pp.ax_lims['tw_x_lims'])
                        print('\t- Set tw_x_lims to',pp.ax_lims['tw_x_lims'])
            #
            # Create the colorbar, but only on the first profile
            if i == 0:
                cbar = plt.colorbar(heatmap, ax=ax)
                # Format the colorbar ticks, if necessary
                if clr_map == 'dt_start' or clr_map == 'dt_end':
                    loc = mpl.dates.AutoDateLocator()
                    cbar.ax.yaxis.set_major_locator(loc)
                    cbar.ax.yaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(loc))
                # Change the colorbar limits, if necessary
                try:
                    ax_lims_keys = list(pp.ax_lims.keys())
                    if 'c_lims' in ax_lims_keys:
                        cbar.mappable.set_clim(vmin=min(pp.ax_lims['c_lims']), vmax=max(pp.ax_lims['c_lims']))
                        print('\t- Set c_lims to',pp.ax_lims['c_lims'])
                except:
                    foo = 2
                # Invert colorbar if necessary
                if clr_map in y_invert_vars:
                    cbar.ax.invert_yaxis()
                cbar.set_label(pp.clabel)
        if clr_map == 'clr_all_same':
            mrk_alpha = 0.9
            # Plot a background line for each profile
            ax.plot(xvar, pf_df[y_key], color=var_clr, linestyle=l_style, label=pf_label, alpha=pf_line_alpha, zorder=1)
            if plot_pts:
                # Plot every point the same color, size, and marker
                ax.scatter(xvar, pf_df[y_key], color=var_clr, s=pf_mrk_size, marker=mkr, alpha=pf_mrk_alpha)
            # Plot maximum
            if TC_max_key:
                press_TC_max = np.unique(np.array(pf_df['press_TC_max'].values))
                print('\t- Plotting TC_max:',TC_max,'press_TC_max:',press_TC_max)
                # ax.scatter(TC_max, press_TC_max, color=var_clr, s=pf_mrk_size*5, marker='^', zorder=5)
                # Plot the partially filled in triangle
                ax.plot(TC_max, press_TC_max, color=AW_clr, markersize=pf_mrk_size*2, marker=AW_mrk, fillstyle='top', markerfacecoloralt=AW_facealtclr, markeredgecolor=AW_edgeclr, linewidth=0, zorder=4)
            # Plot minimum
            if TC_min_key:
                press_TC_min = np.unique(np.array(pf_df['press_TC_min'].values))
                print('\t- Plotting TC_min:',TC_min,'press_TC_min:',press_TC_min)
                # ax.scatter(TC_min, press_TC_min, color=var_clr, s=pf_mrk_size*5, marker='v', zorder=5)
                # Plot the partially filled in triangle
                ax.plot(TC_min, press_TC_min, color=LHW_clr, markersize=pf_mrk_size*2, marker=LHW_mrk, fillstyle='bottom', markerfacecoloralt=LHW_facealtclr, markeredgecolor=LHW_edgeclr, linewidth=0, zorder=4)
            # Plot on twin axes, if specified
            if not isinstance(tw_x_key, type(None)):
                tw_ax_y.plot(tvar, pf_df[y_key], color=tw_clr, linestyle=l_style, alpha=pf_line_alpha, zorder=1)
                # Check whether to adjust the twin axes limits
                if not isinstance(pp.ax_lims, type(None)):
                    if 'tw_x_lims' in pp.ax_lims.keys():
                        tw_ax_y.set_ylim(pp.ax_lims['tw_x_lims'])
                        print('\t- Set tw_x_lims to',pp.ax_lims['tw_x_lims'])
                if plot_pts:
                    tw_ax_y.scatter(tvar, pf_df[y_key], color=tw_clr, s=pf_mrk_size, marker=mkr, alpha=mrk_alpha)
                if TC_max_key:
                    print('\t- Plotting tw_TC_max:',tw_TC_max,'press_TC_max:',press_TC_max)
                    # tw_ax_y.scatter(tw_TC_max, press_TC_max, color=tw_clr, s=pf_mrk_size*5, marker='^', zorder=5)
                    # Plot the partially filled in triangle
                    tw_ax_y.plot(tw_TC_max, press_TC_max, color=AW_clr, markersize=pf_mrk_size*2, marker=AW_mrk, fillstyle='top', markerfacecoloralt=AW_facealtclr, markeredgecolor=AW_edgeclr, linewidth=0, zorder=4)
                    tw_ax_y.axhline(press_TC_max, color=AW_facealtclr, linestyle='--', zorder=3)
                if TC_min_key:
                    print('\t- Plotting tw_TC_min:',tw_TC_min,'press_TC_min:',press_TC_min)
                    # tw_ax_y.scatter(tw_TC_min, press_TC_min, color=tw_clr, s=pf_mrk_size*5, marker='v', zorder=5)
                    # Plot the partially filled in triangle
                    tw_ax_y.plot(tw_TC_min, press_TC_min, color=LHW_clr, markersize=pf_mrk_size*2, marker=LHW_mrk, fillstyle='bottom', markerfacecoloralt=LHW_facealtclr, markeredgecolor=LHW_edgeclr, linewidth=0, zorder=4)
                    tw_ax_y.axhline(press_TC_min, color=LHW_facealtclr, linestyle='--', zorder=3)
            #
        if clr_map == 'cluster':
            # Plot a background line for each profile
            ax.plot(xvar, pf_df[y_key], color=var_clr, linestyle=l_style, label=pf_label, alpha=pf_line_alpha, zorder=1)
            # Make a dataframe with adjusted xvar and tvar
            df_clstrs = pd.DataFrame({x_key:xvar, tw_x_key:tvar, y_key:pf_df[y_key], 'cluster':pf_df['cluster'], 'clst_prob':pf_df['clst_prob']})
            # Get a list of unique cluster numbers, but delete the noise point label "-1"
            cluster_numbers = np.unique(np.array(df_clstrs['cluster'].values, dtype=int))
            cluster_numbers = np.delete(cluster_numbers, np.where(cluster_numbers == -1))
            print('\tcluster_numbers:',cluster_numbers)
            # Plot noise points first
            if plt_noise:
                # ax.scatter(df_clstrs[df_clstrs.cluster==-1][x_key], df_clstrs[df_clstrs.cluster==-1][y_key], color=noise_clr, s=pf_mrk_size, marker=std_marker, alpha=noise_alpha, zorder=2)
                ax.scatter(df_clstrs[df_clstrs.cluster==-1][x_key], df_clstrs[df_clstrs.cluster==-1][y_key], color=std_clr, s=pf_mrk_size, marker=std_marker, alpha=pf_alpha, zorder=1)
            # Plot on twin axes, if specified
            if not isinstance(tw_x_key, type(None)):
                tw_ax_y.plot(tvar, pf_df[y_key], color=tw_clr, linestyle=l_style, label=pf_label, alpha=pf_line_alpha, zorder=1)
                # Check whether to adjust the twin axes limits
                if not isinstance(pp.ax_lims, type(None)):
                    if 'tw_x_lims' in pp.ax_lims.keys():
                        tw_ax_y.set_ylim(pp.ax_lims['tw_x_lims'])
                        print('\t- Set tw_x_lims to',pp.ax_lims['tw_x_lims'])
                if plt_noise:
                    tw_ax_y.scatter(df_clstrs[df_clstrs.cluster==-1][tw_x_key], df_clstrs[df_clstrs.cluster==-1][y_key], color=std_clr, s=pf_mrk_size, marker=std_marker, alpha=noise_alpha, zorder=2)
            # Loop through each cluster
            # if False:
            for i in cluster_numbers:
                # Decide on the color and symbol, don't go off the end of the arrays
                my_clr = distinct_clrs[i%len(distinct_clrs)]
                my_mkr = mpl_mrks[i%len(mpl_mrks)]
                my_mkr = r"${}$".format(str(i))
                # print('\t\tcluster:',i,'my_clr:',my_clr,'my_mkr:',my_mkr)
                # Get relevant data
                x_data = df_clstrs[df_clstrs.cluster == i][x_key]
                y_data = df_clstrs[df_clstrs.cluster == i][y_key]
                alphas = df_clstrs[df_clstrs.cluster == i]['clst_prob']
                # Plot the points for this cluster with the specified color, marker, and alpha value
                # ax.scatter(x_data, y_data, color=my_clr, s=pf_mrk_size, marker=my_mkr, alpha=alphas, zorder=5)
                ax.scatter(x_data, y_data, color=my_clr, s=pf_mrk_size, marker=my_mkr, alpha=pf_alpha, zorder=5)
                # Plot on twin axes, if specified
                if not isinstance(tw_x_key, type(None)):
                    t_data = df_clstrs[df_clstrs.cluster == i][tw_x_key]
                    tw_ax_y.scatter(t_data, y_data, color=my_clr, s=pf_mrk_size, marker=my_mkr, alpha=pf_alpha, zorder=5)
                    # Check whether to adjust the twin axes limits
                    if not isinstance(pp.ax_lims, type(None)):
                        if 'tw_x_lims' in pp.ax_lims.keys():
                            tw_ax_y.set_ylim(pp.ax_lims['tw_x_lims'])
                            print('\t- Set tw_x_lims to',pp.ax_lims['tw_x_lims'])
                #
            #
        #
        if xv_span > xv_span_max:
            xv_span_max = xv_span
        if tw_span > tw_span_max:
            tw_span_max = tw_span
    #
    # print('\t- xv_span_max:',xv_span_max,'tw_span_max:',tw_span_max)
    # print('\t- x_span_arr:',x_span_arr)
    # Build the return dictionary
    ret_dict['xv_span_max'] = xv_span_max
    ret_dict['xv_span_avg'] = np.mean(x_span_arr)
    ret_dict['tw_span_max'] = tw_span_max
    ret_dict['left_bound'] = left_bound
    ret_dict['right_bound'] = right_bound
    ret_dict['tw_left_bound'] = tw_left_bound
    ret_dict['x_pad'] = x_pad
    ret_dict['tw_x_pad'] = tw_x_pad
    ret_dict['twin_high'] = twin_high
    return ret_dict

################################################################################

def plot_waterfall(ax, a_group, fig, ax_pos, pp, clr_map=None):
    """
    Takes in an Analysis_Group object which has the data and plotting parameters
    to produce a subplot of individual profiles in a 3D waterfall layout. Returns 
    the x and y labels and the subplot title

    ax              The axis on which to make the plot
    a_group         A Analysis_Group object containing the info to create this subplot
    fig             The figure in which ax is contained
    ax_pos          A tuple of the ax (rows, cols, linear number of this subplot)
    pp              The Plot_Parameters object for a_group
    clr_map         A string to determine what color map to use in the plot
    """
    print('in plot_waterfall()')
    legend = pp.legend
    scale = pp.plot_scale
    ax_lims = pp.ax_lims
    line_alpha = 0.8
    if scale == 'by_pf':
        print('Cannot use',pp.plot_type,'with plot scale by_pf')
        exit(0)
    clr_map = pp.clr_map
    ## Make a plot of the given profiles vs. the vertical
    # Set the main x, y, and z data keys
    #   NOTE: Later I switch the y and z axes so that variables 
    #         passed to y end up on the vertical axis
    x_key = pp.x_vars[0]
    y_key = pp.y_vars[0]
    z_key = pp.z_vars[0]
    var_clr = get_var_color(x_key)
    # Check for histogram
    if x_key == 'hist' or y_key == 'hist':
        print('Cannot plot histograms with profiles plot type')
        exit(0)
    # Concatonate all the pandas data frames together
    df = pd.concat(a_group.data_frames)
    # Get extra args dictionary, if it exists
    try:
        extra_args = pp.extra_args
        # print('extra_args:',extra_args)
        if 'fit_vars' in extra_args.keys():
            fit_vars = extra_args['fit_vars']
        else:
            fit_vars = False
        if 're_run_clstr' in extra_args.keys():
            re_run_clstr = extra_args['re_run_clstr']
        else:
            re_run_clstr = True
    except:
        extra_args = False
        fit_vars = False
    # Check whether to normalize by subtracting a polyfit2d
    if fit_vars:
        df = calc_fit_vars(df, (x_key, y_key, z_key, clr_map), fit_vars)
    # Set up z(y) axis
    if isinstance(z_key, type(None)):
        z_key = 'dt_start'
        z_label = a_group.data_set.var_attr_dicts[0]['dt_start']['label']
        df[z_key] = mpl.dates.date2num(df[z_key])
    elif z_key in ['dt_start', 'dt_end']:
        df[z_key] = mpl.dates.date2num(df[z_key])
    else:
        df[z_key] = df[z_key]
    # Drop duplicates
    df.drop_duplicates(subset=[x_key, y_key, z_key], keep='first', inplace=True)
    # In order to plot in 3D, need to remove the current axis
    ax.remove()
    # And replace axis with one that has 3 dimensions
    ax = fig.add_subplot(ax_pos, projection='3d')
    # Check for twin x and y data keys
    try:
        tw_x_key = pp.x_vars[1]
        print('Warning: Cannot add twin axes on a 3D plot')
    except:
        foo = 2
    try:
        tw_y_key = pp.y_vars[1]
        print('Warning: Cannot add twin axes on a 3D plot')
    except:
        foo = 2
    # Invert y-axis if specified
    if y_key in y_invert_vars:
        invert_y_axis = True
    else:
        invert_y_axis = False
    # Make a blank list for dataframes of each profile
    profile_dfs = []
    # Check whether to run the clustering algorithm
    #   Need to run the clustering algorithm BEFORE filtering to specified range
    cluster_this = False
    # Set the keys for CT max markers
    if x_key == 'CT':
        TC_max_key = 'CT_TC_max'
    elif x_key == 'SA':
        TC_max_key = 'SA_TC_max'
    else:
        TC_max_key = False
    #   Find all the plotting variables
    plot_vars = [x_key,y_key,clr_map]
    for var in plot_vars:
        if var in clstr_vars:
            cluster_this = True
        #
    if cluster_this:
        m_pts, m_cls, cl_x_var, cl_y_var, cl_z_var, plot_slopes, b_a_w_plt = get_cluster_args(pp)
        df, rel_val, m_pts, m_cls, ell = HDBSCAN_(a_group.data_set.arr_of_ds, df, cl_x_var, cl_y_var, cl_z_var, m_pts, m_cls=m_cls, extra_cl_vars=plot_vars, re_run_clstr=re_run_clstr)
        print('\t- Plot slopes:',plot_slopes)
    # Filter to specified range if applicable
    if not isinstance(a_group.plt_params.ax_lims, type(None)):
        print('Warning: Specifying ax_lims does not work as expected in waterfall plots')
        try:
            y_lims = a_group.plt_params.ax_lims['y_lims']
            # Get endpoints of vertical range, make sure they are positive
            y_lims = [abs(ele) for ele in y_lims]
            y_max = max(y_lims)
            y_min = min(y_lims)
            # Filter the data frame to the specified vertical range
            df = df[(df[y_key] < y_max) & (df[y_key] > y_min)]
        except:
            foo = 2
        #
    # Clean out the null values
    df = df[df[x_key].notnull() & df[y_key].notnull()]
    # Get notes
    # notes_string = ''.join(df.notes.unique())
    notes_string = ''
    # Check for extra arguments
    if extra_args:
        # Check whether to narrow down the profiles to plot
        try:
            pfs_to_plot = extra_args['pfs_to_plot']
            # Make a temporary list of dataframes for the profiles to plot
            tmp_df_list = []
            for pf_no in pfs_to_plot:
                tmp_df_list.append(df[df['prof_no'] == pf_no])
            df = pd.concat(tmp_df_list)
        except:
            foo = 2
        # Check whether to narrow down the datetimes to plot
        try:
            dts_to_plot = extra_args['dts_to_plot']
            # Make a temporary list of dataframes for the profiles to plot
            print('\t- Getting specific profiles')
            tmp_df_list = []
            for dt in dts_to_plot:
                tmp_df_list.append(df[df['dt_start'] == dt])
            print('\t- Done getting specific profiles')
            df = pd.concat(tmp_df_list)
        except:
            print('\t- No specific profiles to plot')
            foo = 2
            # extra_args = None
        try:
            plt_noise = extra_args['plt_noise']
        except:
            plt_noise = True
        try:
            sort_clstrs = pp.extra_args['sort_clstrs']
        except:
            sort_clstrs = True
        try:
            plot_pts = extra_args['plot_pts']
        except:
            plot_pts = True
        try:
            clstrs_to_plot = extra_args['clstrs_to_plot']
        except:
            clstrs_to_plot = []
    else:
        plt_noise = True
        sort_clstrs = True
        plot_pts = True
        clstrs_to_plot = []
    # Re-order the cluster labels, if specified
    if sort_clstrs:
        try:
            # Find the cluster labels that exist in the dataframe
            cluster_numbers = np.unique(np.array(df['cluster'].values, dtype=int))
            #   Delete the noise point label "-1"
            cluster_numbers = np.delete(cluster_numbers, np.where(cluster_numbers == -1))
            n_clusters = int(len(cluster_numbers))
            df = sort_clusters(df, cluster_numbers, ax)
        except:
            foo = 2
    if len(clstrs_to_plot) > 0:
        n_clusters = int(np.nanmax(df['cluster']+1))
        df = filter_to_these_clstrs(df, n_clusters, clstrs_to_plot)
    # Find the unique profiles for this instrmt
    #   Make sure to match the variable type
    pfs_in_this_df = []
    for pf_no in df['prof_no'].values:
        if pf_no not in pfs_in_this_df:
            pfs_in_this_df.append(pf_no)
        #
    dts_in_this_df = []
    for dt in df['dt_start'].values:
        if dt not in dts_in_this_df:
            dts_in_this_df.append(dt)
        #
    print('Datetimes in this plot:',len(dts_in_this_df))
    print(dts_in_this_df)
    # Make sure you're not trying to plot too many profiles
    if len(pfs_in_this_df) > 100:
        print('You are trying to plot',len(pfs_in_this_df),'profiles')
        print('That is too many. Try to plot less than 15')
        exit(0)
    else:
        print('\t- Plotting',len(pfs_in_this_df),'profiles')
        print('\t- Profiles to plot:',pfs_in_this_df)
    # Loop through each profile to create a list of dataframes
    for pf_no in pfs_in_this_df:
        # Get just the part of the dataframe for this profile
        pf_df = df[df['prof_no']==pf_no]
        profile_dfs.append(pf_df)
        # print(pf_df)
    #
    # Check to see whether any profiles were actually loaded
    n_pfs = len(profile_dfs)
    if n_pfs < 1:
        print('No profiles loaded')
        # Add a standard title
        plt_title = add_std_title(a_group)
        return pp.xlabels[0], pp.ylabels[0], pp.zlabels[0], plt_title, ax, invert_y_axis
    # 
    # Plot each profile
    for i in range(n_pfs):
        # Pull the dataframe for this profile
        pf_df = profile_dfs[i]
        # Decide on marker and line styles, don't go off the end of the array
        mkr     = mpl_mrks[i%len(mpl_mrks)]
        l_style = l_styles[i%len(l_styles)]
        # Make a label for this profile
        if len(pf_df) > 1:
            try:
                pf_label = pf_df['source'][0]+pf_df['instrmt'][0]+'-'+str(int(pf_df['prof_no'][0]))
            except:
                pf_label = pf_df['source'][0]+pf_df['instrmt'][0]+'-'+str(pf_df['prof_no'][0])
        elif len(pf_df) == 1:
            try:
                pf_label = pf_df['source']+pf_df['instrmt']+'-'+str(int(pf_df['prof_no']))
            except:
                pf_label = pf_df['source']+pf_df['instrmt']+'-'+str(pf_df['prof_no'])
        # Determine the color mapping to be used
        if clr_map in a_group.vars_to_keep and clr_map != 'cluster':
            # Format the dates if necessary
            if clr_map == 'dt_start' or clr_map == 'dt_end':
                cmap_data = mpl.dates.date2num(pf_df[clr_map])
            else:
                cmap_data = pf_df[clr_map]
            # Plot a background line for each profile
            ax.plot(pf_df[x_key], pf_df[z_key], zs=pf_df[y_key], color=std_clr, alpha=line_alpha, label=pf_label, zorder=1)
            # Get the colormap
            this_cmap = get_color_map(clr_map)
            # Plot the points as a heatmap
            heatmap = ax.scatter(pf_df[x_key], pf_df[z_key], zs=pf_df[y_key], c=cmap_data, cmap=this_cmap, s=pf_mrk_size, marker=mkr)
            #
            # Create the colorbar, but only on the first profile
            if i == 0:
                cbar = plt.colorbar(heatmap, ax=ax)
                # Format the colorbar ticks, if necessary
                if clr_map == 'dt_start' or clr_map == 'dt_end':
                    loc = mpl.dates.AutoDateLocator()
                    cbar.ax.yaxis.set_major_locator(loc)
                    cbar.ax.yaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(loc))
                # Change the colorbar limits, if necessary
                try:
                    ax_lims_keys = list(pp.ax_lims.keys())
                    if 'c_lims' in ax_lims_keys:
                        cbar.mappable.set_clim(vmin=min(pp.ax_lims['c_lims']), vmax=max(pp.ax_lims['c_lims']))
                        print('\t- Set c_lims to',pp.ax_lims['c_lims'])
                except:
                    foo = 2
                # Invert colorbar if necessary
                if clr_map in y_invert_vars:
                    cbar.ax.invert_yaxis()
                cbar.set_label(pp.clabel)
        if clr_map == 'clr_all_same':
            mrk_alpha = 0.9
            # Plot a background line for each profile
            #   NOTE: Switching the y and z axis so the y var is on the vertical
            ax.plot(pf_df[x_key], pf_df[z_key], zs=pf_df[y_key], color=std_clr, alpha=line_alpha, label=pf_label, zorder=1)
            # Plot in 3D
            # ax.scatter(df[x_key], df[y_key], zs=df_z_key, color=std_clr, s=m_size, marker=std_marker, alpha=mrk_alpha, zorder=5)
            if plot_pts:
                # Plot every point the same color, size, and marker
                ax.scatter(pf_df[x_key], pf_df[z_key], zs=pf_df[y_key], color=var_clr, s=pf_mrk_size, marker=mkr, alpha=pf_mrk_alpha)
            # Plot maximum
            if False:#TC_max_key:
                press_TC_max = np.unique(np.array(pf_df['press_TC_max'].values))
                print('\t- Plotting TC_max:',TC_max,'press_TC_max:',press_TC_max)
                ax.scatter(TC_max, press_TC_max, zs=pf_df[y_key][0], color=var_clr, s=pf_mrk_size*5, marker='*', zorder=5)
            # Plot on twin axes, if specified
            #
        if clr_map == 'cluster':
            # Plot a background line for each profile
            ax.plot(pf_df[x_key], pf_df[z_key], zs=pf_df[y_key], color=std_clr, alpha=line_alpha, label=pf_label, zorder=1)
            # Make a dataframe of the relevant data
            df_clstrs = pd.DataFrame({x_key:pf_df[x_key], y_key:pf_df[y_key], z_key:pf_df[z_key], 'cluster':pf_df['cluster'], 'clst_prob':pf_df['clst_prob']})
            # Get a list of unique cluster numbers, but delete the noise point label "-1"
            cluster_numbers = np.unique(np.array(df_clstrs['cluster'].values, dtype=int))
            cluster_numbers = np.delete(cluster_numbers, np.where(cluster_numbers == -1))
            # print('\tcluster_numbers:',cluster_numbers)
            # Plot noise points first
            if plt_noise:
                ax.scatter(df_clstrs[df_clstrs.cluster==-1][x_key], df_clstrs[df_clstrs.cluster==-1][z_key], zs=df_clstrs[df_clstrs.cluster==-1][y_key], color=noise_clr, s=pf_mrk_size, marker=std_marker, alpha=noise_alpha, zorder=2)
            # Loop through each cluster
            for i in cluster_numbers:
                # Decide on the color and symbol, don't go off the end of the arrays
                my_clr = distinct_clrs[i%len(distinct_clrs)]
                my_mkr = mpl_mrks[i%len(mpl_mrks)]
                # print('\t\tcluster:',i,'my_clr:',my_clr,'my_mkr:',my_mkr)
                # Get relevant data
                x_data = df_clstrs[df_clstrs.cluster == i][x_key]
                y_data = df_clstrs[df_clstrs.cluster == i][y_key]
                z_data = df_clstrs[df_clstrs.cluster == i][z_key]
                alphas = df_clstrs[df_clstrs.cluster == i]['clst_prob']
                # Plot the points for this cluster with the specified color, marker, and alpha value
                # ax.scatter(x_data, y_data, color=my_clr, s=pf_mrk_size, marker=my_mkr, alpha=alphas, zorder=5)
                ax.scatter(x_data, z_data, zs=y_data, color=my_clr, s=pf_mrk_size, marker=my_mkr, alpha=pf_alpha, zorder=5)
            #
        #
    #
    if True:
        # Change color of the axis label
        ax.xaxis.label.set_color(var_clr)
        # Change color of the ticks
        ax.tick_params(axis='x', colors=var_clr)
    # Make the panes in the background transparent
    # ax.w_xaxis.set_pane_color([1.0, 1.0, 1.0, 0.05])
    # ax.w_yaxis.set_pane_color([1.0, 1.0, 1.0, 0.05])
    # ax.w_zaxis.set_pane_color([1.0, 1.0, 1.0, 0.05])
    # Turn the panes transparent
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    # Give the pane edges color
    # ax.w_zaxis.pane.set_edgecolor('r')
    # ax.w_zaxis.line.set_color('r')
    # ax.w_xaxis._axinfo.update({'grid' : {'color': (0, 0, 0, 1)}})
    # ax.w_yaxis._axinfo.update({'grid' : {'color': (0, 0, 0, 1)}})
    # ax.w_zaxis._axinfo.update({'grid' : {'color': (0, 0, 0, 1)}})
    # Set the grid transparency
    # ax.grid(False)
    # ax.w_xaxis.grid(False)
    # ax.w_yaxis.grid(False)
    # ax.w_zaxis.grid(False)
    # Add legend
    if legend:
        lgnd = ax.legend()
        # Only add the notes_string if it contains something
        if len(notes_string) > 1:
            handles, labels = ax.get_legend_handles_labels()
            handles.append(mpl.patches.Patch(color='none'))
            labels.append(notes_string)
            lgnd = ax.legend(handles=handles, labels=labels)
        # Need to change the marker size for each label in the legend individually
        for hndl in lgnd.legendHandles:
            hndl._sizes = [lgnd_mrk_size]
        #
    # Add a standard title
    plt_title = add_std_title(a_group)
    # Invert z axis if applicable
    if invert_y_axis:
        ax.invert_zaxis()
    # Format the axes for datetimes, if necessary
    format_datetime_axes(pp.add_grid, x_key, z_key, ax, z_key=y_key)
    # Since the y and z axes are flipped, return False for invert_y_axis
    return pp.xlabels[0], pp.zlabels[0], pp.ylabels[0], plt_title, ax, False

################################################################################

def polyfit2d(x_data, y_data, z_data, kx=3, ky=3, order=3):
    """
    Finds a polynomial fit of the z data across the x_data and y_data

    x_data      1D array of the x values
    y_data      1D array of the y values
    z_data      1D array of the z values
    kx          int, maximum order of polynomial in x
    ky          int, maximum order of polynomial in y (bug note: needs to be equal to kx)
    order       int or None, default is None
                    If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.

    Return paramters from np.linalg.lstsq:
        soln: np.ndarray
            Array of polynomial coefficients.
        residuals: np.ndarray
        rank: int
        s: np.ndarray
    """
    # Prepare coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx+1, ky+1))
    # Prepare solution array
    a = np.zeros((coeffs.size, x_data.size))

    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x_data)
        else:
            arr = coeffs[i, j] * x_data**i * y_data**j
        a[index] = arr.ravel()

    # Do leastsq fitting and save the result
    return np.linalg.lstsq(a.T, np.ravel(z_data), rcond=None)

################################################################################

def reco_polyfit2d(X, Y, coeffs, kx, ky, order):
    """
    Calculates the Z values on X and Y for the polynomial based on the  
    coefficients calculated by polyfit2d, from the variable soln

    X           An array of x values
    Y           An array of y values (needs to be the same size as X)
    coeffs      An array of coefficients from polyfit2d (soln)
    kx          int, maximum order of polynomial in x
    ky          int, maximum order of polynomial in y (bug note: needs to be equal to kx)
    order       int or None
                    If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
    """
    # Prepare blank variable on which to build the solution
    recro = 0
    # Make a string on which to build the equation
    eq_string = 'Z = '
    # Cycle through the coefficients
    c = 0
    for j in range(ky+1):
        for i in range(kx+1):
            # do not include powers greater than order
            if i + j <= order:
                # print('i:',i,'j:',j,'c:',c)
                if i == 0:
                    if j == 0:
                        recro += coeffs[c]
                        eq_string += format_sci_notation(coeffs[c])
                    elif j == 1:
                        recro += coeffs[c]*(Y)
                        eq_string += ' + ' + format_sci_notation(coeffs[c]) + '*Y'
                    else:
                        recro += coeffs[c]*(Y**j)
                        eq_string += ' + ' + format_sci_notation(coeffs[c]) + '*Y^' + str(j)
                elif i == 1:
                    if j == 0:
                        recro += coeffs[c]*(X)
                        eq_string += ' + ' + format_sci_notation(coeffs[c]) + '*X'
                    elif j == 1:
                        recro += coeffs[c]*(X)*(Y)
                        eq_string += ' + ' + format_sci_notation(coeffs[c]) + '*XY'
                    else:
                        recro += coeffs[c]*(X)*(Y**j)
                        eq_string += ' + ' + format_sci_notation(coeffs[c]) + '*XY^' + str(j)
                else:
                    if j == 0:
                        recro += coeffs[c]*(X**i)
                        eq_string += ' + ' + format_sci_notation(coeffs[c]) + '*X^' + str(i)
                    elif j == 1:
                        recro += coeffs[c]*(X**i)*(Y)
                        eq_string += ' + ' + format_sci_notation(coeffs[c]) + '*X^' + str(i) + 'Y'
                    else:
                        recro += coeffs[c]*(X**i)*(Y**j)
                        eq_string += ' + ' + format_sci_notation(coeffs[c]) + '*X^' + str(i) + 'Y^' + str(j)
            c += 1
    print('\t- Fitted equation:')
    print('\t'+eq_string)
    return recro, eq_string

################################################################################

def calc_fit_vars(df, plt_vars, fit_vars, kx=3, ky=3, order=3):
    """
    Takes in a dataframe and the keys for the variable to calculate (plt_var) and
    the variables on which to calculate the fit (fit_vars). Finds a polynomial fit 
    for plt_var data on fit_vars, the adds a new column to the dataframe which
    is the original plt_var data minus the fit

    pp          The Plot_Parameters object for a_group
    x_data      1D array of the x values
    y_data      1D array of the y values
    z_data      1D array of the z values
    kx          int, maximum order of polynomial in x
    ky          int, maximum order of polynomial in y (bug note: needs to be equal to kx)
    order       int or None, default is None
                    If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
    """
    print('in calc_fit_vars')
    # print('plt_vars:',plt_vars)
    # print('fit_vars:',fit_vars)
    # Find which of the plot variables has `-fit` in it
    fit_these_vars = []
    # these_split_vars = []
    fit_vars_dict = {}
    plt_var = None
    for var in plt_vars:
        # print('var:',var)
        if not isinstance(var, type(None)):
            if '-fit' in var:
                plt_var = var
                # Split the suffix from the original variable (assumes a hyphen split)
                split_var = var.split('-', 1)
                var_str = split_var[0]
                affix = split_var[1]
                if 'trd' in var_str:
                    # Split the prefix from the original variable (assumes an underscore split)
                    split_var2 = var_str.split('_', 1)
                    var_str = split_var2[1]
                # Add var_str to list of vars to calculate fit
                print('adding',var_str,'to list of vars to calculate fit')
                fit_vars_dict[var] = [var_str, split_var]
                fit_these_vars.append(var_str)
                # these_split_vars.append(split_var)
            elif 'fit_' in var:
                plt_var = var
                # Split the suffix from the original variable (assumes an underscore split)
                split_var = var.split('_', 1)
                var_str = split_var[1]
                affix = split_var[0]
                # Add var_str to list of vars to calculate fit
                print('adding',var_str,'to list of vars to calculate fit')
                fit_vars_dict[var] = [var_str, split_var]
                fit_these_vars.append(var_str)
                # these_split_vars.append(split_var)
    if isinstance(plt_var, type(None)):
        print('Did not find `fit` in z_key, aborting script')
        exit(0)
    print('fit_these_vars:',fit_vars_dict)
    # Remove rows in dataframe where any fit_these_vars variable is NaN
    # df = df[df[fit_these_vars].notnull().all(axis=1)]
    df = df.dropna(subset=fit_these_vars)
    # Get the x and y data based on the keys
    x_data = df[fit_vars[0]]
    y_data = df[fit_vars[1]]
    # Calculate the fit for each variable
    for var in fit_vars_dict.keys():
    # for i in range(len(fit_these_vars)):
        # var_str = fit_these_vars[i]
        # split_var = these_split_vars[i]
        # this_plt_var = plt_vars[i]
        var_str = fit_vars_dict[var][0]
        split_var = fit_vars_dict[var][1]
        print('\t- Calculating',var,'on',fit_vars)
        # Get the z data based on the key
        z_data = df[var_str]
        # Find the solution to the polyfit
        #   Note: residuals will be empty because this is not an over-determined problem
        soln, residuals, rank, s = polyfit2d(x_data, y_data, z_data, kx, ky, order)
        # Use the solution to calculate the fitted z values
        fitted_z, eq_string = reco_polyfit2d(x_data, y_data, soln, kx, ky, order)
        # See whether to return the fit, or the residual
        if split_var[0] == 'fit':
            # Add new column for fitted z
            df[var] = fitted_z
        elif split_var[1] == 'fit':
            # Add new column for difference between z and fitted z
            df[var] = z_data - fitted_z
        # Add new column for the equation string
        df[var_str+'_fit_eq'] = eq_string
    return df

    # # Remove rows in dataframe where the variable is NaN
    # df = df[df[var_str].notnull()]
    # # Get the x, y, and z data based on the keys
    # x_data = df[fit_vars[0]]
    # y_data = df[fit_vars[1]]
    # z_data = df[var_str]
    # # print('x_data.shape:',x_data.shape)
    # # print('y_data.shape:',y_data.shape)
    # # print('z_data number of nans',z_data.isna().sum())
    # # print('z_data.shape:',z_data.shape)
    # # Find the solution to the polyfit
    # soln, residuals, rank, s = polyfit2d(x_data, y_data, z_data, kx, ky, order)
    # # Use the solution to calculate the fitted z values
    # fitted_z, eq_string = reco_polyfit2d(x_data, y_data, soln, kx, ky, order)
    # # See whether to return the fit, or the residual
    # if split_var[0] == 'fit':
    #     # Add new column for fitted z
    #     df[plt_var] = fitted_z
    # elif split_var[1] == 'fit':
    #     # Add new column for difference between z and fitted z
    #     df[plt_var] = z_data - fitted_z
    # # Add new column for the equation string
    # df[var_str+'_fit_eq'] = eq_string
    # # print(df)
    # return df

################################################################################

def plot_polyfit2d(ax, pp, x_data, y_data, z_data, kx=3, ky=3, order=3, n_grid_pts=50, in_3D=True, plot_type=None):
    """
    Takes in x, y, and z data on an axis. Finds a polynomial fit of the z data 
    and plots the result as a surface.

    ax          The axis on which to make the plot
    pp          The Plot_Parameters object for a_group
    x_data      1D array of the x values
    y_data      1D array of the y values
    z_data      1D array of the z values
    kx          int, maximum order of polynomial in x
    ky          int, maximum order of polynomial in y (bug note: needs to be equal to kx)
    order       int or None, default is None
                    If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
                    If int, coefficients up to a maximum of kx+ky <= order are considered.
    n_grid_pts  int of the number of points per x and y range to make the grid for the 
                    fit surface
    """
    # print('in plot_polyfit2d')
    # print('x_data.shape:',x_data.shape)
    # print('y_data.shape:',y_data.shape)
    # print('z_data.shape:',z_data.shape)
    # Prepare axis transform, if necessary
    # print('plot_type:',plot_type)
    if plot_type == 'map':
        this_transform = ccrs.PlateCarree()
    else:
        this_transform = None
    # Find the solution to the polyfit
    soln, residuals, rank, s = polyfit2d(x_data, y_data, z_data, kx, ky, order)
    # Make a grid for the solution to plot on 
    x_range = np.linspace(min(x_data), max(x_data), n_grid_pts)
    y_range = np.linspace(min(y_data), max(y_data), n_grid_pts)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    # Either of these work, very slight differences
    # fitted_surf = np.polynomial.polynomial.polygrid2d(x_grid, y_grid, soln.reshape((kx+1,ky+1)))
    fitted_surf, eq_string = reco_polyfit2d(x_grid, y_grid, soln, kx, ky, order)
    if in_3D:
        # Plot fit as a surface
        surf = ax.plot_trisurf(x_grid.flatten(), y_grid.flatten(), fitted_surf.flatten(), cmap=get_color_map(pp.z_vars[0]), linewidth=0, alpha=0.5)
    else:
        # Plot the contours
        try:
            # if c_lims are provided
            CS = ax.contourf(x_grid, y_grid, fitted_surf, n_grid_pts, cmap=get_color_map(pp.z_vars[0]), vmin=min(pp.ax_lims['c_lims']), vmax=max(pp.ax_lims['c_lims']), transform=this_transform)
        except:
            # if c_lims are not provided
            CS = ax.contourf(x_grid, y_grid, fitted_surf, n_grid_pts, cmap=get_color_map(pp.z_vars[0]), vmin=min(z_data), vmax=max(z_data), transform=this_transform)
        # Find the min / max values and their locations
        min_idx = np.argmin(fitted_surf)
        max_idx = np.argmax(fitted_surf)
        min_val = fitted_surf.flatten()[min_idx]
        max_val = fitted_surf.flatten()[max_idx]
        min_x = x_grid.flatten()[min_idx]
        max_x = x_grid.flatten()[max_idx]
        min_y = y_grid.flatten()[min_idx]
        max_y = y_grid.flatten()[max_idx]
        print('\t- Minimum value of',min_val,'at x:',min_x,'y:',min_y)
        print('\t- Maximum value of',max_val,'at x:',max_x,'y:',max_y)
        # Plot a marker at the maximum value, but only if the fit was to pressure or depth
        if pp.clr_map == 'depth' or pp.clr_map == 'press' or pp.clr_map == 'press_TC_max' or pp.clr_map == 'press_TC_min':
            ax.plot(max_x, max_y, 'r*', markersize=10, zorder=15, transform=this_transform)

################################################################################

def get_cluster_args(pp):
    """
    Finds cluster-realted variables in the extra_args dictionary

    pp              The Plot_Parameters object for a_group
    """
    # Get the dictionary stored in extra_args
    cluster_plt_dict = pp.extra_args
    # print('cluster_plt_dict:',cluster_plt_dict)
    # Get min_samples parameter, if it was given
    try:
        m_pts = cluster_plt_dict['m_pts']
        if m_pts != 'auto':
            m_pts = int(m_pts)
    except:
        m_pts = None
    # Get min_cluster_size parameter, if it was given
    try:
        m_cls = cluster_plt_dict['m_cls']
        if not isinstance(m_cls, type('')):
            m_cls = int(m_cls)
    except:
        m_cls = None
    # Get x and y parameters, if given
    try:
        cl_x_var = cluster_plt_dict['cl_x_var']
        cl_y_var = cluster_plt_dict['cl_y_var']
    except:
        cl_x_var = None
        cl_y_var = None
    # Get z parameter, if given
    try:
        cl_z_var = cluster_plt_dict['cl_z_var']
    except:
        cl_z_var = None
    # Get the boolean for whether to plot trendline slopes or not
    try:
        plot_slopes = cluster_plt_dict['plot_slopes']
    except:
        plot_slopes = False
    # Check whether or not the box and whisker plots were called for
    try:
        b_a_w_plt = cluster_plt_dict['b_a_w_plt']
    except:
        b_a_w_plt = False
    return m_pts, m_cls, cl_x_var, cl_y_var, cl_z_var, plot_slopes, b_a_w_plt

################################################################################

def HDBSCAN_(arr_of_ds, df, x_key, y_key, z_key, m_pts, m_cls='auto', extra_cl_vars=[None], param_sweep=False, re_run_clstr=True):
    """
    Runs the HDBSCAN algorithm on the set of data specified. Returns a pandas
    dataframe with columns for x_key, y_key, 'cluster', and 'clst_prob', a
    rough measure of the DBCV score from `relative_validity_`, and the ell_size

    run_group   The Analysis_Group object to run HDBSCAN on
    df          A pandas data frame with x_key and y_key as equal length columns
    x_key       String of the name of the column to use on the x-axis
    y_key       String of the name of the column to use on the y-axis
    z_key       String of the name of the column to use on the z-axis (optional)
    m_pts       An integer, 'min_samples', number of points in neighborhood for a core point
    m_cls       An integer, 'min_cluster_size', the minimum number of points for a cluster
    extra_cl_vars   A list of extra variables to potentially calculate
    """
    # print('-- in HDBSCAN')
    # print('-- m_pts:',m_pts)
    # print('-- m_cls:',m_cls)
    # print('-- extra_cl_vars:',extra_cl_vars)
    # print('-- df columns:',df.columns.values.tolist())
    # Find the value of ell, the moving average window
    ell_sizes = []
    # for ds in run_group.data_set.arr_of_ds:
    for ds in arr_of_ds:
        this_ell = ds.attrs['Moving average window']
        # Remove non-numeric characters from the string
        this_ell = re.sub("[^0-9^.]", "", this_ell)
        # Add to list as an integer
        ell_sizes.append(int(this_ell))
    if len(ell_sizes) > 1:
        print('\t- ell_sizes:',np.unique(ell_sizes))
    ell_size = ell_sizes[0]
    # If I've manually set re_run_cluster to False, then don't re-run the clustering
    if re_run_clstr == False:
        re_run = False
    # Whether to run a parameter sweep
    if param_sweep:
        re_run = True
        print('-- Parameter sweep, re_run:',re_run)
    # Check to make sure there is actually a 'cluster' column in the dataframe
    elif re_run_clstr != False and not 'cluster' in df.columns.values.tolist():
        re_run = True
        print('-- `cluster` not in df columns, re_run:',re_run)
    # Check to make sure the global clustering attributes make sense
    else:
        # Make a copy of the global clustering attribute dictionary
        gcattr_dict = {'Last clustered':[],
                       'Clustering x-axis':[],
                       'Clustering y-axis':[],
                       'Clustering z-axis':[],
                       'Clustering m_pts':[],
                       'Clustering m_cls':[],
                       'Moving average window':[],
                       'Clustering filters':[],
                       'Clustering DBCV':[]}
        # Get the global clustering attributes from each dataset
        # for ds in run_group.data_set.arr_of_ds:
        for ds in arr_of_ds:
            for attr in gcattr_dict.keys():
                gcattr_dict[attr].append(ds.attrs[attr])
        # If any attribute has multiple different values, need to re-run HDBSCAN
        re_run = False
        for attr in gcattr_dict.keys():
            gcattr_dict[attr] = list(set(gcattr_dict[attr]))
            if attr not in ['Last clustered', 'Clustering m_pts', 'Clustering m_cls', 'Clustering DBCV'] and len(gcattr_dict[attr]) > 1:
                re_run = True
                print('-- found multiple distinct values of',attr,'in list, re_run:',re_run)
        # if gcattr_dict['Last clustered'][0] == 'Never':
        #     re_run = True
        #     print('-- `Last clustered` attr is `Never`, re_run:',re_run)
    print('\t- Re-run HDBSCAN:',re_run)
    if re_run:
        # print('in HDBSCAN_(), m_pts:',m_pts)
        print('len(df):',len(df))
        if m_pts == 'auto':
            # Selecting a rule of thumb for number m_pts based on number of points
            # m_pts = int(0.00129*len(df) + 181)
            # m_pts = int(7*(len(df)**(1/3)))
            m_pts = int(len(df) / 250 - 150)
        if m_cls == 'auto':
            m_cls = m_pts
        print('\t\tClustering x-axis:',x_key)
        print('\t\tClustering y-axis:',y_key)
        print('\t\tClustering z-axis:',z_key)
        print('\t\tClustering m_pts: ',m_pts)
        print('\t\tClustering m_cls: ',m_cls)
        print('\t\tMoving average window:',ell_size)
        # Set the parameters of the HDBSCAN algorithm
        #   Note: must set gen_min_span_tree=True or you can't get `relative_validity_`
        get_DBCV = True
        clst_sel_met = 'leaf'
        hdbscan_1 = hdbscan.HDBSCAN(gen_min_span_tree=get_DBCV, min_cluster_size=m_cls, min_samples=m_pts, cluster_selection_method=clst_sel_met, leaf_size=m_pts)
        print('\t\thdbscan_1.gen_min_span_tree:',hdbscan_1.gen_min_span_tree)
        print('\t\thdbscan_1.cluster_selection_method:',hdbscan_1.cluster_selection_method)
        try:
            print('\t\thdbscan_1.leaf_size:',hdbscan_1.leaf_size)
        except:
            foo = 2
        # The datetime variables need to be scaled to match
        dt_scale_factor = 10000
        for var in [x_key, y_key, z_key]:
            if var in ['dt_start','dt_end']:
                print('\t\tRescaling '+var+' variable for clustering')
                df[var] = df[var] / dt_scale_factor
        # Run the HDBSCAN algorithm
        ## Start timing the clustering
        import time
        tic = time.perf_counter()
        if isinstance(z_key, type(None)):
            # Run in 2D
            hdbscan_1.fit_predict(df[[x_key,y_key]])
        else:
            # Run in 3D
            hdbscan_1.fit_predict(df[[x_key,y_key,z_key]])
        toc = time.perf_counter()
        print(f'\t\tClustering took {toc - tic:0.4f} seconds')
        # Undo the scaling
        for var in [x_key, y_key, z_key]:
            if var in ['dt_start','dt_end']:
                print('\t\tUnscaling '+var+' variable after clustering')
                df[var] = df[var] * dt_scale_factor
        # Add the cluster labels and probabilities to the dataframe
        df['cluster']   = hdbscan_1.labels_
        df['clst_prob'] = hdbscan_1.probabilities_
        if get_DBCV:
            rel_val = hdbscan_1.relative_validity_
            print('\t\tDBCV:',rel_val)
        else:
            rel_val = -999
        # Determine whether there are any new variables to calculate
        new_cl_vars = list(set(extra_cl_vars) & set(clstr_vars))
        # print('\t\t- new_cl_vars:',new_cl_vars)
        # Don't need to calculate `cluster` so remove it if its there
        if 'cluster' in new_cl_vars:
            new_cl_vars.remove('cluster')
        if len(new_cl_vars) > 0:
            print('\t\tCalculating extra clustering variables')
            df = calc_extra_cl_vars(df, new_cl_vars)
        # Write this dataframe and DBCV value back to the analysis group
        # if not param_sweep:
        #     run_group.data_frames = [df]
        #     run_group.data_set.arr_of_ds[0].attrs['Clustering DBCV'] = rel_val
        return df, rel_val, m_pts, m_cls, ell_size
    else:
        # Use the clustering results that are already in the dataframe
        print('\t- Using clustering results from file')
        print('\t\tClustering x-axis:',gcattr_dict['Clustering x-axis'])
        print('\t\tClustering y-axis:',gcattr_dict['Clustering y-axis'])
        m_pts = int(gcattr_dict['Clustering m_pts'][0])
        print('\t\tClustering m_pts: ',gcattr_dict['Clustering m_pts'])
        m_cls = int(gcattr_dict['Clustering m_cls'][0])
        print('\t\tClustering m_cls: ',gcattr_dict['Clustering m_cls'])
        print('\t\tMoving average window:',gcattr_dict['Moving average window'])
        print('\t\tClustering filters:',gcattr_dict['Clustering filters'])
        print('\t\tClustering DBCV:  ',gcattr_dict['Clustering DBCV'])
        # Determine whether there are any new variables to calculate
        new_cl_vars = list(set(extra_cl_vars) & set(clstr_vars))
        # Don't need to calculate `cluster` so remove it if its there
        if 'cluster' in new_cl_vars:
            new_cl_vars.remove('cluster')
        if len(new_cl_vars) > 0:
            print('\t\tCalculating extra clustering variables')
            df = calc_extra_cl_vars(df, new_cl_vars)
        return df, gcattr_dict['Clustering DBCV'][0], m_pts, m_cls, ell_size

################################################################################

def calc_extra_cl_vars(df, new_cl_vars):
    """
    Takes in an already-clustered pandas data frame and a list of variables and,
    if there are extra variables to calculate, it will add those to the data frame

    df                  A pandas data frame of the data to plot
    new_cl_vars         A list of clustering-related variables to calculate
    """
    print('\t- Calculating extra cluster variables')
    print('\t\t- new_cl_vars:',new_cl_vars)
    # print(np.unique(np.array(df['cluster'].values)))
    # Get list of clusters
    # clstr_ids = np.unique(np.array(df['cluster'].values))
    # # Find the number of clusters
    # n_clusters = len(clstr_ids)
    # Check for variables to calculate
    for this_var in new_cl_vars:
        # Skip any variables that are just 'None"
        if isinstance(this_var, type(None)):
            continue
        print('\t\t\tCalculating',this_var)
        # Split the prefix from the original variable (assumes an underscore split)
        split_var = this_var.split('_', 1)
        prefix = split_var[0]
        try:
            var = split_var[1]
        except:
            var = None
        # print('prefix:',prefix,'- var:',var)
        # print('df[var]:',df[var])
        # Make a new blank column in the data frame for this variable
        df[this_var] = None
        # Calculate the new values based on the prefix
        if prefix == 'pca':
            # Calculate the per profile cluster average version of the variable
            #   Reduces the number of points to just one per cluster per profile
            # Make a new column with the combination of instrument and profile number
            #   This avoids accidentally lumping two profiles with the same number from
            #   different instruments together, which would give the incorrect result
            df['instrmt-prof_no'] = df.instrmt.map(str) + ' ' + df.prof_no.map(str)
            instrmt_pf_nos = np.unique(np.array(df['instrmt-prof_no'].values))
            # Loop over each profile
            for pf in instrmt_pf_nos:
                # Find the data from just this profile
                df_this_pf = df[df['instrmt-prof_no']==pf]
                # Get a list of clusters in this profile
                clstr_ids = np.unique(np.array(df_this_pf['cluster'].values))
                clstr_ids = clstr_ids[~np.isnan(clstr_ids)]
                # Remove the noise points
                clstr_ids = clstr_ids[clstr_ids != -1]
                # Loop over every cluster in this profile
                for i in clstr_ids:
                    # Find the data from this cluster
                    df_this_pf_this_cluster = df_this_pf[df_this_pf['cluster']==i]
                    # Find the mean of this var for this cluster for this profile
                    pf_clstr_mean = np.mean(df_this_pf_this_cluster[var].values)
                    # Put that value back into the original dataframe
                    #   Need to make a mask first for some reason
                    this_pf_this_cluster = (df['instrmt-prof_no']==pf) & (df['cluster']==i)
                    df.loc[this_pf_this_cluster, this_var] = pf_clstr_mean
        elif prefix == 'pcs':
            # Calculate the per profile cluster span version of the variable
            #   Reduces the number of points to just one per cluster per profile
            # Make a new column with the combination of instrument and profile number
            #   This avoids accidentally lumping two profiles with the same number from
            #   different instruments together, which would give the incorrect result
            df['instrmt-prof_no'] = df.instrmt.map(str) + ' ' + df.prof_no.map(str)
            instrmt_pf_nos = np.unique(np.array(df['instrmt-prof_no'].values))
            n_profiles = len(instrmt_pf_nos)
            if n_profiles > 0:
                digits = int(np.log10(n_profiles))+1
            else:
                digits = 1
            print('\t\t\t\tCalculating',this_var,'for',n_profiles,'profiles')
            j = 0
            # Loop over each profile
            for pf in instrmt_pf_nos:
                j += 1
                print('\t\t\t\tCalculating',this_var,'for profile',pf,'('+str(j).rjust(digits),'/',str(n_profiles)+')', end="\r")
                # Find the data from just this profile
                df_this_pf = df[df['instrmt-prof_no']==pf]
                # Get a list of clusters in this profile
                clstr_ids = np.unique(np.array(df_this_pf['cluster'].values))
                clstr_ids = clstr_ids[~np.isnan(clstr_ids)]
                # Remove the noise points
                clstr_ids = clstr_ids[clstr_ids != -1]
                # Loop over every cluster in this profile
                for i in clstr_ids:
                    # Find the data from this cluster
                    df_this_pf_this_cluster = df_this_pf[df_this_pf['cluster']==i]
                    # Find the span of this var for this cluster for this profile
                    pf_clstr_span = max(df_this_pf_this_cluster[var].values) - min(df_this_pf_this_cluster[var].values)
                    # Replace any values of zero with `None`
                    # if pf_clstr_span == 0:
                    #     pf_clstr_span = None
                    # Put that value back into the original dataframe
                    #   Need to make a mask first for some reason
                    this_pf_this_cluster = (df['instrmt-prof_no']==pf) & (df['cluster']==i)
                    df.loc[this_pf_this_cluster, this_var] = pf_clstr_span
        elif prefix == 'cmc':
            # Calculate the cluster mean-centered version of the variable
            #   Should not change the number of points to display
            # Get a list of clusters in this dataframe
            clstr_ids = np.unique(np.array(df['cluster'].values))
            clstr_ids = clstr_ids[~np.isnan(clstr_ids)]
            # Remove the noise points
            clstr_ids = clstr_ids[clstr_ids != -1]
            # Loop over each cluster
            for i in clstr_ids:
                # Find the data from this cluster
                df_this_cluster = df[df['cluster']==i].copy()
                # Find the mean of this var for this cluster
                clstr_mean = np.mean(df_this_cluster[var].values)
                # Calculate normalized values
                df_this_cluster[this_var] = df_this_cluster[var] - clstr_mean
                # Put those values back into the original dataframe
                df.loc[df['cluster']==i, this_var] = df_this_cluster[this_var]
        elif prefix == 'ca':
            # Calculate the cluster average version of the variable
            #   Reduces the number of points to just one per cluster
            # Get a list of clusters in this dataframe
            clstr_ids = np.unique(np.array(df['cluster'].values))
            clstr_ids = clstr_ids[~np.isnan(clstr_ids)]
            # Remove the noise points
            clstr_ids = clstr_ids[clstr_ids != -1]
            # Loop over each cluster
            for i in clstr_ids:
                # Find the data from this cluster
                df_this_cluster = df[df['cluster']==i].copy()
                # Find the mean of this var for this cluster
                if var in ['dt_start','dt_end']:
                    these_values = np.array(df_this_cluster[var].values)
                    these_values = these_values.astype('datetime64', copy=False)
                    these_values.sort()
                    # print('these_values:',these_values)
                    # print('type(these_values[0]):',type(these_values[0]))
                    clstr_mean = datetime.strftime(these_values[len(these_values)//2].astype(datetime), '%Y-%m-%d %H:%M:%S')
                    print('clstr_mean:',clstr_mean)
                    # print('type(clstr_mean):',type(clstr_mean))
                    # exit(0)
                else:
                    clstr_mean = np.mean(df_this_cluster[var].values)
                # Put those values back into the original dataframe
                df.loc[df['cluster']==i, this_var] = clstr_mean
                # print(str(i)+','+str(clstr_mean))
        elif prefix == 'av':
            # Calculate the average of the variable
            # Find the mean of this var for this cluster
            if var in ['dt_start','dt_end']:
                these_values = np.array(df[var].values)
                these_values = these_values.astype('datetime64', copy=False)
                these_values.sort()
                # print('these_values:',these_values)
                # print('type(these_values[0]):',type(these_values[0]))
                this_mean = datetime.strftime(these_values[len(these_values)//2].astype(datetime), '%Y-%m-%d %H:%M:%S')
                # print('this_mean:',this_mean)
                # print('type(this_mean):',type(this_mean))
                # exit(0)
            else:
                this_mean = np.mean(df[var].values)
            # Put this value back in the original dataframe
            df[this_var] = this_mean
        elif prefix == 'nzca':
            # Calculate the cluster average version of the variable, neglecting all zero values
            #   Reduces the number of points to just one per cluster
            #   At the same time, calculate the standard deviation, `nzscd`
            # Get a list of clusters in this dataframe
            clstr_ids = np.unique(np.array(df['cluster'].values))
            clstr_ids = clstr_ids[~np.isnan(clstr_ids)]
            # Remove the noise points
            clstr_ids = clstr_ids[clstr_ids != -1]
            # Loop over each cluster
            for i in clstr_ids:
                # Find the data from this cluster
                df_this_cluster = df[df['cluster']==i].copy()
                # Neglect all zero values
                df_this_cluster = df_this_cluster[df_this_cluster[var] != 0]
                # print('\t\t- Neglected zero values for cluster',i,'new len:',len(df_this_cluster))
                # Find the mean of this var for this cluster
                if var in ['dt_start','dt_end']:
                    these_values = np.array(df_this_cluster[var].values)
                    these_values = these_values.astype('datetime64', copy=False)
                    these_values.sort()
                    # print('these_values:',these_values)
                    # print('type(these_values[0]):',type(these_values[0]))
                    clstr_mean = datetime.strftime(these_values[len(these_values)//2].astype(datetime), '%Y-%m-%d %H:%M:%S')
                    print('clstr_mean:',clstr_mean)
                    # print('type(clstr_mean):',type(clstr_mean))
                    clstr_std = None
                    print('Warning: calculating `nzcsd_'+var+'` not implemented, all values are `None`')
                    # exit(0)
                else:
                    clstr_mean = np.mean(df_this_cluster[var].values)
                    clstr_std  = np.std(df_this_cluster[var].values)
                # Put those values back into the original dataframe
                df.loc[df['cluster']==i, this_var] = clstr_mean
                df.loc[df['cluster']==i, 'nzcsd_'+var] = clstr_std
                # print(str(i)+','+str(clstr_mean))
        elif prefix == 'cs':
            # Calculate the cluster span version of the variable
            #   Reduces the number of points to just one per cluster
            # Get a list of clusters in this dataframe
            clstr_ids = np.unique(np.array(df['cluster'].values))
            clstr_ids = clstr_ids[~np.isnan(clstr_ids)]
            # Remove the noise points
            clstr_ids = clstr_ids[clstr_ids != -1]
            # Loop over each cluster
            for i in clstr_ids:
                # Find the data from this cluster
                df_this_cluster = df[df['cluster']==i].copy()
                # Find the mean of this var for this cluster
                clstr_span = max(df_this_cluster[var].values) - min(df_this_cluster[var].values)
                # Put those values back into the original dataframe
                df.loc[df['cluster']==i, this_var] = clstr_span
        elif prefix == 'csd':
            # Calculate the cluster standard deviation of the variable
            #   Reduces the number of points to just one per cluster
            # Get a list of clusters in this dataframe
            clstr_ids = np.unique(np.array(df['cluster'].values))
            clstr_ids = clstr_ids[~np.isnan(clstr_ids)]
            # Remove the noise points
            clstr_ids = clstr_ids[clstr_ids != -1]
            # Loop over each cluster
            for i in clstr_ids:
                # Find the data from this cluster
                df_this_cluster = df[df['cluster']==i].copy()
                # Find the std of this var for this cluster
                clstr_std = np.std(df_this_cluster[var].values)
                # Put those values back into the original dataframe
                df.loc[df['cluster']==i, this_var] = clstr_std
                # print(str(i)+','+str(clstr_std))
        elif prefix == 'cmm':
            # Find the min/max of each cluster for the variable
            #   Reduces the number of points to just one per cluster
            # Get a list of clusters in this dataframe
            clstr_ids = np.unique(np.array(df['cluster'].values))
            clstr_ids = clstr_ids[~np.isnan(clstr_ids)]
            # Remove the noise points
            clstr_ids = clstr_ids[clstr_ids != -1]
            # Loop over each cluster
            for i in clstr_ids:
                # Find the data from this cluster
                df_this_cluster = df[df['cluster']==i].copy()
                # Find the min/max of this var for this cluster
                clstr_min = min(df_this_cluster[var].values)
                clstr_max = max(df_this_cluster[var].values)
                # Put those values back into the original dataframe
                #   Won't actually use the data in `this_var` so I'll make it obvious it's to be ignored
                df.loc[df['cluster']==i, this_var] = -999
                df.loc[df['cluster']==i, 'cmin_'+var] = clstr_min
                df.loc[df['cluster']==i, 'cmax_'+var] = clstr_max
            #
        elif prefix == 'nir':
            # Find the normalized inter-cluster range for the variable
            #   Reduces the number of points to just one per cluster, minus one
            #   because it depends on the difference between adjacent clusters
            # Get a list of clusters in this dataframe
            clstr_ids = np.unique(np.array(df['cluster'].values))
            clstr_ids = clstr_ids[~np.isnan(clstr_ids)]
            # Remove the noise points
            clstr_ids = clstr_ids[clstr_ids != -1]
            # print('\t\t- clstr_ids without noise:',clstr_ids)
            # Create dataframe to store means and standard deviations of each cluster
            clstr_df = pd.DataFrame(data=clstr_ids, columns=['cluster'])
            clstr_df['clstr_mean'] = None
            clstr_df['clstr_rnge'] = None
            # Loop over each cluster
            for i in clstr_ids:
                # Find the data from this cluster
                df_this_cluster = df[df['cluster']==i].copy()
                # Find the mean and standard deviation of this var for this cluster
                clstr_mean = np.mean(df_this_cluster[var].values)
                clstr_min = min(df_this_cluster[var].values)
                clstr_max = max(df_this_cluster[var].values)
                clstr_std = np.std(df_this_cluster[var].values)
                # print(i,',',clstr_std)
                clstr_rnge = abs(clstr_max - clstr_min)
                # Put those values into the cluster dataframe
                clstr_df.loc[clstr_df['cluster']==i, 'clstr_mean'] = clstr_mean
                clstr_df.loc[clstr_df['cluster']==i, 'clstr_rnge'] = clstr_rnge
            # Sort the cluster dataframe by the mean values
            sorted_clstr_df = clstr_df.sort_values(by='clstr_mean')
            # Get a list of clusters in this dataframe
            sorted_clstr_ids = np.unique(np.array(sorted_clstr_df['cluster'].values))
            # print('\t\t- sorted_clstr_ids:',sorted_clstr_ids)
            # Find normalized inter-cluster range for the first cluster in sorted order
            clstr_id_here  = sorted_clstr_df['cluster'].values[0]
            clstr_id_below = sorted_clstr_df['cluster'].values[1]
            clstr_mean_here = sorted_clstr_df.loc[sorted_clstr_df['cluster']==clstr_id_here, 'clstr_mean'].values[0]
            clstr_mean_below = sorted_clstr_df.loc[sorted_clstr_df['cluster']==clstr_id_below, 'clstr_mean'].values[0]
            this_diff = abs(clstr_mean_below - clstr_mean_here)
            this_rnge = sorted_clstr_df.loc[sorted_clstr_df['cluster']==clstr_id_here, 'clstr_rnge'].values[0]
            this_nir = this_rnge / this_diff
            # print('cluster,rnge,diff,nir,diff_below')
            # print(clstr_id_here,',',this_rnge,',',this_diff,',',this_nir,',',this_diff)
            # Put that value back into the original dataframe
            df.loc[df['cluster']==clstr_id_here, this_var] = this_nir
            # Loop over each middle cluster, finding the normalized inter-cluster ranges
            for i in range(1,len(sorted_clstr_ids)-1):
                # Get cluster id's of this cluster, plus those above and below
                clstr_id_above = sorted_clstr_df['cluster'].values[i-1]
                clstr_id_here  = sorted_clstr_df['cluster'].values[i]
                clstr_id_below = sorted_clstr_df['cluster'].values[i+1]
                # print('clstr_id_above:',clstr_id_above,'clstr_id_here:',clstr_id_here,'clstr_id_below:',clstr_id_below)
                # Get the mean value for each of the three clusters for this variable
                clstr_mean_above = sorted_clstr_df.loc[sorted_clstr_df['cluster']==clstr_id_above, 'clstr_mean'].values[0]
                clstr_mean_here = sorted_clstr_df.loc[sorted_clstr_df['cluster']==clstr_id_here, 'clstr_mean'].values[0]
                clstr_mean_below = sorted_clstr_df.loc[sorted_clstr_df['cluster']==clstr_id_below, 'clstr_mean'].values[0]
                # print('clstr_mean_above:',clstr_mean_above,'clstr_mean_here:',clstr_mean_here,'clstr_mean_below:',clstr_mean_below)
                # Find the distances from this cluster to the clusters above and below
                diff_above = abs(clstr_mean_above - clstr_mean_here)
                diff_below = abs(clstr_mean_below - clstr_mean_here)
                # Find minimum of the distances above and below
                this_diff = min(diff_above, diff_below)
                # this_diff = np.mean([diff_above, diff_below])
                # Calculate the normalized inter-cluster range for this cluster
                this_rnge = sorted_clstr_df.loc[sorted_clstr_df['cluster']==clstr_id_here, 'clstr_rnge'].values[0]
                # print('this_rnge:',this_rnge)
                this_nir = this_rnge / this_diff
                # print(clstr_id_here,',',this_rnge,',',this_diff,',',this_nir,',',diff_below)
                # Put that value back into the original dataframe
                df.loc[df['cluster']==clstr_id_here, this_var] = this_nir
                # exit(0)
            # Find normalized inter-cluster range for the last cluster
            clstr_id_above  = sorted_clstr_df['cluster'].values[-2]
            clstr_id_here = sorted_clstr_df['cluster'].values[-1]
            clstr_mean_here = sorted_clstr_df.loc[sorted_clstr_df['cluster']==clstr_id_here, 'clstr_mean'].values[0]
            clstr_mean_above = sorted_clstr_df.loc[sorted_clstr_df['cluster']==clstr_id_above, 'clstr_mean'].values[0]
            this_diff = abs(clstr_mean_above - clstr_mean_here)
            this_nir = sorted_clstr_df.loc[sorted_clstr_df['cluster']==clstr_id_here, 'clstr_rnge'].values[0] / this_diff
            # print(clstr_id_here,',',this_rnge,',',this_diff,',',this_nir)
            # Put that value back into the original dataframe
            df.loc[df['cluster']==clstr_id_here, this_var] = this_nir
            #
        elif prefix == 'trd':
            # Find the trend vs. dt_start of each cluster for the variable
            #   Reduces the number of points to just one per cluster
            # Get a list of clusters in this dataframe
            clstr_ids = np.unique(np.array(df['cluster'].values))
            clstr_ids = clstr_ids[~np.isnan(clstr_ids)]
            # Remove the noise points
            clstr_ids = clstr_ids[clstr_ids != -1]
            print('\t- Finding trend in var:',var)
            adjustment_factor = 365.25
            per_unit = '/yr'
            # Decide what kind of regression to use
            plot_slopes = 'OLS'
            # Loop over each cluster
            for i in clstr_ids:
                # Find the data from this cluster
                df_this_cluster = df[df['cluster']==i].copy()
                x_data = np.array(df_this_cluster['dt_start'].values)
                x_data = mpl.dates.date2num(x_data)
                y_data = np.array(df_this_cluster[var].values, dtype=float)
                print('\t\t- Cluster:',i)
                # print('\t\t- x_data:',x_data)
                # print('\t\t- y_data:',y_data)
                # Find the trend vs. dt_start of this var for this cluster
                if plot_slopes == 'OLS':
                    # Find the slope of the ordinary least-squares of the points for this cluster
                    m, c, rvalue, pvalue, sd_m = stats.linregress(x_data, y_data)
                    # m, c, sd_m, sd_c = reg.slope, reg.intercept, reg.stderr, reg.intercept_stderr
                    print('\t\t- Slope is',m*adjustment_factor,'+/-',sd_m*adjustment_factor,per_unit) # Note, units for dt_start are in days, so use adjustment_factor to get years
                    print('\t\t- R^2 value is',rvalue)
                    df.loc[df['cluster']==i, 'trdR2_'+var] = rvalue
                else:
                    # Find the slope of the total least-squares of the points for this cluster
                    m, c, sd_m, sd_c = orthoregress(x_data, y_data)
                    print('\t\t- Slope is',m*adjustment_factor,'+/-',sd_m*adjustment_factor,per_unit) # Note, units for dt_start are in days, so use adjustment_factor to get years
                # Put those values back into the original dataframe
                df.loc[df['cluster']==i, this_var] = m*adjustment_factor
                df.loc[df['cluster']==i, 'trdsd_'+var] = sd_m*adjustment_factor
                # print('cluster '+str(i)+', '+str(m))
            #
            # Make sure that I've calculated cRL and nir_SA as well
            if 'cRL' not in new_cl_vars:
                new_cl_vars.append('cRL')
            if 'nir_SA' not in new_cl_vars:
                new_cl_vars.append('nir_SA')
        elif prefix == 'atrd':
            # Find the trend vs. dt_start for the variable
            print('\t- Finding a trend in var:',var)
            adjustment_factor = 365.25
            per_unit = '/yr'
            # Decide what kind of regression to use
            plot_slopes = 'OLS'
            # Find the data
            x_data = np.array(df['dt_start'].values)
            x_data = mpl.dates.date2num(x_data)
            y_data = np.array(df[var].values, dtype=float)
            # print('\t\t- x_data:',x_data)
            # print('\t\t- y_data:',y_data)
            # Find the trend vs. dt_start of this var
            if plot_slopes == 'OLS':
                # Find the slope of the ordinary least-squares of the points
                m, c, rvalue, pvalue, sd_m = stats.linregress(x_data, y_data)
                # m, c, sd_m, sd_c = reg.slope, reg.intercept, reg.stderr, reg.intercept_stderr
                print('\t\t- Slope is',m*adjustment_factor,'+/-',sd_m*adjustment_factor,per_unit) # Note, units for dt_start are in days, so use adjustment_factor to get years
                print('\t\t- R^2 value is',rvalue)
                df['trdR2_'+var] = rvalue
            else:
                # Find the slope of the total least-squares of the points
                m, c, sd_m, sd_c = orthoregress(x_data, y_data)
                print('\t\t- Slope is',m*adjustment_factor,'+/-',sd_m*adjustment_factor,per_unit) # Note, units for dt_start are in days, so use adjustment_factor to get years
            # Put this value to the original dataframe
            df[this_var] = m*adjustment_factor
            df['trdsd_'+var] = sd_m*adjustment_factor
        elif prefix == 'nztrd':
            # Find the trend vs. dt_start of each cluster for the variable, neglecting all zero values
            #   Reduces the number of points to just one per cluster
            # Get a list of clusters in this dataframe
            clstr_ids = np.unique(np.array(df['cluster'].values))
            clstr_ids = clstr_ids[~np.isnan(clstr_ids)]
            # Remove the noise points
            clstr_ids = clstr_ids[clstr_ids != -1]
            print('\t- Finding trend in var:',var)
            adjustment_factor = 365.25
            per_unit = '/yr'
            # Decide what kind of regression to use
            plot_slopes = 'OLS'
            # Loop over each cluster
            for i in clstr_ids:
                # Find the data from this cluster
                df_this_cluster = df[df['cluster']==i].copy()
                # Remove any zero values
                df_this_cluster = df_this_cluster[df_this_cluster[var] != 0]
                x_data = np.array(df_this_cluster['dt_start'].values)
                x_data = mpl.dates.date2num(x_data)
                y_data = np.array(df_this_cluster[var].values, dtype=float)
                print('\t\t- Cluster:',i)
                # print('\t\t- x_data:',x_data)
                # print('\t\t- y_data:',y_data)
                # Find the trend vs. dt_start of this var for this cluster
                if plot_slopes == 'OLS':
                    # Find the slope of the ordinary least-squares of the points for this cluster
                    m, c, rvalue, pvalue, sd_m = stats.linregress(x_data, y_data)
                    # m, c, sd_m, sd_c = reg.slope, reg.intercept, reg.stderr, reg.intercept_stderr
                    print('\t\t- Slope is',m*adjustment_factor,'+/-',sd_m*adjustment_factor,per_unit) # Note, units for dt_start are in days, so use adjustment_factor to get years
                    print('\t\t- R^2 value is',rvalue)
                else:
                    # Find the slope of the total least-squares of the points for this cluster
                    m, c, sd_m, sd_c = orthoregress(x_data, y_data)
                    print('\t\t- Slope is',m*adjustment_factor,'+/-',sd_m*adjustment_factor,per_unit) # Note, units for dt_start are in days, so use adjustment_factor to get years
                # Put those values back into the original dataframe
                df.loc[df['cluster']==i, this_var] = m*adjustment_factor
                df.loc[df['cluster']==i, 'nztrdsd_'+var] = sd_m*adjustment_factor
                df.loc[df['cluster']==i, 'nztrdR2_'+var] = rvalue
                # print('cluster '+str(i)+', '+str(m))
            #
            # Make sure that I've calculated cRL and nir_SA as well
            if 'cRL' not in new_cl_vars:
                new_cl_vars.append('cRL')
            if 'nir_SA' not in new_cl_vars:
                new_cl_vars.append('nir_SA')
        if this_var == 'cRL':
            # Find the lateral density ratio R_L for each cluster
            #   Reduces the number of points to just one per cluster
            # Get a list of clusters in this dataframe
            clstr_ids = np.unique(np.array(df['cluster'].values))
            clstr_ids = clstr_ids[~np.isnan(clstr_ids)]
            # Remove the noise points
            clstr_ids = clstr_ids[clstr_ids != -1]
            # Loop over each cluster
            for i in clstr_ids:
                # Find the data from this cluster
                df_this_cluster = df[df['cluster']==i].copy()
                # Find the variables needed
                alphas = df_this_cluster['alpha'].values
                temps  = df_this_cluster['CT'].values
                betas  = df_this_cluster['beta'].values
                salts  = df_this_cluster['SA'].values
                # Calculate variables needed
                aTs = alphas * temps
                BSs = betas * salts
                # Find the slope of this cluster in aT-BS space
                # m, c = np.linalg.lstsq(np.array([BSs, np.ones(len(BSs))]).T, aTs, rcond=None)[0]
                # Find the slope of the total least-squares of the points for this cluster
                m, c, sd_m, sd_c = orthoregress(BSs, aTs)
                # The lateral density ratio is the inverse of the slope
                this_cRL = 1/m
                # print('cluster:',i,'cRL:',this_cRL)
                # Put those values back into the original dataframe
                df.loc[df['cluster']==i, this_var] = this_cRL
            #
        elif this_var == 'n_points':
            # Find the number of points for each cluster
            #   Reduces the number of points to just one per cluster
            # Get a list of clusters in this dataframe
            clstr_ids = np.unique(np.array(df['cluster'].values))
            clstr_ids = clstr_ids[~np.isnan(clstr_ids)]
            # Remove the noise points
            clstr_ids = clstr_ids[clstr_ids != -1]
            # Loop over each cluster
            for i in clstr_ids:
                # Find the data from this cluster
                df_this_cluster = df[df['cluster']==i].copy()
                # Find the number of points for this cluster
                this_n_points = len(df_this_cluster)
                # Put those values back into the original dataframe
                df.loc[df['cluster']==i, this_var] = this_n_points
            #
        #
    #
    # Remove rows where the plot variables are null
    for this_var in new_cl_vars:
        if 'nir_' not in this_var:
            df = df[df[this_var].notnull()]
    return df

################################################################################

def find_outliers(df, var_keys, mrk_outliers, threshold=2, outlier_type = 'None'):
    """
    Finds any outliers in the dataframe with respect to the x and y keys

    df              A pandas data frame
    var_keys        A list of strings of the names of the columns on which to 
                        find outliers
    mrk_outliers    Either True, or 'ends' (to mask out first and last cluster ids before calculations)
    threshold       The threshold zscore for which to consider an outlier
    """
    if outlier_type == 'MZS':        # Modified Z-Score
        threshold = 3.5 # recommended by Iglewicz and Hoaglin 1993
    for v_key in var_keys:
        # Get the values of the variable for this key
        # v_values = np.array(df[df['cluster']<=120][v_key].values, dtype=np.float64)
        v_values = np.array(df[v_key].values, dtype=np.float64)
        # print(v_key,v_values)
        # Get the cluster ids for this variable
        v_cids = np.array(df['cluster'].values, dtype=np.int64)
        # Mask out cluster ids 120 or higher
        v_cids = np.ma.masked_where(v_cids>=120, v_cids)
        if mrk_outliers == 'ends':
            # print('before')
            # print('v_cids:',v_cids)
            # Mask out the first and last cluster
            cid_min = 0
            cid_max = max(v_cids)
            v_cids = np.ma.masked_where((v_cids==cid_min) | (v_cids==cid_max), v_cids)
            # print('after')
            # print('v_cids:',v_cids)
            # Make a column to specify those end points should be considered outliers
            df['out_ends'] = (df['cluster'] == cid_min) | (df['cluster'] == cid_max)
        else:
            df['out_ends'] = False
        # Apply the mask to the values
        v_values = np.ma.masked_where(v_cids.mask, v_values)
        # print('v_values:',v_values)
        if outlier_type == 'zscore':
            # Find the zscores 
            v_zscores = stats.mstats.zscore(v_values)
            # Put those values back into the dataframe
            df['zs_'+ v_key] = v_zscores
            # print('v_zscores:',v_zscores)
            # Make a column of True/False whether that row is an outlier
            df['out_'+v_key] = (df['zs_'+v_key] > threshold) | (df['zs_'+v_key] < -threshold)
        elif outlier_type == 'MZS':
            # Find the MADs (median absolute deviation) in a vector
            v_MADs = np.ma.median(np.abs(v_values - np.ma.median(v_values)))
            # Find the modified z-scores in a vector
            v_MZSs = 0.6745 * (v_values - np.ma.median(v_values)) / v_MADs
            # Put those values back into the dataframe
            df['mzs_'+ v_key] = v_MZSs
            # Make a column of True/False whether that row is an outlier
            df['out_'+v_key] = (df['mzs_'+v_key] > threshold) | (df['mzs_'+v_key] < -threshold)
        elif outlier_type == 'None':
            # Make a column where all values are False
            df['out_'+v_key] = False
        # Set the clusters with ids 120 or higher to True in outlier column
        df['out_'+v_key].loc[df['cluster']>=120] = True
        # Set the first and last clusters to True in outlier column
        #   This is because they were not divided correctly due to the rolling average
        # df['out_'+v_key].loc[df['cluster']==cid_min] = True
        # df['out_'+v_key].loc[df['cluster']==cid_max] = True
    # print(df)
    # exit(0)
    return df

################################################################################

def mark_outliers(ax, df, x_key, y_key, clr_map, mrk_outliers, find_all=False, threshold=2, mrk_clr='purple', mrk_shape=std_marker, mk_size=mrk_size, mrk_for_other_vars=[]):
    """
    Finds and marks outliers in the dataframe with respect to the x and y keys
    on the provided axis

    ax              The axis on which to plot the outlier markers
    df              A pandas data frame
    x_key           A string of the variable from df to use on the x axis
    y_key           A string of the variable from df to use on the y axis
    clr_map         A string of the color map to use
    mrk_outliers    True/False as whether to mark the outliers (or 'pre-calced', or 'ends')
    find_all        True/False as whether to find outliers in both cRL and nir_SP
    threshold       The threshold zscore for which to consider an outlier
    mrk_clr         The color in which to mark the outliers
    mk_size         The size of the marker to use to mark the outliers
    mrk_for_other_vars
    """
    # Find outliers
    if 'cRL' in mrk_for_other_vars and 'nir_SA' in mrk_for_other_vars:
        print('\t- Marking outliers in cRL and nir_SA')
        if mrk_outliers != 'pre-calced':
            df = find_outliers(df, ['cRL', 'nir_SA'], mrk_outliers, threshold)
        # Set the values of all rows for 'out_'+x_key to False
        ## If I don't do this, then the values not set to True below are NaN
        df['out_'+x_key] = False
        # Set rows of 'out_'+x_key to True if either 'out_cRL', 'out_nir_SA', or 'out_ends' is True
        df.loc[df['out_cRL']==True, 'out_'+x_key] = True
        df.loc[df['out_nir_SA']==True, 'out_'+x_key] = True
        # Plot the outliers in 'cRL', 'nir_SA', and 'ends' in different colors
        cRL_x_data = np.array(df[df['out_cRL']==True][x_key].values, dtype=np.float64)
        cRL_y_data = np.array(df[df['out_cRL']==True][y_key].values, dtype=np.float64)
        nir_x_data = np.array(df[df['out_nir_SA']==True][x_key].values, dtype=np.float64)
        nir_y_data = np.array(df[df['out_nir_SA']==True][y_key].values, dtype=np.float64)
        if mrk_outliers == 'ends':
            df.loc[df['out_ends']==True, 'out_'+x_key] = True
            end_x_data = np.array(df[df['out_ends']==True][x_key].values, dtype=np.float64)
            end_y_data = np.array(df[df['out_ends']==True][y_key].values, dtype=np.float64)
        if x_key in ['ca_FH_cumul']:
            # Don't plot the outliers
            foo = 2
        else:
            if clr_map == 'clr_all_same':
                # Plot over the outliers in a different color
                ax.scatter(cRL_x_data, cRL_y_data, color=out_clr_RL, s=mk_size, marker='x', alpha=mrk_alpha, zorder=4, label=r'$R_L$ outlier')
                ax.scatter(nir_x_data, nir_y_data, color=out_clr_IR, s=mk_size, marker='+', alpha=mrk_alpha, zorder=4)
                if mrk_outliers == 'ends':
                    ax.scatter(end_x_data, end_y_data, color=out_clr_end, s=mk_size, marker='o', alpha=mrk_alpha, zorder=4, label=r'Endpoint')
            else:
                ax.scatter(cRL_x_data, cRL_y_data, edgecolors=out_clr_RL, s=mk_size*5, marker='o', facecolors='none', zorder=2)
                ax.scatter(nir_x_data, nir_y_data, edgecolors=out_clr_IR, s=mk_size*6, marker='o', facecolors='none', zorder=2)
                if mrk_outliers == 'ends':
                    ax.scatter(end_x_data, end_y_data, edgecolors=out_clr_end, s=mk_size*6, marker='o', facecolors='none', zorder=2)
        # Get data with outliers
        x_data = np.array(df[df['out_'+x_key]==True][x_key].values, dtype=np.float64)
        y_data = np.array(df[df['out_'+x_key]==True][y_key].values, dtype=np.float64)
    else:
        print('\t- Marking outliers in',x_key)
        if mrk_outliers != 'pre-calced':
            df = find_outliers(df, [x_key], mrk_outliers, threshold)
        if mrk_outliers == 'ends':
            end_x_data = np.array(df[df['out_ends']==True][x_key].values, dtype=np.float64)
            end_y_data = np.array(df[df['out_ends']==True][y_key].values, dtype=np.float64)
        # Find the marker color
        if x_key == 'cRL':
            mrk_clr = out_clr_RL
        elif x_key == 'nir_SA':
            mrk_clr = out_clr_IR
        else:
            mrk_clr = 'r'
        # Get data with outliers
        print('\t- Marking outliers in',x_key)
        x_data = np.array(df[df['out_'+x_key]==True][x_key].values, dtype=np.float64)
        y_data = np.array(df[df['out_'+x_key]==True][y_key].values, dtype=np.float64)
        # Mark the outliers
        if clr_map == 'clr_all_same':
            # Plot over the outliers in a different color
            ax.scatter(x_data, y_data, color=mrk_clr, s=mk_size, marker=mrk_shape, zorder=4)
            if mrk_outliers == 'ends':
                df.loc[df['out_ends']==True, 'out_'+x_key] = True
                ax.scatter(end_x_data, end_y_data, color=out_clr_end, s=mk_size, marker='o', alpha=mrk_alpha, zorder=4, label=r'Endpoint')
        else:
            ax.scatter(x_data, y_data, edgecolors=mrk_clr, s=mk_size*5, marker='o', facecolors='none', zorder=2)
            if mrk_outliers == 'ends':
                df.loc[df['out_ends']==True, 'out_'+x_key] = True
                ax.scatter(end_x_data, end_y_data, edgecolors=out_clr_end, s=mk_size*6, marker='o', facecolors='none', zorder=2)
    # Print the outliers
    if len(x_data) == 0:
        print('No outliers found')
        return
    else:
        print('Found outliers:')
        print(x_data)
        print(y_data)
    # Run it again
    # if mrk_clr == 'r':
    if False:
        mark_outliers(ax, df.loc[df['out_'+x_key]==False], x_key, y_key, clr_map, mrk_outliers, find_all, threshold, mrk_clr='b', mk_size=mk_size)

################################################################################

def plot_clusters(a_group, ax, pp, df, x_key, y_key, z_key, cl_x_var, cl_y_var, cl_z_var, clr_map, m_pts, m_cls, rel_val, ell, box_and_whisker=True, plot_slopes=False):
    """
    Plots the clusters found by HDBSCAN on the x-y plane

    ax              The axis on which to plot
    pp
    df              A pandas data frame output from HDBSCAN_
    x_key           String of the name of the column to use on the x-axis
    y_key           String of the name of the column to use on the y-axis
    z_key           String of the name of the column to use on the y-axis
    cl_x_var        String of the name of the column used on x-axis of clustering
    cl_y_var        String of the name of the column used on y-axis of clustering
    cl_z_var        String of the name of the column used on z-axis of clustering
    clr_map         String of the name of the colormap to use (ex: 'clusters')
    m_pts       An integer, 'min_samples', number of points in neighborhood for a core point
    m_cls       An integer, 'min_cluster_size', the minimum number of points for a cluster
    rel_val
    ell
    box_and_whisker True/False whether to include the box and whisker plot
    plot_slopes     True/False whether to plot lines of least-squares slopes for
                        each cluster
    """
    # Needed to make changes to a copy of the global variable
    global mrk_size
    m_size = mrk_size3
    # Find extra arguments, if given
    legend = pp.legend
    if not isinstance(pp.extra_args, type(None)):
        try:
            plt_noise = pp.extra_args['plt_noise']
        except:
            plt_noise = True
        try:
            sort_clstrs = pp.extra_args['sort_clstrs']
        except:
            sort_clstrs = True
        try:
            relab_these = pp.extra_args['relab_these']
        except:
            relab_these = False
        try:
            plot_centroid = pp.extra_args['plot_centroid']
        except:
            plot_centroid = None
        try:
            clstrs_to_plot = pp.extra_args['clstrs_to_plot']
        except:
            clstrs_to_plot = []
        try:
            mrk_outliers = pp.extra_args['mark_outliers']
        except:
            mrk_outliers = False
        try:
            fit_vars = pp.extra_args['fit_vars']
        except:
            fit_vars = False
        try:
            mark_LHW_AW = pp.extra_args['mark_LHW_AW']
        except:
            mark_LHW_AW = False
        try:
            use_raster = pp.extra_args['use_raster']
        except:
            use_raster = False
        try:
            mark_left_right_clusters = pp.extra_args['mark_LR']
        except:
            mark_left_right_clusters = True
    else:
        plt_noise = True
        sort_clstrs = True
        relab_these = False
        plot_centroid = False
        clstrs_to_plot = []
        mrk_outliers = False
        fit_vars = False
        mark_LHW_AW = False
        use_raster = False
        mark_left_right_clusters = True
    # Decide whether to plot the centroid or not
    if isinstance(plot_centroid, type(None)):
        if x_key in pf_vars or y_key in pf_vars or z_key in pf_vars:
            plot_centroid = False
        elif y_key == 'CT' or z_key == 'CT':
            plot_centroid = False
        else:
            plot_centroid = True
    # Find the cluster labels that exist in the dataframe
    cluster_numbers = np.unique(np.array(df['cluster'].values, dtype=int))
    #   Delete the noise point label "-1"
    cluster_numbers = np.delete(cluster_numbers, np.where(cluster_numbers == -1))
    # Remove rows where the plot variables are null
    for var in [x_key, y_key, z_key]:
        if not isinstance(var, type(None)):
            df = df[df[var].notnull()]
    # Re-order the cluster labels, if specified
    if sort_clstrs:
        df = sort_clusters(df, cluster_numbers, ax)
    # Relabel clusters, if specified
    if relab_these:
        df, cluster_numbers = relabel_clusters(df, cluster_numbers, relab_these)
    # Noise points are labeled as -1
    # Plot noise points first
    df_noise = df[df.cluster==-1]
    if isinstance(z_key, type(None)):
        df_noise_z_key = 0
        plot_3d = False
    elif z_key in ['dt_start', 'dt_end']:
        df_noise_z_key = mpl.dates.date2num(df_noise[z_key])
        plot_3d = True
    else:
        df_noise_z_key = df_noise[z_key]
        plot_3d = True
    if plt_noise:
        if plot_3d == False:
            # Plot in 2D
            if use_raster:
                ax.plot(df_noise[x_key], df_noise[y_key], color=std_clr, markersize=m_size, marker=std_marker, alpha=noise_alpha, zorder=1, rasterized=True)
            else:
                ax.scatter(df_noise[x_key], df_noise[y_key], color=std_clr, s=m_size, marker=std_marker, alpha=noise_alpha, zorder=1)
        else:
            # Plot in 3D
            ax.scatter(df_noise[x_key], df_noise[y_key], zs=df_noise_z_key, color=std_clr, s=m_size, marker=std_marker, alpha=noise_alpha, zorder=1)
            plot_centroid = False
            # plot_slopes = False
    n_noise_pts = len(df_noise)
    # Check whether to just plot one point per cluster
    if x_key in ca_vars:
        # Drop duplicates to have just one row per cluster
        df.drop_duplicates(subset=[x_key], keep='first', inplace=True)
        plot_centroid = False
        mrk_size = cent_mrk_size
        plot_slopes = False
    if y_key in ca_vars:
        # Drop duplicates to have just one row per cluster
        df.drop_duplicates(subset=[y_key], keep='first', inplace=True)
        plot_centroid = False
        m_size = cent_mrk_size
        # plot_slopes = False
    if 'cRL' in [x_key, y_key]:
        # Drop duplicates to have just one row per cluster
        df.drop_duplicates(subset=['cRL'], keep='first', inplace=True)
        plot_centroid = False
        m_size = cent_mrk_size
        # plot_slopes = False
    if 'cRl' in [x_key, y_key]:
        # Drop duplicates to have just one row per cluster
        df.drop_duplicates(subset=['cRl'], keep='first', inplace=True)
        plot_centroid = False
        m_size = cent_mrk_size
        plot_slopes = False
    if x_key in cmm_vars:
        # Split the prefix from the original variable (assumes an underscore split)
        split_var = x_key.split('_', 1)
        var = split_var[1]
        # Drop duplicates to have just one row per cluster
        df.drop_duplicates(subset=['cmin_'+var,'cmax_'+var], keep='first', inplace=True)
        # Calculate
        df['cmm_mid'] = (df['cmax_'+var] + df['cmin_'+var]) / 2
        df['bar_len'] = (df['cmax_'+var] - df['cmin_'+var]) / 2
        x_key = 'cmm_mid'
        plot_centroid = False
        plot_slopes = False
    if y_key in cmm_vars:
        # Split the prefix from the original variable (assumes an underscore split)
        split_var = y_key.split('_', 1)
        var = split_var[1]
        # Drop duplicates to have just one row per cluster
        df.drop_duplicates(subset=['cmin_'+var,'cmax_'+var], keep='first', inplace=True)
        # Calculate
        df['cmm_mid'] = (df['cmax_'+var] + df['cmin_'+var]) / 2
        df['bar_len'] = (df['cmax_'+var] - df['cmin_'+var]) / 2
        y_key = 'cmm_mid'
        plot_centroid = False
        plot_slopes = False
    # Look for outliers
    # if 'nir' in x_key or 'cRL' in x_key or 'trd' in x_key:
    #     mrk_outliers = True
    # else:
    #     mrk_outliers = False
    print('\t- Mark outliers:',mrk_outliers)
    # Recalculate the cumulative heat flux without outliers
    if 'ca_FH_cumul' in x_key:
        # Mark outliers
        mark_outliers(ax, df, x_key, y_key, clr_map, mrk_outliers, mk_size=m_size, mrk_clr='red', mrk_for_other_vars=[])#['cRL', 'nir_SA'])
        # Remove the outliers from the df
        df = df[df['out_'+x_key]==False]
        # Sort the cluster average heat flux by the y_key variable
        df = df.sort_values(by=y_key)
        # Calculate the cumulative heat flux, starting with the lowest value of y_key
        df['ca_FH_cumul'] = df['ca_FH'].cumsum()
    # Set variable as to whether to invert the y axis
    invert_y_axis = False
    # Which colormap?
    if clr_map == 'cluster':
        # Make blank lists to record values
        pts_per_cluster = []
        # clstr_means = []
        # clstr_stdvs = []
        # Loop through each cluster
        for i in cluster_numbers:
            # Decide on the color and symbol, don't go off the end of the arrays
            my_clr = distinct_clrs[i%len(distinct_clrs)]
            if m_size == cent_mrk_size:
                my_mkr = r"${}$".format(str(i))
                m_alpha = 1
            else:
                my_mkr = mpl_mrks[i%len(mpl_mrks)]
                # m_size = cent_mrk_size
                # my_mkr = r"${}$".format(str(i))
                m_alpha = mrk_alpha3/2
            # Find the data from this cluster
            df_this_cluster = df[df['cluster']==i]
            # If this cluster has no points, skip it
            if len(df_this_cluster) < 1:
                continue
            # print(df_this_cluster)
            if fit_vars:
                df_this_cluster = calc_fit_vars(df_this_cluster, (x_key, y_key, z_key, clr_map), fit_vars)
            # Get relevant data
            x_data = df_this_cluster[x_key] 
            y_data = df_this_cluster[y_key] 
            if isinstance(z_key, type(None)):
                df_z_key = 0
            # elif z_key in ['dt_start', 'dt_end']:
            elif 'dt_start' in z_key or 'dt_end' in z_key:
                df_z_key = mpl.dates.date2num(df_this_cluster[z_key])
            else:
                df_z_key = df_this_cluster[z_key]
            alphas = df_this_cluster['clst_prob'] #df[df.cluster == i]['clst_prob']
            # Plot the points for this cluster with the specified color, marker, and alpha value
            #   Having an issue with actually using the alphas from above without a TypeError
            if x_key == 'cmm_mid':
                xerrs = df[df.cluster == i]['bar_len']
                ax.errorbar(x_data, y_data, xerr=xerrs, color=my_clr, capsize=l_cap_size)
            elif y_key == 'cmm_mid':
                yerrs = df[df.cluster == i]['bar_len']
                ax.errorbar(x_data, y_data, yerr=yerrs, color=my_clr, capsize=l_cap_size)
            elif 'nir' in x_key or 'cRL' in x_key or 'cRl' in x_key:
                my_mkr = r"${}$".format(str(i))
                # Plot a backing circle for the cluster number if it's a light color
                if my_clr in [jackson_clr[13], jackson_clr[14]]:
                    ax.scatter(x_data, y_data, color=std_clr, s=m_size*1.1, marker='o', alpha=0.5, zorder=5)
                # Plot the cluster number
                ax.scatter(x_data, y_data, color=my_clr, s=m_size, marker=my_mkr, alpha=1, zorder=5)
            else:
                if plot_3d == False:
                    # Plot in 2D')
                    if use_raster:
                        ax.plot(x_data, y_data, color=my_clr, markersize=m_size, marker=my_mkr, alpha=m_alpha, zorder=5, rasterized=True)
                    else:
                        ax.scatter(x_data, y_data, color=my_clr, s=m_size, marker=my_mkr, alpha=m_alpha, zorder=5)
                    # If using a number as a marker, plot a backing circle
                    if my_mkr == r"${}$".format(str(i)):
                        # Plot a backing circle for the cluster number if it's a light color
                        if my_clr in [jackson_clr[13], jackson_clr[14]]:
                            ax.scatter(x_data, y_data, color=std_clr, s=m_size*1.1, marker='o', alpha=0.5, zorder=4)
                else:
                    # Plot in 3D
                    ax.scatter(x_data, y_data, zs=df_z_key, color=my_clr, s=m_size, marker=my_mkr, alpha=m_alpha, zorder=5)
            # Plot the centroid of this cluster
            if plot_centroid:
                x_mean = np.mean(x_data)
                x_stdv = np.std(x_data)
                y_mean = np.mean(y_data)
                y_stdv = np.std(y_data)
                # This plots a circle upon which to put the centroid symbol
                ax.scatter(x_mean, y_mean, color=std_clr, s=cent_mrk_size*1.3, marker='o', zorder=9)
                # This will plot a marker at the centroid
                # ax.scatter(x_mean, y_mean, color=cnt_clr, s=cent_mrk_size, marker=my_mkr, zorder=10)
                # This will plot the cluster number at the centroid
                ax.scatter(x_mean, y_mean, color=cnt_clr, s=cent_mrk_size, marker=r"${}$".format(str(i)), zorder=10)
            # print('plot slopes is',plot_slopes)
            if plot_slopes and x_key != 'cRL' and 'ca_' not in x_key and 'nir' not in x_key and 'trd' not in x_key:
                print('\t- Finding slope for cluster',i)
                if plot_3d == False:
                    add_linear_slope(ax, pp, df_this_cluster, x_data, y_data, x_key, y_key, my_clr, plot_slopes, anno_prefix=str(i)+': ')
                else:
                    # Fit a 2d polynomial to the z data
                    plot_polyfit2d(ax, pp, x_data, y_data, df_z_key)
            #
            # Record the number of points in this cluster
            pts_per_cluster.append(len(x_data))
            # Record mean and standard deviation to calculate normalized inter-cluster range
            # clstr_means.append(y_mean)
            # clstr_stdvs.append(y_stdv)
        print('\t- Used marker size:',m_size,'and alpha:',m_alpha)
        # Mark outliers, if specified
        if x_key != 'cRL' and x_key != 'nir_SA':
            other_out_vars = ['cRL', 'nir_SA']
            other_out_vars = []
        else:
            other_out_vars = []
        if mrk_outliers:
            mark_outliers(ax, df, x_key, y_key, clr_map, mrk_outliers, mk_size=m_size, mrk_clr='red', mrk_for_other_vars=other_out_vars)
        # Add cluster markers on left and right-hand sides if plotting vs time
        # if x_key in ['dt_start', 'dt_end'] and m_size != cent_mrk_size:
        if x_key in ['dt_start', 'dt_end'] and mark_left_right_clusters:
            # Select date on which to place the cluster numbers on the left-hand side
            x_place = mpl.dates.date2num(datetime.fromisoformat('2005-07-01'))
            these_clst_ids = []
            # Define the date bounds for BGR0506
            BGR0506_start = mpl.dates.date2num(datetime.fromisoformat('2005-08-15'))
            BGR0506_end = mpl.dates.date2num(datetime.fromisoformat('2006-08-15'))
            # List layers to mark
            mark_these_ones = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
            # mark_these_ones = [0, 3, 9, 18, 27, 35, 39, 47, 49]
            # Just the clusters that appear in BGR0506
            # mark_these_ones = [0, 4, 6, 7, 10, 13, 15, 17, 19, 22, 23, 26, 30, 31, 32, 34, 37, 39, 42, 44, 46, 50, 52, 54, 57, 58, 60, 62, 63, 64, 67, 69, 70, 72, 75, 76, 77, 78, 79, 81, 83, 84, 86, 88, 90, 94, 96, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147]
            # for i in cluster_numbers:
            for i in mark_these_ones:
                # Find the data from this cluster
                df_this_cluster = df[df['cluster']==i]
                # Get data just between 2005-08-15 and 2006-08-15
                df_this_cluster = df_this_cluster[(df_this_cluster['dt_start'] > BGR0506_start) & (df_this_cluster['dt_start'] < BGR0506_end)]
                # Get relevant data
                y_data = df_this_cluster[y_key] 
                y_mean = np.mean(y_data)
                # Decide on the color and symbol, don't go off the end of the arrays
                my_clr = distinct_clrs[i%len(distinct_clrs)]
                # Plot a backing circle for the cluster number if it's a light color
                if my_clr in [jackson_clr[13], jackson_clr[14]]:
                    ax.scatter(x_place, y_mean, color=std_clr, s=cent_mrk_size*1.3, marker='o', alpha=0.5, zorder=9)
                # This will plot the cluster number on the left-hand side
                ax.scatter(x_place, y_mean, color=my_clr, s=cent_mrk_size, marker=r"${}$".format(str(i)), zorder=10)
                these_clst_ids.append(i)
                # Check whether to plot symbols for the LHW and AW
                if mark_LHW_AW:
                    # Parse the bounds keys
                    x_LHW_key, y_LHW_key, x_AW_key, y_AW_key = parse_LHW_AW_keys(x_key, y_key)
                    # Load the bounds data
                    this_BGR = 'BGR_all'
                    # this_BGR = 'BGR1516'
                    bnds_df = pl.load(open('outputs/'+this_BGR+'_LHW_AW_properties.pickle', 'rb'))
                    # Find the mean y values for the LHW and AW
                    y_LHW_data = bnds_df[y_LHW_key]
                    y_LHW_mean = np.mean(y_LHW_data)
                    y_AW_data = bnds_df[y_AW_key]
                    y_AW_mean = np.mean(y_AW_data)
                    # Plot the LHW and AW
                    this_mrk_size = np.sqrt(cent_mrk_size)
                    ax.plot(x_place, y_LHW_mean, color=LHW_clr, markersize=this_mrk_size, marker=LHW_mrk, fillstyle='bottom', markerfacecoloralt=LHW_facealtclr, markeredgecolor=LHW_edgeclr, linewidth=0, zorder=10)
                    ax.plot(x_place, y_AW_mean, color=AW_clr, markersize=this_mrk_size, marker=AW_mrk, fillstyle='top', markerfacecoloralt=AW_facealtclr, markeredgecolor=AW_edgeclr, linewidth=0, zorder=10)
            # print('Cluster numbers:',these_clst_ids)
            # Select date on which to place the cluster numbers on the right-hand side
            x_place = mpl.dates.date2num(datetime.fromisoformat('2022-10-01'))
            these_clst_ids = []
            # Define the date bounds for BGR2122
            BGR2122_start = mpl.dates.date2num(datetime.fromisoformat('2021-08-15'))
            BGR2122_end = mpl.dates.date2num(datetime.fromisoformat('2022-08-15'))
            # Just the clusters that appear in BGR2122
            # mark_these_ones = [6, 7, 10, 14, 17, 19, 23, 24, 29, 31, 32, 35, 37, 40, 44, 46, 47, 52, 53, 55, 57, 58, 60, 62, 63, 65, 67, 69, 70, 72, 73, 76, 77, 78, 79, 80, 81, 83, 85, 88, 90, 92, 94, 96, 97, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137]
            # for i in cluster_numbers:
            for i in mark_these_ones:
                # Find the data from this cluster
                df_this_cluster = df[df['cluster']==i]
                # Get data just between 2005-08-15 and 2006-08-15
                df_this_cluster = df_this_cluster[(df_this_cluster['dt_start'] > BGR2122_start) & (df_this_cluster['dt_start'] < BGR2122_end)]
                # Get relevant data
                y_data = df_this_cluster[y_key] 
                y_mean = np.mean(y_data)
                # Decide on the color and symbol, don't go off the end of the arrays
                my_clr = distinct_clrs[i%len(distinct_clrs)]
                # Plot a backing circle for the cluster number if it's a light color
                if my_clr in [jackson_clr[13], jackson_clr[14]]:
                    ax.scatter(x_place, y_mean, color=std_clr, s=cent_mrk_size*1.3, marker='o', alpha=0.5, zorder=9)
                # This will plot the cluster number on the right-hand side
                ax.scatter(x_place, y_mean, color=my_clr, s=cent_mrk_size, marker=r"${}$".format(str(i)), zorder=10)
                these_clst_ids.append(i)
                # Check whether to plot symbols for the LHW and AW
                if mark_LHW_AW:
                    # Plot the LHW and AW
                    ax.plot(x_place, y_LHW_mean, color=LHW_clr, markersize=this_mrk_size, marker=LHW_mrk, fillstyle='bottom', markerfacecoloralt=LHW_facealtclr, markeredgecolor=LHW_edgeclr, linewidth=0, zorder=10)
                    ax.plot(x_place, y_AW_mean, color=AW_clr, markersize=this_mrk_size, marker=AW_mrk, fillstyle='top', markerfacecoloralt=AW_facealtclr, markeredgecolor=AW_edgeclr, linewidth=0, zorder=10)
            # print('Cluster numbers:',these_clst_ids)
        # Add a line at nir_var = 1, if plotting nir
        if 'nir_' in x_key:
            # Get bounds of axes
            #   If there are x_lims and y_lims provided, use those
            try:
                x_bnds = pp.ax_lims['x_lims']
            except:
                x_bnds = ax.get_xbound()
            try:
                y_bnds = pp.ax_lims['y_lims']
            except:
                y_bnds = ax.get_ybound()
            # x_bnds = ax.get_xbound()
            # y_bnds = ax.get_ybound()
            # # If there are x_lims and y_lims provided, use those instead
            # if not isinstance(pp.ax_lims['x_lims'], type(None)):
            #     x_bnds = pp.ax_lims['x_lims']
            # if not isinstance(pp.ax_lims['y_lims'], type(None)):
            #     y_bnds = pp.ax_lims['y_lims']
            x_span = abs(x_bnds[1] - x_bnds[0])
            y_span = abs(y_bnds[1] - y_bnds[0])
            # print('x_bnds:',x_bnds)
            # print('y_bnds:',y_bnds)
            # print('x_span:',x_span)
            # print('y_span:',y_span)
            # Add line at nir_var = -1
            print('\t- Adding line at '+x_key+' = 1')
            ax.axvline(1, color=std_clr, alpha=0.5, linestyle='-')
            # anno_x = 1+x_span/8
            # anno_y = min(y_bnds)+y_span/2
            anno_x = min(x_bnds) + (2/5)*x_span
            anno_y = min(y_bnds) + 4*y_span/5
            print('\t\t- Annotating at',anno_x,anno_y)
            anno_text = ax.annotate(r'$\mathbf{IR_{S_A}=1}$', xy=(anno_x,anno_y), xycoords='data', color=std_clr, fontweight='bold', alpha=0.5, bbox=anno_bbox, zorder=12)
            anno_text.set_fontsize(font_size_lgnd)
        # Add some lines if plotting cRL or trd
        if plot_slopes and (x_key == 'cRL' or 'ca_' in x_key or 'nir' in x_key or 'trd' in x_key):
        # if plot_slopes:
            # Get bounds of axes
            #   If there are x_lims and y_lims provided, use those
            try:
                x_bnds = pp.ax_lims['x_lims']
            except:
                x_bnds = ax.get_xbound()
            try:
                y_bnds = pp.ax_lims['y_lims']
            except:
                y_bnds = ax.get_ybound()
            x_span = abs(x_bnds[1] - x_bnds[0])
            y_span = abs(y_bnds[1] - y_bnds[0])
            # Get the data without the outliers
            if mrk_outliers:
                x_data = df[df['out_'+x_key] == False][x_key].astype('float')
                y_data = df[df['out_'+x_key] == False][y_key].astype('float')
            else:
                x_data = df[x_key].astype('float')
                y_data = df[y_key].astype('float')
            # Plot polynomial fit line
            if x_key == 'cRL':
                print('\t- Adding 2nd degrees polynomial fit line:',plot_slopes)
                # Use polyfit 
                deg = 2
                z = np.polyfit(y_data, x_data, deg)
                x_fit = np.poly1d(z)
                # Get the means and standard deviations
                x_mean = np.mean(x_data)
                x_stdv = np.std(x_data)
                y_mean = np.mean(y_data)
                y_stdv = np.std(y_data)
                # Find R^2 (coefficient of determination) value which is 1 
                #   minus the ratio of the sum of squares of residuals over 
                #   the total sum of squares
                x_fit_data = x_fit(y_data)
                ss_res = ((x_data-x_fit_data)**2).sum()
                ss_tot = ((x_data-x_mean)**2).sum()
                R2 = 1 - ss_res/ss_tot
                print('\t- Sum of square residuals:',ss_res)
                print('\t- Total sum of squares:',ss_tot)
                print('\t- R^2:',R2)
                print('\t- Median cRL:',np.median(x_data))
                print('\t- Mean cRL:',np.mean(x_data))
                print('\t- Standard deviation of cRL:',np.std(x_data))
                # Make an array of values to plot log fit line 
                y_arr = np.linspace(y_bnds[0], y_bnds[1], 50)
                # Plot the fit
                ax.plot(x_fit(y_arr), y_arr, color=alt_std_clr)
                # Add annotation to say what the line is
                line_label = r"$\mathbf{R_L = " + format_sci_notation(z[0]) + " p^2 " + format_sci_notation(z[1])+ "p + " + format_sci_notation(z[2])+"}$"
                # anno_x = x_mean+0.5*x_stdv
                anno_x = min(x_bnds) + (2/5)*x_span
                anno_y = min(y_bnds) + 4*y_span/5
                # anno_x = min(x_bnds) + 2*x_span/4
                # anno_y = min(y_bnds)+y_span/2
                anno_text = ax.annotate(line_label, xy=(anno_x,anno_y), xycoords='data', color=alt_std_clr, fontweight='bold', bbox=anno_bbox, zorder=12)
                anno_text.set_fontsize(font_size_lgnd)
                # Limit the y axis
                ax.set_ylim((y_bnds[0],y_bnds[1]))
            # Plot linear fit line
            else: 
                add_linear_slope(ax, pp, df, x_data, y_data, x_key, y_key, alt_std_clr, plot_slopes)
            #
        # Add legend to report the total number of points and notes on the data
        if legend:
            if mark_LHW_AW:
                # Add standard legend
                add_std_legend(ax, df, x_key, mark_LHW_AW=mark_LHW_AW, lgd_loc=legend)
            else:
                n_pts_patch   = mpl.patches.Patch(color='none', label=str(len(df[x_key]))+' points')
                m_pts_patch = mpl.patches.Patch(color='none', label=r'$m_{pts}$: '+str(m_pts) + r', $\ell$: '+str(ell))
                n_clusters = int(len(cluster_numbers))
                n_clstr_patch = mpl.lines.Line2D([],[],color=cnt_clr, label=r'$n_{cl}$: '+str(n_clusters) + r', $m_{cls}$: '+str(m_cls), marker='*', linewidth=0)
                n_noise_patch = mpl.patches.Patch(color=std_clr, label=r'$n_{noise pts}$: '+str(n_noise_pts), alpha=noise_alpha, edgecolor=None)
                rel_val_patch = mpl.patches.Patch(color='none', label='DBCV: %.4f'%(rel_val))
                ax.legend(handles=[n_pts_patch, m_pts_patch, n_clstr_patch, n_noise_patch, rel_val_patch])
        # Add inset axis for box-and-whisker plot
        if box_and_whisker:
            inset_ax = inset_axes(ax, width="30%", height="10%", loc=4)
            inset_ax.boxplot(pts_per_cluster, vert=0, sym='X')
            #   Adjust the look of the inset axis
            inset_ax.xaxis.set_ticks_position('top')
            inset_ax.set_xticks([int(min(pts_per_cluster)),int(np.median(pts_per_cluster)),int(max(pts_per_cluster))])
            inset_ax.spines['left'].set_visible(False)
            inset_ax.get_yaxis().set_visible(False)
            inset_ax.spines['right'].set_visible(False)
            inset_ax.spines['bottom'].set_visible(False)
            inset_ax.set_xlabel('Points per cluster')
            inset_ax.xaxis.set_label_position('top')
        #
        # Invert y-axis if specified
        if y_key in y_invert_vars:
            invert_y_axis = True
    # Set as raster, if specified
    if use_raster:
        # ax.set_rasterization_zorder(25)
        print('\t- Rasterized axis')
    return invert_y_axis, [df], m_pts, m_cls, rel_val

################################################################################

def sort_clusters(df, cluster_numbers, ax=None, order_by='SA', use_PDF=False):
    """
    Redoes the cluster labels so they are sorted in some way

    df              A pandas data frame output from HDBSCAN_
    cluster_numbers A list of all the labels for the clusterst in df
    ax              The axis on which to draw lines, if applicable
    order_by        String of the variable by which to sort clusters
    use_PDF         True/False whether to sort by the valleys in a probability distribution function
    """
    n_clusters = int(len(cluster_numbers))
    s_key = order_by
    if use_PDF == False:
        print('\t- Sorting clusters by',order_by)
        ## Figure out the order
        sorting_arr = []
        # Loop through each cluster
        for i in cluster_numbers:
            # Find the data from this cluster
            df_this_cluster = df[df['cluster']==i]
            # Get relevant data for sorting
            s_data = df_this_cluster[s_key] 
            s_mean = np.mean(s_data)
            # Change cluster id to temp number in the original dataframe
            #   Need to make a mask first for some reason
            this_cluster_mask = (df['cluster']==i)
            temp_i = int(i - (n_clusters+2))
            df.loc[this_cluster_mask, 'cluster'] = temp_i
            # Add cluster to the sorting array
            sorting_arr.append((temp_i, s_mean))
        # Make the sorting array into a pandas dataframe
        sorting_df = pd.DataFrame(sorting_arr, columns=['temp_i','s_mean']).sort_values(by=['s_mean'])
        # print(sorting_df)
        ## Re-assign cluster labels based on sorted order
        # Loop through each cluster
        i = 0
        for temp_i in sorting_df['temp_i']:
            # Make a mask to locate the rows in the dataframe to change
            this_cluster_mask = (df['cluster']==temp_i)
            # Replace the cluster id
            df.loc[this_cluster_mask, 'cluster'] = i
            # Increment i
            i += 1
        #
    if use_PDF:
        # Find the gaps in the valleys of the histogram and order clusters that way
        #   Following Lu et al. 2022
        noise_threshold = 0
        # Find the data which is not noise
        df_no_noise = df[df['cluster']!=-1]
        # Get the histogram parameters
        h_var, res_bins, median, mean, std_dev = get_hist_params(df_no_noise, s_key, n_h_bins=1000)
        # Make the histogram
        # hist_vals, hist_bins, patches = ax.hist(h_var, bins=res_bins)
        hist_vals, hist_bins = np.histogram(h_var, bins=res_bins)
        # Estimate the PDF using the histogram
        hist_vals = hist_vals / len(df_no_noise)
        print('hist_vals:',hist_vals)
        # print('hist_bins:',len(hist_bins))
        # Gather bin edges to define cluster boundaries
        clstr_bnds = []
        for i in range(len(hist_vals)-1):
            # Get the edges and values for this bin and the next one
            this_bin = hist_bins[i]
            this_val = hist_vals[i]
            next_bin = hist_bins[i+1]
            next_val = hist_vals[i+1]
            # Check to see what the far left edge is like
            if i == 0:
                # If the left-most bin isn't below the threshold, 
                #   set that as the first cluster boundary
                if not this_val <= noise_threshold:
                    clstr_bnds.append(this_bin)
                l_edge = this_bin
                # else:
                #     l_edge = None
            # If this bin is above the threshold but the next is below, then this is a left edge
            if this_val > noise_threshold and next_val <= noise_threshold:
                l_edge = next_bin
            # If this bin is below the threshold but the next isn't, then this is a right edge
            if this_val <= noise_threshold and next_val > noise_threshold:
                r_edge = this_bin
                if isinstance(l_edge, type(None)):
                    clstr_bnds.append(r_edge)
                    continue
                else:
                    # Find the midpoint between the left and right edge
                    mid_point = (r_edge - l_edge)/2 + l_edge
                    # Add this midpoint as a new cluster boundary
                    clstr_bnds.append(mid_point)
                    continue
            # Check to see what the far right edge is like
            if i == len(hist_vals)-2:
                # If the right-most bin isn't below the threshold, 
                #   set that as the last cluster boundary
                if not this_val <= noise_threshold:
                    clstr_bnds.append(this_bin)
                else:
                    # Add midpoint as last cluster boundary
                    clstr_bnds.append((this_bin - l_edge)/2 + l_edge)
                #
            #
        # Add vertical lines to mark all the cluster boundaries
        for bnd in clstr_bnds:
            ax.axvline(bnd, color='r', linestyle='--')
        # Adjust cluster labels, looping through each pair of cluster bounds
        for i in range(len(clstr_bnds)-1):
            # Make a mask to locate the rows in the dataframe to change
            this_cluster_mask = (df['cluster']!=-1) & (df[s_key]>clstr_bnds[i]) & (df[s_key]<clstr_bnds[i+1])
            # Replace the cluster id
            df.loc[this_cluster_mask, 'cluster'] = i
        #
    return df

################################################################################

def relabel_clusters(df, cluster_numbers, relab_these=None):
    """
    Relabels the clusters, swapping the labels in the keys of the relab_these
    dictionary for those in the values

    df              A pandas data frame output from HDBSCAN_
    relab_these     A dictionary of cluster label swaps to make
    """
    print('\t- Relabeling clusters')
    # Confirm that all the keys of relab_these are cluster labels that exist
    for i in relab_these.keys():
        if i not in cluster_numbers:
            print('Error: Cluster number',i,'not found, aborting script')
            exit(0)
    # Prepare second dictionary 
    relab_these2 = {}
    # Loop through each cluster in the keys of relab_these
    for i in relab_these.keys():
        print('\t\t- Changing cluster',i,'to be cluster',relab_these[i])
        if i not in cluster_numbers:
            print('This cluster label is not available')
            continue
        # Change cluster id to a temp number in the original dataframe
        #   Need to make a mask first for some reason
        this_cluster_mask = (df['cluster']==i)
        temp_i = int(-i - 2)
        df.loc[this_cluster_mask, 'cluster'] = temp_i
        # Change the key in relab_these to the temp number
        relab_these2.update({temp_i:relab_these[i]})
    # Loop through the temp cluster labels now in the keys of relab_these
    for i in relab_these2.keys():
        # Change cluster id to the value of this dictionary key
        #   Need to make a mask first for some reason
        this_cluster_mask = (df['cluster']==i)
        df.loc[this_cluster_mask, 'cluster'] = relab_these2[i]
    # Find the cluster labels that exist in the dataframe
    cluster_numbers = np.unique(np.array(df['cluster'].values, dtype=int))
    #   Delete the noise point label "-1"
    cluster_numbers = np.delete(cluster_numbers, np.where(cluster_numbers == -1))
    return df, cluster_numbers

################################################################################

def filter_to_these_clstrs(df, cluster_numbers, clstrs_to_plot=[]):
    """
    Filters to just the clusters listed

    df              A pandas data frame output from HDBSCAN_
    n_clusters      The number of clusters
    clstrs_to_plot  If None, returns all clusters. If array, returns clusters with the matching ids
    """
    # Check whether to select only some clusters to plot
    if len(clstrs_to_plot) > 0:
        print('\t- Only displaying these clusters:',clstrs_to_plot)
        # Make a temporary list of dataframes for the profiles to plot
        tmp_df_list = []
        for i in cluster_numbers:
            if i in clstrs_to_plot:
                # Add the rows with this cluster id to the list
                tmp_df_list.append(df[df['cluster'] == i])
        if len(tmp_df_list) == 0:
            print('No points in clusters',clstrs_to_plot,'found')
            # Make blank dataframe with the same columns as df
            tmp_df = pd.DataFrame(index=df.index, columns=df.columns)
            # Remove all rows with nan values
            tmp_df = tmp_df.dropna()
            # Append this to the list
            tmp_df_list.append(tmp_df)
            # exit(0)
        df = pd.concat(tmp_df_list)
    return df

################################################################################

def plot_clstr_param_sweep(ax, tw_ax_x, a_group, plt_title=None):
    """
    Plots the DBCV and number of clusters found by HDBSCAN vs. either n_pfs, 
    m_pts, m_cls, or ell_size

    ax              The axis on which to plot
    tw_ax_x         The twin x axis on which to plot
    a_group         An Analysis_Group object containing the info to create this subplot
    plt_title       A string to use as the title of this subplot
    """
    # print('in plot_clstr_param_sweep()')
    ## Get relevant parameters for the plot
    pp = a_group.plt_params
    # Concatonate all the pandas data frames together
    df = pd.concat(a_group.data_frames)
    # Get the dictionary stored in extra_args
    cluster_plt_dict = pp.extra_args
    # Set the main x and y data keys
    x_key = pp.x_vars[0]
    y_key = pp.y_vars[0]
    # Check for twin axis data key
    try:
        tw_y_key = pp.y_vars[1]
    except:
        tw_y_key = None
    # Get cluster arguments
    m_pts, m_cls, cl_x_var, cl_y_var, cl_z_var, plot_slopes, b_a_w_plt = get_cluster_args(pp)
    # Get the parameter sweep tuple, expect the form [start,stop,step]
    try:
        cl_ps_tuple = cluster_plt_dict['cl_ps_tuple']
    except:
        print('Cannot find `cl_ps_tuple` in `extra_args`, aborting script')
        exit(0)
    # Build the array for the x_var axis
    x_var_array = np.arange(cl_ps_tuple[0], cl_ps_tuple[1], cl_ps_tuple[2])
    # Check for a z variable
    try:
        z_key = cluster_plt_dict['z_var']
        z_list = cluster_plt_dict['z_list']
    except:
        z_key = None
        z_list = [0]
    # Open a text file to record values from the parameter sweep
    sweep_txt_file = 'outputs/'+plt_title+'_ps.csv'
    f = open(sweep_txt_file,'w')
    f.write('Parameter Sweep for '+plt_title+'\n')
    f.write(datetime.now().strftime("%I:%M%p on %B %d, %Y")+'\n')
    f.write('m_pts,ell_size,n_clusters,DBCV,m_cls\n')
    f.close()
    # If limiting the number of pfs, find total number of pfs in the given df
    #   In the multi-index of df, level 0 is 'Time'
    if x_key == 'n_pfs':
        pf_nos = np.unique(np.array(df['prof_no'].values, dtype=type('')))
        number_of_pfs = len(pf_nos)
        print('\tNumber of profiles:',number_of_pfs)
        x_var_array = x_var_array[x_var_array <= number_of_pfs]
    if z_key == 'n_pfs':
        pf_nos = np.unique(np.array(df['prof_no'].values, dtype=type('')))
        number_of_pfs = len(pf_nos)
        print('\tNumber of profiles:',number_of_pfs)
        z_list = np.array(z_list)
        z_list = z_list[z_list <= number_of_pfs]
    #
    print('\tPlotting these x values of',x_key,':',x_var_array)
    if z_key:
        print('\tPlotting these z values of',z_key,':',z_list)
    z_len = len(z_list)
    x_len = len(x_var_array)
    for i in range(z_len):
        y_var_array = []
        tw_y_var_array = []
        lines = []
        # Divide the runs among the processes
        # x_var_array = comm.scatter(x_var_array, root=0)
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
            if z_key == 'ell_size':
                # Need to apply moving average window to original data, before
                #   the data filters were applied, so make a new Analysis_Group
                a_group.profile_filters.m_avg_win = z_list[i]
                new_a_group = Analysis_Group(a_group.data_set, a_group.profile_filters, a_group.plt_params)
                this_df = pd.concat(new_a_group.data_frames)
                zlabel = r'$\ell=$'+str(z_list[i])+' dbar'
            if x_key == 'n_pfs':
                this_df = this_df[this_df['prof_no'] <= pf_nos[x-1]].copy()
                xlabel = 'Number of profiles included'
            if z_key == 'n_pfs':
                this_df = this_df[this_df['prof_no'] <= pf_nos[z_list[i]-1]].copy()
                zlabel = r'$n_{profiles}=$'+str(z_list[i])
            if x_key == 'm_pts':
                # min_samples must be an integer
                m_pts = int(x)
                xlabel = r'$m_{pts}$'
            elif x_key == 'm_cls':
                # min_cluster_size must be an integer, or None
                if not isinstance(x, type(None)):
                    m_cls = int(x)
                else:
                    m_cls = x
                xlabel = r'$m_{cls}$'
            if z_key == 'm_pts':
                # min_samples must be an integer
                m_pts = int(z_list[i])
                zlabel = r'$m_{pts}=$: '+str(m_pts)
            elif z_key == 'm_cls':
                # min_cluster_size must be an integer, or None
                if not isinstance(z_list[i], type(None)):
                    m_cls = int(z_list[i])
                else:
                    m_cls = z_list[i]
                zlabel = r'$m_{cls}=$'+str(m_cls)
            # Run the HDBSCAN algorithm on the provided dataframe
            try:
                new_df, rel_val, m_pts_out, m_cls_out, ell = HDBSCAN_(a_group.data_set.arr_of_ds, this_df, cl_x_var, cl_y_var, cl_z_var, m_pts, m_cls=m_cls, param_sweep=True)
            except:
                break
            # Record outputs to plot
            if y_key == 'DBCV':
                # relative_validity_ is a rough measure of DBCV
                y_var_array.append(rel_val)
                ylabel = 'DBCV'
            elif y_key == 'n_clusters':
                # Clusters are labeled starting from 0, so total number of clusters is
                #   the largest label plus 1
                y_var_array.append(int(np.nanmax(new_df['cluster']+1)))
                ylabel = 'Number of clusters'
            if tw_y_key:
                if tw_y_key == 'DBCV':
                    # relative_validity_ is a rough measure of DBCV
                    tw_y_var_array.append(rel_val)
                    tw_ylabel = 'DBCV'
                elif tw_y_key == 'n_clusters':
                    # Clusters are labeled starting from 0, so total number of clusters is
                    #   the largest label plus 1
                    tw_y_var_array.append(int(np.nanmax(new_df['cluster']+1)))
                    tw_ylabel = 'Number of clusters'
                #
            #
            lines.append(str(m_pts_out)+','+str(ell)+','+str(int(np.nanmax(new_df['cluster']+1)))+','+str(rel_val)+','+str(m_cls_out)+'\n')
        # Plot the data
        if True:
            ax.plot(x_var_array, y_var_array, color=std_clr, linestyle=l_styles[i], label=zlabel)
            # Add gridlines
            ax.grid(color=std_clr, linestyle='--', alpha=grid_alpha)
            if tw_y_key:
                if z_key:
                    tw_ax_x.plot(x_var_array, tw_y_var_array, color=alt_std_clr, linestyle=l_styles[i])
                else: # If not plotting multiple z values, make the twin axis line a different linestyle
                    tw_ax_x.plot(x_var_array, tw_y_var_array, color=alt_std_clr, linestyle=l_styles[i+1])
                tw_ax_x.set_ylabel(tw_ylabel)
                # Change color of the axis label on the twin axis
                tw_ax_x.yaxis.label.set_color(alt_std_clr)
                # Change color of the ticks on the twin axis
                tw_ax_x.tick_params(axis='y', colors=alt_std_clr)
                # Add gridlines
                # tw_ax_x.grid(color=alt_std_clr, linestyle='--', alpha=grid_alpha+0.3, axis='y')
        # Gather the data from all the processes
        f = open(sweep_txt_file,'a')
        f.write(''.join(lines))
        f.close()
    if z_key:
        ax.legend()
    return xlabel, ylabel

################################################################################
################################################################################
################################################################################
