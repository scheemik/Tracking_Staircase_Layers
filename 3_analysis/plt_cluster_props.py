"""
Author: Mikhail Schee
Created: 2024-05-03

This script will calculate cluster properties for the BGR data set and save the 
pandas dataframe to a pickle file

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
# For storing figure objects in files (can use `pickle` instead if need be)
import dill as pl
# For formatting data into dataframes
import pandas as pd
# For adding subplot labels a, b, c, ...
import string
# Import the Thermodynamic Equation of Seawater 2010 (TEOS-10) from GSW
# For finding density and heat capacity
import gsw

# For importing custom functions from files, need to add `.` and `..` to path
import sys
sys.path.append("..")
sys.path.append(".")
import importlib
# For custom analysis functions
ahf = importlib.import_module('.0_helper_files.analysis_helper_functions', package='Tracking_Staircase_Layers')
# For common BGR parameters
bps = importlib.import_module('.0_helper_files.BGR_params', package='Tracking_Staircase_Layers')
# For common BGR objects
bob = importlib.import_module('.0_helper_files.BGR_objects', package='Tracking_Staircase_Layers')

# Specify which BGR data set to use
# this_BGR = 'BGR1516'
this_BGR = 'BGR_all'

# Specify which connection method to use
connect_by = 'SA_divs'
# connect_by = 'manual'

filename = this_BGR+'_'+connect_by

################################################################################
# Declare variables for plotting
################################################################################
dark_mode = False

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

# Beaufort Gyre Region (BGR), see Shibley2022
lon_BGR = bps.lon_BGR
lat_BGR = bps.lat_BGR

################################################################################

# Unpickle the data frames from file
# if connect_by == 'SA_divs':
    # df = pl.load(open('outputs/'+this_BGR+'_SA_divs_cluster_properties_og.pickle', 'rb')) # before 2024-11-13
    # df = pl.load(open('outputs/'+this_BGR+'_SA_divs_cluster_properties.pickle', 'rb')) # before 2025-02-11
df = pl.load(open('outputs/'+filename+'_layer_properties.pickle', 'rb'))
# elif connect_by == 'manual':    
#     df = pl.load(open('outputs/'+this_BGR+'_layer_properties.pickle', 'rb'))
bnds_df = pl.load(open('outputs/'+this_BGR+'_LHW_AW_properties.pickle', 'rb'))

# print('df.columns:')
# print(list(df.columns))
# print(df.press_fit_eq)
# exit(0)

# # Sort df by ca_SA
# # df = df.sort_values(by='ca_SA', ascending=True)
# # Print just the coefficient of X from CT_fit_eq
# #   First, cut the equation at the first '*X '
# df['CT_fit_eq_X'] = df['CT_fit_eq'].str.split('X ').str[0]
# #   Then, cut the equation at the first ' '
# df['CT_fit_eq_X'] = df['CT_fit_eq_X'].str.split(' ').str[4]
# print(df['CT_fit_eq_X'])
# exit(0)

# Print the CT_fit_eq for cluster 69
## Get the row for cluster 69 from df
# clust_69 = df.loc[df['cluster'] == 69]
# print(clust_69['CT_fit_eq'])
# exit(0)

################################################################################
# Summary function #############################################################
################################################################################

# print('bnds_df.columns:')
# print(bnds_df.columns)

def print_LHW_AW_vals(bnds_df):
    """
    Prints out values of the LHW and AW cores in a readable format
    """
    # For the LHW and AW
    for bound_str in ['TC_min', 'TC_max']:
        print('- For '+bound_str+':')
        # For each of the three main variables
        for var_str in ['SA', 'CT', 'press']:
            # Find the average values
            print('\t- Average '+var_str+': '+str(bnds_df['av_'+var_str+'_'+bound_str].values[0]))
            # Find the trend in time
            print('\t- Fit-corrected trend in '+var_str+': '+str(bnds_df['trd_'+var_str+'_'+bound_str+'-fit'].values[0]) + ' +/- ' + str(bnds_df['trdsd_'+var_str+'_'+bound_str+'-fit'].values[0]) + ' with R^2 of ' + str(bnds_df['trdR2_'+var_str+'_'+bound_str+'-fit'].values[0]))
# print_LHW_AW_vals(bnds_df)

def props_stats_summary(df, props_list, mrk_outliers):
    """
    A function that will calculate and output a summary of statistics of the 
    given layer properties for the given dataset.

    df          The dataset with which to calculate properties
    props_list  The list of variables of which to summarize
    """
    # print('df.columns:')
    # print(list(df.columns))
    # Find and mark outliers 
    df = ahf.find_outliers(df, ['cRL', 'nir_SA'], mrk_outliers)
    # Sort the dataframe by cluster average salinity
    df.sort_values(by=['ca_SA'], inplace=True)
    # Remove rows of outlier layers
    df.drop(df[df.out_cRL==True].index, inplace=True)
    df.drop(df[df.out_nir_SA==True].index, inplace=True)
    if mrk_outliers == 'ends':
        df.drop(df[df.out_ends==True].index, inplace=True)
    # Verify that the correct cluster numbers are retained
    # print('df[cluster]:')
    # this_list = list(df['cluster'].values)
    # print('length:',len(this_list))
    # print(this_list)
    # Loop over each property in the list and find it's statistics
    for prop_var in props_list:
        print('For '+prop_var+':')
        if 'delta' in prop_var:
            # Get var name
            this_var = prop_var[6:]
            # Calculate difference between neighboring clusters
            df[prop_var] = df['ca_'+this_var].diff()
        this_array = np.array(df[prop_var].values, dtype=np.float64)
        this_median = np.nanmedian(this_array)
        print('\tMedian: ' + str(this_median))
        this_mean = np.nanmean(this_array)
        print('\tMean: ' + str(this_mean))
        this_stdev = np.nanstd(this_array)
        print('\tStdev: ' + str(this_stdev))

# props_stats_summary(df, ['cRL', 'delta_press', 'cs_press', 'csd_press', 'delta_SA', 'cs_SA', 'csd_SA', 'delta_CT', 'cs_CT', 'csd_CT', 'nzca_pcs_press', 'nzcsd_pcs_press', 'nir_SA', 'nir_CT', 'nir_press'], 'ends')

def format_sci_notation2(x, ndp=1):
    """
    Formats a number into scientific notation

    x       The number to format
    """
    sci_lims_f = (-3,3) # ahf.sci_lims
    pm_val = None
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
            return r'${x:0.{ndp:d}f}$'.format(x=x, ndp=ndp)
        # Check to see whether it's outside the scientific notation exponent limits
        if int(e) < min(sci_lims_f) or int(e) > max(sci_lims_f):
            return r'${m:s}{{\times}}10^{{{e:d}}}$'.format(m=m, e=int(e))
        else:
            return r'${x:0.{ndp:d}f}$'.format(x=x, ndp=ndp)
    else:
        try:
            # Find magnitude and base 10 exponent for x
            s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
            m, e = s.split('e')
        except:
            print('Warning: could not format',x,'with',ndp,'decimal places and pm_val:',pm_val)
            return r'${x:0.{ndp:d}f}{{\pm}}{pm_val:0.{ndp:d}f}$'.format(x=x, ndp=ndp, pm_val=pm_val)
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
            return r'$({m:s}{{\pm}}{pm_m:s}){{\times}} 10^{{{e:d}}}$'.format(m=m, pm_m=pm_m, e=int(e))
        else:
            return r'${x:0.{ndp:d}f}{{\pm}}{pm_val:0.{ndp:d}f}$'.format(x=x, ndp=new_ndp, pm_val=pm_val)

def layer_stats(df, var_list):
    """
    A function that will calculate and output a csv table of the given variables
    for each layer in the given dataframe

    df          The dataset with which to calculate properties
    var_list    The list of variables of which to summarize
    """
    # print('df.columns:')
    # print(list(df.columns))
    lines = []
    # Make sure 'cluster' is the first column
    var_list = ['cluster'] + var_list
    # Remove duplicates, preserving order
    var_list = list(dict.fromkeys(var_list))
    # Get just those variable's columns from the dataframe
    if 'SA_min' in var_list:
        import copy
        this_var_list = copy.deepcopy(var_list)
        this_var_list.remove('SA_min')
        this_df = df[this_var_list]
    else:
        this_df = df[var_list]
    # Sort this dataframe by `cluster`
    this_df = this_df.sort_values(by=['cluster'])
    # Make a list of columns to remove
    remove_these_cols = []
    # Format all the values into scientific notation
    for var in var_list:# list(this_df.columns):
        # sci_lims_these = ahf.sci_lims
        sci_lims_these = (-1,3)
        # Add a column for the salinity divisions
        if var == 'SA_min':
            # Get the SA divisions from file
            SA_divs = bps.BGR_HPC_SA_divs
            # Add in the minimum salinity boundary to match the length (49 up to 50)
            SA_mins = [min(bps.S_range_LHW_AW)] + SA_divs
            this_df[var] = np.vectorize(format_sci_notation2)(SA_mins, ndp=3)
            var_csd = False
        # If the column is independent
        if var in ['cRL','nir_SA'] or 'R2' in var:
            this_df[var] = this_df[var].map(format_sci_notation2)
            var_csd = False
        # If the column is a cluster average
        elif 'nzca_' in var:
            # Find the corresponding string for the cluster standard deviation
            var_csd = 'nzcsd_'+var[5:]
        elif 'ca_' in var:
            # Find the corresponding string for the cluster standard deviation
            var_csd = 'csd_'+var[3:]
            sci_lims_these = (-3,3)
        # If the column is a cluster trend
        elif 'nztrd_' in var:
            # Find the corresponding string for the cluster standard error
            var_csd = 'nztrdsd_'+var[6:]
            # sci_lims_these = (-2,3)
        elif 'trd_' in var:
            # Find the corresponding string for the cluster standard error
            var_csd = 'trdsd_'+var[4:]
            # sci_lims_these = (-2,3)
        else:
            var_csd = False
        if var_csd:
            # print(var,'and',var_csd)
            # Enter the formatted values row by row
            for i in range(len(this_df)):
                var_val = this_df.iloc[i, this_df.columns.get_loc(var)]
                var_csd_val = this_df.iloc[i, this_df.columns.get_loc(var_csd)]
                # print(var_val,'and',var_csd_val)
                val_formatted = ahf.format_sci_notation(var_val, ndp=1, sci_lims_f=sci_lims_these, pm_val=var_csd_val, condense=True)
                this_df.iloc[i, this_df.columns.get_loc(var)] = r'$'+val_formatted+'$'
            remove_these_cols.append(var_csd)
    # Remove unneeded columns
    this_df.drop(columns=remove_these_cols, inplace=True)
    # Format cluster ids as integers
    this_df['cluster'] = pd.to_numeric(this_df['cluster'], downcast='integer')
    # Write this out to a csv
    print('Writing to outputs/'+filename+'_layer_properties.csv')
    this_df.to_csv('outputs/'+filename+'_layer_properties.csv', index=False)

# layer_stats(df, ['cRL', 'nir_SA', 'SA_min', 'ca_SA', 'csd_SA', 'ca_CT', 'csd_CT', 'ca_press', 'csd_press', 'nzca_pcs_press', 'nzcsd_pcs_press', 'trd_CT', 'trdsd_CT', 'trdR2_CT', 'trd_CT-fit', 'trdsd_CT-fit', 'trdR2_CT-fit', 'trd_press', 'trdsd_press', 'trdR2_press', 'trd_press-fit', 'trdsd_press-fit', 'trdR2_press-fit', 'nztrd_pcs_press', 'nztrdsd_pcs_press', 'nztrdR2_pcs_press'])
# 
# exit(0)
################################################################################
# Plotting functions ###########################################################
################################################################################

def mark_the_LHW_and_AW(ax, x_key, y_key):
    """
    Marks the LHW and AW clusters on the plot

    ax          A matplotlib axis object
    x_key       A string of the x key
    y_key       A string of the y key
    """
    # Parse the bounds keys
    x_LHW_key, y_LHW_key, x_AW_key, y_AW_key = ahf.parse_LHW_AW_keys(x_key, y_key)
    # Find the x and y values for the LHW and AW
    x_LHW = bnds_df[x_LHW_key].values[0]
    y_LHW = bnds_df[y_LHW_key].values[0]
    x_AW  = bnds_df[x_AW_key].values[0]
    y_AW  = bnds_df[y_AW_key].values[0]
    # Plot the LHW and AW
    s_to_markersize = np.sqrt(100)
    ax.plot(x_LHW, y_LHW, color=ahf.LHW_clr, markersize=s_to_markersize, marker=ahf.LHW_mrk, fillstyle='bottom', markerfacecoloralt=ahf.LHW_facealtclr, markeredgecolor=ahf.LHW_edgeclr, linewidth=0, zorder=10)
    ax.plot(x_AW, y_AW, color=ahf.AW_clr, markersize=s_to_markersize, marker=ahf.AW_mrk, fillstyle='top', markerfacecoloralt=ahf.AW_facealtclr, markeredgecolor=ahf.AW_edgeclr, linewidth=0, zorder=10)

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
    for group in groups_to_plot:
        pp = group.plt_params
        u_x_vars = np.unique(pp.x_vars)
        u_x_vars = np.delete(u_x_vars, np.where(u_x_vars == None))
        u_y_vars = np.unique(pp.y_vars)
        u_y_vars = np.delete(u_y_vars, np.where(u_y_vars == None))
        if len(u_x_vars) > 0:
            for var in u_x_vars:
                x_keys.append(var)
        if len(u_y_vars) > 0:
            for var in u_y_vars:
                y_keys.append(var)
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
            print('\t- Set share_x_axis to False')
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
            print('\t- Set share_y_axis to False')
        #
    else:
        use_same_y_axis = False
    print('\t- Using same x axis:',use_same_x_axis)
    print('\t- Using same y axis:',use_same_y_axis)
    tight_layout_h_pad = 1.0 #None
    tight_layout_w_pad = 0 #None
    if n_subplots == 1:
        if isinstance(row_col_list, type(None)):
            rows, cols, f_ratio, f_size = n_row_col_dict[str(n_subplots)]
        else:
            rows, cols, f_ratio, f_size = row_col_list
        n_subplots = int(np.floor(n_subplots))
        fig, ax = ahf.set_fig_axes([1], [1], fig_ratio=f_ratio, fig_size=f_size)
        xlabel, ylabel, plt_title, ax, invert_y_axis = make_subplot(ax, groups_to_plot[0], fig, 111)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
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
        if n_subplots == 8:
            fig, axes = ahf.set_fig_axes([1,1], [0.6,1,1,0.6], fig_ratio=f_ratio, fig_size=f_size, share_x_axis=use_same_x_axis, share_y_axis=use_same_y_axis)
        else:
            fig, axes = ahf.set_fig_axes([1]*rows, [1]*cols, fig_ratio=f_ratio, fig_size=f_size, share_x_axis=use_same_x_axis, share_y_axis=use_same_y_axis)
        # Create a blank array for the axis labels and axis limits
        xlabels = [[None for x in range(cols)] for y in range(rows)]
        ylabels = [[None for x in range(cols)] for y in range(rows)]
        x_lims = [[None for x in range(cols)] for y in range(rows)]
        y_lims = [[None for x in range(cols)] for y in range(rows)]
        for i in range(n_subplots):
            subplot_label_x = -0.13
            subplot_label_y = -0.18
            print('- Subplot '+string.ascii_lowercase[i])
            if rows > 1 and cols > 1:
                i_ax = (i//cols,i%cols)
            else:
                i_ax = i
            ax_pos = int(str(rows)+str(cols)+str(i+1))
            xlabel, ylabel, plt_title, ax, invert_y_axis = make_subplot(axes[i_ax], groups_to_plot[i], fig, ax_pos)
            xlabels[i//cols][i%cols] = xlabel
            ylabels[i//cols][i%cols] = ylabel
            # Find the plot type
            plot_type = groups_to_plot[i].plt_params.plot_type
            if use_same_x_axis:
                # If on the top row
                if i < cols:# == 0:
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
                # If in the far left column
                if i%cols == 0:
                    tight_layout_w_pad = 0 #-1
                    ax.set_ylabel(ylabel)
                    # subplot_label_x = -0.05
                    # subplot_label_y = -0.07
                    subplot_label_x = -0.07
                    subplot_label_y = -0.1
                else:
                    # subplot_label_x = -0.03
                    # subplot_label_y = -0.07
                    subplot_label_x = -0.03
                    subplot_label_y = -0.1
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
                # Invert y-axis if specified
                if invert_y_axis:
                    ax.invert_yaxis()
                    print('\t- Inverting y-axis-')
            if n_subplots == 8:
                subplot_label_x = -0.20
            # Label subplots a, b, c, ...
            print('\t- Adding subplot label with offsets:',subplot_label_x, subplot_label_y)
            ax.text(subplot_label_x, subplot_label_y, r'\textbf{('+string.ascii_lowercase[i]+')}', transform=ax.transAxes, size=ahf.font_size_labels, fontweight='bold')
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
    print('- Adjusting subplot spacing with h_pad:',tight_layout_h_pad,'w_pad:',tight_layout_w_pad)
    plt.tight_layout(pad=0.4, h_pad=tight_layout_h_pad, w_pad=tight_layout_w_pad)
    #
    if filename != None:
        print('- Saving figure to outputs/'+filename)
        if '.png' in filename or '.pdf' in filename:
            plt.savefig('outputs/'+filename, dpi=600)
        elif '.pickle' in filename:
            pl.dump(fig, open('outputs/'+filename, 'wb'))
        else:
            print('File extension not recognized in',filename)
    else:
        print('- Displaying figure')
        plt.show()

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
    first_dfs = pp.first_dfs
    # Check the plot type
    if plot_type == 'map':
        x_label, y_label, z_label, plt_title, ax, invert_y_axis = ahf.make_subplot(ax, a_group, fig, ax_pos)
        return x_label, y_label, plt_title, ax, invert_y_axis
    # Check the extra arguments
    add_isos = False
    errorbars = False
    fit_vars = False
    log_axes = 'None'
    mpi_run = False
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
            re_run_clstr = False
        if 'mark_LHW_AW' in extra_args.keys():
            mark_LHW_AW = extra_args['mark_LHW_AW']
        else:
            mark_LHW_AW = False
        if 'mark_zero' in extra_args.keys():
            mark_zero = extra_args['mark_zero']
        else:
            mark_zero = True
    else:
        extra_args = False
        plot_slopes = False
        mv_avg = False
        mark_LHW_AW = False
    # Add a title
    plt_title = a_group.plot_title
    ## Make a standard x vs. y scatter plot
    # Set the main x and y data keys
    x_key = pp.x_vars[0]
    y_key = pp.y_vars[0]
    z_key = pp.z_vars[0]
    # print('test print of plot vars 1:',x_key, y_key, z_key, clr_map)
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
        tw_ax_y.ticklabel_format(style='sci', scilimits=ahf.sci_lims, useMathText=True)
    except:
        tw_x_key = None
        tw_ax_y  = None
    try:
        tw_y_key = pp.y_vars[1]
        tw_ax_x  = ax.twinx()
        # Set ticklabel format for twin axis
        tw_ax_x.ticklabel_format(style='sci', scilimits=ahf.sci_lims, useMathText=True)
    except:
        tw_y_key = None
        tw_ax_x  = None
    # print('tw_x_key:',tw_x_key,'tw_y_key:',tw_y_key)
    # Concatonate all the pandas data frames together
    df = pd.concat(a_group.data_frames)
    # Format the dates if necessary
    if x_key in ['dt_start', 'dt_end']:
        df[x_key] = mpl.dates.date2num(df[x_key])
    if y_key in ['dt_start', 'dt_end']:
        df[y_key] = mpl.dates.date2num(df[y_key])
    # Check for variables to be calculated
    if 'percnztrd_pcs_press' in pp.x_vars or 'percnztrd_pcs_press' in pp.y_vars:
        # Calculate the percent trend in non-zero thickness
        df['percnztrd_pcs_press'] = df['nztrd_pcs_press']/df['nzca_pcs_press']*100
        df['percnztrdsd_pcs_press'] = df['nztrdsd_pcs_press']/df['nzca_pcs_press']*100
    if 'ca_rho' in pp.x_vars or 'ca_rho' in pp.y_vars:
        # Calculate the cluster average density
        df['ca_rho'] = gsw.rho(df['ca_SA'].values, df['ca_CT'].values, df['ca_press'].values)
    if 'ca_cp' in pp.x_vars or 'ca_cp' in pp.y_vars:
        # First, calculate the exact temperature
        df['ca_iT'] = gsw.t_from_CT(df['ca_SA'].values, df['ca_CT'].values, df['ca_press'].values)
        # Calculate the cluster average heat capacity
        df['ca_cp'] = gsw.cp_t_exact(df['ca_SA'].values, df['ca_iT'].values, df['ca_press'].values)
    if 'ca_FH' in pp.x_vars or 'ca_FH' in pp.y_vars:
        ## Calculating the heat flux, need to calculate the components first
        # First, calculate the exact temperature
        df['ca_iT'] = gsw.t_from_CT(df['ca_SA'].values, df['ca_CT'].values, df['ca_press'].values)
        # Calculate the cluster average heat capacity
        df['ca_cp'] = gsw.cp_t_exact(df['ca_SA'].values, df['ca_iT'].values, df['ca_press'].values)
        # Calculate the cluster average density
        df['ca_rho'] = gsw.rho(df['ca_SA'].values, df['ca_CT'].values, df['ca_press'].values)
        # Calculate the conversion factor to change year to seconds
        yr_to_s = 365.25*24*60*60
        # Calculate the heat flux
        df['ca_FH'] = df['nzca_pcs_press']*df['ca_cp']*df['ca_rho']*df['trd_CT-fit']/yr_to_s
    if 'ca_FH_cumul' in pp.x_vars:
        ## Calculating the cumulative heat flux, need to calculate the components first
        # First, calculate the exact temperature
        df['ca_iT'] = gsw.t_from_CT(df['ca_SA'].values, df['ca_CT'].values, df['ca_press'].values)
        # Calculate the cluster average heat capacity
        df['ca_cp'] = gsw.cp_t_exact(df['ca_SA'].values, df['ca_iT'].values, df['ca_press'].values)
        # Calculate the cluster average density
        df['ca_rho'] = gsw.rho(df['ca_SA'].values, df['ca_CT'].values, df['ca_press'].values)
        # Calculate the conversion factor to change year to seconds
        yr_to_s = 365.25*24*60*60
        # Calculate the heat flux
        df['ca_FH'] = df['nzca_pcs_press']*df['ca_cp']*df['ca_rho']*df['trd_CT-fit']/yr_to_s
        # Sort the cluster average heat flux by the y_key variable
        df = df.sort_values(by=y_key)
        # Calculate the cumulative heat flux, starting with the lowest value of y_key
        df['ca_FH_cumul'] = df['ca_FH'].cumsum()
    # Check whether to calculate first differences
    if any(first_dfs):
        print('\t- Calculating first differences')
        if first_dfs[0]:
            # Sort the dataframe by the x_key variable
            df = df.sort_values(by=x_key)
            # Calculate the first differences
            df[x_key] = df[x_key].diff()
        if first_dfs[1]:
            # Sort the dataframe by the y_key variable
            df = df.sort_values(by=y_key)
            # Calculate the first differences
            df[y_key] = df[y_key].diff()
    # Print the variables in the df
    # print('df.columns:')
    # print(list(df.columns))
    # Add a vertical line at zero
    if mark_zero:
        ax.axvline(0, color=std_clr, linestyle='--', alpha=0.7, zorder=1)
    # Determine the color mapping to be used
    if clr_map == 'clr_all_same':
        # Check for histogram
        if plot_hist:
            print('\t- Plotting histogram')
            x_label, y_label, z_label, plt_title, ax, invert_y_axis = ahf.plot_histogram(a_group, ax, pp, df, x_key, y_key, clr_map, legend=pp.legend, txk=tw_x_key, tay=tw_ax_y, tyk=tw_y_key, tax=tw_ax_x)
            return x_label, y_label, plt_title, ax, invert_y_axis
        # Set whether to plot in 3D
        if isinstance(z_key, type(None)):
            df_z_key = 0
            # Drop duplicates
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
        m_size, m_alpha = ahf.get_marker_size_and_alpha(len(df[x_key]))
        m_size = m_size*2
        # # Plot every point the same color, size, and marker
        # if plot_3d == False:
        #     # Plot in 2D
        #     ax.scatter(df[x_key], df[y_key], color=std_clr, s=m_size, marker=ahf.std_marker, alpha=m_alpha, zorder=5)
        #     # Output the x and y data, along with cluster id, to a pickle file
        #     # df[[x_key, y_key, 'cluster']].to_pickle('outputs/'+this_BGR+'_x_y_cluster_id.pickle')
        #     # Take moving average of the data
        #     if x_key in ['dt_start', 'dt_end'] and mv_avg:
        #         print('\t- Taking moving average of the data')
        #         # Make a copy of the dataframe
        #         time_df = df.copy()
        #         # Need to convert dt_start back to datetime to take rolling average
        #         time_df[x_key] = mpl.dates.num2date(time_df[x_key])
        #         # Set dt_start as the index
        #         time_df = time_df.set_index(x_key)
        #         # Sort by the dt_start index
        #         time_df.sort_index(inplace=True)
        #         # Take the moving average of the data
        #         time_df[y_key+'_mv_avg'] = time_df[y_key].rolling(mv_avg, min_periods=1).mean()
        #         # Convert the original dataframe's dt_start back to numbers for plotting
        #         time_df.reset_index(level=x_key, inplace=True)
        #         time_df[x_key] = mpl.dates.date2num(time_df[x_key])
        #         ax.plot(time_df[x_key], time_df[y_key+'_mv_avg'], color='b', zorder=6)
        # else:
        #     # Plot in 3D
        #     ax.scatter(df[x_key], df[y_key], zs=df_z_key, color=std_clr, s=m_size, marker=ahf.std_marker, alpha=m_alpha, zorder=5)
        # Add errorbars, if specified
        if errorbars:
            if isinstance(x_key, type('str')) and '_' in x_key:
                # Split the prefix from the original variable (assumes an underscore split)
                split_var = x_key.split('_', 1)
                prefix = split_var[0]
                var_str = split_var[1]
                if prefix == 'ca' and 'FH' not in var_str:
                    x_err_key = 'csd_'+var_str
                elif prefix == 'nzca':
                    x_err_key = 'nzcsd_'+var_str
                elif prefix == 'trd':
                    x_err_key = 'trdsd_'+var_str
                elif prefix == 'nztrd':
                    x_err_key = 'nztrdsd_'+var_str
                elif prefix == 'percnztrd':
                    x_err_key = 'percnztrdsd_'+var_str
                else:
                    x_err_key = False
            if isinstance(y_key, type('str')) and '_' in y_key:
                # Split the prefix from the original variable (assumes an underscore split)
                split_var = y_key.split('_', 1)
                prefix = split_var[0]
                var_str = split_var[1]
                if prefix == 'ca' and 'FH' not in var_str:
                    y_err_key = 'csd_'+var_str
                elif prefix == 'trd':
                    y_err_key = 'trdsd_'+var_str
                else:
                    y_err_key = False
            print('\t- x_err_key:',x_err_key)
            print('\t- y_err_key:',y_err_key)
            if x_err_key == False and y_err_key == False:
                errorbars = False
            print('\t- Adding errorbars:',errorbars)
        else:
            x_err_key = False
            y_err_key = False
        # Check whether to mark outliers
        print('\t- Marking outliers:',mrk_outliers)
        if mrk_outliers:
            # if 'trd_' in x_key:
            #     other_out_vars = ['cRL', 'nir_SA']
            # else:
            #     other_out_vars = []
            # ahf.mark_outliers(ax, df, x_key, y_key, clr_map, mrk_outliers, mrk_for_other_vars=other_out_vars)
            if x_key != 'cRL' and x_key != 'nir_SA':
                ahf.mark_outliers(ax, df, x_key, y_key, clr_map, mrk_outliers, mrk_for_other_vars=['cRL', 'nir_SA'])
            else:
                ahf.mark_outliers(ax, df, x_key, y_key, clr_map, mrk_outliers)
            # ahf.mark_outliers(ax, df, x_key, y_key, clr_map, mrk_outliers, mrk_for_other_vars=other_out_vars)
            # Get data without outliers
            if True: #plot_slopes:
                x_data = np.array(df[df['out_'+x_key]==False][x_key].values, dtype=np.float64)
                y_data = np.array(df[df['out_'+x_key]==False][y_key].values, dtype=np.float64)
                if x_err_key:
                    xerr_data = np.array(df[df['out_'+x_key]==False][x_err_key].values, dtype=np.float64)
                if y_err_key:
                    yerr_data = np.array(df[df['out_'+x_key]==False][y_err_key].values, dtype=np.float64)
                # Print the median of the non-outlier data in x and y
                print('\t- Non-outlier data in x:')
                print(list(x_data))
                print('\t- Non-outlier data in y:')
                print(list(y_data))
                print('\t- Median of non-outlier data in x:',np.median(x_data))
                print('\t- Mean of non-outlier data in x:',np.mean(x_data))
                print('\t- Std dev of non-outlier data in x:',np.std(x_data))
                print('\t- Median of non-outlier data in y:',np.median(y_data))
                print('\t- Std dev of non-outlier data in y:',np.std(y_data))
                # Recalculate the cumulative heat flux without outliers
                if 'ca_FH_cumul' in x_key:
                    # Remove the outliers from the df
                    df = df[df['out_'+x_key]==False]
                    # Sort the cluster average heat flux by the y_key variable
                    df = df.sort_values(by=y_key)
                    # Calculate the cumulative heat flux, starting with the lowest value of y_key
                    df['ca_FH_cumul'] = df['ca_FH'].cumsum()
        else:
            # Get data
            if True: #plot_slopes:
                x_data = np.array(df[x_key].values, dtype=np.float64)
                y_data = np.array(df[y_key].values, dtype=np.float64)
                if x_err_key:
                    xerr_data = np.array(df[x_err_key].values, dtype=np.float64)
                if y_err_key:
                    yerr_data = np.array(df[y_err_key].values, dtype=np.float64)
        
        # Plot every point the same color, size, and marker
        if plot_3d == False:
            # Plot in 2D
            ax.scatter(x_data, y_data, color=std_clr, s=m_size, marker=ahf.std_marker, alpha=m_alpha, zorder=5)
            # Output the x and y data, along with cluster id, to a pickle file
            # df[[x_key, y_key, 'cluster']].to_pickle('outputs/'+this_BGR+'_x_y_cluster_id.pickle')
            # Take moving average of the data
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
            ax.scatter(x_data, y_data, zs=df_z_key, color=std_clr, s=m_size, marker=ahf.std_marker, alpha=m_alpha, zorder=5)
        # Check whether to mark LHW and AW
        print('\t- Marking LHW and AW:',mark_LHW_AW)
        if mark_LHW_AW:
            mark_the_LHW_and_AW(ax, x_key, y_key)
        if plot_slopes:
            if plot_3d == False:
                if x_key == 'cRL':
                    print('\t- Adding 2nd degrees polynomial fit line:',plot_slopes)
                    # Get bounds of axes
                    x_bnds = ax.get_xbound()
                    y_bnds = ax.get_ybound()
                    x_span = abs(x_bnds[1] - x_bnds[0])
                    y_span = abs(y_bnds[1] - y_bnds[0])
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
                    # Make an array of values to plot log fit line 
                    y_arr = np.linspace(y_bnds[0], y_bnds[1], 50)
                    # Plot the fit
                    ax.plot(x_fit(y_arr), y_arr, color=alt_std_clr)
                    # Add annotation to say what the line is
                    line_label = r"$\mathbf{R_L = " + ahf.format_sci_notation(z[0]) + " p^2 + " + ahf.format_sci_notation(z[1])+ "p " + ahf.format_sci_notation(z[2])+"}$"
                    print('line_label:',line_label)
                    ann_x_loc = x_mean+0.5*x_stdv
                    ann_x_loc = min(x_bnds) + (1/3)*x_span
                    anno_text = ax.annotate(line_label, xy=(ann_x_loc,y_span/3+min(y_bnds)), xycoords='data', color=alt_std_clr, fontweight='bold', bbox=ahf.anno_bbox, zorder=12)
                    anno_text.set_fontsize(ahf.font_size_lgnd)
                    # Limit the y axis
                    ax.set_ylim((y_bnds[0],y_bnds[1]))
                # Plot linear fit line
                else: 
                    ahf.add_linear_slope(ax, pp, df, x_data, y_data, x_key, y_key, alt_std_clr, plot_slopes)
            else:
                # Fit a 2d polynomial to the z data
                ahf.plot_polyfit2d(ax, pp, x_data, y_data, df_z_key)
        # Add a line at nir_var = 1, if plotting nir
        if 'nir_' in x_key:
            # Get bounds of axes
            x_bnds = ax.get_xbound()
            y_bnds = ax.get_ybound()
            x_span = abs(x_bnds[1] - x_bnds[0])
            y_span = abs(y_bnds[1] - y_bnds[0])
            # Add line at nir_var = -1
            ax.axvline(1, color=std_clr, alpha=0.5, linestyle='-')
            if x_key == 'nir_press':
                annotate_str = r'$\mathbf{IR_{p}=1}$'
            elif x_key == 'nir_SA':
                annotate_str = r'$\mathbf{IR_{S_A}=1}$'
            elif x_key == 'nir_CT':
                annotate_str = r'$\mathbf{IR_{\Theta}=1}$'
            else:
                annotate_str = r'$\mathbf{IR=1}$'
            anno_text = ax.annotate(annotate_str, xy=(1+x_span/10,y_bnds[0]+y_span/5), xycoords='data', color=std_clr, fontweight='bold', alpha=0.5, bbox=ahf.anno_bbox, zorder=12)
            anno_text.set_fontsize(ahf.font_size_lgnd)
        # Add the integrated heat flux line if 'ca_FH_cumul' in x_key
        if False:#'ca_FH_cumul' in x_key:
            # Get bounds of y axis
            y_bnds = ax.get_ybound()
            # Make a array of values to plot the line
            y_arr = np.linspace(y_bnds[0], y_bnds[1], 76)
            # Calculate the x values using the best fit line
            x_fit = -0.005060675582875437*y_data + 0.1750440331911853
            # Take the cumulative sum of the x values
            x_cumsum = np.cumsum(x_fit)
            # Plot this cumulative sum
            ax.plot(x_cumsum, y_data, color=alt_std_clr)

            # Calculate the x values using the best fit line
            x_fit = -0.005060675582875437*y_arr + 0.1750440331911853
            # Take the cumulative sum of the x values
            x_cumsum = np.cumsum(x_fit)
            # Plot this cumulative sum
            ax.plot(x_cumsum, y_arr, color='r')

            # Calculate the values of the integrated heat flux
            x_arr = -0.005060675582875437*(y_arr**2)/2 + 0.1750440331911853*y_arr - 0.0810936 -2.9
            # Add curve for integrated heat flux
            ax.plot(x_arr, y_arr, color=alt_std_clr)
            # Add annotation to say what the line is
            # line_label = "Integrated Heat Flux"
            # ann_x_loc = x_bnds[0]+0.5*x_span
            # ax.annotate(line_label, xy=(ann_x_loc,y_bnds[0]+y_span/5), xycoords='data', color=std_clr, fontweight='bold', alpha=0.5, bbox=ahf.anno_bbox, zorder=12)
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
            # Make an array of values to plot log fit line 
            y_arr = np.linspace(y_bnds[0], y_bnds[1], 50)
            # Plot the fit
            ax.plot(x_fit(y_arr), y_arr, color='r')
            # Add annotation to say what the line is
            line_label = "$x = " + ahf.format_sci_notation(z[0]) + " y^2 + " + ahf.format_sci_notation(z[1])+ "y " + ahf.format_sci_notation(z[2])+"$"
            print('\t- Line label:',line_label)
        # Invert y-axis if specified
        if y_key in ahf.y_invert_vars:
            invert_y_axis = True
        else:
            invert_y_axis = False
        # Plot on twin axes, if specified
        if not isinstance(tw_x_key, type(None)):
            tw_clr = ahf.get_var_color(tw_x_key)
            if tw_clr == std_clr:
                tw_clr = 'purple'
            tw_ax_y.scatter(df[tw_x_key], df[y_key], color=tw_clr, s=m_size*0.1, marker='s', alpha=m_alpha)
            tw_ax_y.set_xlabel(pp.xlabels[1])
            # Change color of the ticks on the twin axis
            tw_ax_y.tick_params(axis='x', colors=tw_clr)
            # Change color of the axis label on the twin axis
            tw_ax_y.xaxis.label.set_color(tw_clr)
            # Check whether to adjust the twin axes limits
            if not isinstance(pp.ax_lims, type(None)):
                if 'tw_x_lims' in pp.ax_lims.keys():
                    tw_ax_y.set_xlim(pp.ax_lims['tw_x_lims'])
                    print('\t- Set tw_x_lims to',pp.ax_lims['tw_x_lims'])
        elif not isinstance(tw_y_key, type(None)):
            tw_clr = ahf.get_var_color(tw_y_key)
            if tw_clr == std_clr:
                tw_clr = 'purple'
            tw_ax_x.scatter(df[x_key], df[tw_y_key], color=tw_clr, s=m_size*0.1, marker='s', alpha=m_alpha)
            # Invert y-axis if specified
            if tw_y_key in ahf.y_invert_vars:
                tw_ax_x.invert_yaxis()
            tw_ax_x.set_ylabel(pp.ylabels[1])
            # Change color of the ticks on the twin axis
            tw_ax_x.tick_params(axis='y', colors=tw_clr)
            # Change color of the axis label on the twin axis
            tw_ax_x.yaxis.label.set_color(tw_clr)
            # Check whether to adjust the twin axes limits
            if not isinstance(pp.ax_lims, type(None)):
                if 'tw_y_lims' in pp.ax_lims.keys():
                    tw_ax_x.set_ylim(pp.ax_lims['tw_y_lims'])
                    print('\t- Set tw_y_lims to',pp.ax_lims['tw_y_lims'])
        # Add errorbars, if specified
        if errorbars:
            if x_err_key:
                ax.errorbar(x_data, y_data, xerr=xerr_data, color=std_clr, capsize=ahf.l_cap_size, fmt='none', alpha=m_alpha, zorder=4)
                # print('errbar data:',list(xerr_data))
            if y_err_key:
                ax.errorbar(x_data, y_data, yerr=yerr_data, color=std_clr, capsize=ahf.l_cap_size, fmt='none', alpha=m_alpha, zorder=4)
        # Add a standard legend
        if pp.legend:
            ahf.add_std_legend(ax, df, x_key, mark_LHW_AW=mark_LHW_AW)
        # Format the axes for datetimes, if necessary
        ahf.format_datetime_axes(True, x_key, y_key, ax, tw_x_key, tw_ax_y, tw_y_key, tw_ax_x, aug_zorder=4)
        # Check whether to plot isopycnals
        # if add_isos:
        #     add_isopycnals(ax, df, x_key, y_key, p_ref=isopycnals, place_isos=place_isos, tw_x_key=tw_x_key, tw_ax_y=tw_ax_y, tw_y_key=tw_y_key, tw_ax_x=tw_ax_x)
        # Check whether to change any axes to log scale
        # if len(log_axes) == 3:
        #     if log_axes[0]:
        #         ax.set_xscale('log')
        #     if log_axes[1]:
        #         ax.set_yscale('log')
        #     if log_axes[2]:
        #         ax.set_zscale('log')
        return pp.xlabels[0], pp.ylabels[0], plt_title, ax, invert_y_axis
    elif clr_map == 'source':
        if plot_hist:
            return ahf.plot_histogram(a_group, ax, pp, df, x_key, y_key,clr_map, legend=pp.legend)
        # Get unique sources
        these_sources = np.unique(df['source'])
        # If there aren't that many points, make the markers bigger
        m_size, m_alpha = ahf.get_marker_size_and_alpha(len(df[x_key]))
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
            ax.scatter(this_df[x_key], this_df[y_key], color=my_clr, s=m_size, marker=ahf.std_marker, alpha=m_alpha, zorder=5)
            i += 1
            # Add legend to report the total number of points for this instrmt
            lgnd_label = source+': '+str(len(this_df[x_key]))+' points'
            lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label))
        # Invert y-axis if specified
        if y_key in ahf.y_invert_vars:
            invert_y_axis = True
        else:
            invert_y_axis = False
        notes_string = ''.join(this_df.notes.unique())
        # Only add the notes_string if it contains something
        if len(notes_string) > 1:
            notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
            lgnd_hndls.append(notes_patch)
        # Format the axes for datetimes, if necessary
        ahf.format_datetime_axes(True, x_key, y_key, ax)
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
        # plt_title = ahf.add_std_title(a_group)
        return pp.xlabels[0], pp.ylabels[0], plt_title, ax, invert_y_axis
    elif clr_map == 'clr_by_dataset':
        if plot_hist:
            return ahf.plot_histogram(a_group, ax, pp, df, x_key, y_key, clr_map, legend=pp.legend)
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
            m_size, m_alpha = ahf.get_marker_size_and_alpha(len(this_df[x_key]))
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
            ax.scatter(this_df[x_key], this_df[y_key], color=my_clr, s=m_size, marker=ahf.std_marker, alpha=m_alpha, zorder=5)
            # Add error bars if applicable
            if errorbars:
                if not isinstance(x_err_key, type(None)):
                    ax.errorbar(this_df[x_key], this_df[y_key], xerr=this_df[x_err_key], color=my_clr, capsize=l_cap_size, fmt='none')
                if not isinstance(y_err_key, type(None)):
                    ax.errorbar(this_df[x_key], this_df[y_key], yerr=this_df[y_err_key], color=my_clr, capsize=l_cap_size, fmt='none')
            # Add legend to report the total number of points for this instrmt
            lgnd_label = df_labels[i]+': '+str(len(this_df[x_key]))+' points'
            lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label))
            i += 1
        # Invert y-axis if specified
        if y_key in ahf.y_invert_vars:
            invert_y_axis = True
        else:
            invert_y_axis = False
        notes_string = ''.join(this_df.notes.unique())
        # Only add the notes_string if it contains something
        if len(notes_string) > 1:
            notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
            lgnd_hndls.append(notes_patch)
        # Format the axes for datetimes, if necessary
        ahf.format_datetime_axes(True, x_key, y_key, ax)
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
        # plt_title = ahf.add_std_title(a_group)
        return pp.xlabels[0], pp.ylabels[0], plt_title, ax, invert_y_axis
    elif clr_map == 'instrmt':
        if plot_hist:
            return ahf.plot_histogram(a_group, ax, pp, df, x_key, y_key, clr_map, legend=pp.legend)
        # Add column where the source-instrmt combination ensures uniqueness
        df['source-instrmt'] = df['source']+' '+df['instrmt'].astype("string")
        # Get unique instrmts
        these_instrmts = np.unique(df['source-instrmt'])
        if len(these_instrmts) < 1:
            print('No instruments included, aborting script')
            exit(0)
        # If there aren't that many points, make the markers bigger
        m_size, m_alpha = ahf.get_marker_size_and_alpha(len(df[x_key]))
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
            ax.scatter(this_df[x_key], this_df[y_key], color=my_clr, s=m_size, marker=ahf.std_marker, alpha=m_alpha, zorder=5)
            i += 1
            # Add legend to report the total number of points for this instrmt
            lgnd_label = instrmt+': '+str(len(this_df[x_key]))+' points'
            lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label))
            notes_string = ''.join(this_df.notes.unique())
        # Invert y-axis if specified
        if y_key in ahf.y_invert_vars:
            invert_y_axis = True
        else:
            invert_y_axis = False
        # Only add the notes_string if it contains something
        if len(notes_string) > 1:
            notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
            lgnd_hndls.append(notes_patch)
        # Add legend with custom handles
        if pp.legend:
            lgnd = ax.legend(handles=lgnd_hndls)
        # Format the axes for datetimes, if necessary
        ahf.format_datetime_axes(True, x_key, y_key, ax)
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
        # plt_title = ahf.add_std_title(a_group)
        return pp.xlabels[0], pp.ylabels[0], plt_title, ax, invert_y_axis
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
        
        # Check whether to change the colorbar axis to log scale
        cbar_scale = None
        if len(log_axes) == 3:
            if log_axes[2]:
                cbar_scale = mpl.colors.LogNorm()
                if clr_min < 0:
                    clr_min = None
                # clr_min = None
                # clr_max = 100
        # Make the 2D histogram, the number of bins really changes the outcome
        heatmap = ax.hist2d(df[x_key], df[y_key], bins=xy_bins, cmap=ahf.get_color_map(clr_map), cmin=clr_min, cmax=clr_max, norm=cbar_scale)
        # Invert y-axis if specified
        if y_key in ahf.y_invert_vars:
            invert_y_axis = True
        else:
            invert_y_axis = False
        # `hist2d` returns a tuple, the index 3 of which is the mappable for a colorbar
        cbar = plt.colorbar(heatmap[3], ax=ax, extend=clr_ext)
        cbar.set_label('points per pixel')
        # Add legend to report the total number of points and pixels on the plot
        n_pts_patch  = mpl.patches.Patch(color='none', label=str(len(df[x_key]))+' points')
        pixel_patch  = mpl.patches.Patch(color='none', label=str(xy_bins)+'x'+str(xy_bins)+' pixels')
        notes_string = ''.join(df.notes.unique())
        # Only add the notes_string if it contains something
        if pp.legend:
            if len(notes_string) > 1:
                notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
                ax.legend(handles=[n_pts_patch, pixel_patch, notes_patch])
            else:
                ax.legend(handles=[n_pts_patch, pixel_patch])
        # Format the axes for datetimes, if necessary
        ahf.format_datetime_axes(True, x_key, y_key, ax)
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
        # plt_title = ahf.add_std_title(a_group)
        return pp.xlabels[0], pp.ylabels[0], plt_title, ax, invert_y_axis
    elif clr_map == 'cluster':
        # Add a standard title
        # plt_title = ahf.add_std_title(a_group)
        #   Legend is handled inside plot_clusters()
        # Check for histogram
        if plot_hist:
            return ahf.plot_histogram(a_group, ax, pp, df, x_key, y_key, clr_map, legend=pp.legend, txk=tw_x_key, tay=tw_ax_y, tyk=tw_y_key, tax=tw_ax_x, clstr_dict={'m_pts':m_pts, 'rel_val':rel_val})
        # Format the dates if necessary (don't need this because it's done above)
        if x_key in ['dt_start', 'dt_end']:
            try:
                print(df[x_key])
                df[x_key] = mpl.dates.date2num(df[x_key])
            except:
                foo = 2
        if y_key in ['dt_start', 'dt_end']:
            df[y_key] = mpl.dates.date2num(df[y_key])
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
        # Check whether to mark LHW and AW
        print('\t- Marking LHW and AW:',mark_LHW_AW)
        if mark_LHW_AW:
            mark_the_LHW_and_AW(ax, x_key, y_key)
            pp.mark_LHW_AW = False
        # Make sure to assign m_pts and DBCV to the analysis group to enable writing out to netcdf
        invert_y_axis, a_group.data_frames, foo0, foo1, foo2 = ahf.plot_clusters(a_group, ax, pp, df, x_key, y_key, z_key, None, None, None, clr_map, None, None, None, None, box_and_whisker=False, plot_slopes=plot_slopes)
        # Format the axes for datetimes, if necessary
        ahf.format_datetime_axes(True, x_key, y_key, ax)
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
        return pp.xlabels[0], pp.ylabels[0], plt_title, ax, invert_y_axis
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
        this_cmap = ahf.get_color_map(clr_map)
        #
        if plot_hist:
            return ahf.plot_histogram(a_group, ax, pp, df, x_key, y_key, clr_map=clr_map, legend=pp.legend)
        # If there aren't that many points, make the markers bigger
        m_size, m_alpha = ahf.get_marker_size_and_alpha(len(df[x_key]))
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
            heatmap = ax.scatter(df[x_key], df[y_key], zs=df_z_key, c=cmap_data, cmap=this_cmap, s=m_size, marker=ahf.std_marker, zorder=5)
            if plot_slopes:
                # Fit a 2d polynomial to the z data
                ahf.plot_polyfit2d(ax, pp, df[x_key], df[y_key], df_z_key)
        else:
            if plot_slopes and x_key=='lon' and y_key=='lat':
                # Fit a 2d polynomial to the z data
                ahf.plot_polyfit2d(ax, pp, df[x_key], df[y_key], df[clr_map], in_3D=False)
                # Plot the scatter
                heatmap = ax.scatter(df[x_key], df[y_key], c=cmap_data, cmap=this_cmap, s=m_size, marker=ahf.std_marker, zorder=5, alpha=ahf.map_alpha)
            elif plot_slopes:
                # Plot the scatter
                heatmap = ax.scatter(df[x_key], df[y_key], c=cmap_data, cmap=this_cmap, s=m_size, marker=ahf.std_marker, zorder=5, alpha=ahf.map_alpha)
                # Plot a linear slope
                ahf.add_linear_slope(ax, pp, df, df[x_key], df[y_key], x_key, y_key, alt_std_clr, plot_slopes)
            else:
                # Plot the scatter
                heatmap = ax.scatter(df[x_key], df[y_key], c=cmap_data, cmap=this_cmap, s=m_size, marker=ahf.std_marker, zorder=5)
        # Invert y-axis if specified
        if y_key in ahf.y_invert_vars:
            invert_y_axis = True
        else:
            invert_y_axis = False
        # Plot on twin axes, if specified
        if not isinstance(tw_x_key, type(None)):
            tw_clr = ahf.get_var_color(tw_x_key)
            if tw_clr == std_clr:
                tw_clr = alt_std_clr
            # Add backing x to distinguish from main axis
            tw_ax_y.scatter(df[tw_x_key], df[y_key], color=tw_clr, s=m_size*10, marker='x', zorder=3)
            tw_ax_y.scatter(df[tw_x_key], df[y_key], c=cmap_data, cmap=this_cmap, s=m_size, marker=ahf.std_marker, zorder=4)
            tw_ax_y.set_xlabel(pp.xlabels[1])
            # Change color of the axis label on the twin axis
            tw_ax_y.xaxis.label.set_color(tw_clr)
            # Change color of the ticks on the twin axis
            tw_ax_y.tick_params(axis='x', colors=tw_clr)
            # Create the colorbar
            cbar = plt.colorbar(heatmap, ax=tw_ax_y)
        elif not isinstance(tw_y_key, type(None)):
            tw_clr = ahf.get_var_color(tw_y_key)
            if tw_clr == std_clr:
                tw_clr = alt_std_clr
            # Add backing x to distinguish from main axis
            tw_ax_x.scatter(df[x_key], df[tw_y_key], color=tw_clr, s=m_size*10, marker='x', zorder=3)
            tw_ax_x.scatter(df[x_key], df[tw_y_key], c=cmap_data, cmap=this_cmap, s=m_size, marker=ahf.std_marker, zorder=4)
            # Invert y-axis if specified
            if tw_y_key in ahf.y_invert_vars:
                tw_ax_x.invert_yaxis()
            tw_ax_x.set_ylabel(pp.ylabels[1])
            # Change color of the axis label on the twin axis
            tw_ax_x.yaxis.label.set_color(tw_clr)
            # Change color of the ticks on the twin axis
            tw_ax_x.tick_params(axis='y', colors=tw_clr)
            # Create the colorbar
            cbar = plt.colorbar(heatmap, ax=tw_ax_x)
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
                cbar.mappable.set_clim(pp.ax_lims['c_lims'])
                print('\t- Set c_lims to',pp.ax_lims['c_lims'])
        except:
            foo = 2
        # Invert colorbar if necessary
        if clr_map in ahf.y_invert_vars:
            cbar.ax.invert_yaxis()
        cbar.set_label(pp.clabel)
        # Add a standard legend
        if pp.legend:
            ahf.add_std_legend(ax, df, x_key)
        # Format the axes for datetimes, if necessary
        ahf.format_datetime_axes(True, x_key, y_key, ax, tw_x_key, tw_ax_y, tw_y_key, tw_ax_x)
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
        # plt_title = add_std_title(a_group)
        return pp.xlabels[0], pp.ylabels[0], plt_title, ax, invert_y_axis
    else:
        # Did not provide a valid clr_map
        print('Colormap',clr_map,'not valid')
        exit(0)
    #
    print('ERROR: Should not have gotten here. Aborting script')
    exit(0)

################################################################################

this_clr_map = 'clr_all_same'
this_clr_map = 'cluster'

################################################################################
# LHW and AW plots
################################################################################
pfs_LHW_AW = bob.pfs_LHW_and_AW
press_lims_LHW = [300,150]
press_fit_lims_LHW = [75,-75]
press_lims_AW = [550,350]
press_fit_lims_AW = [100,-100]
SA_lims_LHW = [34.105, 34.095]
SA_lims_AW = [35.03, 34.97]
CT_lims_LHW = [-0.8, -1.4]
CT_lims_AW = [0.97, 0.57]
LHW_AW_legend = False
#*# Maps of LHW and AW cores in pressure
if False:
    print('')
    print('- Creating maps of LHW and AW cores')
    # Make the Plot Parameters
    pp_LHW_map = ahf.Plot_Parameters(plot_type='map', clr_map='press_TC_min', extra_args={'map_extent':'Western_Arctic'}, ax_lims={'c_lims':press_lims_LHW}, legend=LHW_AW_legend)
    pp_AW_map = ahf.Plot_Parameters(plot_type='map', clr_map='press_TC_max', extra_args={'map_extent':'Western_Arctic'}, ax_lims={'c_lims':press_lims_AW}, legend=LHW_AW_legend)
    pp_LHW_CT_map = ahf.Plot_Parameters(plot_type='map', clr_map='CT_TC_min', extra_args={'map_extent':'Western_Arctic'}, ax_lims={'c_lims':CT_lims_LHW}, legend=LHW_AW_legend)
    pp_AW_CT_map = ahf.Plot_Parameters(plot_type='map', clr_map='CT_TC_max', extra_args={'map_extent':'Western_Arctic'}, ax_lims={'c_lims':CT_lims_AW}, legend=LHW_AW_legend)
    # Make the subplot groups
    group_LHW_map = ahf.Analysis_Group2([bnds_df], pp_LHW_map, plot_title=r'LHW core')
    group_AW_map = ahf.Analysis_Group2([bnds_df], pp_AW_map, plot_title=r'AW core')
    # group_LHW_CT_map = ahf.Analysis_Group2([bnds_df], pp_LHW_CT_map, plot_title=r'LHW core')
    # group_AW_CT_map = ahf.Analysis_Group2([bnds_df], pp_AW_CT_map, plot_title=r'AW core')
    # Make the figure
    make_figure([group_LHW_map, group_AW_map], use_same_x_axis=False, use_same_y_axis=False, row_col_list=[1,2, 0.45, 1.6], filename='s_LHW_and_AW_press_maps.png')
    # make_figure([group_LHW_map, group_AW_map, group_LHW_CT_map, group_AW_CT_map], use_same_x_axis=False, use_same_y_axis=False, row_col_list=[2,2, 0.8, 1.6], filename='LHW_and_AW_press_maps.pdf')
#*# Maps of LHW and AW cores in SA and CT
if False:
    print('')
    print('- Creating maps of LHW and AW cores in SA and CT')
    # Make the Plot Parameters
    pp_LHW_SA_map = ahf.Plot_Parameters(plot_type='map', clr_map='SA_TC_min', extra_args={'map_extent':'Western_Arctic'}, ax_lims={'c_lims':SA_lims_LHW}, legend=LHW_AW_legend)
    pp_LHW_CT_map = ahf.Plot_Parameters(plot_type='map', clr_map='CT_TC_min', extra_args={'map_extent':'Western_Arctic'}, ax_lims={'c_lims':CT_lims_LHW}, legend=LHW_AW_legend)
    pp_AW_SA_map = ahf.Plot_Parameters(plot_type='map', clr_map='SA_TC_max', extra_args={'map_extent':'Western_Arctic'}, ax_lims={'c_lims':SA_lims_AW}, legend=LHW_AW_legend)
    pp_AW_CT_map = ahf.Plot_Parameters(plot_type='map', clr_map='CT_TC_max', extra_args={'map_extent':'Western_Arctic'}, ax_lims={'c_lims':CT_lims_AW}, legend=LHW_AW_legend)
    # Make the subplot groups
    group_LHW_SA_map = ahf.Analysis_Group2([bnds_df], pp_LHW_SA_map, plot_title=r'LHW core, $S_A$')
    group_LHW_CT_map = ahf.Analysis_Group2([bnds_df], pp_LHW_CT_map, plot_title=r'LHW core, $\Theta$')
    group_AW_SA_map = ahf.Analysis_Group2([bnds_df], pp_AW_SA_map, plot_title=r'AW core, $S_A$')
    group_AW_CT_map = ahf.Analysis_Group2([bnds_df], pp_AW_CT_map, plot_title=r'AW core, $\Theta$')
    # Make the figure
    make_figure([group_LHW_SA_map, group_LHW_CT_map, group_AW_SA_map, group_AW_CT_map], use_same_x_axis=False, use_same_y_axis=False, row_col_list=[2,2, 0.8, 1.6], filename='s_LHW_AW_maps_SA_CT.png')
#*# Lat-Lon plots of LHW and AW cores in SA and CT
if False:
    print('')
    print('- Creating lat-lon plots of LHW and AW cores in SA and CT')
    # Make the Plot Parameters
    pp_LHW_SA_map = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='CT_TC_min', legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':SA_lims_LHW})
    pp_LHW_CT_map = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='CT_TC_min', legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':CT_lims_LHW})
    pp_AW_SA_map = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='SA_TC_max', legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':SA_lims_AW})
    pp_AW_CT_map = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='CT_TC_max', legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':CT_lims_AW})
    # Make the subplot groups
    group_LHW_SA_map = ahf.Analysis_Group2([bnds_df], pp_LHW_SA_map, plot_title=r'LHW core, $S_A$')
    group_LHW_CT_map = ahf.Analysis_Group2([bnds_df], pp_LHW_CT_map, plot_title=r'LHW core, $\Theta$')
    group_AW_SA_map = ahf.Analysis_Group2([bnds_df], pp_AW_SA_map, plot_title=r'AW core, $S_A$')
    group_AW_CT_map = ahf.Analysis_Group2([bnds_df], pp_AW_CT_map, plot_title=r'AW core, $\Theta$')
    # Make the figure
    make_figure([group_LHW_SA_map, group_LHW_CT_map, group_AW_SA_map, group_AW_CT_map], use_same_x_axis=False, use_same_y_axis=False, row_col_list=[2,2, 0.8, 1.6], filename='s_LHW_AW_lat-lon_plots_SA_CT.png')
#*# Plot press_TC_max and press_TC_min histograms, maps with polyfits, residuals, and histograms of fits
if False:
    print('')
    print('- Creating a histogram and map of LHW and AW cores')
    # Make the Plot Parameters
    # Top of the thermocline, LHW core
    pp_LHW_hist = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['press_TC_min'], extra_args={'plot_slopes':False}, ax_lims={'y_lims':press_lims_LHW}, legend=LHW_AW_legend)
    pp_LHW_fit_map = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='press_TC_min', extra_args={'plot_slopes':True, 'extra_vars_to_keep':[]}, ax_lims={'x_lims':bps.lon_BGR, 'y_lims':bps.lat_BGR, 'c_lims':press_lims_LHW}, legend=LHW_AW_legend)
    pp_LHW_res_map = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='press_TC_min-fit', extra_args={'plot_slopes':False, 'extra_vars_to_keep':['press_TC_min'], 'fit_vars':['lon','lat']}, ax_lims={'x_lims':bps.lon_BGR, 'y_lims':bps.lat_BGR, 'c_lims':press_fit_lims_LHW}, legend=LHW_AW_legend)
    pp_LHW_res_hist = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['press_TC_min-fit'], extra_args={'plot_slopes':False, 'extra_vars_to_keep':['press_TC_min'], 'fit_vars':['lon','lat']}, ax_lims={'y_lims':press_fit_lims_LHW}, legend=LHW_AW_legend)
    ## Bottom of the thermocline, AW core
    pp_AW_hist = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['press_TC_max'], extra_args={'plot_slopes':False}, ax_lims={'y_lims':press_lims_AW}, legend=LHW_AW_legend)
    pp_AW_fit_map = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='press_TC_max', extra_args={'plot_slopes':True, 'extra_vars_to_keep':[]}, ax_lims={'x_lims':bps.lon_BGR, 'y_lims':bps.lat_BGR, 'c_lims':press_lims_AW}, legend=LHW_AW_legend)
    pp_AW_res_map = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='press_TC_max-fit', extra_args={'plot_slopes':False, 'extra_vars_to_keep':['press_TC_max'], 'fit_vars':['lon','lat']}, ax_lims={'x_lims':bps.lon_BGR, 'y_lims':bps.lat_BGR, 'c_lims':press_fit_lims_AW}, legend=LHW_AW_legend)
    pp_AW_res_hist = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['press_TC_max-fit'], extra_args={'plot_slopes':False, 'extra_vars_to_keep':['press_TC_max'], 'fit_vars':['lon','lat']}, ax_lims={'y_lims':press_fit_lims_AW}, legend=LHW_AW_legend)
    # Make the subplot groups
    group_LHW_hist = ahf.Analysis_Group2([bnds_df], pp_LHW_hist, plot_title=r'LHW core')#, $p(S_A\approx34.1)$')
    group_LHW_fit_map = ahf.Analysis_Group2([bnds_df], pp_LHW_fit_map, plot_title=r'$p(S_A\approx34.1)$ and polyfit2d')
    group_LHW_res_map = ahf.Analysis_Group2([bnds_df], pp_LHW_res_map, plot_title=r'Residuals')
    group_LHW_res_hist = ahf.Analysis_Group2([bnds_df], pp_LHW_res_hist, plot_title=r'Residuals')
    group_AW_hist = ahf.Analysis_Group2([bnds_df], pp_AW_hist, plot_title=r'AW core')#, $p(\Theta_{max})$')
    group_AW_fit_map = ahf.Analysis_Group2([bnds_df], pp_AW_fit_map, plot_title=r'$p(\Theta_{max})$ and  polyfit2d')
    group_AW_res_map = ahf.Analysis_Group2([bnds_df], pp_AW_res_map, plot_title=r'Residuals')
    group_AW_res_hist = ahf.Analysis_Group2([bnds_df], pp_AW_res_hist, plot_title=r'Residuals')
    # Make the figure
    make_figure([group_LHW_hist, group_LHW_fit_map, group_LHW_res_map, group_LHW_res_hist, group_AW_hist, group_AW_fit_map, group_AW_res_map, group_AW_res_hist], use_same_x_axis=False, use_same_y_axis=False, row_col_list=[2,4, 0.5, 1.6], filename='f3_LHW_and_AW_hists_and_polyfits.pdf')
#*# Tracking LHW and AW cores over time
if False:
    print('')
    print('- Creating a plot of LHW and AW cores over time')
    # Make the Plot Parameters
    across_x_var = 'dt_start'
    pp_LHW = ahf.Plot_Parameters(x_vars=[across_x_var], y_vars=['press_TC_min'], legend=LHW_AW_legend, extra_args={'plot_slopes':'OLS', 'mv_avg':'30D'}, ax_lims={'x_lims':bps.date_range_dict[this_BGR], 'y_lims':press_lims_LHW})
    pp_LHW_fit = ahf.Plot_Parameters(x_vars=[across_x_var], y_vars=['press_TC_min-fit'], legend=LHW_AW_legend, extra_args={'plot_slopes':'OLS', 'mv_avg':'30D', 'fit_vars':['lon','lat']}, ax_lims={'x_lims':bps.date_range_dict[this_BGR], 'y_lims':press_fit_lims_LHW})
    pp_AW = ahf.Plot_Parameters(x_vars=[across_x_var], y_vars=['press_TC_max'], legend=LHW_AW_legend, extra_args={'plot_slopes':'OLS', 'mv_avg':'30D'}, ax_lims={'x_lims':bps.date_range_dict[this_BGR], 'y_lims':press_lims_AW})
    pp_AW_fit = ahf.Plot_Parameters(x_vars=[across_x_var], y_vars=['press_TC_max-fit'], legend=LHW_AW_legend, extra_args={'plot_slopes':'OLS', 'mv_avg':'30D', 'fit_vars':['lon','lat']}, ax_lims={'x_lims':bps.date_range_dict[this_BGR], 'y_lims':press_fit_lims_AW},)
    # Make the subplot groups
    group_LHW = ahf.Analysis_Group2([bnds_df], pp_LHW, plot_title=r'LHW core, $p(S_A\approx34.1)$')
    group_LHW_fit = ahf.Analysis_Group2([bnds_df], pp_LHW_fit, plot_title=r'$p(S_A\approx34.1)$ - polyfit2d')
    group_AW = ahf.Analysis_Group2([bnds_df], pp_AW, plot_title=r'AW core, $p(\Theta_{max})$')
    group_AW_fit = ahf.Analysis_Group2([bnds_df], pp_AW_fit, plot_title=r'$p(\Theta_{max})$ - polyfit2d')
    # Make the figure
    make_figure([group_LHW, group_LHW_fit, group_AW, group_AW_fit], row_col_list=[2,2, 0.48, 1.5], filename='f4_LHW_and_AW_trends_in_time.pdf')
################################################################################
# Histograms of the cluster properties
# Histogram of cluster average temperature
if False:
    this_ca_var = 'ca_CT'
    these_y_lims = None #[355,190]
    # this_ca_var = 'ca_SA'
    # these_y_lims = [bps.S_range_LHW_AW[1], bps.S_range_LHW_AW[0]]
    # Make the plot parameters
    pp_ca_hist = ahf.Plot_Parameters(x_vars=['hist'], y_vars=[this_ca_var], first_dfs=[False,True], extra_args={'n_h_bins':50, 're_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':False, 'mark_outliers':False, 'extra_vars_to_keep':['cluster', 'press']}, legend=True)#, ax_lims={'x_lims':[-100,100], 'y_lims':these_y_lims})
    # Make the subplot groups
    group_ca_hist = ahf.Analysis_Group2([df], pp_ca_hist, plot_title='')#this_BGR)
    # Make the figure
    make_figure([group_ca_hist])
################################################################################
################################################################################
# Evaluating the clusters
#**# Plot of nir_SA and cRL for all clusters (Supplementary Materials)
if False:
    this_ca_var = 'ca_press'
    these_y_lims = [423,200]
    # this_ca_var = 'ca_SA'
    # these_y_lims = [bps.S_range_LHW_AW[1], bps.S_range_LHW_AW[0]]
    # Make the plot parameters
    pp_nir = ahf.Plot_Parameters(x_vars=['nir_SA'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':False, 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press']}, legend=False, ax_lims={'x_lims':[0,1.75], 'y_lims':these_y_lims}) # 'x_lims':[-5,160]
    pp_cRL = ahf.Plot_Parameters(x_vars=['cRL'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':True, 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press']}, legend=False, ax_lims={'x_lims':[-57,0], 'y_lims':these_y_lims}) # 'x_lims':[-100,100]
    # Make the subplot groups
    group_nir = ahf.Analysis_Group2([df], pp_nir, plot_title='')#this_BGR)
    group_cRL = ahf.Analysis_Group2([df], pp_cRL, plot_title='')#this_BGR)
    # Make the figure
    make_figure([group_nir, group_cRL], row_col_list=[1,2, 0.48, 1.25], filename='s6_BGR_all_RL_IR_SA_vs_'+this_ca_var+'.png')
################################################################################
# The IR and spans in all variables for all clusters
# Make plots of the IR and cluster spans in the cluster properties
plot_slopes = None
add_legend = True
this_ca_var = 'ca_press'
if False:
# for this_prefix in ['nir_', 'cs_', 'csd_']:
    for this_clr_map in ['clr_all_same']:#, 'cluster']:
        # Make the Plot Parameters
        pp_press = ahf.Plot_Parameters(x_vars=[this_prefix+'press'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'cRL','nir_SA']}, legend=add_legend)
        pp_SA = ahf.Plot_Parameters(x_vars=[this_prefix+'SA'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'cRL','nir_SA']}, legend=False)
        pp_CT = ahf.Plot_Parameters(x_vars=[this_prefix+'CT'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'cRL','nir_SA']}, legend=False)
        pp_sig = ahf.Plot_Parameters(x_vars=[this_prefix+'sigma'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'cRL','nir_SA']}, legend=False)
        # Make the subplot groups
        group_press = ahf.Analysis_Group2([df], pp_press)
        group_SA = ahf.Analysis_Group2([df], pp_SA)
        group_CT = ahf.Analysis_Group2([df], pp_CT)
        group_sig = ahf.Analysis_Group2([df], pp_sig)
        # Make the figure
        make_figure([group_press, group_SA, group_CT, group_sig], row_col_list=[1,4, 0.3, 1.03])#, filename='fit4-trends_vs_'+this_ca_var+'_w_clrmap_'+this_clr_map+'.png')
        # make_figure([group_press_trends, group_SA_trends, group_CT_trends], row_col_list=[1,3, 0.35, 1.03], filename='fit-trends_vs_'+this_ca_var+'_w_clrmap_'+this_clr_map+'.png')
################################################################################
# The properties of the clusters
# Make plots of the polyfit2d trends in the cluster properties
plot_slopes = 'OLS'
these_ax_lims = None#{'y_lims':[355,190]}
add_legend = True
if False:
# for this_ca_var in ['ca_SA']:#, 'ca_press', 'ca_SA', 'ca_CT', 'ca_sigma']:
    for this_clr_map in ['clr_all_same', 'cluster']:
        # Make the Plot Parameters
        pp_press_trends = ahf.Plot_Parameters(x_vars=['ca_press'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'cRL','nir_SA'], 'mark_LHW_AW':True}, ax_lims=these_ax_lims, legend=add_legend)
        pp_SA_trends = ahf.Plot_Parameters(x_vars=['ca_SA'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'cRL','nir_SA'], 'mark_LHW_AW':True}, legend=False)
        pp_CT_trends = ahf.Plot_Parameters(x_vars=['ca_CT'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'cRL','nir_SA'], 'mark_LHW_AW':True}, legend=False)
        pp_sig_trends = ahf.Plot_Parameters(x_vars=['ca_sigma'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'cRL','nir_SA'], 'mark_LHW_AW':True}, legend=False)
        # Make the subplot groups
        group_press_trends = ahf.Analysis_Group2([df], pp_press_trends)
        group_SA_trends = ahf.Analysis_Group2([df], pp_SA_trends)
        group_CT_trends = ahf.Analysis_Group2([df], pp_CT_trends)
        group_sig_trends = ahf.Analysis_Group2([df], pp_sig_trends)
        # Make the figure
        make_figure([group_press_trends, group_SA_trends, group_CT_trends, group_sig_trends], row_col_list=[1,4, 0.3, 1.03], filename='ca_vars_vs_'+this_ca_var+'_w_clrmap_'+this_clr_map+'.pdf')
        # make_figure([group_press_trends, group_SA_trends, group_CT_trends], row_col_list=[1,3, 0.35, 1.03], filename='fit-trends_vs_'+this_ca_var+'_w_clrmap_'+this_clr_map+'.png')
    these_ax_lims = None
################################################################################
# Thicknesses of the layers
# Plot of pcs_press for all profiles against this_ca_var
this_clr_map = 'clr_all_same'
this_clr_map = 'cluster'
# this_ca_var = 'ca_press'
# this_vert_var = 'press'
this_ca_var = 'ca_SA'
this_vert_var = 'SA'
if False:
    # Load the per-profile data
    df_per_pf = pl.load(open('outputs/'+filename+'_pf_cluster_properties.pickle', 'rb'))
    # To compare to Shibley et al. 2019 Figure 6b: 
    #   Use only ITP13 between August 2007 - August 2008 (ITP13 has data in this time period only)
    plot_Shibley2019_fig6b = False
    if plot_Shibley2019_fig6b:
        import datetime as dt
        df_per_pf = df_per_pf[(df_per_pf['instrmt'] == '13')]
        # print('df_per_pf[instrmt]:',df_per_pf['instrmt'])
        # print('type of df_per_pf[instrmt]:',type(df_per_pf['instrmt'].values[0]))
        # exit(0)
        these_x_lims = [0.5,4.5]
        these_y_lims = [300,170]
        these_x_lims = [0,15]
        these_y_lims = [350,170]
        this_plt_title = 'ITP 13'
    else:
        these_x_lims = [0, 16]
        if this_ca_var == 'ca_press':
            these_y_lims = [450,150]
        elif this_ca_var == 'ca_SA':
            if this_clr_map == 'clr_all_same':
                these_y_lims = [bps.S_range_LHW_AW[1], bps.S_range_LHW_AW[0]]
                these_x_lims = [0,8]
            elif this_clr_map == 'cluster':
                these_y_lims = [bps.S_range_LHW_AW[1], bps.S_range_LHW_AW[0]]
                these_x_lims = [0,20]
        this_plt_title = ''
    # Find outliers in df
    df = ahf.find_outliers(df, ['cRL', 'nir_SA'], mrk_outliers='ends')
    # Based on df, get values of 'cRL' and 'nir_SA' for df_per_pf
    df_per_pf['out_cRL'] = None
    df_per_pf['out_nir_SA'] = None
    df_per_pf['out_ends'] = None
    # Check columns of df
    # print('df columns:',df.columns)
    # print('df_per_pf columns:',df_per_pf.columns)
    # Mark the outlier clusters in df_per_pf
    for i in np.unique(df_per_pf['cluster']):
        if i in df['cluster'].values:
            # Mark the cluster as an outlier if it is an outlier in the original dataframe
            df_per_pf.loc[df_per_pf['cluster'] == i, 'out_cRL'] = df.loc[df['cluster'] == i, 'out_cRL'].values[0]
            df_per_pf.loc[df_per_pf['cluster'] == i, 'out_nir_SA'] = df.loc[df['cluster'] == i, 'out_nir_SA'].values[0]
            df_per_pf.loc[df_per_pf['cluster'] == i, 'out_ends'] = df.loc[df['cluster'] == i, 'out_ends'].values[0]
    # Remove all rows with 'cRL' or 'nir_SA' outliers
    df_per_pf = df_per_pf[(df_per_pf['out_cRL'] == False) & (df_per_pf['out_nir_SA'] == False) & (df_per_pf['out_ends'] == False)]
    # Make a copy of the dataframe
    nzdf_per_pf = df_per_pf.copy()
    # Make a column in the dataframe of pcs_press without zeros
    nzdf_per_pf = nzdf_per_pf[nzdf_per_pf['pcs_press'] != 0]
    nzdf_per_pf['nzpcs_press'] = nzdf_per_pf['pcs_press']
    # Change the new columns to be numpy float32
    nzdf_per_pf['pcs_press'] = nzdf_per_pf['pcs_press'].astype(np.float32)
    nzdf_per_pf['nzpcs_press'] = nzdf_per_pf['nzpcs_press'].astype(np.float32)
    # Check columns of df
    # print('df columns:',df.columns)
    # print('df_per_pf columns:',df_per_pf.columns)
    print('nzdf_per_pf columns:',nzdf_per_pf.columns)
    """
    Note to myself: it doesn't make sense to take the polyfit2d of pcs_press, it just shifts it over
    """
    # Plot of nzpcs_press for all profiles against press
    if False:
        # Make the plot parameters
        pp_nzpcs = ahf.Plot_Parameters(x_vars=['nzpcs_press'], y_vars=[this_vert_var], clr_map='clr_all_same', extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':False, 'extra_vars_to_keep':['cluster', 'press']}, legend=False, ax_lims={'x_lims':these_x_lims, 'y_lims':these_y_lims})
        # pp_nzpcs = ahf.Plot_Parameters(x_vars=['nzpcs_press'], y_vars=['press'], clr_map='cluster', extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':False, 'mark_outliers':False, 'extra_vars_to_keep':['cluster', 'press']}, legend=False, ax_lims={'x_lims':these_x_lims, 'y_lims':these_y_lims})
        # Make the subplot groups
        group_nzpcs = ahf.Analysis_Group2([nzdf_per_pf], pp_nzpcs, plot_title=this_plt_title)
        # Make the figure
        make_figure([group_nzpcs], row_col_list=[1,1, 0.35, 1.1], filename='C10_ITP13_compare_to_Shibley2019fig6b.pdf')

    # Plot of trends in pcs_press for each cluster, with zero values and without
    if False:
        # Make the plot parameters
        pp_nzpcs = ahf.Plot_Parameters(x_vars=['nztrd_pcs_press'], y_vars=[this_ca_var], clr_map='clr_all_same', extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=True)
        pp_pcs = ahf.Plot_Parameters(x_vars=['trd_pcs_press'], y_vars=[this_ca_var], clr_map='clr_all_same', extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=True)#, ax_lims={'x_lims':[0.5,4.5], 'y_lims':[300,170]})
        # Make the subplot groups
        group_nzpcs = ahf.Analysis_Group2([df], pp_nzpcs, plot_title=this_BGR)
        group_pcs   = ahf.Analysis_Group2([df], pp_pcs, plot_title=this_BGR)
        # Make the figure
        make_figure([group_nzpcs, group_pcs])#, group_nzpcs_fit])#, row_col_list=[1,1, 0.8, 1.25])
    
    #*# Plots of thicknesses all layers, cluster average thickness, and trends in thickness
    if False:
        # Make the plot parameters
        """
        Note to myself: I've already taken out the outlier clusters in nzdf_per_pf
        """
        pp_nzpcs = ahf.Plot_Parameters(x_vars=['nzpcs_press'], y_vars=[this_vert_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':False, 'mark_outliers':False, 'plot_centroid':False, 'extra_vars_to_keep':['cluster', 'press']}, legend=False, ax_lims={'x_lims':these_x_lims, 'y_lims':these_y_lims})
        pp_ca_nzpcs = ahf.Plot_Parameters(x_vars=['nzca_pcs_press'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False, ax_lims={'x_lims':these_x_lims, 'y_lims':these_y_lims})
        # pp_trd_nzpcs = ahf.Plot_Parameters(x_vars=['nztrd_pcs_press'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False, ax_lims={'x_lims':[-2,2], 'y_lims':these_y_lims})
        pp_trd_nzpcs = ahf.Plot_Parameters(x_vars=['percnztrd_pcs_press'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False, ax_lims={'x_lims':[-40,40], 'y_lims':these_y_lims})
        # Make the subplot groups
        group_nzpcs    = ahf.Analysis_Group2([nzdf_per_pf], pp_nzpcs, plot_title='')
        group_ca_nzpcs = ahf.Analysis_Group2([df], pp_ca_nzpcs, plot_title='')
        group_trd_nzpcs= ahf.Analysis_Group2([df], pp_trd_nzpcs, plot_title='')
        # Make the figure
        make_figure([group_nzpcs, group_ca_nzpcs, group_trd_nzpcs], row_col_list=[1,3, 0.3, 1.02], filename='s_BGR_all_layer_thickness_'+this_clr_map+'.png')# row_col_list=[1,2, 0.45, 1.2])

    #**# Plots of thicknesses all layers, heat flux, and cumulative heat flux (Supplementary Materials)
    if True:
        # Make the plot parameters
        """
        Note to myself: I've already taken out the outlier clusters in nzdf_per_pf
        """
        pp_nzpcs = ahf.Plot_Parameters(x_vars=['nzpcs_press'], y_vars=[this_vert_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':False, 'mark_outliers':False, 'plot_centroid':False, 'extra_vars_to_keep':['cluster', 'press']}, legend=False, ax_lims={'x_lims':these_x_lims, 'y_lims':these_y_lims})
        # pp_ca_nzpcs = ahf.Plot_Parameters(x_vars=['nzca_pcs_press'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False, ax_lims={'x_lims':these_x_lims, 'y_lims':these_y_lims})
        pp_trd_nzpcs = ahf.Plot_Parameters(x_vars=['nztrd_pcs_press'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL'], 'errorbars':True}, legend=False, ax_lims={'x_lims':[-0.5,0.5], 'y_lims':these_y_lims})
        pp_perctrd_nzpcs = ahf.Plot_Parameters(x_vars=['percnztrd_pcs_press'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL'], 'errorbars':True}, legend=False, ax_lims={'y_lims':these_y_lims})
        # Make the subplot groups
        group_nzpcs    = ahf.Analysis_Group2([nzdf_per_pf], pp_nzpcs, plot_title='')
        group_trd_nzpcs = ahf.Analysis_Group2([df], pp_trd_nzpcs, plot_title='')
        group_perctrd_nzpcs= ahf.Analysis_Group2([df], pp_perctrd_nzpcs, plot_title='')
        # Make the figure
        make_figure([group_nzpcs, group_trd_nzpcs, group_perctrd_nzpcs], row_col_list=[1,3, 0.3, 1.02], filename='s12_BGR_all_thickness_'+this_clr_map+'.png')# row_col_list=[1,2, 0.45, 1.2])
    
    # Plot of thickness for just one cluster
    this_clstr_id = 5
    if False:
        # Reduce nzdf_per_pf to only that cluster
        nzdf_per_pf = nzdf_per_pf[nzdf_per_pf['cluster'] == this_clstr_id]
        # Make the plot parameters
        pp_nzpcs = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['nzpcs_press'], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':False, 'plot_centroid':False, 'extra_vars_to_keep':['cluster', 'press']}, legend=False)#, ax_lims={'x_lims':these_x_lims, 'y_lims':these_y_lims})
        # Make the subplot groups
        group_nzpcs = ahf.Analysis_Group2([nzdf_per_pf], pp_nzpcs, plot_title='Cluster '+str(this_clstr_id))
        # Make the figure
        make_figure([group_nzpcs], row_col_list=[1,1, 0.35, 1.1])

    #*# Maps of layer thicknesses
    if False:
        # Narrow nzdf_per_pf to only cluster 63
        # nzdf_per_pf = nzdf_per_pf[nzdf_per_pf['cluster'] == 63]
        # Find the mean nzpcs_press for each profile
        nzdf_per_pf = nzdf_per_pf.groupby('dt_start').mean()
        nzdf_per_pf['notes'] = ['']*len(nzdf_per_pf)
        # Rename nzpcs_press to nzpcs_press_mean
        nzdf_per_pf['nzpcs_press_mean'] = nzdf_per_pf['nzpcs_press']
        # Make the plot parameters
        pp_pcs_map = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='nzpcs_press_mean', extra_args={'plot_slopes':True, 'extra_vars_to_keep':[]}, ax_lims={'x_lims':bps.lon_BGR, 'y_lims':bps.lat_BGR, 'c_lims':[0,3]}, legend=False)
        pp_pcs_hist = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['nzpcs_press_mean'], extra_args={'plot_slopes':False}, ax_lims={'y_lims':[0,3]}, legend=False)
        # Make the subplot groups
        group_pcs_map = ahf.Analysis_Group2([nzdf_per_pf], pp_pcs_map, plot_title='')
        group_pcs_hist = ahf.Analysis_Group2([nzdf_per_pf], pp_pcs_hist, plot_title='')
        # Make the figure
        make_figure([group_pcs_map, group_pcs_hist], filename='C11_BGR_all_nzpcs_press_mean_per_pf.pdf')
################################################################################
# Calculating heat flux in W/m^2
#**# Plots of cluster averages of components of the heat flux (Supplementary Materials)
#   height, isobaric heat capacity, temperature trend in time, density
this_ca_var = 'ca_SA'
# this_clr_map = 'clr_all_same'
if False:
    # Make the Plot Parameters
    pp_height = ahf.Plot_Parameters(x_vars=['nzca_pcs_press'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False)
    pp_cp = ahf.Plot_Parameters(x_vars=['ca_cp'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'mark_zero':False, 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False)
    pp_trd_CT = ahf.Plot_Parameters(x_vars=['trd_CT-fit'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False)
    pp_rho = ahf.Plot_Parameters(x_vars=['ca_rho'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'mark_zero':False, 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False)
    pp_FH = ahf.Plot_Parameters(x_vars=['ca_FH'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False)
    pp_FH_cumul = ahf.Plot_Parameters(x_vars=['ca_FH_cumul'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':False, 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False)
    # Make the subplot groups
    group_height = ahf.Analysis_Group2([df], pp_height, plot_title=r'$\overline{\Delta h}$')
    group_cp = ahf.Analysis_Group2([df], pp_cp, plot_title=r'$\overline{C_p}$')
    group_trd_CT = ahf.Analysis_Group2([df], pp_trd_CT, plot_title=r'$\overline{\text{d}_t \Theta}$')
    group_rho = ahf.Analysis_Group2([df], pp_rho, plot_title=r'$\overline{\rho}$')
    group_FH = ahf.Analysis_Group2([df], pp_FH, plot_title=r'$\overline{\Delta\mathcal{F}_H} = \overline{\Delta h} \cdot \overline{C_p} \cdot \overline{\text{d}_t \Theta} \cdot \overline{\rho}$')
    group_FH_cumul = ahf.Analysis_Group2([df], pp_FH_cumul, plot_title='')
    # Make the figure
    # make_figure([group_height, group_cp])
    # make_figure([group_height, group_cp, group_trd_CT, group_rho], row_col_list=[2,2, 0.6, 1.6], filename='C12_BGR_all_FH_comps_vs_SA.pdf')
    make_figure([group_height, group_cp, group_trd_CT, group_rho, group_FH, group_FH_cumul], row_col_list=[2,3, 0.6, 1.6], filename='s11_BGR_all_FH_comps_vs_SA.png')
# Plotting net heat flux against three different vertical axes
if False:
    # Make the Plot Parameters
    pp_FH_v_SA = ahf.Plot_Parameters(x_vars=['ca_FH'], y_vars=['ca_SA'], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False)
    pp_FH_v_CT = ahf.Plot_Parameters(x_vars=['ca_FH'], y_vars=['ca_CT'], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False)
    pp_FH_v_press = ahf.Plot_Parameters(x_vars=['ca_FH'], y_vars=['ca_press'], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False)
    # Make the subplot groups
    group_FH_SA = ahf.Analysis_Group2([df], pp_FH_v_SA, plot_title='')
    group_FH_CT = ahf.Analysis_Group2([df], pp_FH_v_CT, plot_title='')
    group_FH_press = ahf.Analysis_Group2([df], pp_FH_v_press, plot_title='')
    # Make the figure
    make_figure([group_FH_SA, group_FH_CT, group_FH_press])
# Plot of the heat flux in W/m^2
if False:
    # Make the Plot Parameters
    pp_FH = ahf.Plot_Parameters(x_vars=['ca_FH'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False)
    pp_FH_cumul = ahf.Plot_Parameters(x_vars=['ca_FH_cumul'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':False, 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False)
    # Make the subplot groups
    group_FH = ahf.Analysis_Group2([df], pp_FH, plot_title='')
    group_FH_cumul = ahf.Analysis_Group2([df], pp_FH_cumul, plot_title='')
    # Make the figure
    make_figure([group_FH, group_FH_cumul])#, filename='f11_FH_and_cumul_vs_SA.pdf')#, row_col_list=[1,1, 0.3, 1.04])
################################################################################
################################################################################
################################################################################
# Results summary plot
#***# Make plots of the polyfit2d trends in the cluster properties 
x_lims_dict = {
                'ca_press':[200,450],
                'ca_depth':[200,450],
                'ca_CT':[-1.2,0.9],
                'nzca_pcs_press':[-0.2,7]
}
plot_slopes = 'OLS'
add_legend = False
if False:
# for this_clr_map in ['clr_all_same']:#, 'clr_all_same', 'cluster']:
    groups_to_plot = []
    # Add in plots of cluster averages
    # this_ca_var = 'ca_press'
    this_ca_var = 'ca_SA'
    if this_ca_var == 'ca_press':
        these_y_lims = [355,190]
    elif this_ca_var == 'ca_SA':
        these_y_lims = [bps.S_range_LHW_AW[1], bps.S_range_LHW_AW[0]]
    # if False:
    for this_plt_var in ['ca_CT', 'ca_press', 'nzca_pcs_press']:# 'ca_SA']:
        # Make the Plot Parameters
        if this_plt_var == 'nzca_pcs_press':
            mrk_LHW_AW = False
        else:
            mrk_LHW_AW = True
        if this_plt_var == 'ca_press':
            lgnd = False #True
        else:
            lgnd = add_legend
        pp_ca_plot = ahf.Plot_Parameters(x_vars=[this_plt_var], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL','nir_SA'], 'mark_LHW_AW':mrk_LHW_AW, 'errorbars':True}, legend=lgnd, ax_lims={'x_lims':x_lims_dict[this_plt_var], 'y_lims':these_y_lims})
        # Make the subplot groups
        groups_to_plot.append(ahf.Analysis_Group2([df], pp_ca_plot, plot_title=''))
    # Add in plots of trends over time
    # if False:
    for this_ca_var in ['ca_SA']:#, 'ca_press']:#, 'ca_SA', 'ca_CT', 'ca_sigma']:
        # Make the Plot Parameters
        ## Just one cluster, maps (og and fit) of press, SA, or CT
        if False:
        # for this_cluster_id in [63]:
        # for this_cluster_id in [27]:
            ds_this_BGR = ahf.Data_Set(bps.BGR_HPC_SA_div_dict[this_BGR], bob.dfs_all)
            lon_BGR = bps.lon_BGR
            lat_BGR = bps.lat_BGR
            # Make title and file name prefix
            this_cluster_title = 'Cluster '+str(this_cluster_id)
            # Get the cluster ranges dictionary
            clstr_ranges_dict = bps.BGR_all_clstr_plt_ranges[this_cluster_id]
            # Make the profile filters for this cluster
            pfs_these_clstrs = ahf.Profile_Filters(clstrs_to_plot=[this_cluster_id])
            # 
            pp_press_map = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='press', legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['press', 'SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':clstr_ranges_dict['press_lims']})
            #
            pp_SA_map = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='SA', legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':clstr_ranges_dict['SA_lims']})
            #
            pp_CT_map = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], clr_map='CT', legend=False, extra_args={'sort_clstrs':False, 'plot_slopes':True, 'extra_vars_to_keep':['CT', 'SA','cluster']}, ax_lims={'x_lims':lon_BGR, 'y_lims':lat_BGR, 'c_lims':clstr_ranges_dict['CT_lims']})
        if this_ca_var == 'ca_press':
            these_y_lims = [400,190]
            legend_for_press_plot = True
        elif this_ca_var == 'ca_SA':
            these_y_lims = [bps.S_range_LHW_AW[1], bps.S_range_LHW_AW[0]]
            legend_for_press_plot = False
        ## Trends in vars
        pp_press_trends = ahf.Plot_Parameters(x_vars=['trd_press-fit'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'cRL','nir_SA'], 'mark_LHW_AW':True, 'errorbars':True}, legend=legend_for_press_plot, ax_lims={'y_lims':these_y_lims})
        #
        pp_SA_trends = ahf.Plot_Parameters(x_vars=['trd_SA-fit'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'cRL','nir_SA'], 'mark_LHW_AW':True, 'errorbars':True}, legend=add_legend, ax_lims={'y_lims':these_y_lims})
        #
        pp_CT_trends = ahf.Plot_Parameters(x_vars=['trd_CT-fit',], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':'ends', 'extra_vars_to_keep':['press', 'cluster', 'cRL','nir_SA'], 'mark_LHW_AW':True, 'errorbars':True}, legend=add_legend, ax_lims={'y_lims':these_y_lims, 'tw_x_lims':[-0.00056,0.002]})
        #
        pp_sig_trends = ahf.Plot_Parameters(x_vars=['trd_sigma-fit'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'cRL','nir_SA'], 'mark_LHW_AW':True, 'errorbars':True}, legend=add_legend, ax_lims={'y_lims':these_y_lims})
        #
        pp_FH = ahf.Plot_Parameters(x_vars=['ca_FH'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL'], 'errorbars':True}, legend=add_legend, ax_lims={'y_lims':these_y_lims})
        #
        pp_FH_cumul = ahf.Plot_Parameters(x_vars=['ca_FH_cumul'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':False, 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL'], 'errorbars':True}, legend=add_legend, ax_lims={'y_lims':these_y_lims})
        # 
        ## Layer thicknesses
        these_x_lims = [0, 16]
        if this_ca_var == 'ca_SA':
            if this_clr_map == 'clr_all_same':
                these_x_lims = [0,8]
            elif this_clr_map == 'cluster':
                these_x_lims = [0,20]
        # Load the per-profile data
        if False:
            df_per_pf = pl.load(open('outputs/'+filename+'_pf_cluster_properties.pickle', 'rb'))
            # Find outliers in df
            df = ahf.find_outliers(df, ['cRL', 'nir_SA'])
            # Based on df, get values of 'cRL' and 'nir_SA' for df_per_pf
            df_per_pf['out_cRL'] = None
            df_per_pf['out_nir_SA'] = None
            # Check columns of df
            # print('df columns:',df.columns)
            # print('df_per_pf columns:',df_per_pf.columns)
            # Mark the outlier clusters in df_per_pf
            for i in np.unique(df_per_pf['cluster']):
                if i in df['cluster'].values:
                    # Mark the cluster as an outlier if it is an outlier in the original dataframe
                    df_per_pf.loc[df_per_pf['cluster'] == i, 'out_cRL'] = df.loc[df['cluster'] == i, 'out_cRL'].values[0]
                    df_per_pf.loc[df_per_pf['cluster'] == i, 'out_nir_SA'] = df.loc[df['cluster'] == i, 'out_nir_SA'].values[0]
            # Remove all rows with 'cRL' or 'nir_SA' outliers
            df_per_pf = df_per_pf[(df_per_pf['out_cRL'] == False) & (df_per_pf['out_nir_SA'] == False)]
            # Make a copy of the dataframe
            nzdf_per_pf = df_per_pf.copy()
            # Make a column in the dataframe of pcs_press without zeros
            nzdf_per_pf = nzdf_per_pf[nzdf_per_pf['pcs_press'] != 0]
            nzdf_per_pf['nzpcs_press'] = nzdf_per_pf['pcs_press']
            # Change the new columns to be numpy float32
            nzdf_per_pf['pcs_press'] = nzdf_per_pf['pcs_press'].astype(np.float32)
            nzdf_per_pf['nzpcs_press'] = nzdf_per_pf['nzpcs_press'].astype(np.float32)
        # Make the plot parameters
        pp_nzpcs = ahf.Plot_Parameters(x_vars=['nzpcs_press'], y_vars=[this_vert_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':False, 'mark_outliers':False, 'plot_centroid':False, 'extra_vars_to_keep':['cluster', 'press'], 'errorbars':True}, legend=False, ax_lims={'x_lims':these_x_lims, 'y_lims':these_y_lims})
        #
        pp_ca_nzpcs = ahf.Plot_Parameters(x_vars=['nzca_pcs_press'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL'], 'errorbars':True}, legend=False, ax_lims={'x_lims':these_x_lims, 'y_lims':these_y_lims})
        #
        # pp_trd_nzpcs = ahf.Plot_Parameters(x_vars=['nztrd_pcs_press'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL'], 'errorbars':True}, legend=False, ax_lims={'x_lims':[-0.5,0.5], 'y_lims':these_y_lims})
        pp_trd_nzpcs = ahf.Plot_Parameters(x_vars=['percnztrd_pcs_press'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':'ends', 'extra_vars_to_keep':['cluster', 'press', 'cRL'], 'errorbars':True}, legend=False, ax_lims={'y_lims':these_y_lims})
        #
        # Make the subplot groups
        # ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_press_map, plot_title=this_cluster_title),
        # ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_SA_map, plot_title=this_cluster_title),
        # ahf.Analysis_Group(ds_this_BGR, pfs_these_clstrs, pp_CT_map, plot_title=this_cluster_title),
        # groups_to_plot.append(ahf.Analysis_Group2([df], pp_SA_trends, plot_title=''))
        groups_to_plot.append(ahf.Analysis_Group2([df], pp_CT_trends, plot_title=''))           #**
        # groups_to_plot.append(ahf.Analysis_Group2([df], pp_FH, plot_title=''))
        # groups_to_plot.append(ahf.Analysis_Group2([df], pp_FH_cumul, plot_title=''))
        groups_to_plot.append(ahf.Analysis_Group2([df], pp_press_trends, plot_title=''))        #**
        # groups_to_plot.append(ahf.Analysis_Group2([nzdf_per_pf], pp_nzpcs, plot_title=''))
        # groups_to_plot.append(ahf.Analysis_Group2([df], pp_ca_nzpcs, plot_title=''))
        groups_to_plot.append(ahf.Analysis_Group2([df], pp_trd_nzpcs, plot_title=''))           #**
        # ahf.Analysis_Group2([df], pp_sig_trends, plot_title=''),
    # Make the figure
    # row_col_list=[1,4, 0.3, 1.03])
    # , row_col_list=[2,2, 0.8, 1.03]
    # make_figure(groups_to_plot, filename='f5_results_summary_w_clrmap_'+this_clr_map+'.png')
    make_figure(groups_to_plot, filename='3_Results_'+filename+'_'+this_clr_map+'.png')
################################################################################
