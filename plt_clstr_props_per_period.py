"""
Author: Mikhail Schee
Created: 2024-07-03

This script will plot cluster properties for the BGR data set per period using the pickle file 
created by get_clstr_props2.py.

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
import pandas as pd
# For adding subplot labels a, b, c, ...
import string
import analysis_helper_functions as ahf
import dill as pl
# Import the Thermodynamic Equation of Seawater 2010 (TEOS-10) from GSW
# For finding density and heat capacity
import gsw

# For common BGR parameters
import BGR_params as bps
# For common BGR objects
import BGR_objects as bob

# Specify which BGR data set to use
# this_BGR = 'BGR0506'
this_BGR = 'BGR_all'

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

################################################################################

# Unpickle the data frame from a file
df = pl.load(open('outputs/'+this_BGR+'_cluster_properties_unrelab.pickle', 'rb'))
# df = pl.load(open('outputs/'+this_BGR+'_pf_cluster_properties.pickle', 'rb'))
# df = pl.load(open('outputs/'+this_BGR+'_cluster_properties2.pickle', 'rb'))
# df = pl.load(open('outputs/'+this_BGR+'_cluster_properties-pf.pickle', 'rb'))
# bnds_df = pl.load(open('outputs/'+this_BGR+'_LHW_AW_properties.pickle', 'rb'))

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
# Plotting functions ###########################################################
################################################################################

def get_axis_labels(pp):
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
    # Get x axis labels
    if not isinstance(pp.x_vars, type(None)):
        for i in range(len(pp.x_vars)):
            pp.xlabels[i] = make_var_label(pp.x_vars[i], ax_labels, ax_labels_no_units)
    else:
        pp.xlabels[0] = None
        pp.xlabels[1] = None
    # Get y axis labels
    if not isinstance(pp.y_vars, type(None)):
        for i in range(len(pp.y_vars)):
            pp.ylabels[i] = make_var_label(pp.y_vars[i], ax_labels, ax_labels_no_units)
    else:
        pp.ylabels[0] = None
        pp.ylabels[1] = None
    # Get colormap label
    if not isinstance(pp.clr_map, type(None)):
        try:
            pp.clabel = var_attr_dicts[0][pp.clr_map]['label']
        except:
            pp.clabel = make_var_label(pp.clr_map, ax_labels, ax_labels_no_units)

    return pp

def make_var_label(var_key, ax_labels, ax_labels_no_units):
    """
    Takes in a variable key and returns the corresponding label

    var_key         A string of the variable key
    ax_labels       A dictionary of variable keys and their labels
    """
    # Check for certain modifications to variables,
    #   check longer strings first to avoid mismatching percnztrd_pcs_press
    # Check for percent non-zero trend in profile cluster span variables
    if 'percnztrd_pcs_' in var_key:
        # Take out the first 13 characters of the string to leave the original variable name
        var_str = var_key[13:]
        # return 'Percent non-zero trend in '+ ax_labels[var_str] + '/year'
        return 'Trend in layer thickness (\%/year)'
    # Check for non-zero cluster average of profile cluster span variables
    if 'nzca_pcs_' in var_key:
        # Take out the first 9 characters of the string to leave the original variable name
        var_str = var_key[9:]
        # return 'NZCA/CS/P of '+ ax_labels[var_str]
        return 'Cluster average thickness in '+ ax_labels[var_str]
    # Check for cluster average of profile cluster span variables
    if 'ca_pcs_' in var_key:
        # Take out the first 7 characters of the string to leave the original variable name
        var_str = var_key[7:]
        # return 'Profile cluster span of '+ ax_labels[var_str]
        return 'CA/CS/P of '+ ax_labels[var_str]
    # Check for non-zero trend in profile cluster span variables
    if 'nztrd_pcs_' in var_key:
        # Take out the first 10 characters of the string to leave the original variable name
        var_str = var_key[10:]
        # return 'NZ Trend in CS/P '+ ax_labels[var_str] + '/year'
        return 'Trend in layer thickness in '+ ax_labels[var_str] + '/year'
    # Check for trend in profile cluster span variables
    if 'trd_pcs_' in var_key:
        # Take out the first 8 characters of the string to leave the original variable name
        var_str = var_key[8:]
        return 'Trend in CS/P '+ ax_labels[var_str] + '/year'
    # Check for profile cluster average variables
    if 'pca_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'CA/P of '+ ax_labels[var_str]
        # return 'Profile cluster average of '+ ax_labels[var_str]
    # Check for non-zero profile cluster span variables
    if 'nzpcs_' in var_key:
        # Take out the first 6 characters of the string to leave the original variable name
        var_str = var_key[6:]
        # Check for variables normalized by subtracting a polyfit2d
        if '-fit' in var_str:
            # Take out the first 3 characters of the string to leave the original variable name
            var_str = var_str[:-4]
            return 'NZCS/P of '+ ax_labels_no_units[var_str] + ' $-$ polyfit2d'
        elif 'mean' in var_str:
            # Take out the last 5 characters of the string to leave the original variable name
            var_str = var_str[:-5]
            return 'Mean layer thickness in '+ ax_labels[var_str]
        else:
            # return 'NZCS/P of '+ ax_labels[var_str]
            return 'Layer thickness in '+ ax_labels[var_str]
        # return 'Non-zero profile cluster span of '+ ax_labels[var_str]
    # Check for profile cluster span variables
    if 'pcs_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        # Check for variables normalized by subtracting a polyfit2d
        if '-fit' in var_str:
            # Take out the first 3 characters of the string to leave the original variable name
            var_str = var_str[:-4]
            return 'CS/P of '+ ax_labels_no_units[var_str] + ' $-$ polyfit2d'
        else:
            # return 'Profile cluster span of '+ ax_labels[var_str]
            return 'CS/P of '+ ax_labels[var_str]
    # Check for cluster mean-centered variables
    elif 'cmc_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'Cluster mean-centered '+ ax_labels[var_str]
    # Check for local anomaly variables
    elif 'la_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[3:]
        return r"$\Theta'$ ($^\circ$C)"
        # return 'Local anomaly of '+ ax_labels[var_str]
    # Check for local anomaly variables
    elif 'max_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'Maximum '+ ax_labels[var_str]
    # Check for local anomaly variables
    elif 'min_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'Minimum '+ ax_labels[var_str]
    # Check for cluster average variables
    elif 'ca_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[3:]
        if var_str == 'FH_cumul':
            return ax_labels[var_str]
        else:
            return 'Cluster average of '+ ax_labels[var_str]
    # Check for cluster span variables
    elif 'cs_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[3:]
        return 'Cluster span of '+ ax_labels[var_str]
    # Check for cluster standard deviation variables
    elif 'csd_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'Cluster standard deviation of '+ ax_labels[var_str]
    # Check for cluster min/max variables
    elif 'cmm_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'Cluster min/max of '+ ax_labels[var_str]
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
        # return r'Normalized inter-cluster range $IR$ of '+ ax_labels[var_str]
    # Check for trend variables
    elif 'trd_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[4:]
        # Check whether also normalizing by subtracting a polyfit2d
        if '-fit' in var_str:
            # Take out the first 3 characters of the string to leave the original variable name
            var_str = var_str[:-4]
            return '('+ ax_labels_no_units[var_str] + ' $-$ polyfit2d)/year'
        else:  
            return ''+ ax_labels[var_str] + '/year'
    # Check for the polyfit2d of a variable
    elif 'fit_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'polyfit2d of ' + ax_labels[var_str]
    # Check for variables normalized by subtracting a polyfit2d
    elif '-fit' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[:-4]
        return ax_labels_no_units[var_str] + ' $-$ polyfit2d'
    if var_key in ax_labels.keys():
        return ax_labels[var_key]
    else:
        return 'None'

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
        self.plt_params = get_axis_labels(plt_params)
        self.plot_title = plot_title

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
                      '5':[2,3, 0.5, 2.00], '6':[2,3, 0.5, 2.00],
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
            re_run_clstr = True
        if 'mark_LHW_AW' in extra_args.keys():
            mark_LHW_AW = extra_args['mark_LHW_AW']
        else:
            mark_LHW_AW = False
        if 'all_BGR' in extra_args.keys():
            all_BGR = extra_args['all_BGR']
        else:
            all_BGR = False
        if 'connect_p' in extra_args.keys():
            connect_p = extra_args['connect_p']
        else:
            connect_p = False
    else:
        extra_args = False
        plot_slopes = False
        mrk_outliers = False
        mv_avg = False
        mark_LHW_AW = False
        all_BGR = False
        connect_p = False
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
    except:
        tw_x_key = None
        tw_ax_y  = None
    try:
        tw_y_key = pp.y_vars[1]
        tw_ax_x  = ax.twinx()
    except:
        tw_y_key = None
        tw_ax_x  = None
    # Concatonate all the pandas data frames together
    df = pd.concat(a_group.data_frames)
    # Format the dates if necessary
    if x_key in ['dt_start', 'dt_end']:
        df[x_key] = mpl.dates.date2num(df[x_key])
    if y_key in ['dt_start', 'dt_end']:
        df[y_key] = mpl.dates.date2num(df[y_key])
    # Check for variables to be calculated
    if 'hist' in x_key or 'hist' in y_key:
        # Add a blank column to the dataframe
        df['hist'] = np.nan
    # Loop over each dataset
    if all_BGR:
        these_BGRs = ['BGR_all']
    else:
        these_BGRs = np.unique(df['dataset'])
    n_BGRs = len(these_BGRs)
    dfs = []
    for b in range(n_BGRs):
        this_BGR = these_BGRs[b]
        print('- Plotting data from',this_BGR)
        # Get the data for this dataset
        if this_BGR == 'BGR_all':
            this_df = df
        else:
            this_df = df[df['dataset'] == this_BGR]
            these_clusters = np.unique(this_df['cluster'])
            # Find nearest neighbors in the next and previous periods
            if x_key in ['dt_start', 'dt_end', 'hist'] and y_key in ['ca_SA']:
                # Add columns to this_df to store the nearest neighbor data
                #   Forward nearest neighbor
                this_df[y_key+'_fnn_cid'] = np.nan
                this_df[y_key+'_fnn_x'] = np.nan
                this_df[y_key+'_fnn_y'] = np.nan
                #   Backward nearest neighbor
                this_df[y_key+'_bnn_cid'] = np.nan
                this_df[y_key+'_bnn_x'] = np.nan
                this_df[y_key+'_bnn_y'] = np.nan
                # For the first period, just calculate the forward nearest neighbors
                if b == 0:
                    # Get the data for the next dataset
                    next_df = df[df['dataset'] == these_BGRs[b+1]]
                    # Loop over each cluster
                    for c in range(len(these_clusters)):
                        this_cluster = these_clusters[c]
                        # Get the y data for this cluster
                        this_y_val = this_df[this_df['cluster'] == this_cluster][y_key].values[0]
                        # Find the nearest neighbor in the next period
                        df_sort = next_df.iloc[(next_df[y_key]-this_y_val).abs().argsort()[:1]]
                        # print('\t- '+this_BGR+' cluster',this_cluster,'has y value',this_y_val)
                        # print('\t\t- Nearest neighbor in '+these_BGRs[b+1]+': cluster',df_sort['cluster'].values[0],'with y value',df_sort[y_key].values[0])
                        # Record the nearest neighbor's cluster id, x value, and y value
                        this_df.loc[this_df['cluster'] == this_cluster, y_key+'_fnn_cid'] = df_sort['cluster'].values[0]
                        this_df.loc[this_df['cluster'] == this_cluster, y_key+'_fnn_x'] = df_sort[x_key].values[0]
                        this_df.loc[this_df['cluster'] == this_cluster, y_key+'_fnn_y'] = df_sort[y_key].values[0]
                    if clr_map == 'cluster': 
                        quiver_nn(ax, this_df, x_key, y_key, fnn=True, bnn=False)
                # For middle periods, calculate both nearest neighbors
                elif b > 0 and b < n_BGRs-1:
                    # Get the data for the previous and next datasets
                    prev_df = df[df['dataset'] == these_BGRs[b-1]]
                    next_df = df[df['dataset'] == these_BGRs[b+1]]
                    # Loop over each cluster
                    for c in range(len(these_clusters)):
                        this_cluster = these_clusters[c]
                        # Get the y data for this cluster
                        this_y_val = this_df[this_df['cluster'] == this_cluster][y_key].values[0]
                        # Find the nearest neighbor in the previous period
                        df_sort = prev_df.iloc[(prev_df[y_key]-this_y_val).abs().argsort()[:1]]
                        # Record the nearest neighbor's cluster id, x value, and y value
                        this_df.loc[this_df['cluster'] == this_cluster, y_key+'_bnn_cid'] = df_sort['cluster'].values[0]
                        this_df.loc[this_df['cluster'] == this_cluster, y_key+'_bnn_x'] = df_sort[x_key].values[0]
                        this_df.loc[this_df['cluster'] == this_cluster, y_key+'_bnn_y'] = df_sort[y_key].values[0]
                        # Find the nearest neighbor in the next period
                        df_sort = next_df.iloc[(next_df[y_key]-this_y_val).abs().argsort()[:1]]
                        # Record the nearest neighbor's cluster id, x value, and y value
                        this_df.loc[this_df['cluster'] == this_cluster, y_key+'_fnn_cid'] = df_sort['cluster'].values[0]
                        this_df.loc[this_df['cluster'] == this_cluster, y_key+'_fnn_x'] = df_sort[x_key].values[0]
                        this_df.loc[this_df['cluster'] == this_cluster, y_key+'_fnn_y'] = df_sort[y_key].values[0]
                    if clr_map == 'cluster': 
                        quiver_nn(ax, this_df, x_key, y_key, fnn=True, bnn=True)
                # For the last period, just calculate the backward nearest neighbors
                elif b == n_BGRs-1:
                    # Get the data for the previous dataset
                    prev_df = df[df['dataset'] == these_BGRs[b-1]]
                    # Loop over each cluster
                    for c in range(len(these_clusters)):
                        this_cluster = these_clusters[c]
                        # Get the y data for this cluster
                        this_y_val = this_df[this_df['cluster'] == this_cluster][y_key].values[0]
                        # Find the nearest neighbor in the previous period
                        df_sort = prev_df.iloc[(prev_df[y_key]-this_y_val).abs().argsort()[:1]]
                        # Record the nearest neighbor's cluster id, x value, and y value
                        this_df.loc[this_df['cluster'] == this_cluster, y_key+'_bnn_cid'] = df_sort['cluster'].values[0]
                        this_df.loc[this_df['cluster'] == this_cluster, y_key+'_bnn_x'] = df_sort[x_key].values[0]
                        this_df.loc[this_df['cluster'] == this_cluster, y_key+'_bnn_y'] = df_sort[y_key].values[0]
                    if clr_map == 'cluster': 
                        quiver_nn(ax, this_df, x_key, y_key, fnn=False, bnn=True)
                #
            #
        # Determine the color mapping to be used
        if clr_map == 'clr_all_same':
            # Check for histogram
            if plot_hist:
                print('\t- Plotting histogram')
                x_label, y_label, z_label, plt_title, ax, invert_y_axis = ahf.plot_histogram(a_group, ax, pp, this_df, x_key, y_key, clr_map, legend=pp.legend, txk=tw_x_key, tay=tw_ax_y, tyk=tw_y_key, tax=tw_ax_x)
                return x_label, y_label, plt_title, ax, invert_y_axis
            # Set whether to plot in 3D
            if isinstance(z_key, type(None)):
                df_z_key = 0
                # Drop duplicates
                this_df.drop_duplicates(subset=[x_key, y_key], keep='first', inplace=True)
                plot_3d = False
            else:
                df_z_key = this_df[z_key]
                # Drop duplicates
                this_df.drop_duplicates(subset=[x_key, y_key, z_key], keep='first', inplace=True)
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
            m_size, m_alpha = ahf.get_marker_size_and_alpha(len(this_df[x_key]))
            m_size = m_size*2
            # Check whether to mark outliers
            print('\t- Marking outliers:',mrk_outliers)
            if mrk_outliers:
                # if 'trd_' in x_key:
                #     other_out_vars = ['cRL', 'nir_SA']
                # else:
                #     other_out_vars = []
                # ahf.mark_outliers(ax, this_df, x_key, y_key, clr_map, mrk_outliers, mrk_for_other_vars=other_out_vars)
                if x_key != 'cRL' and x_key != 'nir_SA':
                    ahf.mark_outliers(ax, this_df, x_key, y_key, clr_map, mrk_outliers, mrk_for_other_vars=['cRL', 'nir_SA'])
                else:
                    ahf.mark_outliers(ax, this_df, x_key, y_key, clr_map, mrk_outliers)
                # ahf.mark_outliers(ax, this_df, x_key, y_key, clr_map, mrk_outliers, mrk_for_other_vars=other_out_vars)
                # Get data without outliers
                if True: #plot_slopes:
                    x_data = np.array(this_df[this_df['out_'+x_key]==False][x_key].values, dtype=np.float64)
                    y_data = np.array(this_df[this_df['out_'+x_key]==False][y_key].values, dtype=np.float64)
                    # Print the median of the non-outlier data in x and y
                    print('\t- Non-outlier data in x:')
                    print(x_data)
                    print('\t- Non-outlier data in y:')
                    print(y_data)
                    print('\t- Median of non-outlier data in x:',np.median(x_data))
                    print('\t- Mean of non-outlier data in x:',np.mean(x_data))
                    print('\t- Std dev of non-outlier data in x:',np.std(x_data))
                    print('\t- Median of non-outlier data in y:',np.median(y_data))
                    print('\t- Std dev of non-outlier data in y:',np.std(y_data))
                    # Recalculate the cumulative heat flux without outliers
                    if 'ca_FH_cumul' in x_key:
                        # Remove the outliers from the this_df
                        this_df = this_df[this_df['out_'+x_key]==False]
                        # Sort the cluster average heat flux by the y_key variable
                        this_df = this_df.sort_values(by=y_key)
                        # Calculate the cumulative heat flux, starting with the lowest value of y_key
                        this_df['ca_FH_cumul'] = this_df['ca_FH'].cumsum()
            else:
                # Get data
                if True: #plot_slopes:
                    x_data = np.array(this_df[x_key].values, dtype=np.float64)
                    y_data = np.array(this_df[y_key].values, dtype=np.float64)
            
            # Plot every point the same color, size, and marker
            if plot_3d == False:
                # Plot in 2D
                ax.scatter(x_data, y_data, color=std_clr, s=m_size, marker=ahf.std_marker, alpha=m_alpha, zorder=5)
                # Output the x and y data, along with cluster id, to a pickle file
                # this_df[[x_key, y_key, 'cluster']].to_pickle('outputs/'+this_BGR+'_x_y_cluster_id.pickle')
                # Take moving average of the data
                if x_key in ['dt_start', 'dt_end'] and mv_avg:
                    print('\t- Taking moving average of the data')
                    # Make a copy of the dataframe
                    time_df = this_df.copy()
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
                        afnn_x_loc = x_mean+0.5*x_stdv
                        afnn_x_loc = min(x_bnds) + (1/3)*x_span
                        ax.annotate(line_label, xy=(afnn_x_loc,y_span/3+min(y_bnds)), xycoords='data', color=alt_std_clr, fontweight='bold', bbox=ahf.anno_bbox, zorder=12)
                        anno_text.set_fontsize(ahf.font_size_lgnd)
                        # Limit the y axis
                        ax.set_ylim((y_bnds[0],y_bnds[1]))
                    # Plot linear fit line
                    else: 
                        ahf.add_linear_slope(ax, pp, this_df, x_data, y_data, x_key, y_key, alt_std_clr, plot_slopes)
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
                ax.annotate(annotate_str, xy=(1+x_span/10,y_bnds[0]+y_span/5), xycoords='data', color=std_clr, fontweight='bold', alpha=0.5, bbox=ahf.anno_bbox, zorder=12)
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
                # afnn_x_loc = x_bnds[0]+0.5*x_span
                # ax.annotate(line_label, xy=(afnn_x_loc,y_bnds[0]+y_span/5), xycoords='data', color=std_clr, fontweight='bold', alpha=0.5, bbox=ahf.anno_bbox, zorder=12)
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
                tw_clr = get_var_color(tw_x_key)
                if tw_clr == std_clr:
                    tw_clr = alt_std_clr
                tw_ax_y.scatter(this_df[tw_x_key], this_df[y_key], color=tw_clr, s=m_size, marker=ahf.std_marker, alpha=m_alpha)
                tw_ax_y.set_xlabel(pp.xlabels[1])
                # Change color of the ticks on the twin axis
                tw_ax_y.tick_params(axis='x', colors=tw_clr)
            elif not isinstance(tw_y_key, type(None)):
                tw_clr = get_var_color(tw_y_key)
                if tw_clr == std_clr:
                    tw_clr = alt_std_clr
                tw_ax_x.scatter(this_df[x_key], this_df[tw_y_key], color=tw_clr, s=m_size, marker=ahf.std_marker, alpha=m_alpha)
                # Invert y-axis if specified
                if tw_y_key in ahf.y_invert_vars:
                    tw_ax_x.invert_yaxis()
                tw_ax_x.set_ylabel(pp.ylabels[1])
                # Change color of the ticks on the twin axis
                tw_ax_x.tick_params(axis='y', colors=tw_clr)
            # Add a standard legend
            if pp.legend:
                ahf.add_std_legend(ax, this_df, x_key, mark_LHW_AW=mark_LHW_AW)
            # Format the axes for datetimes, if necessary
            ahf.format_datetime_axes(pp.add_grid, x_key, y_key, ax, tw_x_key, tw_ax_y, tw_y_key, tw_ax_x, aug_zorder=4)
        elif clr_map == 'source':
            if plot_hist:
                return ahf.plot_histogram(a_group, ax, pp, this_df, x_key, y_key,clr_map, legend=pp.legend)
            # Get unique sources
            these_sources = np.unique(this_df['source'])
            # If there aren't that many points, make the markers bigger
            m_size, m_alpha = ahf.get_marker_size_and_alpha(len(this_df[x_key]))
            # if len(this_df[x_key]) < 1000:
            #     m_size = map_mrk_size
            # elif len(this_df[x_key]) < 100000:
            #     m_size = mrk_size2
            # else: 
            #     m_size = mrk_size
            i = 0
            lgnd_hndls = []
            for source in these_sources:
                # Decide on the color, don't go off the end of the array
                my_clr = distinct_clrs[i%len(distinct_clrs)]
                # Get the data for just this source
                this_source_df = this_df[this_df['source'] == source] 
                # Plot every point from this df the same color, size, and marker
                ax.scatter(this_source_df[x_key], this_source_df[y_key], color=my_clr, s=m_size, marker=ahf.std_marker, alpha=m_alpha, zorder=5)
                i += 1
                # Add legend to report the total number of points for this instrmt
                lgnd_label = source+': '+str(len(this_source_df[x_key]))+' points'
                lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label))
            # Invert y-axis if specified
            if y_key in ahf.y_invert_vars:
                invert_y_axis = True
            else:
                invert_y_axis = False
            notes_string = ''.join(this_source_df.notes.unique())
            # Only add the notes_string if it contains something
            if len(notes_string) > 1:
                notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
                lgnd_hndls.append(notes_patch)
            # Format the axes for datetimes, if necessary
            ahf.format_datetime_axes(pp.add_grid, x_key, y_key, ax)
            # Add legend with custom handles
            if pp.legend:
                lgnd = ax.legend(handles=lgnd_hndls)
            # Check whether to plot isopycnals
            if add_isos:
                add_isopycnals(ax, this_df, x_key, y_key, p_ref=isopycnals, place_isos=place_isos, tw_x_key=tw_x_key, tw_ax_y=tw_ax_y, tw_y_key=tw_y_key, tw_ax_x=tw_ax_x)
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
            # return pp.xlabels[0], pp.ylabels[0], plt_title, ax, invert_y_axis
        elif clr_map == 'clr_by_dataset':
            if plot_hist:
                return ahf.plot_histogram(a_group, ax, pp, this_df, x_key, y_key, clr_map, legend=pp.legend)
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
            ahf.format_datetime_axes(pp.add_grid, x_key, y_key, ax)
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
            # return pp.xlabels[0], pp.ylabels[0], plt_title, ax, invert_y_axis
        elif clr_map == 'instrmt':
            if plot_hist:
                return ahf.plot_histogram(a_group, ax, pp, this_df, x_key, y_key, clr_map, legend=pp.legend)
            # Add column where the source-instrmt combination ensures uniqueness
            this_df['source-instrmt'] = this_df['source']+' '+this_df['instrmt'].astype("string")
            # Get unique instrmts
            these_instrmts = np.unique(this_df['source-instrmt'])
            if len(these_instrmts) < 1:
                print('No instruments included, aborting script')
                exit(0)
            # If there aren't that many points, make the markers bigger
            m_size, m_alpha = ahf.get_marker_size_and_alpha(len(this_df[x_key]))
            # if len(this_df[x_key]) < 1000:
            #     m_size = map_mrk_size
            # elif len(this_df[x_key]) < 100000:
            #     m_size = mrk_size2
            # else: 
            #     m_size = mrk_size
            i = 0
            lgnd_hndls = []
            # Loop through each instrument
            for instrmt in these_instrmts:
                # Get the data for just this instrmt
                this_instrmt_df = this_df[this_df['source-instrmt'] == instrmt]
                # Decide on the color, don't go off the end of the array
                my_clr = distinct_clrs[i%len(distinct_clrs)]
                # Plot every point from this df the same color, size, and marker
                ax.scatter(this_instrmt_df[x_key], this_instrmt_df[y_key], color=my_clr, s=m_size, marker=ahf.std_marker, alpha=m_alpha, zorder=5)
                i += 1
                # Add legend to report the total number of points for this instrmt
                lgnd_label = instrmt+': '+str(len(this_instrmt_df[x_key]))+' points'
                lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label))
                notes_string = ''.join(this_instrmt_df.notes.unique())
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
            ahf.format_datetime_axes(pp.add_grid, x_key, y_key, ax)
            # Check whether to plot isopycnals
            if add_isos:
                add_isopycnals(ax, this_df, x_key, y_key, p_ref=isopycnals, place_isos=place_isos, tw_x_key=tw_x_key, tw_ax_y=tw_ax_y, tw_y_key=tw_y_key, tw_ax_x=tw_ax_x)
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
            # return pp.xlabels[0], pp.ylabels[0], plt_title, ax, invert_y_axis
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
            heatmap = ax.hist2d(this_df[x_key], this_df[y_key], bins=xy_bins, cmap=ahf.get_color_map(clr_map), cmin=clr_min, cmax=clr_max, norm=cbar_scale)
            # Invert y-axis if specified
            if y_key in ahf.y_invert_vars:
                invert_y_axis = True
            else:
                invert_y_axis = False
            # `hist2d` returns a tuple, the index 3 of which is the mappable for a colorbar
            cbar = plt.colorbar(heatmap[3], ax=ax, extend=clr_ext)
            cbar.set_label('points per pixel')
            # Add legend to report the total number of points and pixels on the plot
            n_pts_patch  = mpl.patches.Patch(color='none', label=str(len(this_df[x_key]))+' points')
            pixel_patch  = mpl.patches.Patch(color='none', label=str(xy_bins)+'x'+str(xy_bins)+' pixels')
            notes_string = ''.join(this_df.notes.unique())
            # Only add the notes_string if it contains something
            if pp.legend:
                if len(notes_string) > 1:
                    notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
                    ax.legend(handles=[n_pts_patch, pixel_patch, notes_patch])
                else:
                    ax.legend(handles=[n_pts_patch, pixel_patch])
            # Format the axes for datetimes, if necessary
            ahf.format_datetime_axes(pp.add_grid, x_key, y_key, ax)
            # Check whether to plot isopycnals
            if add_isos:
                add_isopycnals(ax, this_df, x_key, y_key, p_ref=isopycnals, place_isos=place_isos, tw_x_key=tw_x_key, tw_ax_y=tw_ax_y, tw_y_key=tw_y_key, tw_ax_x=tw_ax_x)
            # Check whether to change any axes to log scale
            if len(log_axes) == 3:
                if log_axes[0]:
                    ax.set_xscale('log')
                if log_axes[1]:
                    ax.set_yscale('log')
            # Add a standard title
            # plt_title = ahf.add_std_title(a_group)
            # return pp.xlabels[0], pp.ylabels[0], plt_title, ax, invert_y_axis
        elif clr_map == 'cluster':
            # Add a standard title
            # plt_title = ahf.add_std_title(a_group)
            #   Legend is handled inside plot_clusters()
            # Check for histogram
            if plot_hist:
                return ahf.plot_histogram(a_group, ax, pp, this_df, x_key, y_key, clr_map, legend=pp.legend, txk=tw_x_key, tay=tw_ax_y, tyk=tw_y_key, tax=tw_ax_x, clstr_dict={'m_pts':m_pts, 'rel_val':rel_val})
            # Format the dates if necessary (don't need this because it's done above)
            if x_key in ['dt_start', 'dt_end']:
                try:
                    # print(this_df[x_key])
                    this_df[x_key] = mpl.dates.date2num(this_df[x_key])
                except:
                    foo = 2
            if y_key in ['dt_start', 'dt_end']:
                this_df[y_key] = mpl.dates.date2num(this_df[y_key])
            if isinstance(z_key, type(None)):
                df_z_key = 0
                # Drop duplicates
                # this_df.drop_duplicates(subset=[x_key, y_key], keep='first', inplace=True)
                plot_3d = False
            elif z_key in ['dt_start', 'dt_end']:
                df_z_key = mpl.dates.date2num(this_df[z_key])
                # Drop duplicates
                this_df.drop_duplicates(subset=[x_key, y_key, z_key], keep='first', inplace=True)
                plot_3d = True
            else:
                df_z_key = this_df[z_key]
                # Drop duplicates
                this_df.drop_duplicates(subset=[x_key, y_key, z_key], keep='first', inplace=True)
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
            invert_y_axis, a_group.data_frames, foo0, foo1, foo2 = ahf.plot_clusters(a_group, ax, pp, this_df, x_key, y_key, z_key, None, None, None, clr_map, None, None, None, None, box_and_whisker=False, plot_slopes=plot_slopes)
            # Format the axes for datetimes, if necessary
            ahf.format_datetime_axes(pp.add_grid, x_key, y_key, ax)
            # Check whether to plot isopycnals
            if add_isos:
                add_isopycnals(ax, this_df, x_key, y_key, p_ref=isopycnals, place_isos=place_isos, tw_x_key=tw_x_key, tw_ax_y=tw_ax_y, tw_y_key=tw_y_key, tw_ax_x=tw_ax_x)
            # Check whether to change any axes to log scale
            if len(log_axes) == 3:
                if log_axes[0]:
                    ax.set_xscale('log')
                if log_axes[1]:
                    ax.set_yscale('log')
                if log_axes[2]:
                    ax.set_zscale('log')
            # return pp.xlabels[0], pp.ylabels[0], plt_title, ax, invert_y_axis
            #
        elif clr_map in np.array(this_df.columns):
            if pp.plot_scale == 'by_pf':
                if clr_map not in pf_vars:
                    print('Cannot use',clr_map,'with plot scale by_pf')
                    exit(0)
            # Set whether to plot in 3D
            if isinstance(z_key, type(None)):
                plot_3d = False
            else:
                df_z_key = this_df[z_key]
                plot_3d = True
            # Format color map dates if necessary
            if clr_map in ['dt_start', 'dt_end']:
                cmap_data = mpl.dates.date2num(this_df[clr_map])
            else:
                cmap_data = this_df[clr_map]
            # Get the colormap
            this_cmap = ahf.get_color_map(clr_map)
            #
            if plot_hist:
                return ahf.plot_histogram(a_group, ax, pp, this_df, x_key, y_key, clr_map=clr_map, legend=pp.legend)
            # If there aren't that many points, make the markers bigger
            m_size, m_alpha = ahf.get_marker_size_and_alpha(len(this_df[x_key]))
            # if len(this_df[x_key]) < 1000:
            #     m_size = map_mrk_size
            # elif len(this_df[x_key]) < 100000:
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
                heatmap = ax.scatter(this_df[x_key], this_df[y_key], zs=df_z_key, c=cmap_data, cmap=this_cmap, s=m_size, marker=ahf.std_marker, zorder=5)
                if plot_slopes:
                    # Fit a 2d polynomial to the z data
                    ahf.plot_polyfit2d(ax, pp, this_df[x_key], this_df[y_key], df_z_key)
            else:
                if plot_slopes and x_key=='lon' and y_key=='lat':
                    # Fit a 2d polynomial to the z data
                    ahf.plot_polyfit2d(ax, pp, this_df[x_key], this_df[y_key], this_df[clr_map], in_3D=False)
                    # Plot the scatter
                    heatmap = ax.scatter(this_df[x_key], this_df[y_key], c=cmap_data, cmap=this_cmap, s=m_size, marker=ahf.std_marker, zorder=5, alpha=ahf.map_alpha)
                elif plot_slopes:
                    # Plot the scatter
                    heatmap = ax.scatter(this_df[x_key], this_df[y_key], c=cmap_data, cmap=this_cmap, s=m_size, marker=ahf.std_marker, zorder=5, alpha=ahf.map_alpha)
                    # Plot a linear slope
                    ahf.add_linear_slope(ax, pp, this_df, this_df[x_key], this_df[y_key], x_key, y_key, alt_std_clr, plot_slopes)
                else:
                    # Plot the scatter
                    heatmap = ax.scatter(this_df[x_key], this_df[y_key], c=cmap_data, cmap=this_cmap, s=m_size, marker=ahf.std_marker, zorder=5)
            # Invert y-axis if specified
            if y_key in ahf.y_invert_vars:
                invert_y_axis = True
            else:
                invert_y_axis = False
            # Plot on twin axes, if specified
            if not isinstance(tw_x_key, type(None)):
                tw_clr = get_var_color(tw_x_key)
                if tw_clr == std_clr:
                    tw_clr = alt_std_clr
                # Add backing x to distinguish from main axis
                tw_ax_y.scatter(this_df[tw_x_key], this_df[y_key], color=tw_clr, s=m_size*10, marker='x', zorder=3)
                tw_ax_y.scatter(this_df[tw_x_key], this_df[y_key], c=cmap_data, cmap=this_cmap, s=m_size, marker=ahf.std_marker, zorder=4)
                tw_ax_y.set_xlabel(pp.xlabels[1])
                # Change color of the axis label on the twin axis
                tw_ax_y.xaxis.label.set_color(tw_clr)
                # Change color of the ticks on the twin axis
                tw_ax_y.tick_params(axis='x', colors=tw_clr)
                # Create the colorbar
                cbar = plt.colorbar(heatmap, ax=tw_ax_y)
            elif not isinstance(tw_y_key, type(None)):
                tw_clr = get_var_color(tw_y_key)
                if tw_clr == std_clr:
                    tw_clr = alt_std_clr
                # Add backing x to distinguish from main axis
                tw_ax_x.scatter(this_df[x_key], this_df[tw_y_key], color=tw_clr, s=m_size*10, marker='x', zorder=3)
                tw_ax_x.scatter(this_df[x_key], this_df[tw_y_key], c=cmap_data, cmap=this_cmap, s=m_size, marker=ahf.std_marker, zorder=4)
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
                ahf.add_std_legend(ax, this_df, x_key)
            # Format the axes for datetimes, if necessary
            ahf.format_datetime_axes(pp.add_grid, x_key, y_key, ax, tw_x_key, tw_ax_y, tw_y_key, tw_ax_x)
            # Check whether to plot isopycnals
            if add_isos:
                add_isopycnals(ax, this_df, x_key, y_key, p_ref=isopycnals, place_isos=place_isos, tw_x_key=tw_x_key, tw_ax_y=tw_ax_y, tw_y_key=tw_y_key, tw_ax_x=tw_ax_x)
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
            # return pp.xlabels[0], pp.ylabels[0], plt_title, ax, invert_y_axis
        # 
        dfs.append(this_df)
    #
    if clr_map == 'fnn':
        # Concatonate all the dataframes
        df = pd.concat(dfs)
        # Calculate the differences between the cluster and its forward nearest neighbors
        df[y_key+'_fnn_diff'] = df[y_key] - df[y_key+'_fnn_y']
        # Change the axis labels
        pp.ylabels[0] = r'NN-diff of ' + pp.ylabels[0]
        # Check for histogram
        if plot_hist:
            print('\t- Plotting histogram')
            x_label, y_label, z_label, plt_title, ax, invert_y_axis = ahf.plot_histogram(a_group, ax, pp, df, x_key, y_key+'_fnn_diff', clr_map, legend=pp.legend, txk=tw_x_key, tay=tw_y_key, tyk=tw_y_key, tax=tw_ax_x)
    elif clr_map == 'bnn':
        # Concatonate all the dataframes
        df = pd.concat(dfs)
        # Calculate the differences between the cluster and its backward nearest neighbors
        df[y_key+'_bnn_diff'] = df[y_key] - df[y_key+'_bnn_y']
        # Change the axis labels
        pp.ylabels[0] = r'NN-diff of ' + pp.ylabels[0]
        # Check for histogram
        if plot_hist:
            print('\t- Plotting histogram')
            x_label, y_label, z_label, plt_title, ax, invert_y_axis = ahf.plot_histogram(a_group, ax, pp, df, x_key, y_key+'_bnn_diff', clr_map, legend=pp.legend, txk=tw_x_key, tay=tw_y_key, tyk=tw_y_key, tax=tw_ax_x)
    # else:
    #     # Did not provide a valid clr_map
    #     print('Colormap',clr_map,'not valid')
    #     exit(0)
    # Check whether to connect clusters across periods
    if connect_p:
        # Concatonate all the dataframes
        df = pd.concat(dfs)
        # Color lines between pairs of clusters that are each others nearest neighbors
        # Add column to dataframe to track mutual nearest neighbors
        df[y_key+'_mnn_cid'] = np.nan
        df[y_key+'_mnn_x'] = np.nan
        df[y_key+'_mnn_y'] = np.nan
        # Loop over each dataset
        these_BGRs = np.unique(df['dataset'])
        n_BGRs = len(these_BGRs)
        dfs = []
        for b in range(n_BGRs-1):
            this_BGR = these_BGRs[b]
            print('- Plotting data from',this_BGR)
            # Get the data for this dataset
            this_df = df[df['dataset'] == this_BGR]
            these_clusters = np.unique(this_df['cluster'])
            # Get the data for the next dataset
            next_df = df[df['dataset'] == these_BGRs[b+1]]
            # Loop over each cluster
            for c in range(len(these_clusters)):
                this_cluster = these_clusters[c]
                # Get the fnn cluster id
                this_fnn_cid = this_df[this_df['cluster'] == this_cluster][y_key+'_fnn_cid'].values[0]
                # Find that cluster in the next dataset, check it's bnn cluster id
                next_bnn_cid = next_df[next_df['cluster'] == this_fnn_cid][y_key+'_bnn_cid'].values[0]
                # If that bnn cluster id is the same as the current cluster id, then they are mutual nearest neighbors
                if next_bnn_cid == this_cluster:
                    # First make a mask
                    this_cluster_mask = (this_df['cluster'] == this_cluster)
                    # Copy the fnn cid, x, and y to the mnn columns
                    this_df.loc[this_cluster_mask, y_key+'_mnn_cid'] = this_fnn_cid
                    this_df.loc[this_cluster_mask, y_key+'_mnn_x'] = this_df[this_df['cluster'] == this_cluster][y_key+'_fnn_x'].values[0]
                    this_df.loc[this_cluster_mask, y_key+'_mnn_y'] = this_df[this_df['cluster'] == this_cluster][y_key+'_fnn_y'].values[0]
            # Print this_df columns for cluster, fnn, and mnn
            # print(this_df[['cluster', y_key+'_fnn_cid', y_key+'_fnn_y', y_key+'_mnn_cid', y_key+'_mnn_y']].sort_values(by=['cluster']).to_string())
            if clr_map == 'cluster': 
                quiver_nn(ax, this_df, x_key, y_key, mnn=True)
    return pp.xlabels[0], pp.ylabels[0], plt_title, ax, invert_y_axis
    print('ERROR: Should not have gotten here. Aborting script')
    exit(0)

################################################################################

def quiver_nn(ax, this_df, x_key, y_key, fnn=True, bnn=True, mnn=False):
    """
    Plot arrows from each cluster to its nearest neighbor

    ax      : the axis to plot on
    this_df : the dataframe with the data
    x_key   : the x variable
    y_key   : the y variable
    fnn     : plot the forward nearest neighbors
    bnn     : plot the backward nearest neighbors
    mnn     : plot the mutual nearest neighbors
    """
    # Use the quiver function to plot arrows from each cluster to its nearest neighbor
    X_ = this_df[x_key].values.astype(np.float64)
    Y_ = this_df[y_key].values.astype(np.float64)
    if fnn:
        U_ = this_df[y_key+'_fnn_x'].values.astype(np.float64) - X_
        V_ = this_df[y_key+'_fnn_y'].values.astype(np.float64) - Y_
        ax.quiver(X_, Y_, U_, V_, angles='xy', scale_units='xy', scale=1, width=0.001, headwidth=5, headlength=7, fc=std_clr, alpha=0.5, zorder=1)
    if bnn:
        U_ = this_df[y_key+'_bnn_x'].values.astype(np.float64) - X_
        V_ = this_df[y_key+'_bnn_y'].values.astype(np.float64) - Y_
        ax.quiver(X_, Y_, U_, V_, angles='xy', scale_units='xy', scale=1, width=0.001, headwidth=5, headlength=7, fc='none', ec='b', hatch='|||||', alpha=1, zorder=3)
    if mnn:
        U_ = this_df[y_key+'_mnn_x'].values.astype(np.float64) - X_
        V_ = this_df[y_key+'_mnn_y'].values.astype(np.float64) - Y_
        ax.quiver(X_, Y_, U_, V_, angles='xy', scale_units='xy', scale=1, width=0.001, headwidth=5, headlength=7, fc='r', alpha=1, zorder=2)

################################################################################

this_clr_map = 'clr_all_same'
this_clr_map = 'cluster'

################################################################################
# Plots investigating nearest neighbors across periods
################################################################################
across_x_var = 'dt_start'
# Tracking nearest neighbors between periods
if True:
    print('')
    print('- Creating a plot showing arrows between nearest neighbors between periods')
    # Make the Plot Parameters
    pp_ca_SA_per_period = ahf.Plot_Parameters(x_vars=[across_x_var], y_vars=['ca_SA'], clr_map='cluster', legend=False, add_grid=False, extra_args={'mark_LR':False, 'connect_p':True})
    # Make the subplot groups
    group_ca_SA_per_period = Analysis_Group2([df], pp_ca_SA_per_period, plot_title=r'')
    # Make the figure
    make_figure([group_ca_SA_per_period])
# Histogram of differences between nearest neighbors between periods
if False:
    print('')
    print('- Creating a plot showing histograms of differences in nearest neighbors between periods')
    # Make the Plot Parameters
    pp_fnn_ca_SA_per_period = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['ca_SA'], clr_map='fnn', legend=True, extra_args={'n_h_bins':100})
    pp_bnn_ca_SA_per_period = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['ca_SA'], clr_map='bnn', legend=True, extra_args={'n_h_bins':100})
    # Make the subplot groups
    group_fnn_ca_SA_per_period = Analysis_Group2([df], pp_fnn_ca_SA_per_period, plot_title=r'Forward Nearest Neighbors')
    group_bnn_ca_SA_per_period = Analysis_Group2([df], pp_bnn_ca_SA_per_period, plot_title=r'Backward Nearest Neighbors')
    # Make the figure
    make_figure([group_fnn_ca_SA_per_period, group_bnn_ca_SA_per_period])
# Histograms of cluster spans and standard deviations
if False:
    print('')
    print('- Creating a plot showing histograms of cluster spans and standard deviations')
    # Make the Plot Parameters
    pp_span_per_period = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['cs_SA'], clr_map='clr_all_same', legend=True, extra_args={'n_h_bins':150, 'all_BGR':True}, ax_lims={'y_lims':[0, 0.04]})
    pp_std_per_period = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['csd_SA'], clr_map='clr_all_same', legend=True, extra_args={'n_h_bins':50, 'all_BGR':True}, ax_lims={'y_lims':[0, 0.04]})
    pp_nir_per_period = ahf.Plot_Parameters(x_vars=['hist'], y_vars=['nir_SA'], clr_map='clr_all_same', legend=True, extra_args={'n_h_bins':150, 'all_BGR':True}, ax_lims={'y_lims':[0, 4]})
    # Make the subplot groups
    group_span_per_period = Analysis_Group2([df], pp_span_per_period, plot_title=r'')
    group_std_per_period = Analysis_Group2([df], pp_std_per_period, plot_title=r'')
    group_nir_per_period = Analysis_Group2([df], pp_nir_per_period, plot_title=r'')
    # Make the figure
    make_figure([group_span_per_period, group_std_per_period, group_nir_per_period])
################################################################################
# Evaluating the clusters
#*# Plot of nir_SA and cRL for all clusters
if False:
    this_ca_var = 'ca_press'
    these_y_lims = [355,190]
    # this_ca_var = 'ca_SA'
    # these_y_lims = [bps.S_range_LHW_AW[1], bps.S_range_LHW_AW[0]]
    # Make the plot parameters
    pp_nir = ahf.Plot_Parameters(x_vars=['nir_SA'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':False, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'press']}, legend=False, ax_lims={'x_lims':[-5,160], 'y_lims':these_y_lims})
    pp_cRL = ahf.Plot_Parameters(x_vars=['cRL'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':True, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'press']}, legend=False, ax_lims={'x_lims':[-100,100], 'y_lims':these_y_lims})
    # Make the subplot groups
    group_nir = Analysis_Group2([df], pp_nir, plot_title='')#this_BGR)
    group_cRL = Analysis_Group2([df], pp_cRL, plot_title='')#this_BGR)
    # Make the figure
    make_figure([group_nir, group_cRL], row_col_list=[1,2, 0.48, 1.25], filename='f7_BGR_all_RL_IR_SA_vs_'+this_ca_var+'.pdf')
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
        group_press = Analysis_Group2([df], pp_press)
        group_SA = Analysis_Group2([df], pp_SA)
        group_CT = Analysis_Group2([df], pp_CT)
        group_sig = Analysis_Group2([df], pp_sig)
        # Make the figure
        make_figure([group_press, group_SA, group_CT, group_sig], row_col_list=[1,4, 0.3, 1.03])#, filename='fit4-trends_vs_'+this_ca_var+'_w_clrmap_'+this_clr_map+'.png')
        # make_figure([group_press_trends, group_SA_trends, group_CT_trends], row_col_list=[1,3, 0.35, 1.03], filename='fit-trends_vs_'+this_ca_var+'_w_clrmap_'+this_clr_map+'.png')
################################################################################
# The properties of the clusters
#*# Make plots of the polyfit2d trends in the cluster properties
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
        group_press_trends = Analysis_Group2([df], pp_press_trends)
        group_SA_trends = Analysis_Group2([df], pp_SA_trends)
        group_CT_trends = Analysis_Group2([df], pp_CT_trends)
        group_sig_trends = Analysis_Group2([df], pp_sig_trends)
        # Make the figure
        make_figure([group_press_trends, group_SA_trends, group_CT_trends, group_sig_trends], row_col_list=[1,4, 0.3, 1.03], filename='ca_vars_vs_'+this_ca_var+'_w_clrmap_'+this_clr_map+'.pdf')
        # make_figure([group_press_trends, group_SA_trends, group_CT_trends], row_col_list=[1,3, 0.35, 1.03], filename='fit-trends_vs_'+this_ca_var+'_w_clrmap_'+this_clr_map+'.png')
    these_ax_lims = None
################################################################################
# The trends of all the clusters
#*# Make plots of the polyfit2d trends in the cluster properties
plot_slopes = 'OLS'
these_ax_lims = None #{'y_lims':[355,190]}
add_legend = True
if False:
# for this_ca_var in ['ca_SA']:#, 'ca_press', 'ca_SA', 'ca_CT', 'ca_sigma']:
    for this_clr_map in ['clr_all_same', 'cluster']:
        # Make the Plot Parameters
        pp_press_trends = ahf.Plot_Parameters(x_vars=['trd_press-fit'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'cRL','nir_SA'], 'mark_LHW_AW':True}, ax_lims=these_ax_lims, legend=add_legend)
        pp_SA_trends = ahf.Plot_Parameters(x_vars=['trd_SA-fit'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'cRL','nir_SA'], 'mark_LHW_AW':True}, legend=False)
        pp_CT_trends = ahf.Plot_Parameters(x_vars=['trd_CT-fit'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'cRL','nir_SA'], 'mark_LHW_AW':True}, legend=False)
        # pp_sig_trends = ahf.Plot_Parameters(x_vars=['trd_sigma-fit'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'cRL','nir_SA'], 'mark_LHW_AW':True}, legend=False)
        # Make the subplot groups
        group_press_trends = Analysis_Group2([df], pp_press_trends)
        group_SA_trends = Analysis_Group2([df], pp_SA_trends)
        group_CT_trends = Analysis_Group2([df], pp_CT_trends)
        # group_sig_trends = Analysis_Group2([df], pp_sig_trends)
        # Make the figure
        # make_figure([group_press_trends, group_SA_trends, group_CT_trends, group_sig_trends], row_col_list=[1,4, 0.3, 1.03])#, filename='fit4-trends_vs_'+this_ca_var+'_w_clrmap_'+this_clr_map+'.png')
        make_figure([group_press_trends, group_SA_trends, group_CT_trends], row_col_list=[1,3, 0.35, 1.03], filename='f9_fit-trends_vs_'+this_ca_var+'_w_clrmap_'+this_clr_map+'.pdf')
    these_ax_lims = None
################################################################################
# Thicknesses of the layers
# Plot of pcs_press for all profiles against this_ca_var
# this_clr_map = 'clr_all_same'
this_clr_map = 'cluster'
# this_ca_var = 'ca_press'
# this_vert_var = 'press'
this_ca_var = 'ca_SA'
this_vert_var = 'SA'
if False:
    # Load the per-profile data
    df_per_pf = pl.load(open('outputs/'+this_BGR+'_pf_cluster_properties.pickle', 'rb'))
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
                these_y_lims = [35.02,34.1]
                these_x_lims = [0,8]
            elif this_clr_map == 'cluster':
                these_y_lims = [35.02,34.1]
                these_x_lims = [0,20]
        this_plt_title = ''
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
        group_nzpcs = Analysis_Group2([nzdf_per_pf], pp_nzpcs, plot_title=this_plt_title)
        # Make the figure
        make_figure([group_nzpcs], row_col_list=[1,1, 0.35, 1.1], filename='C10_ITP13_compare_to_Shibley2019fig6b.pdf')

    # Plot of trends in pcs_press for each cluster, with zero values and without
    if False:
        # Make the plot parameters
        pp_nzpcs = ahf.Plot_Parameters(x_vars=['nztrd_pcs_press'], y_vars=[this_ca_var], clr_map='clr_all_same', extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=True)
        pp_pcs = ahf.Plot_Parameters(x_vars=['trd_pcs_press'], y_vars=[this_ca_var], clr_map='clr_all_same', extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=True)#, ax_lims={'x_lims':[0.5,4.5], 'y_lims':[300,170]})
        # Make the subplot groups
        group_nzpcs = Analysis_Group2([df], pp_nzpcs, plot_title=this_BGR)
        group_pcs   = Analysis_Group2([df], pp_pcs, plot_title=this_BGR)
        # Make the figure
        make_figure([group_nzpcs, group_pcs])#, group_nzpcs_fit])#, row_col_list=[1,1, 0.8, 1.25])
    
    #*# Plots of thicknesses all layers, cluster average thickness, and trends in thickness
    if True:
        # Make the plot parameters
        """
        Note to myself: I've already taken out the outlier clusters in nzdf_per_pf
        """
        pp_nzpcs = ahf.Plot_Parameters(x_vars=['nzpcs_press'], y_vars=[this_vert_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':False, 'mark_outliers':False, 'plot_centroid':False, 'extra_vars_to_keep':['cluster', 'press']}, legend=False, ax_lims={'x_lims':these_x_lims, 'y_lims':these_y_lims})
        pp_ca_nzpcs = ahf.Plot_Parameters(x_vars=['nzca_pcs_press'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False, ax_lims={'x_lims':these_x_lims, 'y_lims':these_y_lims})
        # pp_trd_nzpcs = ahf.Plot_Parameters(x_vars=['nztrd_pcs_press'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False, ax_lims={'x_lims':[-2,2], 'y_lims':these_y_lims})
        pp_trd_nzpcs = ahf.Plot_Parameters(x_vars=['percnztrd_pcs_press'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False, ax_lims={'x_lims':[-40,40], 'y_lims':these_y_lims})
        # Make the subplot groups
        group_nzpcs    = Analysis_Group2([nzdf_per_pf], pp_nzpcs, plot_title='')
        group_ca_nzpcs = Analysis_Group2([df], pp_ca_nzpcs, plot_title='')
        group_trd_nzpcs= Analysis_Group2([df], pp_trd_nzpcs, plot_title='')
        # Make the figure
        make_figure([group_nzpcs, group_ca_nzpcs, group_trd_nzpcs], row_col_list=[1,3, 0.3, 1.02], filename='f10_BGR_all_layer_thickness_'+this_clr_map+'.png')# row_col_list=[1,2, 0.45, 1.2])
    
    # Plot of thickness for just one cluster
    this_clstr_id = 5
    if False:
        # Reduce nzdf_per_pf to only that cluster
        nzdf_per_pf = nzdf_per_pf[nzdf_per_pf['cluster'] == this_clstr_id]
        # Make the plot parameters
        pp_nzpcs = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['nzpcs_press'], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':False, 'plot_centroid':False, 'extra_vars_to_keep':['cluster', 'press']}, legend=False)#, ax_lims={'x_lims':these_x_lims, 'y_lims':these_y_lims})
        # Make the subplot groups
        group_nzpcs = Analysis_Group2([nzdf_per_pf], pp_nzpcs, plot_title='Cluster '+str(this_clstr_id))
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
        group_pcs_map = Analysis_Group2([nzdf_per_pf], pp_pcs_map, plot_title='')
        group_pcs_hist = Analysis_Group2([nzdf_per_pf], pp_pcs_hist, plot_title='')
        # Make the figure
        make_figure([group_pcs_map, group_pcs_hist], filename='C11_BGR_all_nzpcs_press_mean_per_pf.pdf')
################################################################################
# Calculating heat flux in W/m^2
# Plots of cluster averages of components of the heat flux
#   height, isobaric heat capacity, temperature trend in time, density
this_ca_var = 'ca_SA'
this_clr_map = 'clr_all_same'
if False:
    # Make the Plot Parameters
    pp_height = ahf.Plot_Parameters(x_vars=['nzca_pcs_press'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False)
    pp_cp = ahf.Plot_Parameters(x_vars=['ca_cp'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False)
    pp_trd_CT = ahf.Plot_Parameters(x_vars=['trd_CT-fit'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False)
    pp_rho = ahf.Plot_Parameters(x_vars=['ca_rho'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False)
    # Make the subplot groups
    group_height = Analysis_Group2([df], pp_height, plot_title='')
    group_cp = Analysis_Group2([df], pp_cp, plot_title='')
    group_trd_CT = Analysis_Group2([df], pp_trd_CT, plot_title='')
    group_rho = Analysis_Group2([df], pp_rho, plot_title='')
    # Make the figure
    # make_figure([group_height, group_cp])
    make_figure([group_height, group_cp, group_trd_CT, group_rho], row_col_list=[2,2, 0.6, 1.6], filename='C12_BGR_all_FH_comps_vs_SA.pdf')
# Plot of the heat flux in W/m^2
if False:
    # Make the Plot Parameters
    pp_FH = ahf.Plot_Parameters(x_vars=['ca_FH'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':'OLS', 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False)
    pp_FH_cumul = ahf.Plot_Parameters(x_vars=['ca_FH_cumul'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':False, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'press', 'cRL']}, legend=False)
    # Make the subplot groups
    group_FH = Analysis_Group2([df], pp_FH, plot_title='')
    group_FH_cumul = Analysis_Group2([df], pp_FH_cumul, plot_title='')
    # Make the figure
    make_figure([group_FH, group_FH_cumul], filename='f11_FH_and_cumul_vs_SA.pdf')#, row_col_list=[1,1, 0.3, 1.04])


exit(0)

plot_slopes = 'OLS'
filename_prefix = '2024-05-13_'
# Make plots of the polyfit2d trends in the cluster properties
for this_ca_var in ['ca_press', 'ca_SA', 'ca_CT']:
    # Make the Plot Parameters
    pp_press_trends = ahf.Plot_Parameters(x_vars=['trd_press-fit'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'cRL','nir_SA']}, legend=False)
    pp_SA_trends = ahf.Plot_Parameters(x_vars=['trd_SA-fit'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'cRL','nir_SA']}, legend=False)
    pp_CT_trends = ahf.Plot_Parameters(x_vars=['trd_CT-fit'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'cRL','nir_SA']}, legend=False)
    # Make the subplot groups
    group_press_trends = Analysis_Group2([df], pp_press_trends)
    group_SA_trends = Analysis_Group2([df], pp_SA_trends)
    group_CT_trends = Analysis_Group2([df], pp_CT_trends)
    # Make the figure
    make_figure([group_press_trends, group_SA_trends, group_CT_trends], row_col_list=[1,3, 0.35, 1.02], filename=filename_prefix+'fit-trends_vs_'+this_ca_var+'_w_clrmap_'+this_clr_map+'.png')
# Make plots of the trends in the cluster properties
# if False:
for this_ca_var in ['ca_press', 'ca_SA', 'ca_CT']:
    # Make the Plot Parameters
    pp_press_trends = ahf.Plot_Parameters(x_vars=['trd_press'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'cRL','nir_SA']}, legend=False)
    pp_SA_trends = ahf.Plot_Parameters(x_vars=['trd_SA'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'cRL','nir_SA']}, legend=False)
    pp_CT_trends = ahf.Plot_Parameters(x_vars=['trd_CT'], y_vars=[this_ca_var], clr_map=this_clr_map, extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':plot_slopes, 'mark_outliers':True, 'extra_vars_to_keep':['cluster', 'cRL','nir_SA']}, legend=False)
    # Make the subplot groups
    group_press_trends = Analysis_Group2([df], pp_press_trends)
    group_SA_trends = Analysis_Group2([df], pp_SA_trends)
    group_CT_trends = Analysis_Group2([df], pp_CT_trends)
    # Make the figure
    make_figure([group_press_trends, group_SA_trends, group_CT_trends], row_col_list=[1,3, 0.35, 1.02], filename=filename_prefix+'trends_vs_'+this_ca_var+'_w_clrmap_'+this_clr_map+'.png')


