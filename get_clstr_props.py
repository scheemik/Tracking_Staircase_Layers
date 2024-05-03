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
import pandas as pd
# For adding subplot labels a, b, c, ...
import string
import analysis_helper_functions as ahf
import dill as pl

# For common BGR parameters
import BGR_params as bps
# For common BGR objects
import BGR_objects as bob

# Specify which BGR data set to use
this_BGR = 'BGR1516'

################################################################################
# Declare variables for plotting
################################################################################
dark_mode = True

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

# Make the data set
ds_this_BGR = ahf.Data_Set(bps.BGR_HPC_clstrd_dict[this_BGR], bob.dfs_all)
# Make the profile filters
pfs_0 = ahf.Profile_Filters()
# Make the plot parameters
pp_ = ahf.Plot_Parameters(x_vars=['cRL'], y_vars=['clst_prob'], extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':False, 'mark_outliers':False, 'extra_vars_to_keep':['dt_start', 'cluster', 'press', 'SA', 'CT']}, legend=False)
# Make the subplot groups
group_ = ahf.Analysis_Group(ds_this_BGR, pfs_0, pp_)

# Concatonate all the pandas data frames together
df = pd.concat(group_.data_frames)

# Make a list of variables to calculate
calc_vars = ['n_points', 'cRL']
for var in ['press', 'SA', 'CT']:
    calc_vars.append('ca_'+var)
    calc_vars.append('nir_'+var)
    calc_vars.append('trd_'+var)
# Calculate new cluster variables
df = ahf.calc_extra_cl_vars(group_.data_frames[0], calc_vars)

# Drop duplicates to get one row per cluster
df = df.drop_duplicates(subset=['cluster'])
# Drop Time dimension
df = df.droplevel('Time')
# Drop the columns for dt_start and entry
df = df.drop(columns=['dt_start', 'entry'])
# Set this as the new data frame in the analysis group
group_.data_frames = [df]

print('df:')
print(group_.data_frames[0])

# Pickle the data frame to a file
pl.dump(df, open('outputs/'+this_BGR+'_cluster_properties.pickle', 'wb'))

exit(0)

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
                 'hist':r'Occurrences',
                 'press':r'Pressure (dbar)',
                 'SA':r'$S_A$ (g/kg)',
                 'CT':r'$\Theta$ ($^\circ$C)',
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
    # Get x axis labels
    if not isinstance(pp.x_vars, type(None)):
        for i in range(len(pp.x_vars)):
            pp.xlabels[i] = make_var_label(pp.x_vars[i], ax_labels)
    else:
        pp.xlabels[0] = None
        pp.xlabels[1] = None
    # Get y axis labels
    if not isinstance(pp.y_vars, type(None)):
        for i in range(len(pp.y_vars)):
            pp.ylabels[i] = make_var_label(pp.y_vars[i], ax_labels)
    else:
        pp.ylabels[0] = None
        pp.ylabels[1] = None
    return pp

def make_var_label(var_key, ax_labels):
    """
    Takes in a variable key and returns the corresponding label

    var_key         A string of the variable key
    ax_labels       A dictionary of variable keys and their labels
    """
    # Check for certain modifications to variables,
    #   check longer strings first to avoid mismatching
    # Check for profile cluster average variables
    if 'pca_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'CA/P of '+ ax_labels[var_str]
        # return 'Profile cluster average of '+ ax_labels[var_str]
    # Check for profile cluster span variables
    if 'pcs_' in var_key:
        # Take out the first 4 characters of the string to leave the original variable name
        var_str = var_key[4:]
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
        return r'Normalized inter-cluster range $IR_{S_A}$'
        # return r'Normalized inter-cluster range $IR$ of '+ ax_labels[var_str]
    # Check for trend variables
    elif 'trd_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[4:]
        # Check whether also normalizing by subtracting a polyfit2d
        if '-fit' in var_str:
            # Take out the first 3 characters of the string to leave the original variable name
            var_str = var_str[:-4]
            return 'Trend in '+ ax_labels[var_str] + ' - polyfit2d'
        else:  
            return 'Trend in '+ ax_labels[var_str] + '/year'
    # Check for the polyfit2d of a variable
    elif 'fit_' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[4:]
        return 'polyfit2d of ' + ax_labels[var_str]
    # Check for variables normalized by subtracting a polyfit2d
    elif '-fit' in var_key:
        # Take out the first 3 characters of the string to leave the original variable name
        var_str = var_key[:-4]
        return ax_labels[var_str] + ' - polyfit2d'
    if var_key in ax_labels.keys():
        return ax_labels[var_key]
    else:
        return 'None'

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

def make_figure(groups_to_plot, filename=None, use_same_x_axis=False, use_same_y_axis=None, row_col_list=None):
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
                      '3':[3,1, 0.3, 1.70], '4':[2,2, 0.7, 2.00],
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
    tight_layout_h_pad = 1.0 #None
    tight_layout_w_pad = 1.0 #None
    if n_subplots == 1:
        if isinstance(row_col_list, type(None)):
            rows, cols, f_ratio, f_size = n_row_col_dict[str(n_subplots)]
        else:
            rows, cols, f_ratio, f_size = row_col_list
        n_subplots = int(np.floor(n_subplots))
        fig, ax = ahf.set_fig_axes([1], [1], fig_ratio=f_ratio, fig_size=f_size)
        xlabel, ylabel, plt_title, ax, invert_y_axis = make_subplot(ax, groups_to_plot[0])#, fig, 111)
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
        fig, axes = ahf.set_fig_axes([1]*rows, [1]*cols, fig_ratio=f_ratio, fig_size=f_size, share_x_axis=use_same_x_axis, share_y_axis=use_same_y_axis)
        for i in range(n_subplots):
            print('- Subplot '+string.ascii_lowercase[i])
            if rows > 1 and cols > 1:
                i_ax = (i//cols,i%cols)
            else:
                i_ax = i
            ax_pos = int(str(rows)+str(cols)+str(i+1))
            xlabel, ylabel, plt_title, ax, invert_y_axis = make_subplot(axes[i_ax], groups_to_plot[i])#, fig, ax_pos)
            if use_same_x_axis:
                # If on the top row
                if i < cols == 0:
                    tight_layout_h_pad = -1
                    ax.set_title(plt_title)
                # If in the bottom row
                elif i >= n_subplots-cols:
                    ax.set_xlabel(xlabel)
            else:
                ax.set_title(plt_title)
                ax.set_xlabel(xlabel)
            ax.set_title(plt_title)
            if use_same_y_axis:
                # If in the far left column
                if i%cols == 0:
                    tight_layout_w_pad = -1
                    ax.set_ylabel(ylabel)
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
            # Label subplots a, b, c, ...
            ax.text(-0.1, -0.1, '('+string.ascii_lowercase[i]+')', transform=ax.transAxes, size=mpl.rcParams['axes.labelsize'], fontweight='bold')
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
        if '.png' in filename:
            plt.savefig('outputs/'+filename, dpi=1000)
        elif '.pickle' in filename:
            pl.dump(fig, open('outputs/'+filename, 'wb'))
        else:
            print('File extension not recognized in',filename)
    else:
        print('- Displaying figure')
        plt.show()

################################################################################

def make_subplot(ax, a_group):#, fig, ax_pos):
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
    # plot_type = pp.plot_type
    clr_map   = pp.clr_map
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
    # Check for cluster-based variables
    # if x_key in clstr_vars or y_key in clstr_vars or z_key in clstr_vars or tw_x_key in clstr_vars or tw_y_key in clstr_vars or clr_map in clstr_vars:
    #     m_pts, m_cls, cl_x_var, cl_y_var, cl_z_var, plot_slopes, b_a_w_plt = get_cluster_args(pp)
    #     df, rel_val, m_pts, m_cls, ell = HDBSCAN_(a_group.data_set.arr_of_ds, df, cl_x_var, cl_y_var, cl_z_var, m_pts, m_cls=m_cls, extra_cl_vars=[x_key,y_key,z_key,tw_x_key,tw_y_key,clr_map], re_run_clstr=re_run_clstr)
    # print('\t- Plot slopes:',plot_slopes)
    # Check whether to normalize by subtracting a polyfit2d
    # if fit_vars:
    #     df = calc_fit_vars(df, (x_key, y_key, z_key, clr_map), fit_vars)
    # Determine the color mapping to be used
    if clr_map == 'clr_all_same':
        # Check for histogram
        if plot_hist:
            return plot_histogram(a_group, ax, pp, df, x_key, y_key, clr_map, legend=pp.legend, txk=tw_x_key, tay=tw_ax_y, tyk=tw_y_key, tax=tw_ax_x)
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
        # if len(df[x_key]) < 1000:
        #     m_size = map_mrk_size
        # elif len(df[x_key]) < 100000:
        #     m_size = mrk_size2
        # else: 
        #     m_size = mrk_size
        # Plot every point the same color, size, and marker
        if plot_3d == False:
            # Plot in 2D
            ax.scatter(df[x_key], df[y_key], color=std_clr, s=m_size, marker=ahf.std_marker, alpha=m_alpha, zorder=5)
            # Take moving average of the data
            if x_key in ['dt_start', 'dt_end'] and mv_avg:
                print('\t- Taking moving average of the data')
                # Need to convert dt_start back to datetime to take rolling average
                df[x_key] = mpl.dates.num2date(df[x_key])
                # Make a different version of the dataframe with dt_start as the index
                time_df = df.set_index(x_key)
                # Sort by the dt_start index
                time_df.sort_index(inplace=True)
                # Take the moving average of the data
                time_df[y_key+'_mv_avg'] = time_df[y_key].rolling(mv_avg, min_periods=1).mean()
                # Convert the original dataframe's dt_start back to numbers for plotting
                df[x_key] = mpl.dates.date2num(df[x_key])
                time_df.reset_index(level=x_key, inplace=True)
                time_df[x_key] = mpl.dates.date2num(time_df[x_key])
                ax.plot(df[x_key], time_df[y_key+'_mv_avg'], color='b', zorder=6)
        else:
            # Plot in 3D
            ax.scatter(df[x_key], df[y_key], zs=df_z_key, color=std_clr, s=m_size, marker=std_marker, alpha=m_alpha, zorder=5)
        if False:#plot_slopes:
            # Mark outliers
            print('\t- Marking outliers:',mrk_outliers)
            if mrk_outliers:
                if 'trd_' in x_key:
                    other_out_vars = ['cRL', 'nir_SA']
                else:
                    other_out_vars = []
                mark_outliers(ax, df, x_key, y_key, clr_map, mrk_for_other_vars=other_out_vars)
                # Get data without outliers
                x_data = np.array(df[df['out_'+x_key]==False][x_key].values, dtype=np.float64)
                y_data = np.array(df[df['out_'+x_key]==False][y_key].values, dtype=np.float64)
            else:
                # Get data
                x_data = np.array(df[x_key].values, dtype=np.float64)
                y_data = np.array(df[y_key].values, dtype=np.float64)
            if plot_3d == False:
                add_linear_slope(ax, df, x_data, y_data, x_key, y_key, alt_std_clr, plot_slopes)
            else:
                # Fit a 2d polynomial to the z data
                plot_polyfit2d(ax, pp, x_data, y_data, df_z_key)
        # Invert y-axis if specified
        if y_key in ahf.y_invert_vars:
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
        # Add a standard legend
        if pp.legend:
            add_std_legend(ax, df, x_key)
        # Format the axes for datetimes, if necessary
        ahf.format_datetime_axes(x_key, y_key, ax, tw_x_key, tw_ax_y, tw_y_key, tw_ax_x)
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
        # Invert y-axis if specified
        if y_key in y_invert_vars:
            invert_y_axis = True
        notes_string = ''.join(this_df.notes.unique())
        # Only add the notes_string if it contains something
        if len(notes_string) > 1:
            notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
            lgnd_hndls.append(notes_patch)
        # Format the axes for datetimes, if necessary
        format_datetime_axes(x_key, y_key, ax)
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
        return pp.xlabels[0], pp.ylabels[0], pp.zlabels[0], plt_title, ax, invert_y_axis
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
                    ax.errorbar(this_df[x_key], this_df[y_key], xerr=this_df[x_err_key], color=my_clr, capsize=l_cap_size, fmt='none')
                if not isinstance(y_err_key, type(None)):
                    ax.errorbar(this_df[x_key], this_df[y_key], yerr=this_df[y_err_key], color=my_clr, capsize=l_cap_size, fmt='none')
            # Add legend to report the total number of points for this instrmt
            lgnd_label = df_labels[i]+': '+str(len(this_df[x_key]))+' points'
            lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=lgnd_label))
            i += 1
        # Invert y-axis if specified
        if y_key in y_invert_vars:
            invert_y_axis = True
        notes_string = ''.join(this_df.notes.unique())
        # Only add the notes_string if it contains something
        if len(notes_string) > 1:
            notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
            lgnd_hndls.append(notes_patch)
        # Format the axes for datetimes, if necessary
        format_datetime_axes(x_key, y_key, ax)
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
        # Invert y-axis if specified
        if y_key in y_invert_vars:
            invert_y_axis = True
        # Only add the notes_string if it contains something
        if len(notes_string) > 1:
            notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
            lgnd_hndls.append(notes_patch)
        # Add legend with custom handles
        if pp.legend:
            lgnd = ax.legend(handles=lgnd_hndls)
        # Format the axes for datetimes, if necessary
        format_datetime_axes(x_key, y_key, ax)
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
        heatmap = ax.hist2d(df[x_key], df[y_key], bins=xy_bins, cmap=get_color_map(clr_map), cmin=clr_min, cmax=clr_max, norm=cbar_scale)
        # Invert y-axis if specified
        if y_key in y_invert_vars:
            invert_y_axis = True
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
        format_datetime_axes(x_key, y_key, ax)
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
        return pp.xlabels[0], pp.ylabels[0], pp.zlabels[0], plt_title, ax, invert_y_axis
    elif clr_map == 'cluster':
        # Add a standard title
        # plt_title = ahf.add_std_title(a_group)
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
        format_datetime_axes(x_key, y_key, ax)
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
                add_linear_slope(ax, df, df[x_key], df[y_key], x_key, y_key, alt_std_clr, plot_slopes)
            else:
                # Plot the scatter
                heatmap = ax.scatter(df[x_key], df[y_key], c=cmap_data, cmap=this_cmap, s=m_size, marker=std_marker, zorder=5)
        # Invert y-axis if specified
        if y_key in y_invert_vars:
            invert_y_axis = True
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
        cbar.set_label(pp.clabel)
        # Add a standard legend
        if pp.legend:
            add_std_legend(ax, df, x_key)
        # Format the axes for datetimes, if necessary
        format_datetime_axes(x_key, y_key, ax, tw_x_key, tw_ax_y, tw_y_key, tw_ax_x)
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
        return pp.xlabels[0], pp.ylabels[0], pp.zlabels[0], plt_title, ax, invert_y_axis
    else:
        # Did not provide a valid clr_map
        print('Colormap',clr_map,'not valid')
        exit(0)
    #
    print('ERROR: Should not have gotten here. Aborting script')
    exit(0)

################################################################################

# Make the plot parameters
pp_test = ahf.Plot_Parameters(x_vars=['trd_SA'], y_vars=['ca_press'], extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':False, 'mark_outliers':False, 'extra_vars_to_keep':['cluster', 'press']}, legend=False)
pp_test2 = ahf.Plot_Parameters(x_vars=['trd_CT'], y_vars=['ca_press'], extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':False, 'mark_outliers':False, 'extra_vars_to_keep':['cluster', 'press']}, legend=False)
# Make the subplot groups
group_test = Analysis_Group2([df], pp_test, plot_title='This is a Test Title')
group_test2 = Analysis_Group2([df], pp_test2, plot_title='This is also a Test Title')
# Make the figure
make_figure([group_test, group_test2])#, row_col_list=[1,1, 0.8, 1.25])
