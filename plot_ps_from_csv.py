"""
Author: Mikhail Schee
Created: 2023-02-27

This script will take in the name of a csv file and plot the data within it
I specifically made this to do parameter sweeps that take too long to plot
with the functions I made in `analysis_helper_functions.py`

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions in source code must retain the accompanying copyright notice, this list of conditions, and the following disclaimer.
    2. Redistributions in binary form must reproduce the accompanying copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Names of the copyright holders must not be used to endorse or promote products derived from this software without prior written permission from the copyright holders.
    4. If any files are modified, you must cause the modified files to carry prominent notices stating that you changed the files and the date of any change.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Usage:
    plot_cs_from_csv.py CSV1 [CSV2]

Arguments:
    CSV1            # filepath of the csv to plot
    CSV2            # filepath of second optional csv to plot
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
# For adding subplot labels a, b, c, ...
import string
import analysis_helper_functions as ahf
import dill as pl
# Parse input parameters
from docopt import docopt
args = docopt(__doc__)
my_csv = args['CSV1']       # filename of the pickle to unpickle

# Try to load the specified csv
print('- Loading '+my_csv)
try:
    df = pd.read_csv(my_csv, header=2)
except:
    print('Could not load '+my_csv)
    exit(0)
# Check for second csv
try:
    my_csv2 = args['CSV2']
    print('- Loading '+my_csv2)
    try:
        df2 = pd.read_csv(my_csv2, header=2)
    except:
        print('Could not load '+my_csv2)
        exit(0)
except:
    my_csv2 = None
    df2 = None

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
# Plotting functions ###########################################################
################################################################################

class Analysis_Group:
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

def get_axis_labels(pp):
    """
    Adds in the x and y labels based on the plot variables in the 
    Plot_Parameters object

    pp              A custom Plot_Parameters object
    """
    # Get x axis labels
    if not isinstance(pp.x_vars, type(None)):
        for i in range(len(pp.x_vars)):
            if pp.x_vars[i] == 'm_pts':
                pp.xlabels[i] = r'Minimum density threshold $m_{pts}$'
            elif pp.x_vars[i] == 'ell_size':
                pp.xlabels[i] = r'Moving average window $\ell$ (dbar)'
    else:
        pp.xlabels[0] = None
        pp.xlabels[1] = None
    # Get y axis labels
    if not isinstance(pp.y_vars, type(None)):
        for i in range(len(pp.y_vars)):
            if pp.y_vars[i] == 'n_clusters':
                pp.ylabels[i] = r'Number of clusters'
            elif pp.y_vars[i] == 'DBCV':
                pp.ylabels[i] = r'DBCV'
    else:
        pp.ylabels[0] = None
        pp.ylabels[1] = None
    return pp

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
        xlabel, ylabel, plt_title, ax = make_subplot(ax, groups_to_plot[0])#, fig, 111)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
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
            xlabel, ylabel, plt_title, ax = make_subplot(axes[i_ax], groups_to_plot[i])#, fig, ax_pos)
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
            else:
                ax.set_ylabel(ylabel)
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
    # clr_map   = pp.clr_map
    # Concatonate all the pandas data frames together
    df = pd.concat(a_group.data_frames)
    ## Make a standard x vs. y scatter plot
    # Set the main x and y data keys
    x_key = pp.x_vars[0]
    y_key = pp.y_vars[0]
    # Check for twin data key
    try:
        tw_y_key = pp.y_vars[1]
        tw_ax_x  = ax.twinx()
    except:
        tw_y_key = None
        tw_ax_x  = None
    # Add a standard title
    plt_title = ahf.add_std_title(a_group)
    # Plot the parameter sweep
    xlabel, ylabel = plot_clstr_param_sweep(ax, tw_ax_x, df, pp, plt_title)
    return xlabel, ylabel, plt_title, ax

################################################################################

def plot_clstr_param_sweep(ax, tw_ax_x, df, pp, plt_title=None):
    """
    Plots the number of clusters found by HDBSCAN vs. the number of profiles
    included in the data set

    ax              The axis on which to make the plot
    df              A pandas DataFrame
    pp              A custom Plot Parameters object
    """
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
    # Check for a z variable
    try:
        z_key = cluster_plt_dict['z_var']
    except:
        z_key = None
    # Set x and y labels
    ax.set_xlabel(pp.xlabels[0])
    ax.set_ylabel(pp.ylabels[0])
    # Find the different values of the z variable
    z_list = np.unique(df[z_key])
    # Loop through the different z values in the list
    for i in range(len(z_list)):
        # Set z label
        if z_key == 'ell_size':
            zlabel = r'$\ell=$'+str(z_list[i])+' dbar'
        elif z_key == 'm_pts':
            zlabel = r'$m_{pts}=$: '+str(z_list[i])
        # Plot main axis
        ax.plot(df[x_key], df[y_key], color=std_clr, linestyle=l_styles[i], label=zlabel)
        if tw_y_key:
            # Plot twin axis
            tw_ax_x.plot(df[x_key], df[tw_y_key], color=alt_std_clr, linestyle=l_styles[i])
            # tw_ax_x.set_ylabel(tw_ylabel)
            tw_ax_x.set_ylabel(pp.ylabels[1])
            # Change color of the axis label on the twin axis
            tw_ax_x.yaxis.label.set_color(alt_std_clr)
            # Change color of the ticks on the twin axis
            tw_ax_x.tick_params(axis='y', colors=alt_std_clr)
    if z_key:
        ax.legend()
    return pp.xlabels[0], pp.ylabels[0]

################################################################################

# Assemble plot title from the file names
# Split the prefix from the original variable (assumes an underscore split)
split_name = my_csv.split('_', 1)
plt_title = split_name[0].split('/',1)[1]
if not isinstance(my_csv2, type(None)):
    split_name = my_csv2.split('_', 1)
    plt_title = plt_title + ' and ' + split_name[0].split('/',1)[1]
# plt_title = 'BGR'

pp_ps_m_pts = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'z_var':'ell_size'})
pp_ps_ell = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'z_var':'m_pts'})

group_param_sweep1 = Analysis_Group([df], pp_ps_m_pts, plot_title=plt_title)
if not isinstance(df2, type(None)):
    group_param_sweep2 = Analysis_Group([df2], pp_ps_ell, plot_title=plt_title)
    make_figure([group_param_sweep1, group_param_sweep2])#, filename='new_BGOS_test_sweep.pickle')
else:
    make_figure([group_param_sweep1])#, filename='new_BGOS_test_sweep.pickle')