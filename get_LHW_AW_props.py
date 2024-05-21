"""
Author: Mikhail Schee
Created: 2024-05-20

This script will calculate certain properties of the LHW and AW for the BGR data set and save the 
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
# this_BGR = 'BGR1516'
this_BGR = 'BGR_all'

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
ds_this_BGR = ahf.Data_Set(bps.BGRITPs_dict[this_BGR], bob.dfs1)
# Make the plot parameters
pp_ = ahf.Plot_Parameters(x_vars=['lon'], y_vars=['lat'], extra_args={'re_run_clstr':False, 'sort_clstrs':False, 'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':False, 'mark_outliers':False, 'extra_vars_to_keep':['dt_start', 'lon', 'lat', 'press_TC_max', 'press_TC_min', 'SA_TC_max', 'SA_TC_min', 'CT_TC_max', 'CT_TC_min', 'sig_TC_max', 'sig_TC_min']}, legend=False)
# Make the subplot groups
group_ = ahf.Analysis_Group(ds_this_BGR, bob.pfs_LHW_and_AW, pp_)

# Concatonate all the pandas data frames together
df = pd.concat(group_.data_frames)

print('df:')
print(df)
# exit(0)

# Calculate fit variables over lat-lon
print('Calculating fit variables over lat-lon')
df = ahf.calc_fit_vars(df, ('press_TC_max-fit', 'press_TC_min-fit', 'SA_TC_max-fit', 'SA_TC_min-fit', 'CT_TC_max-fit', 'CT_TC_min-fit', 'sig_TC_max-fit', 'sig_TC_min-fit'), ['lon','lat'])
print(df.columns)

# Make a list of variables to calculate
calc_vars = []
for var in ['press', 'SA', 'CT', 'sig']:
    for ext in ['_TC_max', '_TC_min']:
        calc_vars.append('av_'+var+ext)
        calc_vars.append('atrd_'+var+ext)
        calc_vars.append('atrd_'+var+ext+'-fit')
# Calculate new cluster variables
print('Calculating LHW and AW properties')
df = ahf.calc_extra_cl_vars(df, calc_vars)

# # Drop duplicates to get one row per cluster
# df = df.drop_duplicates(subset=['cluster'])
# # Drop Time dimension
# df = df.droplevel('Time')
# # Drop the columns for dt_start and entry
# df = df.drop(columns=['dt_start', 'entry'])

# Remove the `a` from all the column names with `atrd_`
df = df.rename(columns=lambda x: x.replace('atrd_', 'trd_'))

print('df:')
print(df)
print(df.columns)

# Pickle the data frame to a file
pl.dump(df, open('outputs/'+this_BGR+'_LHW_AW_properties.pickle', 'wb'))
