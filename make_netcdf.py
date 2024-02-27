"""
Author: Mikhail Schee
Created: 2022-03-24

This script will load data from various Arctic Ocean profile datasets and
reformat them into netcdf files, one for each instrument.
This script expects the following file structure of the raw data files:

science_data_file_path/
    ITPs/
        itp1/
            itp1cormat/
                cor0001.mat
                cor0002.mat
                ...
            itp1final/
                itp1grd0001.dat
                itp1grd0002.dat
                ...
        itp2/
            ...
        ...

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions in source code must retain the accompanying copyright notice, this list of conditions, and the following disclaimer.
    2. Redistributions in binary form must reproduce the accompanying copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Names of the copyright holders must not be used to endorse or promote products derived from this software without prior written permission from the copyright holders.
    4. If any files are modified, you must cause the modified files to carry prominent notices stating that you changed the files and the date of any change.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
# For searching and listing directories
import os
# For reading the ITP `cormat` files
import mat73
from scipy import io
# For reading netcdf files
import netCDF4 as netcdf
# For matching regular expressions
import re
# Import the Thermodynamic Equation of Seawater 2010 (TEOS-10) from GSW
# For converting from depth to pressure
import gsw

# For test plots
import matplotlib.pyplot as plt
plt.style.use('dark_background')

# Define attribute strings
science_data_file_path = '/Users/Grey/Documents/Research/Science_Data/'
me_string = 'Mikhail Schee, University of Toronto'
doi = 'No DOI yet'

# Choose pressure value
#   The value of CT_max (and therefore press_CT_max and SA_CT_max) must occur
#   at a pressure greater than this value
press_CT_max_threshold = 5

# Choose type of float to store
float_dtype = 'float32'
np_float_type = np.float32

################################################################################

def read_instrmt(source, instrmt_name, instrmt_dir, out_file):
    """
    Reads in all the data for the specified instrument and formats it into a
    single netcdf

    source              string of the name of the data source (ex: 'AIDJEX', 'ITP')
    instrmt_name        string of the name of this instrmt
    instrmt_dir         string of a file path to this instrmt's directory
    out_file            string of the file path in which to save the netcdf
    """
    print('Reading',source,instrmt_name)
    # Select the corresponding read function for the provided data source
    if source == 'AIDJEX':
        read_data_file  = read_AIDJEX_data_file
        attribution     = 'Moritz, Richard. 2020. Salinity, Temperature, Depth profiler data at AIDJEX stations April 1975 through April 1976, Version 1. Boulder, Colorado USA. CanWIN: Canadian Watershed Information Network. https://doi.org/10.34992/4xak-8r05 [Date Accessed: 2023-05-18].'
        source_url      = 'https://canwin-datahub.ad.umanitoba.ca/data/dataset/aidjex'
        og_vert         = 'depth'
        og_temp         = 'iT'
        og_salt         = 'SP'
    elif source == 'SHEBA':
        read_data_file  = read_SHEBA_data_file
        attribution     = 'Need to look this up'
        source_url      = 'URL'
        og_vert         = 'press'
        og_temp         = 'iT'
        og_salt         = 'SP'
    elif source == 'ITP':
        read_data_file  = read_ITP_data_file
        attribution    = 'The Ice-Tethered Profiler data were collected and made available by the Ice-Tethered Profiler Program (Toole et al., 2011; Krishfield et al., 2008) based at the Woods Hole Oceanographic Institution'
        source_url      = 'https://www.whoi.edu/itp'
        og_vert         = 'press'
        og_temp         = 'iT'
        og_salt         = 'SP'
    else:
        print(source,'is not a valid source')
        exit(0)
    # Make blank arrays for data
    list_of_entries         = []
    list_of_pf_nos          = []
    list_of_black_list      = []
    list_of_datetimes_start = []
    list_of_datetimes_end   = []
    list_of_lons            = []
    list_of_lats            = []
    list_of_regs            = []
    list_of_up_casts        = []
    list_of_press_maxs      = []
    list_of_CT_maxs         = []
    list_of_press_CT_maxs   = []
    list_of_SA_CT_maxs      = []
    list_of_press_arrs      = []
    list_of_depth_arrs      = []
    list_of_iT_arrs         = []
    list_of_CT_arrs         = []
    list_of_PT_arrs         = []
    list_of_SP_arrs         = []
    list_of_SA_arrs         = []
    # Keep track of the maximum number of vertical measurements per profile
    max_vert_count = 0
    # Loop through the data files in this instrmt's directory
    data_files = list_data_files(instrmt_dir)
    if isinstance(data_files, type(None)):
        print('Did not find any files for',instrmt_name)
        exit(0)
    else:
        # Keep track of the entry number with i
        i = 0
        for file in data_files:
            if os.path.isfile(instrmt_dir+'/'+file):
                # Read in the data file for this profile
                # print('\t\tReading',file)
                out_dict = read_data_file(instrmt_dir, file, instrmt_name)
                if not isinstance(out_dict, type(None)):
                    # Append that data
                    list_of_entries.append(i)
                    list_of_pf_nos.append(out_dict['prof_no'])
                    list_of_black_list.append(out_dict['black_list'])
                    list_of_datetimes_start.append(out_dict['dt_start'])
                    list_of_datetimes_end.append(out_dict['dt_end'])
                    list_of_lons.append(out_dict['lon'])
                    list_of_lats.append(out_dict['lat'])
                    list_of_regs.append(out_dict['region'])
                    list_of_up_casts.append(out_dict['up_cast'])
                    list_of_press_maxs.append(out_dict['press_max'])
                    list_of_CT_maxs.append(out_dict['CT_max'])
                    list_of_press_CT_maxs.append(out_dict['press_CT_max'])
                    list_of_SA_CT_maxs.append(out_dict['SA_CT_max'])
                    list_of_press_arrs.append(out_dict['press'])
                    list_of_depth_arrs.append(out_dict['depth'])
                    list_of_iT_arrs.append(out_dict['iT'])
                    list_of_CT_arrs.append(out_dict['CT'])
                    list_of_PT_arrs.append(out_dict['PT'])
                    list_of_SP_arrs.append(out_dict['SP'])
                    list_of_SA_arrs.append(out_dict['SA'])
                    # Check for a new maximum vertical dimension length
                    max_vert_count = max(max_vert_count, len(out_dict['depth']))
                    # Increase entry number
                    i += 1
                #
            #
        # Print out total files found
        print('\tRead',i,'data files')
    #
    # Make sure the vertical data are all the same length lists
    for i in range(len(list_of_depth_arrs)):
        list_of_press_arrs[i] = list(list_of_press_arrs[i]) +[None]*(max_vert_count - len(list_of_press_arrs[i]))
        list_of_depth_arrs[i] = list(list_of_depth_arrs[i]) +[None]*(max_vert_count - len(list_of_depth_arrs[i]))
        list_of_iT_arrs[i] = list(list_of_iT_arrs[i]) + [None]*(max_vert_count - len(list_of_iT_arrs[i]))
        list_of_SP_arrs[i] = list(list_of_SP_arrs[i]) + [None]*(max_vert_count - len(list_of_SP_arrs[i]))
        list_of_CT_arrs[i] = list(list_of_CT_arrs[i]) + [None]*(max_vert_count - len(list_of_CT_arrs[i]))
        list_of_PT_arrs[i] = list(list_of_PT_arrs[i]) + [None]*(max_vert_count - len(list_of_PT_arrs[i]))
        list_of_SA_arrs[i] = list(list_of_SA_arrs[i]) + [None]*(max_vert_count - len(list_of_SA_arrs[i]))
    #
    # Make a blank array for each dimension
    Time_blank = [None]*len(list_of_datetimes_start)
    Vertical_blank = [[None]*max_vert_count]*len(list_of_datetimes_start)
    # Make arrays for the source and instrument names
    list_of_sources  = [source]*len(list_of_datetimes_start)
    list_of_instrmts = [instrmt_name]*len(list_of_datetimes_start)

    # Define variables with data and attributes
    nc_vars = {
                'source':(
                        ['Time'],
                        list_of_sources,
                        {
                            'units':'N/A',
                            'label':'Source',
                            'long_name':'Name of the expedition on which the data was collected'
                        }
                ),
                'instrmt':(
                        ['Time'],
                        list_of_instrmts,
                        {
                            'units':'N/A',
                            'label':'Instrument',
                            'long_name':'Instrument identifier'
                        }
                ),
                'entry':(
                        ['Time'],
                        np.array(list_of_entries, dtype=np.int32),
                        {
                            'units':'N/A',
                            'label':'Entry (index)',
                            'long_name':'Sequential entry number',
                            'dtype':'float64'
                        }
                ),
                'prof_no':(
                        ['Time'],
                        np.array(list_of_pf_nos, dtype=np.int32),
                        {
                            'units':'N/A',
                            'label':'Profile number',
                            'long_name':'Profile number',
                            'dtype':'int32'
                        }
                ),
                'BL_yn':(
                        ['Time'],
                        list_of_black_list,
                        {
                            'units':'True/False',
                            'label':'On Black List',
                            'long_name':'Profile is on Black List, True(1) or False(0)'
                        }
                ),
                'dt_start':(
                        ['Time'],
                        list_of_datetimes_start,
                        {
                            'units':'YYYY-MM-DD HH:MM:SS',
                            'label':'Datetime of profile start',
                            'long_name':'Datetime when data for this entry started'
                        }
                ),
                'dt_end':(
                        ['Time'],
                        list_of_datetimes_end,
                        {
                            'units':'YYYY-MM-DD HH:MM:SS',
                            'label':'Datetime of profile end',
                            'long_name':'Datetime when data for this entry ended'
                        }
                ),
                'lon':(
                        ['Time'],
                        list_of_lons,
                        {
                            'units':'Degrees',
                            'label':'Longitude ($^\circ$E+)',
                            'long_name':'Longitude',
                            'dtype':float_dtype
                        }
                ),
                'lat':(
                        ['Time'],
                        list_of_lats,
                        {
                            'units':'Degrees',
                            'label':'Latitude ($^\circ$N)',
                            'long_name':'Latitude',
                            'dtype':float_dtype
                        }
                ),
                'region':(
                        ['Time'],
                        list_of_regs,
                        {
                            'units':'N/A',
                            'label':'Geographical region',
                            'long_name':'Geographical region in which this entry was measured'
                        }
                ),
                'up_cast':(
                        ['Time'],
                        list_of_up_casts,
                        {
                            'units':'N/A',
                            'label':'Upcast or Downcast',
                            'long_name':'Profile taken with instrument going up(True) or down(False)'
                        }
                ),
                'press_max':(
                        ['Time'],
                        np.array(list_of_press_maxs, dtype=np_float_type),
                        {
                            'units':'dbar',
                            'label':'$p_{max}$ (dbar)',
                            'long_name':'Maximum Pressure',
                            'dtype':float_dtype
                        }
                ),
                'CT_max':(
                        ['Time'],
                        np.array(list_of_CT_maxs, dtype=np_float_type),
                        {
                            'units':'degrees Celcius',
                            'label':'$\Theta_{max}$ ($^\circ$C)',
                            'long_name':'Maximum Conservative Temperature',
                            'dtype':float_dtype
                        }
                ),
                'press_CT_max':(
                        ['Time'],
                        np.array(list_of_press_CT_maxs, dtype=np_float_type),
                        {
                            'units':'dbar',
                            'label':'$p(\Theta_{max})$ (dbar)',
                            'long_name':'Pressure at Maximum Conservative Temperature',
                            'dtype':float_dtype
                        }
                ),
                'SA_CT_max':(
                        ['Time'],
                        np.array(list_of_SA_CT_maxs, dtype=np_float_type),
                        {
                            'units':'g/kg',
                            'label':'$S_A(\Theta_{max})$ (g/kg)',
                            'long_name':'Absolute Salinity at Maximum Conservative Temperature',
                            'dtype':float_dtype
                        }
                ),
                'R_rho':(
                        ['Time'],
                        np.array(Time_blank, dtype=np_float_type),
                        {
                            'units':'N/A',
                            'label':'Vertical Density Ratio $R_\\rho$',
                            'long_name':'Vertical Density Ratio',
                            'dtype':float_dtype
                        }
                ),
                'press':(
                        ['Time','Vertical'],
                        np.array(list_of_press_arrs, dtype=np_float_type),
                        {
                            'units':'dbar',
                            'label':'Pressure (dbar)',
                            'long_name':'Pressure',
                            'dtype':float_dtype
                        }
                ),
                'depth':(
                        ['Time','Vertical'],
                        np.array(list_of_depth_arrs, dtype=np_float_type),
                        {
                            'units':'m',
                            'label':'Depth (m)',
                            'long_name':'Depth',
                            'dtype':float_dtype
                        }
                ),
                'iT':(
                        ['Time','Vertical'],
                        np.array(list_of_iT_arrs, dtype=np_float_type),
                        {
                            'units':'degrees Celcius',
                            'label':'in-situ Temperature ($^\circ$C)',
                            'long_name':'in-situ Temperature',
                            'dtype':float_dtype
                        }
                ),
                'CT':(
                        ['Time','Vertical'],
                        np.array(list_of_CT_arrs, dtype=np_float_type),
                        {
                            'units':'degrees Celcius',
                            'label':'$\Theta$ ($^\circ$C)',
                            'long_name':'Conservative Temperature',
                            'dtype':float_dtype
                        }
                ),
                'PT':(
                        ['Time','Vertical'],
                        np.array(list_of_PT_arrs, dtype=np_float_type),
                        {
                            'units':'degrees Celcius',
                            'label':'$\theta$ ($^\circ$C)',
                            'long_name':'Potential Temperature',
                            'dtype':float_dtype
                        }
                ),
                'SP':(
                        ['Time','Vertical'],
                        np.array(list_of_SP_arrs, dtype=np_float_type),
                        {
                            'units':'g/kg',
                            'label':'$S_P$ (g/kg)',
                            'long_name':'Practical Salinity',
                            'dtype':float_dtype
                        }
                ),
                'SA':(
                        ['Time','Vertical'],
                        np.array(list_of_SA_arrs, dtype=np_float_type),
                        {
                            'units':'g/kg',
                            'label':'$S_A$ (g/kg)',
                            'long_name':'Absolute Salinity',
                            'dtype':float_dtype
                        }
                ),
                'sigma':(
                        ['Time','Vertical'],
                        np.array(gsw.sigma1(list_of_SA_arrs, list_of_CT_arrs), dtype=np_float_type),
                        {
                            'units':'kg/m^3',
                            'label':'$\\sigma_1$ (kg/m$^3$)',
                            'long_name':'Density anomaly referenced to 1000 dbar',
                            'dtype':float_dtype
                        }
                ),
                'alpha':(
                        ['Time','Vertical'],
                        np.array(gsw.alpha(list_of_SA_arrs, list_of_CT_arrs, list_of_press_arrs), dtype=np_float_type),
                        {
                            'units':'1/(degrees Celcius)',
                            'label':'$\\alpha$ (1/$^\circ$C)',
                            'long_name':'Thermal expansion coefficient alpha',
                            'dtype':float_dtype
                        }
                ),
                'alpha_PT':(
                        ['Time','Vertical'],
                        np.array(gsw.alpha(list_of_SA_arrs, list_of_PT_arrs, list_of_press_arrs), dtype=np_float_type),
                        {
                            'units':'1/(degrees Celcius)',
                            'label':'$\\alpha_{PT}$ (1/$^\circ$C)',
                            'long_name':'Thermal expansion coefficient alpha wrt PT',
                            'dtype':float_dtype
                        }
                ),
                'alpha_iT':(
                        ['Time','Vertical'],
                        np.array(gsw.alpha_wrt_t_exact(list_of_SA_arrs, list_of_iT_arrs, list_of_press_arrs), dtype=np_float_type),
                        {
                            'units':'1/(degrees Celcius)',
                            'label':'$\\alpha_{iT}$ (1/$^\circ$C)',
                            'long_name':'Thermal expansion coefficient alpha wrt iT',
                            'dtype':float_dtype
                        }
                ),
                'beta':(
                        ['Time','Vertical'],
                        np.array(gsw.beta(list_of_SA_arrs, list_of_CT_arrs, list_of_press_arrs), dtype=np_float_type),
                        {
                            'units':'1/(g/kg)',
                            'label':'$\\beta$ (kg/g)',
                            'long_name':'Saline contraction coefficient beta',
                            'dtype':float_dtype
                        }
                ),
                'beta_PT':(
                        ['Time','Vertical'],
                        np.array(gsw.beta(list_of_SA_arrs, list_of_PT_arrs, list_of_press_arrs), dtype=np_float_type),
                        {
                            'units':'1/(g/kg)',
                            'label':'$\\beta_{PT}$ (kg/g)',
                            'long_name':'Saline contraction coefficient beta wrt PT',
                            'dtype':float_dtype
                        }
                ),
                'ss_mask':(
                        ['Time','Vertical'],
                        np.array(Vertical_blank, dtype=np_float_type),
                        {
                            'units':'N/A',
                            'label':'Sub-sample mask',
                            'long_name':'A mask to apply the sub-sample scheme',
                            'dtype':float_dtype
                        }
                ),
                'ma_iT':(
                        ['Time','Vertical'],
                        np.array(Vertical_blank, dtype=np_float_type),
                        {
                            'units':'degrees Celcius',
                            'label':'Moving average in-situ temperature ($^\circ$C)',
                            'long_name':'Moving average in-situ temperature',
                            'dtype':float_dtype
                        }
                ),
                'ma_CT':(
                        ['Time','Vertical'],
                        np.array(Vertical_blank, dtype=np_float_type),
                        {
                            'units':'degrees Celcius',
                            'label':'Moving average $\Theta$ ($^\circ$C)',
                            'long_name':'Moving average conservative temperature',
                            'dtype':float_dtype
                        }
                ),
                'ma_PT':(
                        ['Time','Vertical'],
                        np.array(Vertical_blank, dtype=np_float_type),
                        {
                            'units':'degrees Celcius',
                            'label':'Moving average $\theta$ ($^\circ$C)',
                            'long_name':'Moving average potential temperature',
                            'dtype':float_dtype
                        }
                ),
                'ma_SP':(
                        ['Time','Vertical'],
                        np.array(Vertical_blank, dtype=np_float_type),
                        {
                            'units':'g/kg',
                            'label':'Moving average $S_P$ (g/kg)',
                            'long_name':'Moving average practical salinity',
                            'dtype':float_dtype
                        }
                ),
                'ma_SA':(
                        ['Time','Vertical'],
                        np.array(Vertical_blank, dtype=np_float_type),
                        {
                            'units':'g/kg',
                            'label':'Moving average $S_A$ (g/kg)',
                            'long_name':'Moving average absolute salinity',
                            'dtype':float_dtype
                        }
                ),
                'ma_sigma':(
                        ['Time','Vertical'],
                        np.array(Vertical_blank, dtype=np_float_type),
                        {
                            'units':'kg/m^3',
                            'label':'Moving average $\\sigma_1$ (kg/m$^3$)',
                            'long_name':'Moving average density anomaly',
                            'dtype':float_dtype
                        }
                ),
                'cluster':(
                        ['Time','Vertical'],
                        Vertical_blank,
                        {
                            'units':'N/A',
                            'label':'Cluster label',
                            'long_name':'Cluster label (-1 means noise points)',
                            'dtype':'int32'
                        }
                ),
                'clst_prob':(
                        ['Time','Vertical'],
                        np.array(Vertical_blank, dtype=np_float_type),
                        {
                            'units':'N/A',
                            'label':'Cluster probability',
                            'long_name':'Probability of being in the labeled cluster',
                            'dtype':float_dtype
                        }
                )
    }
    #

    # Use the ITP number, padded with zeros as a value for the miliseconds
    #   This avoids index collisions when combining netcdfs of multiple ITPs
    if source == 'ITP':
        ms_to_add = instrmt_name.zfill(3)
        # Add this milisecond value to all datetimes in a new variable
        list_of_Times = [this_dt+'.'+ms_to_add for this_dt in list_of_datetimes_start]
    else:
        list_of_Times = list_of_datetimes_start

    # Define coordinates
    nc_coords = {
               'Time':(
                        ['Time'],
                        list_of_Times,
                        {
                            'units':'YYYY-MM-DD HH:MM:SS',
                            'long_name':'Datetime of the start of the measurement',
                            'dtype':'datetime64[ns]'
                        }
                ),
               'Vertical':(
                        ['Vertical'],
                        np.arange(max_vert_count),
                        {
                            'units':'N/A',
                            'long_name':'An index of vertical position (depth or pressure)',
                            'dtype':'int32'
                        }
                )
    }
    #
    # Define global attributes
    nc_attrs = {        # Note: can't store datetime objects in netcdfs
                'Title':'Arctic Ocean Profile Data',
                'Creation date':str(datetime.now()),
                'Source':source,
                'Instrument':instrmt_name,
                'Data Attribution':attribution,
                'Data Source':source_url,
                'Data Organizer':me_string,
                'Reference':doi,
                'Last modified':str(datetime.now()),
                'Last modification':'File was created',
                'Original vertical measure':og_vert,
                'Original temperature measure':og_temp,
                'Original salinity measure':og_salt,
                'Sub-sample scheme':'None',
                'Moving average window':'None',
                'Last clustered':'Never',
                'Clustering x-axis':'None',
                'Clustering y-axis':'None',
                'Clustering z-axis':'None',
                'Clustering m_pts':'None',
                'Clustering m_cls':'None',
                'Clustering filters':'None',
                'Clustering DBCV':'None'
    }

    # Convert into a dataset
    ds = xr.Dataset(data_vars=nc_vars, coords=nc_coords, attrs=nc_attrs)
    # Write out to netcdf
    print('Writing data to',out_file)
    ds.to_netcdf(out_file, 'w')

################################################################################

black_list = {'BigBear': [531, 535, 537, 539, 541, 543, 545, 547, 549],
              'BlueFox': [94, 308, 310],
              'Caribou': [],
              'Snowbird': [443],
              'Seacat': ['SH15200.UP', 'SH03200', 'SH30500', 'SH34100']}

################################################################################
# Loading in data

def list_data_files(file_path):
    """
    Creates a list of all data files within a containing folder

    file_path       The path to the folder containing the data files
    """
    # Search the provided file path
    if os.path.isdir(file_path):
        data_files = os.listdir(file_path)
        # Remove .DS_Store directory from list
        if '.DS_Store' in data_files: data_files.remove('.DS_Store')
        # Restrict to just unique instruments (there shouldn't be any duplicates hopefully)
        data_files = np.unique(data_files)
    else:
        print(file_path, " is not a directory")
        data_files = None
    #
    return data_files

################################################################################
# ITP functions
################################################################################

def make_all_ITP_netcdfs(science_data_file_path, format='cormat'):
    """
    Finds ITP data files for all instruments available and formats them into netcdfs

    science_data_file_path      string of the filepath where the data is stored
    format                      which version of the data files to use
                                    either 'cormat' or 'final'
    """
    # Declare file path
    main_dir = science_data_file_path+'ITPs/'
    # Search the provided file path
    if os.path.isdir(main_dir):
        ITP_dirs = os.listdir(main_dir)
        # Remove .DS_Store directory from list
        if '.DS_Store' in ITP_dirs: ITP_dirs.remove('.DS_Store')
    else:
        print(main_dir, " is not a directory")
        exit(0)
    # Read in data for all ITPs available
    for itp in ITP_dirs:
        if 'itp' in itp:
            # Get just the number for the itp
            itp_number = ''.join(filter(str.isdigit, itp))
            read_instrmt('ITP', itp_number, main_dir+itp+'/'+itp+format, 'netcdfs/ITP_'+itp_number.zfill(3)+'.nc')
        #
    #

################################################################################

def read_ITP_data_file(file_path, file_name, instrmt):
    """
    Reads certain data from an ITP profile file
    Returns a dictionary with that specific information formatted in a manageable way

    file_path           string of a file path to the containing directory
    file_name           string of the file name of a specific file
    instrmt             string of the name of this instrmt (just an unpadded number here)
    """
    # Make sure it isn't a 'sami' file instead of a 'grd' file
    if 'sami' in file_name:
        print('Skipping',file_name)
        return None
    # Get just the proper file name, after the slash
    filename2 = file_name.split('/')[-1]
    # Assuming filename format itpXgrdYYYY.dat where YYYY is always 4 digits
    #   works for `final` format above or `cormat` format corYYYY.mat
    try:
        prof_no = int(filename2[-8:-4])
    except:
        print('Skipping',instrmt,'file',filename2)
        return None
    #
    # Check to see whether loading `final` or `cormat` file format
    if 'cormat' in file_path:
        return read_ITP_cormat(file_path, file_name, instrmt, prof_no)
    elif 'final' in file_path:
        return read_ITP_final(file_path, file_name, instrmt, prof_no)
    #

def read_ITP_cormat(file_path, file_name, instrmt, prof_no):
    """
    Loads the data from an ITP profile file in the `cormat` format
    Returns a dictionary with specific information

    file_path           string of a file path to the containing directory
    file_name           string of the file name of a specific file
    instrmt             The number of the ITP that took the profile measurement
    prof_no             The number identifying this specific profile
    """
    # Check to make sure this one isn't on the black list
    if instrmt in black_list.keys():
        if prof_no in black_list[instrmt]:
            on_black_list = True
        else:
            on_black_list = False
    else:
        on_black_list = False
    #
    # Load cormat file into dictionary with mat73
    #   (specific to version of MATLAB used to make cormat files)
    try:
        dat = mat73.loadmat(file_path+'/'+file_name)
    except:
        dat = io.loadmat(file_path+'/'+file_name)
    # Extract certain data from the object, specific to how the files are formatted
    #   The latitude and longitude values where the profile was taken
    lon = np.array(dat['longitude'], dtype=np_float_type)
    lat = np.array(dat['latitude'], dtype=np_float_type)
    # Make sure lon and lat are single values (ITP25 caused this issue)
    while len(lon.shape) > 0:
        lon = lon[0]
    while len(lat.shape) > 0:
        lat = lat[0]
    #   The date this profile was taken, psdate: profile start or pedate: profile end
    date_MMDDYY_start = dat['psdate']
    date_MMDDYY_end   = dat['pedate']
    #   The time this profile was taken, pstart: profile start or pstop: profile end
    time_HHMMSS_start = dat['pstart']
    time_HHMMSS_end   = dat['pstop']
    # Sometimes, the date comes out as an array. In that case, take the first index
    if not isinstance(date_MMDDYY_start, str):
        try:
            date_MMDDYY_start = date_MMDDYY_start[0]
        except:
            date_MMDDYY_start = None
    if not isinstance(date_MMDDYY_end, str):
        try:
            date_MMDDYY_end   = date_MMDDYY_end[0]
        except:
            date_MMDDYY_end   = None
    if not isinstance(time_HHMMSS_start, str):
        try:
            time_HHMMSS_start = time_HHMMSS_start[0]
        except:
            time_HHMMSS_start = None
    if not isinstance(time_HHMMSS_end, str):
        try:
            time_HHMMSS_end   = time_HHMMSS_end[0]
        except:
            time_HHMMSS_end   = None
    # Attempt to format the datetimes
    try:
        dt_start = str(datetime.strptime(date_MMDDYY_start+' '+time_HHMMSS_start, r'%m/%d/%y %H:%M:%S'))
    except:
        dt_start = None
    try:
        dt_end = str(datetime.strptime(date_MMDDYY_end+' '+time_HHMMSS_end, r'%m/%d/%y %H:%M:%S'))
    except:
        dt_end = None
    # Determine the region
    reg = find_geo_region(lon, lat)
    #
    # If it finds the correct column headers, put data into arrays
    if 'te_adj' in dat and 'sa_adj' in dat and 'pr_filt' in dat:
        press0 = dat['pr_filt'].flatten()
        iT0  = dat['te_adj'].flatten()
        SP0  = dat['sa_adj'].flatten()
        # Check to make sure at least some data is there
        if len(press0) < 2 or np.isnan(press0).all():
            print('\t- no data for',instrmt,prof_no)
            return None
        # Find maximum pressure value
        press_max = max(press0)
        # Convert to absolute salinity (SA), conservative (CT) and potential temperature (PT) 
        pf_vert_len = len(press0)
        SA1 = gsw.SA_from_SP(SP0, press0, [lon]*pf_vert_len, [lat]*pf_vert_len)
        CT1 = gsw.CT_from_t(SA1, iT0, press0)
        PT1 = gsw.pt0_from_t(SA1, iT0, press0)
        # Convert to depth from pressure
        depth = gsw.z_from_p(press0, [lat]*pf_vert_len)
        # Down-casts have an issue with the profiler wake, so note whether the
        #   profile was taken going up or down
        if press0[0] < press0[-1]:
            up_cast = False
        else:
            up_cast = True
        # Get the index of the maximum conservative temperature
        try:
            # Get just the part of the CT1 array where press0 is greater
            #   than press_CT_max_threshold
            temp_CT = np.ma.masked_where(press0 < press_CT_max_threshold, CT1)
            i_CT_max = np.nanargmax(temp_CT)
            CT_max = CT1[i_CT_max]
            press_CT_max = press0[i_CT_max]
            SA_CT_max = SA1[i_CT_max]
        except:
            print('\t\t- Cannot compute CT_max for ITP',instrmt,'profile',prof_no)
            CT_max = None
            press_CT_max = None
            SA_CT_max = None
        # Create output dictionary for this profile
        out_dict = {'prof_no': prof_no,
                    'black_list': on_black_list,
                    'dt_start': dt_start,
                    'dt_end': dt_end,
                    'lon': lon,
                    'lat': lat,
                    'region': reg,
                    'up_cast': up_cast,
                    'press_max': press_max,
                    'CT_max':CT_max,
                    'press_CT_max':press_CT_max,
                    'SA_CT_max':SA_CT_max,
                    'press': press0,
                    'depth': depth,
                    'iT': iT0,
                    'CT': CT1,
                    'PT': PT1,
                    'SP': SP0,
                    'SA': SA1
                    }
        #
        # Return all the relevant values
        return out_dict
    else:
        print('No data found for',instrmt,prof_no)
        return None

def read_ITP_final(file_path, file_name, instrmt, prof_no):
    """
    Loads the data from an ITP profile file in the `final` format
    Returns a dictionary with specific information

    file_path           string of a file path to the containing directory
    file_name           string of the file name of a specific file
    instrmt             The number of the ITP that took the profile measurement
    prof_no             The number identifying this specific profile
    """
    # Check to make sure this one isn't on the black list
    if instrmt in black_list.keys():
        if prof_no in black_list[instrmt]:
            on_black_list = True
        else:
            on_black_list = False
    else:
        on_black_list = False
    #
    # Read data file in as a pandas data object
    #   The arguments used are specific to how the ITP files are formatted
    #   Reading in the file one line at a time because the number of items
    #       on each line is inconsistent between files
    dat0 = pd.read_table(file_path+'/'+file_name, header=None, nrows=2, engine='python', delim_whitespace=True).iloc[1]
    # Extract certain data from the object, specific to how the files are formatted
    #   The date this profile was taken
    try:
        date = str(pd.to_datetime(float(dat0[1])-1, unit='D', origin=str(dat0[0])))
    except:
        date = None
    #   The latitude and longitude values where the profile was taken
    lon = np.array(dat0[2], dtype=np_float_type)
    lat = np.array(dat0[3], dtype=np_float_type)
    # Determine the region
    reg = find_geo_region(lon, lat)
    #
    # Read in data from the file
    dat = pd.read_table(file_path+'/'+file_name,header=2,skipfooter=1,engine='python',delim_whitespace=True)
    # If it finds the correct column headers, put data into arrays
    if 'temperature(C)' and 'salinity' and '%pressure(dbar)' in dat.columns:
        press0 = dat['%pressure(dbar)'][:].values
        iT0  = dat['temperature(C)'][:].values
        SP0  = dat['salinity'][:].values
        # Find maximum pressure value
        press_max = max(press0)
        # Convert to absolute salinity (SA), conservative (CT) and potential temperature (PT) 
        pf_vert_len = len(press0)
        SA1 = gsw.SA_from_SP(SP0, press0, [lon]*pf_vert_len, [lat]*pf_vert_len)
        CT1 = gsw.CT_from_t(SA1, iT0, press0)
        PT1 = gsw.pt0_from_t(SA1, iT0, press0)
        # Convert to depth from pressure
        depth = gsw.z_from_p(press0, [lat]*pf_vert_len)
        # `final` formatted profiles are sorted, so no way to tell which direction
        #   They were taken in. So, just mark all as up-casts
        up_cast = True
        # Get the index of the maximum conservative temperature
        try:
            # Get just the part of the CT1 array where press0 is greater
            #   than press_CT_max_threshold
            temp_CT = np.ma.masked_where(press0 < press_CT_max_threshold, CT1)
            i_CT_max = np.nanargmax(temp_CT)
        except:
            i_CT_max = 0
        # Create output dictionary for this profile
        out_dict = {'prof_no': prof_no,
                    'black_list': on_black_list,
                    'dt_start': date,
                    'dt_end': None,
                    'lon': lon,
                    'lat': lat,
                    'region': reg,
                    'up_cast': up_cast,
                    'press_max': press_max,
                    'CT_max':CT1[i_CT_max],
                    'press_CT_max':press0[i_CT_max],
                    'SA_CT_max':SA1[i_CT_max],
                    'press': press0,
                    'depth': depth,
                    'iT': iT0,
                    'CT': CT1,
                    'PT': PT1,
                    'SP': SP0,
                    'SA': SA1
                    }
        #
        # Return all the relevant values
        return out_dict
    else:
        print('No data found for',instrmt,prof_no)
        return None

################################################################################
# SHEBA functions
################################################################################

def make_SHEBA_netcdfs(science_data_file_path):
    """
    Finds SHEBA data files for all instruments available and formats them into netcdfs

    science_data_file_path      string of the filepath where the data is stored
    """
    # Declare file path
    main_dir = science_data_file_path+'SHEBA/dataset_13_524/Ice_Camp_Ocean_Seacat_CTD_Data-Deep/'
    # Only one instrument, so just read it in
    read_instrmt('SHEBA', 'Seacat', main_dir, 'netcdfs/SHEBA_Seacat.nc')
    #

def read_SHEBA_data_file(file_path, file_name, instrmt):
    """
    Reads certain data from an AIDJEX profile file
    Returns a dictionary with specific information

    file_path           string of a file path to the containing directory
    file_name           string of the file name of a specific file
    instrmt             string of the name of this instrmt
    """
    # All relevant SHEBA Seacat data files have headers with the extension `.HDR`
    #   Only start the reading process if you get one of those
    if not '.HDR' in file_name:
        return None
    # Set the header file (the one with all the meta data)
    header_file = file_name
    # print('Reading in the header file:',header_file)
    # Set the profile ID (the part without `.HDR`)
    prof_no = header_file.replace('.HDR', '')
    # See if there exists data files with that profile ID
    file_list = os.listdir(file_path)
    data_files = None
    for file in file_list:
        if prof_no.lower() in file.lower():
            # Don't set the data file to be the .HDR or .UP file
            if not '.HDR' in file:
                data_file = file
            #
        #
    #
    # Check to make sure this one isn't on the black list
    if prof_no in black_list[instrmt]:
        on_black_list = True
        # I'm not sure what the issue is exactly, but if I attempt to load any
        #   profile on the black list anyway, the code crashes. So just skip them
        return None
    else:
        on_black_list = False
    #
    # Read in header file as an array, each line of the file for each index
    #   Need to use 'rb' to avoid issues with UnicodeDecodeError
    with open(file_path+'/'+header_file, 'rb') as read_file:
        dat0 = read_file.readlines()
    # Declare variables
    # print(header_file)
    lon = None
    lat = None
    date = None
    date_regex = {r'(\d+/\d+/\d+)': "%m/%d/%y",
                  r'(\d+\s[A-Z,a-z]{3}\s\d{4})': "%d %b %Y",
                  r'(\d+\s[A-Z,a-z]{3}\s\d{2})': "%d %b %y"}
    max_cols = 0
    # Set arbitrary values that are greater than the number of columns you could have
    t_index = 99
    s_index = 99
    p_index = 99
    # Take relevant lines and put them into variables
    for line in dat0:
        # Convert to string
        line_str = str(line)
        # Find all the numbers in the line
        line_nmbrs = []
        # Find all whitespace-separated substrings
        line_ss = []
        for ss in line_str.split():
            ss_strp = ss.rstrip('\\n\'').strip()
            line_ss.append(ss_strp)
            if isfloat(ss_strp):
                line_nmbrs.append(ss_strp)
        # Search for a date value, as long as one hasn't been found yet
        if isinstance(date, type(None)):
            # Make sure not to accidentally take the upload date
            if not 'upload' in line_str.lower():
                # Try all the different regular expressions listed above
                for key in date_regex.keys():
                    # If a regex matches, store the date and stop looking (break)
                    match = re.search(key, line_str)
                    if not isinstance(match, type(None)):
                        # Find the date the profile was taken
                        try:
                            date = str(datetime.strptime(match.group(0), date_regex[key]))
                            break
                        except:
                            continue
        if len(line_nmbrs) > 0:
            # Find latitude and longitude
            if 'latitude' in line_str.lower():
                if len(line_nmbrs) == 2:
                    lat = np.array(line_nmbrs[0], dtype=np_float_type) + np.array(line_nmbrs[1], dtype=np_float_type)/100
                else:
                    lat = np.array(line_nmbrs[0], dtype=np_float_type)
                # print('Latitude:',lat)
            if 'longitude' in line_str.lower():
                if len(line_nmbrs) == 2:
                    lon = -np.array(line_nmbrs[0], dtype=np_float_type) + np.array(line_nmbrs[1], dtype=np_float_type)/100
                else:
                    lon = -np.array(line_nmbrs[0], dtype=np_float_type)
                # print('Longitude:',lon)
            # Find the headers of each column
            if 'name' in line_str.lower():
                if 'temperature' in line_str.lower():
                    t_index = int(line_nmbrs[0])
                elif 'salinity' in line_str.lower():
                    s_index = int(line_nmbrs[0])
                elif 'pressure' in line_str.lower() or 'depth' in line_str.lower():
                    p_index = int(line_nmbrs[0])
                # Find the total number of columns in the data file
                if int(line_nmbrs[0]) > max_cols:
                    max_cols = int(line_nmbrs[0])
            #
        #
    #
    # If those columns weren't found, skip this profile
    if t_index + s_index + p_index > 99:
        return None
    # Determine the region
    reg = find_geo_region(lon, lat)
    #
    # Create an empty list to capture data from file
    dat = []
    # Read data file in as a pandas data object
    #   The arguments used are specific to how the SHEBA files are formatted
    # print('Reading in the data file:',data_file)
    # Read in data file as an array, each line of the file for each index
    #   Need to use 'rb' to avoid issues with UnicodeDecodeError
    with open(file_path+'/'+data_file, 'rb') as read_file:
        # Skip the header
        for _ in range(1):
            next(read_file)
        for line in read_file:
            # If the row can be converted into an array of floats,
            #   add it to the dat list
            try:
                line_list = np.array(line.split(), dtype='float')
                dat.append(line_list)
                # Add note as to whether the pressure is increasing or decreasing
            except:
                continue
        #
    # Convert the data into a 2D array
    data = np.array(dat)
    if len(dat) < 1:
        return None
    # If the correct column headers were found, put data into arrays
    press0 = data[:,p_index]
    iT0  = data[:,t_index]
    SP0  = data[:,s_index]
    # Find maximum pressure value
    press_max = max(press0)
    # Convert to SA and CT
    pf_vert_len = len(press0)
    SA1 = gsw.SA_from_SP(SP0, press0, [lon]*pf_vert_len, [lat]*pf_vert_len)
    CT1 = gsw.CT_from_t(SA1, iT0, press0)
    PT1 = gsw.pt0_from_t(SA1, iT0, press0)
    # Convert to depth from pressure
    depth = gsw.z_from_p(press0, [lat]*pf_vert_len)
    # Get the index of the maximum conservative temperature
    try:
        # Get just the part of the CT1 array where press0 is greater
        #   than press_CT_max_threshold
        temp_CT = np.ma.masked_where(press0 < press_CT_max_threshold, CT1)
        i_CT_max = np.nanargmax(temp_CT)
    except:
        i_CT_max = 0
    # If there is data to be put into a dictionary...
    if len(press0) > 1:
        # Many SHEBA profiles contain both up and down casts, so just mark
        #   all as up-casts
        up_cast = True
        # Create output dictionary for this profile
        out_dict = {'prof_no': prof_no,
                    'black_list': on_black_list,
                    'dt_start': date,
                    'dt_end': None,
                    'lon': lon,
                    'lat': lat,
                    'region': reg,
                    'up_cast': up_cast,
                    'press_max': press_max,
                    'CT_max':CT1[i_CT_max],
                    'press_CT_max':press0[i_CT_max],
                    'SA_CT_max':SA1[i_CT_max],
                    'press': press0,
                    'depth': depth,
                    'iT': iT0,
                    'CT': CT1,
                    'PT': PT1,
                    'SP': SP0,
                    'SA': SA1
                    }
        #
        # Return all the relevant values
        return out_dict
    else:
        return None

################################################################################
# AIDJEX functions
################################################################################

def make_all_AIDJEX_netcdfs(science_data_file_path):
    """
    Finds AIDJEX data files for all 4 instruments and formats them into netcdfs

    science_data_file_path      string of the filepath where the data is stored
    """
    # Declare file path
    main_dir = science_data_file_path+'AIDJEX/AIDJEX/'
    # Read in data for all 4 AIDJEX stations
    station_names = ['BigBear', 'BlueFox', 'Caribou', 'Snowbird']
    for station in station_names:
        read_instrmt('AIDJEX', station, main_dir+station, 'netcdfs/AIDJEX_'+station+'.nc')
    #

def read_AIDJEX_data_file(file_path, file_name, instrmt):
    """
    Reads certain data from an AIDJEX profile file
    Returns a dictionary with specific information

    file_path           string of a file path to the containing directory
    file_name           string of the file name of a specific file
    instrmt             string of the name of this instrmt
    """
    # Assuming file name format instrmt_YYY where the profile number,
    #   YYY, is always 3 digits
    prof_no = int(''.join(filter(str.isdigit, file_name)))
    # Check to make sure this one isn't on the black list
    if prof_no in black_list[instrmt]:
        on_black_list = True
    else:
        on_black_list = False
    #
    # Declare variables
    # print('Loading in',file_name)
    lon = None
    lat = None
    # Read data file in as a pandas data object
    #   The arguments used are specific to how the AIDJEX files are formatted
    #   Reading in the file one line at a time because the number of items
    #       on each line is inconsistent between files
    dat0 = pd.read_table(file_path+'/'+file_name, header=None, nrows=1, engine='python', delim_whitespace=True).iloc[0]
    dat1 = pd.read_table(file_path+'/'+file_name, header=0, nrows=1, engine='python', delim_whitespace=True).iloc[0]
    #   The date this profile was taken
    date_string = dat0[3]
    #   The time this profile was taken
    time_string = str(dat0[4]).zfill(4)
    #   Assuming date is in format d/MON/YYYY where d can have 1 or 2 digits,
    #       MON is the first three letters of the month, and YYYY is the year
    #   Assuming time is in format HMM where H can have 1 or 2 digits originally
    #       but is 2 digits because of zfill(4) and MM is the minutes of the hour
    try:
        date = str(datetime.strptime(date_string+' '+time_string, r'%d/%b/%Y %H%M'))
    except:
        date = None
        print('Failed to format date for', file_name)
    #   Longitude and Latitude
    if ('Lat' in dat1[0]) or ('lat' in dat1[0]):
        lat = np.array(dat1[1], dtype=np_float_type)
    if ('Lon' in dat1[2]) or ('lon' in dat1[2]):
        lon = np.array(dat1[3], dtype=np_float_type)
    #       Check to make sure the lon and lat values are valid
    if lat == 99.9999 and lon == 99.9999:
        lat = None
        lon = None
    # Determine the region
    reg = find_geo_region(lon, lat)
    # Read in data from the file
    dat = pd.read_table(file_path+'/'+file_name,header=3,skipfooter=0,engine='python',delim_whitespace=True)
    # If it finds the correct column headers, put data into arrays
    if 'Depth(m)' and 'Temp(C)' and 'Sal(PPT)' in dat.columns:
        depth0 = dat['Depth(m)'][:].values
        iT0    = dat['Temp(C)'][:].values
        SP0    = dat['Sal(PPT)'][:].values
        # Converting requires both a longitude and a latitude
        if isinstance(lon, type(None)):
            lon = -145.3498
        #   Requires a latitude to work, but sometimes `lat` is None, so use an average
        if isinstance(lat, type(None)):
            lat = 76.3519
        # print('Using a lat =',lat,'and lon =',lon,'for profile',prof_no)
        # Calculate pressure from depth using GSW which expects negative depth values, zero being the surface
        press0 = gsw.p_from_z(-depth0, lat)
        # Find maximum pressure value
        press_max = max(press0)
        pf_vert_len = len(press0)
        # Convert to absolute salinity (SA), conservative (CT) and potential temperature (PT) 
        SA1 = gsw.SA_from_SP(SP0, press0, [lon]*pf_vert_len, [lat]*pf_vert_len)
        CT1 = gsw.CT_from_t(SA1, iT0, press0)
        # print('Number of non-nan SA1 values:',np.count_nonzero(~np.isnan(SA1)))
        PT1 = gsw.pt0_from_t(SA1, iT0, press0)
        # Doesn't matter the direction of data collection for AIDJEX, so mark
        #   all profiles as up-casts
        up_cast = True
        # Get the index of the maximum conservative temperature
        try:
            # Get just the part of the CT1 array where press0 is greater
            #   than press_CT_max_threshold
            temp_CT = np.ma.masked_where(press0 < press_CT_max_threshold, CT1)
            i_CT_max = np.nanargmax(temp_CT)
        except:
            i_CT_max = 0
        # Create output dictionary for this profile
        out_dict = {'prof_no': prof_no,
                    'black_list': on_black_list,
                    'dt_start': date,
                    'dt_end': None,
                    'lon': lon,
                    'lat': lat,
                    'region': reg,
                    'up_cast': up_cast,
                    'press_max': press_max,
                    'CT_max':CT1[i_CT_max],
                    'press_CT_max':press0[i_CT_max],
                    'SA_CT_max':SA1[i_CT_max],
                    'press': press0,
                    'depth': depth0,
                    'iT': iT0,
                    'CT': CT1,
                    'PT': PT1,
                    'SP': SP0,
                    'SA': SA1
                    }
        #
        # Return all the relevant values
        return out_dict
    else:
        return None

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def find_geo_region(lon, lat):
    """
    Returns a string of the geographical region of the Arctic Ocean in which
    the coordinates are located. Region boundaries based on:
    Peralta-Ferriz and Woodgate (2015), doi:10.1016/j.pocean.2014.12.005

    lon         longitude value
    lat         latitude value
    """
    if isinstance(lon, type(None)) or isinstance(lat, type(None)):
        return 'error'
    # Chukchi Sea
    elif (lon > -180) & (lon < -155) & (lat > 68) & (lat < 76):
        return 'CS'
    # Southern Beaufort Sea
    elif (lon > -155) & (lon < -120) & (lat > 68) & (lat < 72):
        return 'SBS'
    # Canada Basin
    elif (lon > -155) & (lon < -130) & (lat > 72) & (lat < 84):
        return 'CB'
    # Makarov Basin part 1
    elif (lon > 50) & (lon < 180) & (lat > 83.5) & (lat < 90):
        return 'MB'
    # Makarov Basin part 2
    elif (lon > 141) & (lon < 180) & (lat > 78) & (lat < 90):
        return 'MB'
    # Eurasian Basin part 1
    elif (lon > 30) & (lon < 140) & (lat > 82) & (lat < 90):
        return 'EB'
    # Eurasian Basin part 2
    elif (lon > 110) & (lon < 140) & (lat > 78) & (lat < 82):
        return 'EB'
    # Barents Sea part 1
    elif (lon > 15) & (lon < 60) & (lat > 75) & (lat < 80):
        return 'BS'
    # Barents Sea part 2
    elif (lon > 15) & (lon < 55) & (lat > 67) & (lat < 75):
        return 'BS'
    else:
        return False

################################################################################

## Read instrument makes a netcdf for just the given instrument
if True:
    read_instrmt('ITP', '1', science_data_file_path+'ITPs/itp1/itp1cormat', 'netcdfs/ITP_001.nc')
    read_instrmt('ITP', '2', science_data_file_path+'ITPs/itp2/itp2cormat', 'netcdfs/ITP_002.nc') # Not in time period
    read_instrmt('ITP', '3', science_data_file_path+'ITPs/itp3/itp3cormat', 'netcdfs/ITP_003.nc')
    read_instrmt('ITP', '4', science_data_file_path+'ITPs/itp4/itp4cormat', 'netcdfs/ITP_004.nc')
    read_instrmt('ITP', '5', science_data_file_path+'ITPs/itp5/itp5cormat', 'netcdfs/ITP_005.nc')
    read_instrmt('ITP', '6', science_data_file_path+'ITPs/itp6/itp6cormat', 'netcdfs/ITP_006.nc')
    # read_instrmt('ITP', '7', science_data_file_path+'ITPs/itp7/itp7cormat', 'netcdfs/ITP_007.nc') # Not in BGR
    read_instrmt('ITP', '8', science_data_file_path+'ITPs/itp8/itp8cormat', 'netcdfs/ITP_008.nc')
    # read_instrmt('ITP', '9', science_data_file_path+'ITPs/itp9/itp9cormat', 'netcdfs/ITP_009.nc') # Not in BGR
    # read_instrmt('ITP', '10', science_data_file_path+'ITPs/itp10/itp10cormat', 'netcdfs/ITP_010.nc') # Not in BGR
    read_instrmt('ITP', '11', science_data_file_path+'ITPs/itp11/itp11cormat', 'netcdfs/ITP_011.nc')
    # read_instrmt('ITP', '12', science_data_file_path+'ITPs/itp12/itp12cormat', 'netcdfs/ITP_012.nc') # Not in BGR
    read_instrmt('ITP', '13', science_data_file_path+'ITPs/itp13/itp13cormat', 'netcdfs/ITP_013.nc')
    # read_instrmt('ITP', '14', science_data_file_path+'ITPs/itp14/itp14cormat', 'netcdfs/ITP_014.nc') # Not in BGR
    # read_instrmt('ITP', '15', science_data_file_path+'ITPs/itp15/itp15cormat', 'netcdfs/ITP_015.nc') # Not in BGR
    # read_instrmt('ITP', '16', science_data_file_path+'ITPs/itp16/itp16cormat', 'netcdfs/ITP_016.nc') # Not in BGR
    # read_instrmt('ITP', '17', science_data_file_path+'ITPs/itp17/itp17cormat', 'netcdfs/ITP_017.nc') # Not in BGR
    read_instrmt('ITP', '18', science_data_file_path+'ITPs/itp18/itp18cormat', 'netcdfs/ITP_018.nc')
    # read_instrmt('ITP', '19', science_data_file_path+'ITPs/itp19/itp19cormat', 'netcdfs/ITP_019.nc') # Not in BGR
    read_instrmt('ITP', '21', science_data_file_path+'ITPs/itp21/itp21cormat', 'netcdfs/ITP_021.nc')
    # read_instrmt('ITP', '22', science_data_file_path+'ITPs/itp22/itp22cormat', 'netcdfs/ITP_022.nc') # Not in BGR
    # read_instrmt('ITP', '23', science_data_file_path+'ITPs/itp23/itp23cormat', 'netcdfs/ITP_023.nc') # Not in BGR
    # read_instrmt('ITP', '24', science_data_file_path+'ITPs/itp24/itp24cormat', 'netcdfs/ITP_024.nc') # Not in BGR
    read_instrmt('ITP', '25', science_data_file_path+'ITPs/itp25/itp25cormat', 'netcdfs/ITP_025.nc')
    # read_instrmt('ITP', '26', science_data_file_path+'ITPs/itp26/itp26cormat', 'netcdfs/ITP_026.nc') # Not in BGR
    # read_instrmt('ITP', '27', science_data_file_path+'ITPs/itp27/itp27cormat', 'netcdfs/ITP_027.nc') # Not in BGR
    # read_instrmt('ITP', '28', science_data_file_path+'ITPs/itp28/itp28cormat', 'netcdfs/ITP_028.nc') # Not in BGR
    # read_instrmt('ITP', '29', science_data_file_path+'ITPs/itp29/itp29cormat', 'netcdfs/ITP_029.nc') # Not in BGR
    read_instrmt('ITP', '30', science_data_file_path+'ITPs/itp30/itp30cormat', 'netcdfs/ITP_030.nc')
    read_instrmt('ITP', '32', science_data_file_path+'ITPs/itp32/itp32cormat', 'netcdfs/ITP_032.nc')
    read_instrmt('ITP', '33', science_data_file_path+'ITPs/itp33/itp33cormat', 'netcdfs/ITP_033.nc')
    read_instrmt('ITP', '34', science_data_file_path+'ITPs/itp34/itp34cormat', 'netcdfs/ITP_034.nc')
    read_instrmt('ITP', '35', science_data_file_path+'ITPs/itp35/itp35cormat', 'netcdfs/ITP_035.nc')
    # read_instrmt('ITP', '36', science_data_file_path+'ITPs/itp36/itp36cormat', 'netcdfs/ITP_036.nc') # Not in BGR
    # read_instrmt('ITP', '37', science_data_file_path+'ITPs/itp37/itp37cormat', 'netcdfs/ITP_037.nc') # Not in BGR
    # read_instrmt('ITP', '38', science_data_file_path+'ITPs/itp38/itp38cormat', 'netcdfs/ITP_038.nc') # Not in BGR
    read_instrmt('ITP', '41', science_data_file_path+'ITPs/itp41/itp41cormat', 'netcdfs/ITP_041.nc')
    read_instrmt('ITP', '42', science_data_file_path+'ITPs/itp42/itp42cormat', 'netcdfs/ITP_042.nc')
    read_instrmt('ITP', '43', science_data_file_path+'ITPs/itp43/itp43cormat', 'netcdfs/ITP_043.nc')
    # read_instrmt('ITP', '47', science_data_file_path+'ITPs/itp47/itp47cormat', 'netcdfs/ITP_047.nc') # Not in BGR
    # read_instrmt('ITP', '48', science_data_file_path+'ITPs/itp48/itp48cormat', 'netcdfs/ITP_048.nc') # Not in BGR
    # read_instrmt('ITP', '49', science_data_file_path+'ITPs/itp49/itp49cormat', 'netcdfs/ITP_049.nc') # Not in BGR
    # read_instrmt('ITP', '51', science_data_file_path+'ITPs/itp51/itp51cormat', 'netcdfs/ITP_051.nc') # Not in BGR
    read_instrmt('ITP', '52', science_data_file_path+'ITPs/itp52/itp52cormat', 'netcdfs/ITP_052.nc')
    read_instrmt('ITP', '53', science_data_file_path+'ITPs/itp53/itp53cormat', 'netcdfs/ITP_053.nc')
    read_instrmt('ITP', '54', science_data_file_path+'ITPs/itp54/itp54cormat', 'netcdfs/ITP_054.nc')
    read_instrmt('ITP', '55', science_data_file_path+'ITPs/itp55/itp55cormat', 'netcdfs/ITP_055.nc')
    # read_instrmt('ITP', '56', science_data_file_path+'ITPs/itp56/itp56cormat', 'netcdfs/ITP_056.nc') # Not in BGR
    # read_instrmt('ITP', '57', science_data_file_path+'ITPs/itp57/itp57cormat', 'netcdfs/ITP_057.nc') # Not in BGR
    # read_instrmt('ITP', '58', science_data_file_path+'ITPs/itp58/itp58cormat', 'netcdfs/ITP_058.nc') # Not in BGR
    # read_instrmt('ITP', '59', science_data_file_path+'ITPs/itp59/itp59cormat', 'netcdfs/ITP_059.nc') # Not in BGR
    # read_instrmt('ITP', '60', science_data_file_path+'ITPs/itp60/itp60cormat', 'netcdfs/ITP_060.nc') # Not in BGR
    # read_instrmt('ITP', '61', science_data_file_path+'ITPs/itp61/itp61cormat', 'netcdfs/ITP_061.nc') # Not in BGR
    read_instrmt('ITP', '62', science_data_file_path+'ITPs/itp62/itp62cormat', 'netcdfs/ITP_062.nc')
    # read_instrmt('ITP', '63', science_data_file_path+'ITPs/itp63/itp63cormat', 'netcdfs/ITP_063.nc') # Not in BGR
    read_instrmt('ITP', '64', science_data_file_path+'ITPs/itp64/itp64cormat', 'netcdfs/ITP_064.nc')
    read_instrmt('ITP', '65', science_data_file_path+'ITPs/itp65/itp65cormat', 'netcdfs/ITP_065.nc')
    read_instrmt('ITP', '68', science_data_file_path+'ITPs/itp68/itp68cormat', 'netcdfs/ITP_068.nc')
    read_instrmt('ITP', '69', science_data_file_path+'ITPs/itp69/itp69cormat', 'netcdfs/ITP_069.nc')
    read_instrmt('ITP', '70', science_data_file_path+'ITPs/itp70/itp70cormat', 'netcdfs/ITP_070.nc')
    # read_instrmt('ITP', '72', science_data_file_path+'ITPs/itp72/itp72cormat', 'netcdfs/ITP_072.nc') # Not in BGR
    # read_instrmt('ITP', '73', science_data_file_path+'ITPs/itp73/itp73cormat', 'netcdfs/ITP_073.nc') # Not in BGR
    # read_instrmt('ITP', '74', science_data_file_path+'ITPs/itp74/itp74cormat', 'netcdfs/ITP_074.nc') # Not in BGR
    # read_instrmt('ITP', '75', science_data_file_path+'ITPs/itp75/itp75cormat', 'netcdfs/ITP_075.nc') # Not in BGR
    # read_instrmt('ITP', '76', science_data_file_path+'ITPs/itp76/itp76cormat', 'netcdfs/ITP_076.nc') # Not in BGR
    read_instrmt('ITP', '77', science_data_file_path+'ITPs/itp77/itp77cormat', 'netcdfs/ITP_077.nc')
    read_instrmt('ITP', '78', science_data_file_path+'ITPs/itp78/itp78cormat', 'netcdfs/ITP_078.nc')
    read_instrmt('ITP', '79', science_data_file_path+'ITPs/itp79/itp79cormat', 'netcdfs/ITP_079.nc')
    read_instrmt('ITP', '80', science_data_file_path+'ITPs/itp80/itp80cormat', 'netcdfs/ITP_080.nc')
    read_instrmt('ITP', '81', science_data_file_path+'ITPs/itp81/itp81cormat', 'netcdfs/ITP_081.nc')
    read_instrmt('ITP', '82', science_data_file_path+'ITPs/itp82/itp82cormat', 'netcdfs/ITP_082.nc')
    # read_instrmt('ITP', '83', science_data_file_path+'ITPs/itp83/itp83cormat', 'netcdfs/ITP_083.nc') # Not in BGR
    read_instrmt('ITP', '84', science_data_file_path+'ITPs/itp84/itp84cormat', 'netcdfs/ITP_084.nc')
    read_instrmt('ITP', '85', science_data_file_path+'ITPs/itp85/itp85cormat', 'netcdfs/ITP_085.nc')
    read_instrmt('ITP', '86', science_data_file_path+'ITPs/itp86/itp86cormat', 'netcdfs/ITP_086.nc')
    read_instrmt('ITP', '87', science_data_file_path+'ITPs/itp87/itp87cormat', 'netcdfs/ITP_087.nc')
    read_instrmt('ITP', '88', science_data_file_path+'ITPs/itp88/itp88cormat', 'netcdfs/ITP_088.nc')
    read_instrmt('ITP', '89', science_data_file_path+'ITPs/itp89/itp89cormat', 'netcdfs/ITP_089.nc')
    # read_instrmt('ITP', '90', science_data_file_path+'ITPs/itp90/itp90cormat', 'netcdfs/ITP_090.nc') # Not in BGR
    # read_instrmt('ITP', '91', science_data_file_path+'ITPs/itp91/itp91cormat', 'netcdfs/ITP_091.nc') # Not in BGR
    # read_instrmt('ITP', '92', science_data_file_path+'ITPs/itp92/itp92cormat', 'netcdfs/ITP_092.nc') # Not in BGR
    # read_instrmt('ITP', '94', science_data_file_path+'ITPs/itp94/itp94cormat', 'netcdfs/ITP_094.nc') # Not in BGR
    # read_instrmt('ITP', '95', science_data_file_path+'ITPs/itp95/itp95cormat', 'netcdfs/ITP_095.nc') # Not in BGR
    read_instrmt('ITP', '97', science_data_file_path+'ITPs/itp97/itp97cormat', 'netcdfs/ITP_097.nc')
    # read_instrmt('ITP', '98', science_data_file_path+'ITPs/itp98/itp98cormat', 'netcdfs/ITP_098.nc') # Not in BGR
    read_instrmt('ITP', '99', science_data_file_path+'ITPs/itp99/itp99cormat', 'netcdfs/ITP_099.nc')
    read_instrmt('ITP', '100', science_data_file_path+'ITPs/itp100/itp100cormat', 'netcdfs/ITP_100.nc')
    read_instrmt('ITP', '101', science_data_file_path+'ITPs/itp101/itp101cormat', 'netcdfs/ITP_101.nc')
    # read_instrmt('ITP', '102', science_data_file_path+'ITPs/itp102/itp102cormat', 'netcdfs/ITP_102.nc') # Not in BGR
    read_instrmt('ITP', '103', science_data_file_path+'ITPs/itp103/itp103cormat', 'netcdfs/ITP_103.nc')
    read_instrmt('ITP', '104', science_data_file_path+'ITPs/itp104/itp104cormat', 'netcdfs/ITP_104.nc')
    read_instrmt('ITP', '105', science_data_file_path+'ITPs/itp105/itp105cormat', 'netcdfs/ITP_105.nc')
    read_instrmt('ITP', '107', science_data_file_path+'ITPs/itp107/itp107cormat', 'netcdfs/ITP_107.nc')
    read_instrmt('ITP', '108', science_data_file_path+'ITPs/itp108/itp108cormat', 'netcdfs/ITP_108.nc')
    read_instrmt('ITP', '109', science_data_file_path+'ITPs/itp109/itp109cormat', 'netcdfs/ITP_109.nc')
    read_instrmt('ITP', '110', science_data_file_path+'ITPs/itp110/itp110cormat', 'netcdfs/ITP_110.nc')
    # read_instrmt('ITP', '111', science_data_file_path+'ITPs/itp111/itp111cormat', 'netcdfs/ITP_111.nc') # Not in BGR
    read_instrmt('ITP', '113', science_data_file_path+'ITPs/itp113/itp113cormat', 'netcdfs/ITP_113.nc')
    read_instrmt('ITP', '114', science_data_file_path+'ITPs/itp114/itp114cormat', 'netcdfs/ITP_114.nc')
    # read_instrmt('ITP', '116', science_data_file_path+'ITPs/itp116/itp116cormat', 'netcdfs/ITP_116.nc') # Not in BGR
    read_instrmt('ITP', '117', science_data_file_path+'ITPs/itp117/itp117cormat', 'netcdfs/ITP_117.nc')
    read_instrmt('ITP', '118', science_data_file_path+'ITPs/itp118/itp118cormat', 'netcdfs/ITP_118.nc')
    read_instrmt('ITP', '120', science_data_file_path+'ITPs/itp120/itp120cormat', 'netcdfs/ITP_120.nc')
    read_instrmt('ITP', '121', science_data_file_path+'ITPs/itp121/itp121cormat', 'netcdfs/ITP_121.nc')
    read_instrmt('ITP', '122', science_data_file_path+'ITPs/itp122/itp122cormat', 'netcdfs/ITP_122.nc')
    read_instrmt('ITP', '123', science_data_file_path+'ITPs/itp123/itp123cormat', 'netcdfs/ITP_123.nc')
    # read_instrmt('ITP', '125', science_data_file_path+'ITPs/itp125/itp125cormat', 'netcdfs/ITP_125.nc')
    # read_instrmt('ITP', '128', science_data_file_path+'ITPs/itp128/itp128cormat', 'netcdfs/ITP_128.nc')

## This will make all the netcdfs for ITPs (takes a long time)
# make_all_ITP_netcdfs(science_data_file_path)

## This will make all the netcdfs for SHEBA
# make_SHEBA_netcdfs(science_data_file_path)

## This will make all the netcdfs for AIDJEX
# make_all_AIDJEX_netcdfs(science_data_file_path)
# read_instrmt('AIDJEX', 'BigBear', science_data_file_path+'AIDJEX/AIDJEX/BigBear', 'netcdfs/AIDJEX_BigBear.nc')

exit(0)

## Tests for how the netcdfs turned out

# ds2 = xr.load_dataset('netcdfs/ITP_2.nc')
# ds2['Time'] = pd.DatetimeIndex(ds2['Time'].values)
# # print ds.info() to only see the structure
# print(ds2.info())

# This gets all the data for a profile that has the given `prof_no`
# ds_test = ds2.where(ds2.prof_no==185, drop=True).squeeze()
# print(ds_test)
# ds_test.iT.plot()
# plt.show()

# This gets all the TS data from all profiles between certain depths
# ds_test = ds2[['press','iT','SP']]
# ds_test = ds_test.where(ds_test.press>200, drop=True).squeeze()
# ds_test = ds_test.where(ds_test.press<300, drop=True).squeeze()
# ds_test.plot.scatter(x='SP', y='iT', marker='.', s=0.5)
# plt.show()
