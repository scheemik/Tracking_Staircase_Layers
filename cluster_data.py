"""
Author: Mikhail Schee
Created: 2023-03-06

This script is set up to run the HDBSCAN clustering algorithm on a set of ITP 
data given certain inputs. It will fill in values in the 'cluster' and 
'clst_prob' columns, as well as for the global attributes:
'Last clustered'
'Clustering x-axis'
'Clustering y-axis'
'Clustering m_pts'
'Clustering ranges'
'Clustering DBCV'

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

# Import the Thermodynamic Equation of Seawater 2010 (TEOS-10) from GSW
# For converting from depth to pressure
import gsw

# For custom analysis functions
import analysis_helper_functions as ahf

################################################################################
# Main execution
################################################################################

dfs0 = ahf.Data_Filters()
dfs1 = ahf.Data_Filters(min_press=400)

LHW_S_range = [34.366, 35.5]
pfs_fltrd = ahf.Profile_Filters(lon_range=[-152.9,-133.7], lat_range=[72.6,77.4], SA_range=LHW_S_range, lt_pCT_max=True)
pfs_fltrd_ss = ahf.Profile_Filters(lon_range=[-152.9,-133.7], lat_range=[72.6,77.4], SA_range=LHW_S_range, lt_pCT_max=True, subsample=True)

## Reproducing figures from Timmermans et al. 2008
ITP2_clstr_dict = {#'netcdf_file':'netcdfs/ITP_2.nc',
                   'sources_dict':{'ITP_2':'all'},
                   'data_filters':dfs0,
                   'pfs_object':ahf.Profile_Filters(SP_range=[34.05,34.75]),
                   'cl_x_var':'SP',
                   'cl_y_var':'la_CT',
                   'm_pts':170
                   }

## Reproducing figures from Lu et al. 2022
ITP3_clstr_dict = {#'netcdf_file':'netcdfs/ITP_3.nc',
                   'sources_dict':{'ITP_3':'all'},
                   'data_filters':dfs0,
                   'SP_range':[34.21,34.82],
                   'cl_x_var':'SP',
                   'cl_y_var':'la_CT',
                   'm_pts':580
                   }

## AIDJEX BigBear
ABB_clstr_dict =  {'netcdf_file':'netcdfs/AIDJEX_BigBear.nc',
                   'sources_dict':{'AIDJEX_BigBear':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_fltrd,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'm_pts':300
                   }

## AIDJEX
AJX_clstr_dict =  {'netcdf_file':'netcdfs/AIDJEX_mpts_500_ell_010.nc',
                   'sources_dict':{'AIDJEX_BigBear':'all','AIDJEX_BlueFox':'all','AIDJEX_Caribou':'all','AIDJEX_Snowbird':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_fltrd,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'm_pts':500
                   }
AJX_clstr_dict =  {'netcdf_file':'netcdfs/AIDJEX_mpts_300_ell_100.nc',
                   'sources_dict':{'AIDJEX_BigBear':'all','AIDJEX_BlueFox':'all','AIDJEX_Caribou':'all','AIDJEX_Snowbird':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_fltrd,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'm_pts':300
                   }
AJX_clstr_dict =  {'netcdf_file':'netcdfs/AIDJEX_mpts_490_ell_050.nc',
                   'sources_dict':{'AIDJEX_BigBear':'all','AIDJEX_BlueFox':'all','AIDJEX_Caribou':'all','AIDJEX_Snowbird':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_fltrd,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'm_pts':490
                   }

## BGOS
BGOS_clstr_dict = {'netcdf_file':'netcdfs/BGOS_mpts_440_ell_010.nc',
                   'sources_dict':{'ITP_33':'all','ITP_34':'all','ITP_35':'all','ITP_41':'all','ITP_42':'all','ITP_43':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_fltrd,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'm_pts':440
                   }
BGOS_clstr_dict = {'netcdf_file':'netcdfs/BGOS_mpts_360_ell_100.nc',
                   'sources_dict':{'ITP_33':'all','ITP_34':'all','ITP_35':'all','ITP_41':'all','ITP_42':'all','ITP_43':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_fltrd,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'm_pts':360
                   }
BGOS_clstr_dict = {'netcdf_file':'netcdfs/BGOS_mpts_280_ell_050.nc',
                   'sources_dict':{'ITP_33':'all','ITP_34':'all','ITP_35':'all','ITP_41':'all','ITP_42':'all','ITP_43':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_fltrd,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'm_pts':280
                   }

## BGOSss
BGOSss_clstr_dict = {'netcdf_file':'netcdfs/BGOSss_mpts_260_ell_010.nc',
                   'sources_dict':{'ITP_33':'all','ITP_34':'all','ITP_35':'all','ITP_41':'all','ITP_42':'all','ITP_43':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_fltrd_ss,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'm_pts':260
                   }
BGOSss_clstr_dict = {'netcdf_file':'netcdfs/BGOSss_mpts_220_ell_100.nc',
                   'sources_dict':{'ITP_33':'all','ITP_34':'all','ITP_35':'all','ITP_41':'all','ITP_42':'all','ITP_43':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_fltrd_ss,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'm_pts':220
                   }
BGOSss_clstr_dict = {'netcdf_file':'netcdfs/BGOSss_mpts_340_ell_050.nc',
                   'sources_dict':{'ITP_33':'all','ITP_34':'all','ITP_35':'all','ITP_41':'all','ITP_42':'all','ITP_43':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_fltrd_ss,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'm_pts':340
                   }




# Data filters
this_min_press = 400
dfs1 = ahf.Data_Filters(min_press=this_min_press)
dfs1_BGR_0d = ahf.Data_Filters(min_press=this_min_press, date_range=['2007/08/11 00:00:00','2008/09/18 00:00:00'])
dfs1_BGR_0e = ahf.Data_Filters(min_press=this_min_press, date_range=['2008/09/18 00:00:00','2009/09/01 00:00:00'])
dfs1_BGR_0f = ahf.Data_Filters(min_press=this_min_press, date_range=['2009/08/31 00:00:08','2010/06/11 00:00:00'])
dfs1_BGR_0g = ahf.Data_Filters(min_press=this_min_press, date_range=['2010/06/11 00:00:00','2011/06/09 00:00:00'])
dfs1_BGR_0h = ahf.Data_Filters(min_press=this_min_press, date_range=['2011/06/09 00:00:00','2012/08/07 00:00:00'])
dfs1_BGR_0i = ahf.Data_Filters(min_press=this_min_press, date_range=['2012/08/07 00:00:00','2013/08/25 00:00:00'])
dfs1_BGR_0j = ahf.Data_Filters(min_press=this_min_press, date_range=['2013/08/25 00:00:00','2014/10/10 00:00:00'])
dfs1_BGR_0k = ahf.Data_Filters(min_press=this_min_press, date_range=['2013/08/25 00:00:00','2015/09/05 00:00:00'])
dfs1_BGR_0l = ahf.Data_Filters(min_press=this_min_press, date_range=['2015/09/05 00:00:00','2016/07/10 00:00:00'])
dfs1_BGR_0m = ahf.Data_Filters(min_press=this_min_press, date_range=['2016/07/10 00:00:00','2017/08/29 00:00:00'])
dfs1_BGR_0n = ahf.Data_Filters(min_press=this_min_press, date_range=['2017/08/29 00:00:00','2018/08/17 00:00:00'])
dfs1_BGR_0o = ahf.Data_Filters(min_press=this_min_press, date_range=['2018/08/17 00:00:00','2019/04/23 00:00:00'])
dfs1_BGR_0p = ahf.Data_Filters(min_press=this_min_press, date_range=['2019/04/23 00:00:00','2019/09/10 00:00:00'])
dfs1_BGR_0q = ahf.Data_Filters(min_press=this_min_press, date_range=['2019/09/10 00:00:00','2020/04/22 00:00:00'])
dfs1_BGR_0r = ahf.Data_Filters(min_press=this_min_press, date_range=['2020/04/22 00:00:00','2021/09/03 00:00:00'])
dfs1_BGR_0s = ahf.Data_Filters(min_press=this_min_press, date_range=['2020/07/01 00:00:00','2022/01/01 00:00:00'])
# By year
dfs1_BGR0506 = ahf.Data_Filters(min_press=this_min_press, date_range=['2005/08/15 00:00:00','2006/08/15 00:00:00'])
dfs1_BGR0607 = ahf.Data_Filters(min_press=this_min_press, date_range=['2006/08/15 00:00:00','2007/08/15 00:00:00'])
dfs1_BGR0507 = ahf.Data_Filters(min_press=this_min_press, date_range=['2005/08/15 00:00:00','2007/08/15 00:00:00'])
dfs1_BGR0708 = ahf.Data_Filters(min_press=this_min_press, date_range=['2007/08/15 00:00:00','2008/08/15 00:00:00'])
dfs1_BGR0809 = ahf.Data_Filters(min_press=this_min_press, date_range=['2008/08/15 00:00:00','2009/08/15 00:00:00'])
dfs1_BGR0910 = ahf.Data_Filters(min_press=this_min_press, date_range=['2009/08/15 00:00:00','2010/08/15 00:00:00'])

# Beaufort Gyre Region (BGR), see Shibley2022
lon_BGR = [-160,-130]
lat_BGR = [73,81.5]
LHW_S_range = [34.366, 35.5]
test_S_range = [34.4, 34.6]
pfs_BGR1 = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR, p_range=[1000,5], SA_range=LHW_S_range, lt_pCT_max=True)
pfs_test = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR, p_range=[1000,5], SA_range=test_S_range, lt_pCT_max=True)

## BGR ITPs 0a
BGRa_clstr_dict = {'netcdf_file':'netcdfs/BGRa_mpts_110.nc',
                   'sources_dict':{'ITP_002':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_BGR1,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':110
                   }

## BGR ITPs 0b
BGRb_clstr_dict = {'netcdf_file':'netcdfs/BGRb_mpts_380.nc',
                   'sources_dict':{'ITP_001':'all','ITP_003':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_BGR1,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':380
                   }

## BGR ITPs 0f
BGRf_clstr_dict = {'netcdf_file':'netcdfs/BGRf_mpts_310.nc',
                   'sources_dict':{'ITP_032':'all','ITP_033':'all','ITP_034':'all','ITP_035':'all'},
                   'data_filters':dfs1_BGR_0f,
                   'pfs_object':pfs_BGR1,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':310
                   }

## BGR ITPs 0m
BGRm_clstr_dict = {'netcdf_file':'netcdfs/BGRm_mpts_410.nc',
                   'sources_dict':{'ITP_097':'all','ITP_099':'all'},
                   'data_filters':dfs1_BGR_0m,
                   'pfs_object':pfs_BGR1,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':410
                   }

## BGR ITPs 0n
BGRn_clstr_dict = {'netcdf_file':'netcdfs/BGRn_mpts_240.nc',
                   'sources_dict':{'ITP_097':'all','ITP_100':'all','ITP_101':'all','ITP_108':'all'},
                   'data_filters':dfs1_BGR_0n,
                   'pfs_object':pfs_BGR1,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':240
                   }

## BGR ITPs 0o
BGRo_clstr_dict = {'netcdf_file':'netcdfs/BGRo_mpts_390.nc',
                   'sources_dict':{'ITP_103':'all','ITP_104':'all','ITP_105':'all','ITP_107':'all','ITP_109':'all','ITP_110':'all'},
                   'data_filters':dfs1_BGR_0o,
                   'pfs_object':pfs_BGR1,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':390
                   }

# By year

## BGR ITPs 04
BGR04_clstr_dict = {'netcdf_file':'netcdfs/BGR04.nc',
                   'sources_dict':{'ITP_002':'all'},
                #    'sources_dict':{'BGR04':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'None',
                #    'm_pts':110,
                   }
## BGR ITPs 0506
BGR0506_clstr_dict = {'netcdf_file':'netcdfs/BGR0506.nc',
                #    'sources_dict':{'ITP_001':'all','ITP_003':'all'},
                   'sources_dict':{'BGR0506':'all'},
                   'data_filters':dfs1_BGR0506,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':670
                   }
## BGR ITPs 0607
BGR0607_clstr_dict = {'netcdf_file':'netcdfs/BGR0607.nc',
                   'sources_dict':{'ITP_001':'all','ITP_003':'all','ITP_004':'all','ITP_005':'all','ITP_006':'all','ITP_008':'all'},
                   'data_filters':dfs1_BGR0607,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'None'
                   }

# Test parameter sweep with ITP3
ITP3t_clstr_dict = {'netcdf_file':'netcdfs/ITP3t.nc',
                   'sources_dict':{'ITP_003':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':350
                   }

for clstr_dict in [BGR0607_clstr_dict]:
    gattrs_to_print =  ['Last modified',
                        'Last modification',
                        'Last clustered',
                        'Clustering x-axis',
                        'Clustering y-axis',
                        'Clustering z-axis',
                        'Clustering m_pts',
                        'Clustering filters',
                        'Clustering DBCV']

    gattrs_to_copy = {        # Note: can't store datetime objects in netcdfs
                    'Title':'',
                    'Source':'',
                    'Instrument':'',
                    'Data Attribution':'',
                    'Data Source':'',
                    'Data Organizer':'',
                    'Reference':'',
                    'Original vertical measure':'',
                    'Original temperature measure':'',
                    'Original salinity measure':'',
                    'Sub-sample scheme':'',
                    'Moving average window':''
                    }
    # Find the netcdf to which to write the clustering results
    try:
        my_nc = clstr_dict['netcdf_file']
    except:
        my_nc = False
    # Make new netcdf to store clustering results
    if my_nc != False:
        print('Preparing to write to',my_nc)
        # Will make a new netcdf to store the clustering results
        ds_var_info = {}
        # ds_coord_info = {}
        # Create data set object
        ds_object = ahf.Data_Set(clstr_dict['sources_dict'], clstr_dict['data_filters'])
        # Copy down the global attributes to put into the new netcdf later
        # Loop through each dataset (one for each instrument)
        for ds in ds_object.arr_of_ds:
            # Loop through the variables to get the information
            # print(ds.variables)
            # exit(0)
            for var in ds.variables:
                var_ = ds.variables[var]
                # print('Variable:',var,'attrs:',var_.attrs)
                ds_var_info[var] = var_.attrs
            # print(ds_var_info)
            # print('')
            # print(ds.coords)
            # for coord in ds.coords:
            #     coord_ = ds.coords[coord]
            #     print(coord_.attrs)
            # exit(0)
            # Loop through each of that dataset's global attributes
            for attr in ds.attrs:
                # See if it's an attribute we want to copy
                if attr in gattrs_to_copy:
                    this_attr = str(ds.attrs[attr])
                    # See whether we've already copied it in
                    if this_attr not in gattrs_to_copy[attr]:
                        gattrs_to_copy[attr] = gattrs_to_copy[attr] + ds.attrs[attr] + ' '
                    #
                #
            #
        #
        # print(gattrs_to_copy)
        # exit(0)
        # Create profile filter object
        pfs_object = clstr_dict['pfs_object']
        # Create plot parameters object
        if clstr_dict['m_pts'] == 'None':
            keep_these_vars = ['ma_CT', 'entry', 'prof_no', 'CT', 'SA']
            # keep_these_vars = ['entry', 'prof_no', 'BL_yn', 'dt_start', 'dt_end', 'lon', 'lat', 'region', 'up_cast', 'press_max', 'CT_max', 'press_CT_max', 'SA_CT_max', 'R_rho', 'press', 'depth', 'iT', 'CT', 'PT', 'SP', 'SA', 'sigma', 'alpha', 'beta', 'aiT', 'aCT', 'aPT', 'BSP', 'BSA', 'ss_mask', 'ma_iT', 'ma_CT', 'ma_PT', 'ma_SP', 'ma_SA', 'ma_sigma', 'la_iT', 'la_CT', 'la_PT', 'la_SP', 'la_SA', 'la_sigma']
            pp_clstr = ahf.Plot_Parameters(x_vars=[clstr_dict['cl_x_var']], y_vars=[clstr_dict['cl_y_var']], clr_map='clr_by_instrmt', extra_args={'extra_vars_to_keep':keep_these_vars}, legend=True)
        else:
            # keep_these_vars = ['la_CT', 'entry', 'prof_no', 'CT', 'SA']
            keep_these_vars = ['entry', 'prof_no', 'BL_yn', 'dt_start', 'dt_end', 'lon', 'lat', 'region', 'up_cast', 'press_max', 'CT_max', 'press_CT_max', 'SA_CT_max', 'R_rho', 'press', 'depth', 'iT', 'CT', 'PT', 'SP', 'SA', 'sigma', 'alpha', 'beta', 'aCT', 'BSA', 'ss_mask', 'ma_iT', 'ma_CT', 'ma_PT', 'ma_SP', 'ma_SA', 'ma_sigma', 'la_iT', 'la_CT', 'la_PT', 'la_SP', 'la_SA', 'la_sigma']
            pp_clstr = ahf.Plot_Parameters(x_vars=[clstr_dict['cl_x_var']], y_vars=[clstr_dict['cl_y_var']], clr_map='cluster', extra_args={'b_a_w_plt':True, 'cl_x_var':clstr_dict['cl_x_var'], 'cl_y_var':clstr_dict['cl_y_var'], 'm_pts':clstr_dict['m_pts'], 'extra_vars_to_keep':keep_these_vars}, legend=True)
        # Create analysis group
        print('creating analysis group')
        group_test_clstr = ahf.Analysis_Group(ds_object, pfs_object, pp_clstr)
        # Make a figure to run clustering algorithm and check results
        print('making figure')
        ahf.make_figure([group_test_clstr])

        # Does this netcdf already exist?
        try:
            ds = xr.load_dataset(my_nc)
            print('- Previously,',my_nc)
            # See the variables before
            for attr in gattrs_to_print:
                print('\t',attr+':',ds.attrs[attr])
        except:
            foo = 2
        
        # Pull the dataframes from the analysis group
        dfs = group_test_clstr.data_frames
        # print('len(group_test_clstr.data_frames):',len(dfs))
        # exit(0)
        # Loop through each dataframe each (avoiding index collisions)
        # if 'AIDJEX' in gattrs_to_copy['Source']:
            # print('Avoiding index collisions')
            # for df in dfs:
            #     # Get list of instruments
            #     instrmts = np.unique(np.array(df['instrmt']))
            #     # Need to make the `Time` values unique for each instrument
            #     # Can only change values of columns, not indexes, so reset Time
            #     df.reset_index(inplace=True, level='Time')
            #     # Loop across each instrument
            #     i = 0
            #     for instrmt in instrmts:
            #         i += 1
            #         # Add i seconds to all the time values
            #         df.loc[df['instrmt'] == instrmt, ['Time']] = df.loc[df['instrmt'] == instrmt, ['Time']] + np.timedelta64(i, 's')
            # new_df = pd.concat(dfs)
            # # Make `Time` and index again
            # new_df = new_df.set_index('Time', append=True)
            # # Turn `Vertical` on and then off again as index to reset index order
            # new_df.reset_index(inplace=True, level='Vertical')
            # new_df = new_df.set_index('Vertical', append=True)
        #else:
        new_df = pd.concat(dfs)
        if not 'instrmt' in new_df.index.names:
            print('Avoiding index collisions for ITPs by adding `instrmt` as an index')
            new_df = new_df.set_index('instrmt', append=True)
        else:
            # Remove the superfluous 'instrmt' column
            new_df = new_df.drop('instrmt', axis=1)
        if clstr_dict['m_pts'] == 'None':
            # Remove unnecessary variables
            print('variables in new_df:',list(new_df))
            only_need_these_vars = ['entry','prof_no','CT','ma_CT','SA',]
            for var in list(new_df):
                if not var in only_need_these_vars:
                    new_df = new_df.drop(var, axis=1)
        print('after, vars in new_df:',list(new_df))
        # print('Duplicated indices')
        # print(new_df.index[new_df.index.duplicated()].unique())
        # exit(0)
        # Convert back to xarray dataset
        ds = new_df.to_xarray()
        # Put the global attributes back in
        for attr in gattrs_to_copy:
            ds.attrs[attr] = gattrs_to_copy[attr]
        # Update the global variables:
        ds.attrs['Creation date'] = str(datetime.now())
        ds.attrs['Last modified'] = str(datetime.now())
        ds.attrs['Last modification'] = 'Updated clustering'
        ds.attrs['Last clustered'] = str(datetime.now())
        ds.attrs['Clustering x-axis'] = clstr_dict['cl_x_var']
        ds.attrs['Clustering y-axis'] = clstr_dict['cl_y_var']
        ds.attrs['Clustering z-axis'] = clstr_dict['cl_z_var']
        ds.attrs['Clustering m_pts'] = clstr_dict['m_pts']
        ds.attrs['Clustering filters'] = ahf.print_profile_filters(clstr_dict['pfs_object'])
        ds.attrs['Clustering DBCV'] = group_test_clstr.data_set.arr_of_ds[0].attrs['Clustering DBCV']
        # Add the variable attributes back in
        for var in ds.variables:
            if var in ds_var_info.keys():
                ds.variables[var].attrs = ds_var_info[var]
        # Write out to netcdf
        print('Writing data to',my_nc)
        ds.to_netcdf(my_nc, 'w')
        # Load in with xarray
        ds2 = xr.load_dataset(my_nc)
        # See the variables after
        for attr in gattrs_to_print:
            print('\t',attr+':',ds2.attrs[attr])
        #
    # Overwrite existing netcdf with clustering results
    else:
        # Make sure only one, complete source file was specified
        if len(clstr_dict['sources_dict']) > 1:
            print('Must specify `netcdf_file` in cluster dictionary when using multiple sources')
            exit(0)
        else:
            # Find the netcdf file name of the original
            # Get a list which should be: (name_of_source, 'all')
            this_source = next(iter(clstr_dict['sources_dict'].items()))
            if this_source[1] == 'all':
                my_nc = 'netcdfs/' + this_source[0] + '.nc'
            else:
                print('Must specify `all` from a source to overwrite netcdf file')
                exit(0)
        # Create data set object
        ds_object = ahf.Data_Set(clstr_dict['sources_dict'], clstr_dict['data_filters'])
        # Find the xarray dataset
        ds = ds_object.xarrs[0]
        # Convert a subset of the dataset to a dataframe
        og_df = ds[['cluster', 'clst_prob']].squeeze().to_dataframe()
        # Get the original dimensions of the data to reshape the arrays later
        len0 = len(og_df.index.get_level_values(0).unique())
        len1 = len(og_df.index.get_level_values(1).unique())
        # See the variables before
        for attr in gattrs_to_print:
            print('\t',attr+':',ds.attrs[attr])

        # Create profile filter object
        pfs_object = clstr_dict['pfs_object']
        # Create plot parameters object
        pp_clstr = ahf.Plot_Parameters(x_vars=[clstr_dict['cl_x_var']], y_vars=[clstr_dict['cl_y_var']], clr_map='cluster', extra_args={'b_a_w_plt':False, 'cl_x_var':clstr_dict['cl_x_var'], 'cl_y_var':clstr_dict['cl_y_var'], 'm_pts':clstr_dict['m_pts']}, legend=False)
        # Create analysis group
        group_test_clstr = ahf.Analysis_Group(ds_object, pfs_object, pp_clstr)
        # Make a figure to run clustering algorithm and check results
        ahf.make_figure([group_test_clstr])

        print('making changes')
        # Pull the now modified dataframe from the analysis group
        new_df = group_test_clstr.data_frames[0][['cluster','clst_prob']]
        # Update the original dataframe from the netcdf with the filtered dataframe from plotting
        og_df.update(new_df)
        # Put the clustering variables back into the dataset
        ds['cluster'].values   = og_df['cluster'].values.reshape((len0, len1))
        ds['clst_prob'].values = og_df['clst_prob'].values.reshape((len0, len1))
        # Update the global variables:
        ds.attrs['Last modified'] = str(datetime.now())
        ds.attrs['Last modification'] = 'Updated clustering'
        ds.attrs['Last clustered'] = str(datetime.now())
        ds.attrs['Clustering x-axis'] = clstr_dict['cl_x_var']
        ds.attrs['Clustering y-axis'] = clstr_dict['cl_y_var']
        ds.attrs['Clustering m_pts'] = clstr_dict['m_pts']
        ds.attrs['Clustering filters'] = ahf.print_profile_filters(clstr_dict['pfs_object'])
        ds.attrs['Clustering DBCV'] = group_test_clstr.data_set.arr_of_ds[0].attrs['Clustering DBCV']
        # Write out to netcdf
        print('Writing data to',my_nc)
        ds.to_netcdf(my_nc, 'w')
        # Load in with xarray
        ds2 = xr.load_dataset(my_nc)
        # See the variables after
        for attr in gattrs_to_print:
            print('\t',attr+':',ds2.attrs[attr])


