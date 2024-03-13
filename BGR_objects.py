"""
Author: Mikhail Schee
Created: 2024-03-04

This script makes common objects for the BGR project. It is used `figures.py`, `cluster_data.py`, and `HPC_param_sweep.py`.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions in source code must retain the accompanying copyright notice, this list of conditions, and the following disclaimer.
    2. Redistributions in binary form must reproduce the accompanying copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Names of the copyright holders must not be used to endorse or promote products derived from this software without prior written permission from the copyright holders.
    4. If any files are modified, you must cause the modified files to carry prominent notices stating that you changed the files and the date of any change.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# For custom analysis functions
import analysis_helper_functions as ahf
# For common BGR parameters
import BGR_params as bps

################################################################################
## Profile filter objects
################################################################################

# The profile filters I'm using for this study
pfs_BGR_test = ahf.Profile_Filters(lon_range=bps.lon_BGR, lat_range=bps.lat_BGR, p_range=[1000,5], SA_range=bps.test_S_range, lt_pTC_max=True)

# The profile filters to narrow down to an example area
pfs_ex_area = ahf.Profile_Filters(lon_range=bps.lon_ex_area, lat_range=bps.lat_ex_area, p_range=[1000,5], SA_range=bps.test_S_range, lt_pTC_max=True)

################################################################################
## Data filter objects
################################################################################

this_min_press = bps.this_min_press

# Data filter object using all default settings
dfs0 = ahf.Data_Filters()
# Data filter object that keeps all profiles
dfs_all = ahf.Data_Filters(keep_black_list=True, cast_direction='any')
# Data filter object that limits the minimum pressure
dfs1 = ahf.Data_Filters(min_press=this_min_press)

# Beaufort Gyre Region
## Data filter objects for different time periods
dfs1_BGR_dict = {'BGR04':dfs1,
                 'BGR0506':ahf.Data_Filters(min_press=this_min_press, date_range=['2005/08/15 00:00:00','2006/08/15 00:00:00']),
                 'BGR0607':ahf.Data_Filters(min_press=this_min_press, date_range=['2006/08/15 00:00:00','2007/08/15 00:00:00']),
                 'BGR0708':ahf.Data_Filters(min_press=this_min_press, date_range=['2007/08/15 00:00:00','2008/08/15 00:00:00']),
                 'BGR0809':ahf.Data_Filters(min_press=this_min_press, date_range=['2008/08/15 00:00:00','2009/08/15 00:00:00']),
                 'BGR0910':ahf.Data_Filters(min_press=this_min_press, date_range=['2009/08/15 00:00:00','2010/08/15 00:00:00']),
                 'BGR1011':ahf.Data_Filters(min_press=this_min_press, date_range=['2010/08/15 00:00:00','2011/08/15 00:00:00']),
                 'BGR1112':ahf.Data_Filters(min_press=this_min_press, date_range=['2011/08/15 00:00:00','2012/08/15 00:00:00']),
                 'BGR1213':ahf.Data_Filters(min_press=this_min_press, date_range=['2012/08/15 00:00:00','2013/08/15 00:00:00']),
                 'BGR1314':ahf.Data_Filters(min_press=this_min_press, date_range=['2013/08/15 00:00:00','2014/08/15 00:00:00']),
                 'BGR1415':ahf.Data_Filters(min_press=this_min_press, date_range=['2014/08/15 00:00:00','2015/08/15 00:00:00']),
                 'BGR1516':ahf.Data_Filters(min_press=this_min_press, date_range=['2015/08/15 00:00:00','2016/08/15 00:00:00']),
                 'BGR1617':ahf.Data_Filters(min_press=this_min_press, date_range=['2016/08/15 00:00:00','2017/08/15 00:00:00']),
                 'BGR1718':ahf.Data_Filters(min_press=this_min_press, date_range=['2017/08/15 00:00:00','2018/08/15 00:00:00']),
                 'BGR1819':ahf.Data_Filters(min_press=this_min_press, date_range=['2018/08/15 00:00:00','2019/08/15 00:00:00']),
                 'BGR1920':ahf.Data_Filters(min_press=this_min_press, date_range=['2019/08/15 00:00:00','2020/08/15 00:00:00']),
                 'BGR2021':ahf.Data_Filters(min_press=this_min_press, date_range=['2020/08/15 00:00:00','2021/08/15 00:00:00']),
                 'BGR2122':ahf.Data_Filters(min_press=this_min_press, date_range=['2021/08/15 00:00:00','2022/08/15 00:00:00']),
                 'BGR2223':ahf.Data_Filters(min_press=this_min_press, date_range=['2022/08/15 00:00:00','2023/08/15 00:00:00']),
                 'BGR_all':ahf.Data_Filters(min_press=this_min_press, date_range=['2005/08/15 00:00:00','2023/08/15 00:00:00'])
                 }

dfs1_BGR0506 = ahf.Data_Filters(min_press=this_min_press, date_range=['2005/08/15 00:00:00','2006/08/15 00:00:00'])
dfs1_BGR0607 = ahf.Data_Filters(min_press=this_min_press, date_range=['2006/08/15 00:00:00','2007/08/15 00:00:00'])
dfs1_BGR0708 = ahf.Data_Filters(min_press=this_min_press, date_range=['2007/08/15 00:00:00','2008/08/15 00:00:00'])
dfs1_BGR0809 = ahf.Data_Filters(min_press=this_min_press, date_range=['2008/08/15 00:00:00','2009/08/15 00:00:00'])
dfs1_BGR0910 = ahf.Data_Filters(min_press=this_min_press, date_range=['2009/08/15 00:00:00','2010/08/15 00:00:00'])
dfs1_BGR1011 = ahf.Data_Filters(min_press=this_min_press, date_range=['2010/08/15 00:00:00','2011/08/15 00:00:00'])
dfs1_BGR1112 = ahf.Data_Filters(min_press=this_min_press, date_range=['2011/08/15 00:00:00','2012/08/15 00:00:00'])
dfs1_BGR1213 = ahf.Data_Filters(min_press=this_min_press, date_range=['2012/08/15 00:00:00','2013/08/15 00:00:00'])
dfs1_BGR1314 = ahf.Data_Filters(min_press=this_min_press, date_range=['2013/08/15 00:00:00','2014/08/15 00:00:00'])
dfs1_BGR1415 = ahf.Data_Filters(min_press=this_min_press, date_range=['2014/08/15 00:00:00','2015/08/15 00:00:00'])
dfs1_BGR1516 = ahf.Data_Filters(min_press=this_min_press, date_range=['2015/08/15 00:00:00','2016/08/15 00:00:00'])
dfs1_BGR1617 = ahf.Data_Filters(min_press=this_min_press, date_range=['2016/08/15 00:00:00','2017/08/15 00:00:00'])
dfs1_BGR1718 = ahf.Data_Filters(min_press=this_min_press, date_range=['2017/08/15 00:00:00','2018/08/15 00:00:00'])
dfs1_BGR1819 = ahf.Data_Filters(min_press=this_min_press, date_range=['2018/08/15 00:00:00','2019/08/15 00:00:00'])
dfs1_BGR1920 = ahf.Data_Filters(min_press=this_min_press, date_range=['2019/08/15 00:00:00','2020/08/15 00:00:00'])
dfs1_BGR2021 = ahf.Data_Filters(min_press=this_min_press, date_range=['2020/08/15 00:00:00','2021/08/15 00:00:00'])
dfs1_BGR2122 = ahf.Data_Filters(min_press=this_min_press, date_range=['2021/08/15 00:00:00','2022/08/15 00:00:00'])
dfs1_BGR2223 = ahf.Data_Filters(min_press=this_min_press, date_range=['2022/08/15 00:00:00','2023/08/15 00:00:00'])

## combo
dfs1_BGR0508 = ahf.Data_Filters(min_press=this_min_press, date_range=['2005/08/15 00:00:00','2008/08/15 00:00:00'])
dfs1_BGR0511 = ahf.Data_Filters(min_press=this_min_press, date_range=['2005/08/15 00:00:00','2011/08/15 00:00:00'])