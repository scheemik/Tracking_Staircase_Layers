"""
Author: Mikhail Schee
Created: 2024-03-04

This script holds common parameters for the BGR project. It is used `figures.py`, `cluster_data.py`, and `HPC_param_sweep.py`.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions in source code must retain the accompanying copyright notice, this list of conditions, and the following disclaimer.
    2. Redistributions in binary form must reproduce the accompanying copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Names of the copyright holders must not be used to endorse or promote products derived from this software without prior written permission from the copyright holders.
    4. If any files are modified, you must cause the modified files to carry prominent notices stating that you changed the files and the date of any change.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

################################################################################
# Define commonly used parameters
################################################################################

# The absolute salinity range I've selected for this study
S_range_full_TC = [32.75, 35.03]
S_range_LHW_AW = [34.1, 35.02] # [34.1, 35.03] # [34.366, 35.03]
test_S_range = [34.4, 34.6]

# Beaufort Gyre Region (BGR), see Shibley2022
lon_BGR = [-160,-130]
lat_BGR = [73,81.5]

# Ranges for the LHW and AW
LHW_plt_ranges = {
    'press_lims':[312,150],
    'CT_lims':[-0.78, -1.43],
    'SA_lims':[34.13, 34.09]
}
AW_plt_ranges = {
    'press_lims':[564,336],
    'CT_lims':[1.0, 0.5],
    'SA_lims':[35.05, 34.95]
}
# Add fit ranges to BGR_all_clstr_plt_ranges
var_lims = ['press_lims', 'CT_lims', 'SA_lims']
var_fit_lims = ['press-fit_lims', 'CT-fit_lims', 'SA-fit_lims']
for dict in [LHW_plt_ranges, AW_plt_ranges]:
    for i in range(len(var_lims)):
        # Find range of var lims
        var_lim_range = abs(dict[var_lims[i]][1] - dict[var_lims[i]][0])
        dict[var_fit_lims[i]] = [var_lim_range/2, -var_lim_range/2]

# Example area, chosen to have a few profiles from all time periods
# lon_ex_area = [-141.19, -140.27] # [-143.8,-139.8]
# lat_ex_area = [76.66, 77.33] # [76.6,77.6]
# lon_ex_area = [-143.35, -141.81]
# lat_ex_area = [76.23, 76.837] 
# Different example area, chosen to encompass only one profile which is close to
#   -150.1902, 74.3886, the location of the minimum point for cluster 27's pressure polyfit
temp_buff = 0.25
lon_ex_area = [-150.1902+temp_buff, -150.1902-temp_buff]
lat_ex_area = [74.3886+temp_buff, 74.3886-temp_buff]

# For this study, I'm only using profiles that extend down to at least 400 dbar
this_min_press = 400

################################################################################
# Make dictionaries for what data to load in and analyze
################################################################################

ITPsAll = {
                'ITP_001':'all',
                'ITP_002':'all',
                'ITP_003':'all',
                'ITP_004':'all',
                'ITP_005':'all',
                'ITP_006':'all',
                'ITP_007':'all',
                'ITP_008':'all',
                'ITP_009':'all',
                'ITP_010':'all',
                'ITP_011':'all',
                'ITP_012':'all',
                'ITP_013':'all',
                'ITP_014':'all',
                'ITP_015':'all',
                'ITP_016':'all',
                'ITP_017':'all',
                'ITP_018':'all',
                'ITP_019':'all',
                'ITP_021':'all',
                'ITP_022':'all',
                'ITP_023':'all',
                'ITP_024':'all',
                'ITP_025':'all',
                'ITP_026':'all',
                'ITP_027':'all',
                'ITP_028':'all',
                'ITP_029':'all',
                'ITP_030':'all',
                'ITP_032':'all',
                'ITP_033':'all',
                'ITP_034':'all',
                'ITP_035':'all',
                'ITP_036':'all',
                'ITP_037':'all',
                'ITP_038':'all',
                'ITP_041':'all',
                'ITP_042':'all',
                'ITP_043':'all',
                'ITP_047':'all',
                'ITP_048':'all',
                'ITP_049':'all',
                'ITP_051':'all',
                'ITP_052':'all',
                'ITP_053':'all',
                'ITP_054':'all',
                'ITP_055':'all',
                'ITP_056':'all',
                'ITP_057':'all',
                'ITP_058':'all',
                'ITP_059':'all',
                'ITP_060':'all',
                'ITP_061':'all',
                'ITP_062':'all',
                'ITP_063':'all',
                'ITP_064':'all',
                'ITP_065':'all',
                'ITP_068':'all',
                'ITP_069':'all',
                'ITP_070':'all',
                'ITP_072':'all',
                'ITP_073':'all',
                'ITP_074':'all',
                'ITP_075':'all',
                'ITP_076':'all',
                'ITP_077':'all',
                'ITP_078':'all',
                'ITP_079':'all',
                'ITP_080':'all',
                'ITP_081':'all',
                'ITP_082':'all',
                'ITP_083':'all',
                'ITP_084':'all',
                'ITP_085':'all',
                'ITP_086':'all',
                'ITP_087':'all',
                'ITP_088':'all',
                'ITP_089':'all',
                'ITP_090':'all',
                'ITP_091':'all',
                'ITP_092':'all',
                'ITP_094':'all',
                'ITP_095':'all',
                'ITP_097':'all',
                'ITP_098':'all',
                'ITP_099':'all',
                'ITP_100':'all',
                'ITP_101':'all',
                'ITP_102':'all',
                'ITP_103':'all',
                'ITP_104':'all',
                'ITP_105':'all',
                'ITP_107':'all',
                'ITP_108':'all',
                'ITP_109':'all',
                'ITP_110':'all',
                'ITP_111':'all',
                'ITP_113':'all',
                'ITP_114':'all',
                'ITP_116':'all',
                'ITP_117':'all',
                'ITP_118':'all',
                'ITP_120':'all',
                'ITP_121':'all',
                'ITP_122':'all',
                'ITP_123':'all',
                'ITP_125':'all',
                'ITP_128':'all',
                }


## Date ranges for the different time periods
date_range_dict = {'BGR04':['2004/08/15 00:00:00','2005/08/15 00:00:00'],
                 'BGR0506':['2005/08/15 00:00:00','2006/08/15 00:00:00'],
                 'BGR0607':['2006/08/15 00:00:00','2007/08/15 00:00:00'],
                 'BGR0708':['2007/08/15 00:00:00','2008/08/15 00:00:00'],
                 'BGR0809':['2008/08/15 00:00:00','2009/08/15 00:00:00'],
                 'BGR0910':['2009/08/15 00:00:00','2010/08/15 00:00:00'],
                 'BGR1011':['2010/08/15 00:00:00','2011/08/15 00:00:00'],
                 'BGR1112':['2011/08/15 00:00:00','2012/08/15 00:00:00'],
                 'BGR1213':['2012/08/15 00:00:00','2013/08/15 00:00:00'],
                 'BGR1314':['2013/08/15 00:00:00','2014/08/15 00:00:00'],
                 'BGR1415':['2014/08/15 00:00:00','2015/08/15 00:00:00'],
                 'BGR1516':['2015/08/15 00:00:00','2016/08/15 00:00:00'],
                 'BGR1617':['2016/08/15 00:00:00','2017/08/15 00:00:00'],
                 'BGR1718':['2017/08/15 00:00:00','2018/08/15 00:00:00'],
                 'BGR1819':['2018/08/15 00:00:00','2019/08/15 00:00:00'],
                 'BGR1920':['2019/08/15 00:00:00','2020/08/15 00:00:00'],
                 'BGR2021':['2020/08/15 00:00:00','2021/08/15 00:00:00'],
                 'BGR2122':['2021/08/15 00:00:00','2022/08/15 00:00:00'],
                 'BGR2223':['2022/08/15 00:00:00','2023/08/15 00:00:00'],
                 'BGR_all':['2005/08/15 00:00:00','2022/08/15 00:00:00'],
                 }

# Sets of ITPs within the BGR by year-long time periods
## 2004-08-20 00:00:01 to 2004-09-29 00:00:05, duration: 41 days
BGRITPs04   = {'ITP_002':'all'}
## 2005-08-16 06:00:01 to 2006-08-14 18:00:03, gap: 320 days, duration: 
BGRITPs0506 = { 'ITP_001':'all',
                # 'ITP_002':'all', # Not in this time range
                'ITP_003':'all',
                }
## 2006-08-15 06:00:02 to 2007-08-14 12:00:02, gap: 1 day, duration: 
BGRITPs0607 = { 'ITP_001':'all',
                # 'ITP_002':'all', # Not in this time range
                'ITP_003':'all',
                'ITP_004':'all',
                'ITP_005':'all',
                'ITP_006':'all',
                # 'ITP_007':'all', # Not in BGR
                'ITP_008':'all',
                # 'ITP_009':'all', # Not in BGR
                # 'ITP_010':'all', # Not in BGR
                # 'ITP_011':'all', # Not in this time range
                # 'ITP_012':'all', # Not in BGR
                'ITP_013':'all',  
                # 'ITP_014':'all', # Not in BGR
                # 'ITP_015':'all', # Not in BGR
                # 'ITP_016':'all', # Not in BGR
                # 'ITP_017':'all', # Not in BGR
                }
## 2007-08-15 00:00:01 to 2008-08-14 00:24:17, gap: 1 day, duration: 
BGRITPs0708 = { 'ITP_004':'all',
                'ITP_005':'all',
                'ITP_006':'all',
                # 'ITP_007':'all', # Not in BGR
                'ITP_008':'all',
                # 'ITP_009':'all', # Not in BGR
                # 'ITP_010':'all', # Not in BGR
                # 'ITP_011':'all', # Not in this time period
                # 'ITP_012':'all', # Not in BGR
                'ITP_013':'all',
                # 'ITP_014':'all', # Not in BGR
                # 'ITP_015':'all', # Not in BGR
                # 'ITP_016':'all', # Not in BGR
                # 'ITP_017':'all', # Not in BGR
                'ITP_018':'all',
                # 'ITP_019':'all', # Not in BGR
                'ITP_021':'all',
                # 'ITP_022':'all', # Not in BGR
                # 'ITP_023':'all', # Not in BGR
                # 'ITP_024':'all', # Not in BGR
                # 'ITP_025':'all', # Not in this time period
                # 'ITP_026':'all', # Not in BGR
                # 'ITP_027':'all', # Not in BGR
                # 'ITP_028':'all', # Not in BGR
                # 'ITP_029':'all', # Not in BGR
                'ITP_030':'all',
                }
## 2008-08-15 00:00:03 to 2009-08-12 00:00:07, gap: 1 day, duration: 
BGRITPs0809 = { 'ITP_008':'all',
                # 'ITP_009':'all', # Not in BGR
                # 'ITP_010':'all', # Not in BGR
                'ITP_011':'all',
                # 'ITP_012':'all', # Not in BGR
                # 'ITP_013':'all', # Not in this time period
                # 'ITP_014':'all', # Not in BGR
                # 'ITP_015':'all', # Not in BGR
                # 'ITP_016':'all', # Not in BGR
                # 'ITP_017':'all', # Not in BGR
                'ITP_018':'all',
                # 'ITP_019':'all', # Not in BGR
                'ITP_021':'all',
                # 'ITP_022':'all', # Not in BGR
                # 'ITP_023':'all', # Not in BGR
                # 'ITP_024':'all', # Not in BGR
                'ITP_025':'all',
                # 'ITP_026':'all', # Not in BGR
                # 'ITP_027':'all', # Not in BGR
                # 'ITP_028':'all', # Not in BGR
                # 'ITP_029':'all', # Not in BGR
                'ITP_030':'all',
                }
## 2009-08-19 00:00:07 to 2010-08-14 00:00:06
BGRITPs0910 = { 'ITP_021':'all',
                # 'ITP_022':'all', # Not in BGR
                # 'ITP_023':'all', # Not in BGR
                # 'ITP_024':'all', # Not in BGR
                # 'ITP_025':'all', # Not in this time period
                # 'ITP_026':'all', # Not in BGR
                # 'ITP_027':'all', # Not in BGR
                # 'ITP_028':'all', # Not in BGR
                # 'ITP_029':'all', # Not in BGR
                # 'ITP_030':'all', # Not in this time period
                'ITP_032':'all',
                'ITP_033':'all',
                'ITP_034':'all',
                'ITP_035':'all',
                # 'ITP_036':'all', # Not in BGR
                # 'ITP_037':'all', # Not in BGR
                # 'ITP_038':'all', # Not in BGR
                }
## 2010-08-15 00:00:06 to 2011-08-14 00:00:06
BGRITPs1011 = { 'ITP_033':'all',
                # 'ITP_034':'all', # Not in this time period
                # 'ITP_035':'all', # Not in this time period
                # 'ITP_036':'all', # Not in BGR
                # 'ITP_037':'all', # Not in BGR
                # 'ITP_038':'all', # Not in BGR
                'ITP_041':'all',
                'ITP_042':'all',
                'ITP_043':'all',
                }
## 2011-08-15 00:00:01 to 2012-08-14 00:00:07
BGRITPs1112 = { 'ITP_041':'all',
                # 'ITP_042':'all', # Not in this time period
                # 'ITP_043':'all', # Not in this time period
                # 'ITP_047':'all', # Not in BGR
                # 'ITP_048':'all', # Not in BGR
                # 'ITP_049':'all', # Not in BGR
                # 'ITP_051':'all', # Not in BGR
                'ITP_052':'all',
                'ITP_053':'all',
                'ITP_054':'all',
                'ITP_055':'all',
                }
## 2012-08-15 00:00:06 to 2013-08-12 00:00:06
BGRITPs1213 = { 'ITP_041':'all',
                # 'ITP_042':'all', # Not in this time period
                # 'ITP_043':'all', # Not in this time period
                # 'ITP_052':'all', # Not in this time period
                # 'ITP_053':'all', # Not in this time period
                # 'ITP_054':'all', # Not in this time period
                # 'ITP_055':'all', # Not in this time period
                # 'ITP_056':'all', # Not in BGR
                # 'ITP_057':'all', # Not in BGR
                # 'ITP_058':'all', # Not in BGR
                # 'ITP_059':'all', # Not in BGR
                # 'ITP_060':'all', # Not in BGR
                # 'ITP_061':'all', # Not in BGR
                'ITP_062':'all',
                # 'ITP_063':'all', # Not in BGR
                'ITP_064':'all',
                'ITP_065':'all',
                }
## 2013-08-22 00:02:02 to 2014-08-14 00:02:02
BGRITPs1314 = { 'ITP_064':'all',
                # 'ITP_065':'all', # Not in BGR
                'ITP_068':'all',
                'ITP_069':'all',
                'ITP_070':'all',
                # 'ITP_072':'all', # Not in BGR
                # 'ITP_073':'all', # Not in BGR
                # 'ITP_074':'all', # Not in BGR
                # 'ITP_075':'all', # Not in BGR
                # 'ITP_076':'all', # Not in BGR
                'ITP_077':'all',
                'ITP_078':'all',
                'ITP_079':'all',
                'ITP_080':'all',
                }
## 2014-08-15 00:02:02 to 2015-08-14 00:02:02
BGRITPs1415 = { 'ITP_077':'all',
                # 'ITP_078':'all', # Not in this time period
                'ITP_079':'all',
                'ITP_080':'all',
                'ITP_081':'all',
                'ITP_082':'all',
                # 'ITP_083':'all', # Not in BGR
                'ITP_084':'all',
                'ITP_085':'all',
                'ITP_086':'all',
                'ITP_087':'all',
                }
## 2015-08-15 00:02:01 to 2016-05-02 00:02:01
BGRITPs1516 = { 'ITP_082':'all',
                # 'ITP_083':'all', # Not in BGR
                # 'ITP_084':'all', # Not in this time period
                'ITP_085':'all',
                'ITP_086':'all',
                # 'ITP_087':'all', # Not in this time period
                'ITP_088':'all',
                'ITP_089':'all',
                # 'ITP_090':'all', # Not in BGR
                # 'ITP_091':'all', # Not in BGR
                # 'ITP_092':'all', # Not in BGR
                # 'ITP_094':'all', # Not in BGR
                # 'ITP_095':'all', # Not in BGR
                }
## 2016-10-03 00:02:02 to 2017-08-04 00:02:01
BGRITPs1617 = { 'ITP_097':'all',
                # 'ITP_098':'all', # Not in BGR
                'ITP_099':'all',
                }
## 2017-09-17 00:02:02 to 2018-07-21 00:02:02
BGRITPs1718 = { 'ITP_097':'all',
                # 'ITP_098':'all', # Not in BGR
                # 'ITP_099':'all', # Not in this time period
                'ITP_100':'all',
                'ITP_101':'all',
                # 'ITP_102':'all', # Not in BGR
                # 'ITP_103':'all', # Not in this time period
                # 'ITP_104':'all', # Not in this time period
                # 'ITP_105':'all', # Not in this time period
                # 'ITP_107':'all', # Not in this time period
                'ITP_108':'all',
                }
## 2018-09-19 00:02:03 to 2019-08-14 00:02:02
BGRITPs1819 = { 'ITP_103':'all',
                'ITP_104':'all',
                'ITP_105':'all',
                'ITP_107':'all',
                # 'ITP_108':'all', # Not in this time period
                'ITP_109':'all',
                'ITP_110':'all',
                }
## 2019-08-16 00:02:01 to 2020-08-14 17:33:46
BGRITPs1920 = { 'ITP_105':'all',
                # 'ITP_107':'all', # Not in this time period
                # 'ITP_108':'all', # Not in this time period
                # 'ITP_109':'all', # Not in this time period
                # 'ITP_110':'all', # Not in this time period
                # 'ITP_111':'all',
                'ITP_113':'all',
                'ITP_114':'all',
                # 'ITP_116':'all', # Not in BGR
                'ITP_117':'all',
                'ITP_118':'all',
                }
## 2020-08-18 17:33:46 to 2021-08-14 00:02:02
BGRITPs2021 = { 'ITP_113':'all',
                'ITP_114':'all',
                # 'ITP_116':'all', # Not in BGR
                # 'ITP_117':'all', # Not in this time period
                # 'ITP_118':'all', # Not in this time period
                'ITP_120':'all',
                'ITP_121':'all',
                }
## 2021-08-15 00:02:02 to 2022-08-11 12:02:02
BGRITPs2122 = { 'ITP_120':'all',
                'ITP_121':'all',
                'ITP_122':'all',
                'ITP_123':'all',
                # 'ITP_125':'all',
                # 'ITP_128':'all',
                }
## 2022-08-21 12:02:02 to 2022-12-23 00:02:02
BGRITPs2223 = { 'ITP_122':'all',
                # 'ITP_123':'all',
                # 'ITP_125':'all',
                # 'ITP_128':'all',
                }

## All ITPs in the BGR
BGRITPsAll  = {**BGRITPs0506, **BGRITPs0607, **BGRITPs0708, **BGRITPs0809, **BGRITPs0910, **BGRITPs1011, **BGRITPs1112, **BGRITPs1213, **BGRITPs1314, **BGRITPs1415, **BGRITPs1516, **BGRITPs1617, **BGRITPs1718, **BGRITPs1819, **BGRITPs1920, **BGRITPs2021, **BGRITPs2122, **BGRITPs2223}

BGRITPs_dict = {'BGR04':BGRITPs04,
                'BGR0506':BGRITPs0506,
                'BGR0607':BGRITPs0607,
                'BGR0708':BGRITPs0708,
                'BGR0809':BGRITPs0809,
                'BGR0910':BGRITPs0910,
                'BGR1011':BGRITPs1011,
                'BGR1112':BGRITPs1112,
                'BGR1213':BGRITPs1213,
                'BGR1314':BGRITPs1314,
                'BGR1415':BGRITPs1415,
                'BGR1516':BGRITPs1516,
                'BGR1617':BGRITPs1617,
                'BGR1718':BGRITPs1718,
                'BGR1819':BGRITPs1819,
                'BGR1920':BGRITPs1920,
                'BGR2021':BGRITPs2021,
                'BGR2122':BGRITPs2122,
                'BGR2223':BGRITPs2223,
                'BGR_all':BGRITPsAll,
                }

## Pre-clustered files
# From HPC
BGR_HPC_clstrd_dict = { 'BGR04':{'HPC_BGR04_clstrd':'all'},
                        'BGR0506':{'HPC_BGR0506_clstrd':'all'},
                        'BGR0607':{'HPC_BGR0607_clstrd':'all'},
                        'BGR0708':{'HPC_BGR0708_clstrd':'all'},
                        'BGR0809':{'HPC_BGR0809_clstrd':'all'},
                        'BGR0910':{'HPC_BGR0910_clstrd':'all'},
                        'BGR1011':{'HPC_BGR1011_clstrd':'all'},
                        'BGR1112':{'HPC_BGR1112_clstrd':'all'},
                        'BGR1213':{'HPC_BGR1213_clstrd':'all'},
                        'BGR1314':{'HPC_BGR1314_clstrd':'all'},
                        'BGR1415':{'HPC_BGR1415_clstrd':'all'},
                        'BGR1516':{'HPC_BGR1516_clstrd':'all'},
                        'BGR1617':{'HPC_BGR1617_clstrd':'all'},
                        'BGR1718':{'HPC_BGR1718_clstrd':'all'},
                        'BGR1819':{'HPC_BGR1819_clstrd':'all'},
                        'BGR1920':{'HPC_BGR1920_clstrd':'all'},
                        'BGR2021':{'HPC_BGR2021_clstrd':'all'},
                        'BGR2122':{'HPC_BGR2122_clstrd':'all'},
                        'BGR2223':{'HPC_BGR2223_clstrd':'all'},
                        'BGR_all':{ 
                                    'HPC_BGR0506_clstrd':'all',
                                    'HPC_BGR0607_clstrd':'all',
                                    'HPC_BGR0708_clstrd':'all',
                                    'HPC_BGR0809_clstrd':'all',
                                    'HPC_BGR0910_clstrd':'all',
                                    'HPC_BGR1011_clstrd':'all',
                                    'HPC_BGR1112_clstrd':'all',
                                    'HPC_BGR1213_clstrd':'all',
                                    'HPC_BGR1314_clstrd':'all',
                                    'HPC_BGR1415_clstrd':'all',
                                    'HPC_BGR1516_clstrd':'all',
                                    'HPC_BGR1617_clstrd':'all',
                                    'HPC_BGR1718_clstrd':'all',
                                    'HPC_BGR1819_clstrd':'all',
                                    'HPC_BGR1920_clstrd':'all',
                                    'HPC_BGR2021_clstrd':'all',
                                    'HPC_BGR2122_clstrd':'all',
                                    }
}

# From HPC, before relabeling
BGR_HPC_unrelab_dict = {  'BGR04':{'HPC_BGR04_clstrd_unrelab':'all'},
                        'BGR0506':{'HPC_BGR0506_clstrd_unrelab':'all'},
                        'BGR0607':{'HPC_BGR0607_clstrd_unrelab':'all'},
                        'BGR0708':{'HPC_BGR0708_clstrd_unrelab':'all'},
                        'BGR0809':{'HPC_BGR0809_clstrd_unrelab':'all'},
                        'BGR0910':{'HPC_BGR0910_clstrd_unrelab':'all'},
                        'BGR1011':{'HPC_BGR1011_clstrd_unrelab':'all'},
                        'BGR1112':{'HPC_BGR1112_clstrd_unrelab':'all'},
                        'BGR1213':{'HPC_BGR1213_clstrd_unrelab':'all'},
                        'BGR1314':{'HPC_BGR1314_clstrd_unrelab':'all'},
                        'BGR1415':{'HPC_BGR1415_clstrd_unrelab':'all'},
                        'BGR1516':{'HPC_BGR1516_clstrd_unrelab':'all'},
                        'BGR1617':{'HPC_BGR1617_clstrd_unrelab':'all'},
                        'BGR1718':{'HPC_BGR1718_clstrd_unrelab':'all'},
                        'BGR1819':{'HPC_BGR1819_clstrd_unrelab':'all'},
                        'BGR1920':{'HPC_BGR1920_clstrd_unrelab':'all'},
                        'BGR2021':{'HPC_BGR2021_clstrd_unrelab':'all'},
                        'BGR2122':{'HPC_BGR2122_clstrd_unrelab':'all'},
                        'BGR2223':{'HPC_BGR2223_clstrd_unrelab':'all'},
                        'BGR_all':{ 
                                    'HPC_BGR0506_clstrd_unrelab':'all',
                                    'HPC_BGR0607_clstrd_unrelab':'all',
                                    'HPC_BGR0708_clstrd_unrelab':'all',
                                    'HPC_BGR0809_clstrd_unrelab':'all',
                                    'HPC_BGR0910_clstrd_unrelab':'all',
                                    'HPC_BGR1011_clstrd_unrelab':'all',
                                    'HPC_BGR1112_clstrd_unrelab':'all',
                                    'HPC_BGR1213_clstrd_unrelab':'all',
                                    'HPC_BGR1314_clstrd_unrelab':'all',
                                    'HPC_BGR1415_clstrd_unrelab':'all',
                                    'HPC_BGR1516_clstrd_unrelab':'all',
                                    'HPC_BGR1617_clstrd_unrelab':'all',
                                    'HPC_BGR1718_clstrd_unrelab':'all',
                                    'HPC_BGR1819_clstrd_unrelab':'all',
                                    'HPC_BGR1920_clstrd_unrelab':'all',
                                    'HPC_BGR2021_clstrd_unrelab':'all',
                                    'HPC_BGR2122_clstrd_unrelab':'all',
                                    }
}

## Relabeling clusters
# From HPC
BGR_HPC_relab_dict  = { 
    'BGR04':{},
    'BGR0506':{1:4, 2:6, 3:7, 4:10, 5:13, 6:15, 7:17, 8:19, 9:22, 10:23, 11:26, 12:30, 13:31, 14:32, 15:34, 16:37, 17:39, 18:42, 19:44, 20:46, 21:50, 22:52, 23:54, 24:57, 25:58, 26:60, 27:62, 28:63, 29:64, 30:67, 31:69, 32:70, 33:72, 34:75, 35:76, 36:76, 37:77, 38:78, 39:78, 40:79, 41:81, 42:83, 43:83, 42:83, 44:84, 45:86, 46:88, 47:90, 48:94, 49:94, 50:96, 51:120, 52:121, 53:122, 54:123, 55:124, 56:125, 57:126, 58:127, 59:128, 60:129, 61:130, 62:131, 63:132, 64:133, 65:134, 66:135, 67:136, 68:137, 69:138, 70:139, 71:140, 72:141, 73:142, 74:143, 75:144, 76:145, 77:146, 78:147},
    'BGR0607':{1:4, 2:6, 3:7, 4:10, 5:13, 6:15, 7:17, 8:19, 9:22, 10:23, 11:26, 12:30, 13:31, 14:32, 15:34, 16:37, 17:39, 18:42, 19:44, 20:46, 21:50, 22:52, 23:54, 24:57, 25:58, 26:60, 27:62, 28:63, 29:64, 30:67, 31:69, 32:72, 33:75, 34:77, 35:78, 36:79, 37:82, 38:84, 39:88, 40:88, 41:90, 42:93, 43:96, 44:120, 45:121, 46:122, 47:123, 48:124, 49:125, 50:126, 51:127, 52:128, 53:129, 54:130, 55:131, 56:132, 57:133, 58:134, 59:135, 60:136, 61:137, 62:138, 63:139, 64:140, 65:141, 66:142, 67:143, 68:144},
    'BGR0708':{0:3, 1:12, 2:16, 3:19, 4:21, 5:22, 6:23, 7:26, 8:30, 9:31, 10:34, 11:35, 12:37, 13:39, 14:41, 15:43, 16:44, 17:46, 18:50, 19:52, 20:53, 21:55, 22:57, 23:58, 24:60, 25:62, 26:63, 27:64, 28:67, 29:69, 30:72, 31:75, 32:77, 33:77, 34:79, 35:81, 36:83, 37:84, 38:88, 39:88, 40:90, 41:93, 42:96, 43:120, 44:121, 45:122, 46:123, 47:124, 48:125, 49:126, 50:127, 51:128, 52:129, 53:130, 54:131, 55:132, 56:133, 57:134},
    'BGR0809':{0:3, 1:12, 2:17, 3:19, 4:21, 5:23, 6:28, 7:32, 8:35, 9:37, 10:39, 11:41, 12:43, 13:44, 14:46, 15:50, 16:52, 17:54, 18:57, 19:58, 20:60, 21:62, 22:63, 23:64, 24:67, 25:69, 26:72, 27:75, 28:77, 29:79, 30:81, 31:83, 32:84, 33:87, 34:90, 35:91, 36:93, 37:96, 38:120, 39:121, 40:122, 41:123, 42:124, 43:125, 44:126, 45:127, 46:128, 47:129, 48:130, 49:131, 50:132, 51:133, 52:134, 53:135, 54:136, 55:137, 56:138, 57:139, 58:140, 59:141, 60:142, 61:143, 62:144, 63:145, 64:146},
    'BGR0910':{0:1, 1:5, 2:10, 3:12, 4:17, 5:19, 6:21, 7:22, 8:23, 9:26, 10:30, 11:31, 12:32, 13:35, 14:37, 15:39, 16:41, 17:43, 18:44, 19:46, 20:50, 21:52, 22:53, 23:55, 24:57, 25:58, 26:58, 27:60, 28:62, 29:63, 30:64, 31:66, 32:67, 33:69, 34:70, 35:72, 36:75, 37:77, 38:77, 39:79, 40:80, 41:81, 42:83, 43:84, 44:86, 45:88, 46:90, 47:93, 48:95, 49:95, 50:97, 51:97, 52:97, 53:120, 54:121, 55:122, 56:123, 57:124, 58:125, 59:126, 60:127, 61:128, 62:129, 63:130, 64:131, 65:132, 66:133, 67:134, 68:135, 69:136, 70:137, 71:138, 72:139, 73:140, 74:141, 75:142, 76:143, 77:144, 78:145, 79:146, 80:147, 81:148, 82:149, 83:150, 84:151},
    'BGR1011':{0:4, 1:5, 2:10, 3:12, 4:17, 5:19, 6:21, 7:22, 8:23, 9:26, 10:30, 11:31, 12:32, 13:35, 14:37, 15:39, 16:41, 17:43, 18:44, 19:46, 20:46, 21:50, 22:52, 23:53, 24:55, 25:57, 26:58, 27:58, 28:59, 29:60, 30:62, 31:63, 32:64, 33:66, 34:67, 35:69, 36:70, 37:72, 38:75, 39:76, 40:77, 41:79, 42:80, 43:81, 44:83, 45:84, 46:84, 47:86, 48:88, 49:89, 50:90, 51:91, 52:92, 53:94, 54:95, 55:95, 56:97, 57:97, 58:120, 59:121, 60:122, 61:123, 62:124, 63:125, 64:126, 65:127, 66:128, 67:129, 68:130, 69:131, 70:132, 71:133, 72:134, 73:135, 74:136, 75:137, 76:138, 77:139, 78:140, 79:141, 80:142, 81:143, 82:144},
    'BGR1112':{0:2, 1:8, 2:15, 3:17, 4:19, 5:20, 6:23, 7:27, 8:31, 9:32, 10:35, 11:37, 12:39, 13:41, 14:43, 15:44, 16:46, 17:50, 18:52, 19:54, 20:57, 21:58, 22:60, 23:62, 24:63, 25:64, 26:67, 27:69, 28:72, 29:75, 30:77, 31:79, 32:81, 33:83, 34:84, 35:87, 36:90, 37:93, 38:96, 39:120, 40:121, 41:122, 42:123, 43:124, 44:125, 45:126, 46:127, 47:128, 48:129, 49:130, 50:131, 51:132, 52:133, 53:134},
    'BGR1213':{0:2, 1:11, 2:20, 3:28, 4:32, 5:35, 6:37, 7:40, 8:43, 9:44, 10:46, 11:51, 12:54, 13:57, 14:58, 15:60, 16:62, 17:63, 18:64, 19:67, 20:69, 21:71, 22:75, 23:77, 24:79, 25:81, 26:83, 27:84, 28:86, 29:88, 30:90, 31:93, 32:96, 33:120, 34:121, 35:122, 36:123, 37:124, 38:125, 39:126, 40:127, 41:128, 42:129, 43:130, 44:131, 45:132, 46:133, 47:134, 48:135},
    'BGR1314':{0:2, 1:9, 2:14, 3:18, 4:20, 5:25, 6:31, 7:32, 8:35, 9:37, 10:40, 11:43, 12:44, 13:46, 14:50, 15:52, 16:54, 17:57, 18:58, 19:60, 20:62, 21:63, 22:64, 23:67, 24:69, 25:72, 26:75, 27:76, 28:77, 29:79, 30:81, 31:83, 32:84, 33:86, 34:88, 35:90, 36:93, 37:96, 38:120, 39:121, 40:122, 41:123, 42:124, 43:125, 44:126, 45:127, 46:128},
    'BGR1415':{1:4, 2:6, 3:7, 4:10, 5:13, 6:15, 7:17, 8:19, 9:20, 10:23, 11:27, 12:31, 13:32, 14:35, 15:38,  16:39, 17:41, 18:43, 19:44, 20:46, 21:48, 22:50, 23:52, 24:53, 25:55, 26:57, 27:58, 28:59, 29:60, 30:62, 31:63, 32:64, 33:66, 34:67, 35:69, 36:70, 37:72, 38:75, 39:76, 40:77, 41:79, 42:80, 43:81, 44:81, 45:83, 46:83, 47:84, 48:86, 49:86, 50:88, 51:89, 52:90, 53:92, 54:94, 55:94, 56:95, 57:96, 58:97, 59:120, 60:121, 61:122, 62:123, 63:124, 64:125, 65:126, 66:127, 67:128, 68:129, 69:130, 70:131, 71:132, 72:133, 73:134, 74:135, 75:136, 76:137, 77:138, 78:139, 79:140, 80:141, 81:142, 82:143},
    'BGR1516':{1:4, 2:6, 3:9, 4:14, 5:18, 6:20, 7:23, 8:27, 9:31, 10:32, 11:35, 12:40, 13:43, 14:44, 15:47, 16:50, 17:52, 18:53, 19:56, 20:58, 21:59, 22:61, 23:63, 24:64, 25:66, 26:67, 27:69, 28:70, 29:72, 30:75, 31:76, 32:77, 33:79, 34:81, 35:83, 36:84, 37:86, 38:88, 39:90, 40:93, 41:95, 42:96, 43:97, 44:120, 45:121, 46:122, 47:123, 48:124, 49:125, 50:126, 51:127, 52:128, 53:129, 54:130, 55:131, 56:132, 57:133, 58:134, 59:135, 60:136, 61:137, 62:138, 63:139},
    'BGR1617':{1:4, 2:6, 3:7, 4:10, 5:14, 6:17, 7:20, 8:23, 9:27, 10:31, 11:32, 12:35, 13:38, 14:41, 15:43, 16:45, 17:48, 18:50, 19:52, 20:53, 21:55, 22:57, 23:57, 24:59, 25:60, 26:62, 27:63, 28:64, 29:66, 30:67, 31:69, 32:70, 33:72, 34:75, 35:76, 36:77, 37:78, 38:79, 39:80, 40:81, 41:83, 42:84, 43:86, 44:86, 45:89, 46:90, 47:92, 48:92, 49:95, 50:96, 51:97, 52:97, 53:120, 54:121, 55:122, 56:123, 57:124, 58:125, 59:126, 60:127, 61:128, 62:129, 63:130, 64:131, 65:132, 66:133, 67:134, 68:135, 69:136, 70:137, 71:138, 72:139, 73:140, 74:141, 75:142, 76:143, 77:144, 78:145, 79:146, 80:147, 81:148, 82:149, 83:150, 84:151, 85:152},
    'BGR1718':{0:2, 1:6, 2:12, 3:17, 4:19, 5:24, 6:27, 7:31, 8:32, 9:35, 10:37, 11:39, 12:41, 13:43, 14:44, 15:47, 16:50, 17:52, 18:53, 19:55, 20:57, 21:59, 22:60, 23:62, 24:63, 25:64, 26:66, 27:67, 28:69, 29:70, 30:72, 31:75, 32:76, 33:77, 34:78, 35:79, 36:80, 37:81, 38:83, 39:84, 40:87, 41:89, 42:90, 43:93, 44:96, 44:97, 45:120, 46:121, 47:122, 48:123, 49:124, 50:125, 51:126, 52:127, 53:128, 54:129, 55:130, 56:131, 57:132, 58:133, 59:134, 60:135, 61:136, 62:137, 63:138, 64:139, 65:140, 66:141},
    'BGR1819':{0:4, 1:6, 2:9, 3:14, 4:18, 5:24, 6:27, 7:31, 8:32, 9:35, 10:38, 11:42, 12:44, 13:46, 14:51, 15:53, 16:56, 17:59, 18:61, 19:63, 20:64, 21:66, 22:68, 23:70, 24:74, 25:76, 26:77, 27:79, 28:81, 29:83, 30:83, 31:84, 32:86, 33:86, 34:89, 35:90, 36:92, 37:94, 38:96, 39:97, 40:120, 41:121, 42:122, 43:123, 44:124, 45:125, 46:126, 47:127, 48:128, 49:129, 50:130, 51:131, 52:132, 53:133, 54:134, 55:135, 56:136, 57:137, 58:138, 59:139, 60:140, 61:141, 62:142, 63:143, 64:144, 65:145, 66:146, 67:147, 68:148, 69:149},
    'BGR1920':{0:1, 1:6, 2:14, 3:17, 4:19, 5:24, 6:23, 7:29, 8:33, 9:35, 10:38, 11:42, 12:44, 13:46, 14:50, 15:52, 16:53, 17:55, 18:57, 19:58, 20:60, 21:62, 22:63, 23:64, 24:66, 25:67, 26:69, 27:70, 28:72, 29:73, 30:75, 31:76, 32:77, 33:78, 34:79, 35:80, 36:81, 37:83, 38:84, 39:87, 40:89, 41:90, 42:92, 43:94, 44:95, 45:96, 46:97, 47:120, 48:121, 49:122, 50:123, 51:124, 52:125, 53:126, 54:127, 55:128, 56:129, 57:130, 58:131, 59:132, 60:133},
    'BGR2021':{0:1, 1:6, 2:7, 3:10, 4:14, 5:17, 6:19, 7:24, 8:23, 9:29, 10:31, 11:36, 12:37, 13:39, 14:42, 15:44, 16:46, 17:47, 18:52, 19:53, 20:55, 21:57, 22:58, 23:60, 24:62, 25:63, 26:65, 27:67, 28:69, 29:70, 30:72, 31:74, 32:77, 33:78, 34:79, 35:80, 36:81, 37:83, 38:84, 39:86, 40:88, 41:90, 42:92, 43:94, 44:95, 45:96, 46:97, 47:120, 48:121, 49:122, 50:123, 51:124, 52:125, 53:126, 54:127, 55:128, 56:129, 57:130, 58:131, 59:132, 60:133, 61:134, 62:135, 63:136, 64:137, 65:138, 66:139, 67:140},
    'BGR2122':{0:6, 1:7, 2:10, 3:14, 4:17, 5:19, 6:24, 7:23, 8:29, 9:31, 10:32, 11:35, 12:37, 13:40, 14:44, 15:46, 16:47, 17:52, 18:53, 19:55, 20:57, 21:58, 22:60, 23:62, 24:63, 25:65, 26:67, 27:69, 28:70, 29:72, 30:73, 31:76, 32:77, 33:78, 34:79, 35:80, 36:81, 37:83, 38:85, 39:88, 40:90, 41:92, 42:94, 43:96, 44:97, 45:120, 46:121, 47:122, 48:123, 49:124, 50:125, 51:126, 52:127, 53:128, 54:129, 55:130, 56:131, 57:132, 58:133, 59:134, 60:135, 61:136, 62:137},
    'BGR2223':{},
    'BGR_all':{},
}

# After relabeling based on SA dividers
BGR_HPC_SA_div_dict =  {  'BGR04':{'HPC_BGR04_clstrd_SA_divs':'all'},
                        'BGR0506':{'HPC_BGR0506_clstrd_SA_divs':'all'},
                        'BGR0607':{'HPC_BGR0607_clstrd_SA_divs':'all'},
                        'BGR0708':{'HPC_BGR0708_clstrd_SA_divs':'all'},
                        'BGR0809':{'HPC_BGR0809_clstrd_SA_divs':'all'},
                        'BGR0910':{'HPC_BGR0910_clstrd_SA_divs':'all'},
                        'BGR1011':{'HPC_BGR1011_clstrd_SA_divs':'all'},
                        'BGR1112':{'HPC_BGR1112_clstrd_SA_divs':'all'},
                        'BGR1213':{'HPC_BGR1213_clstrd_SA_divs':'all'},
                        'BGR1314':{'HPC_BGR1314_clstrd_SA_divs':'all'},
                        'BGR1415':{'HPC_BGR1415_clstrd_SA_divs':'all'},
                        'BGR1516':{'HPC_BGR1516_clstrd_SA_divs':'all'},
                        'BGR1617':{'HPC_BGR1617_clstrd_SA_divs':'all'},
                        'BGR1718':{'HPC_BGR1718_clstrd_SA_divs':'all'},
                        'BGR1819':{'HPC_BGR1819_clstrd_SA_divs':'all'},
                        'BGR1920':{'HPC_BGR1920_clstrd_SA_divs':'all'},
                        'BGR2021':{'HPC_BGR2021_clstrd_SA_divs':'all'},
                        'BGR2122':{'HPC_BGR2122_clstrd_SA_divs':'all'},
                        'BGR_all':{ 
                                    'HPC_BGR0506_clstrd_SA_divs':'all',
                                    'HPC_BGR0607_clstrd_SA_divs':'all',
                                    'HPC_BGR0708_clstrd_SA_divs':'all',
                                    'HPC_BGR0809_clstrd_SA_divs':'all',
                                    'HPC_BGR0910_clstrd_SA_divs':'all',
                                    'HPC_BGR1011_clstrd_SA_divs':'all',
                                    'HPC_BGR1112_clstrd_SA_divs':'all',
                                    'HPC_BGR1213_clstrd_SA_divs':'all',
                                    'HPC_BGR1314_clstrd_SA_divs':'all',
                                    'HPC_BGR1415_clstrd_SA_divs':'all',
                                    'HPC_BGR1516_clstrd_SA_divs':'all',
                                    'HPC_BGR1617_clstrd_SA_divs':'all',
                                    'HPC_BGR1718_clstrd_SA_divs':'all',
                                    'HPC_BGR1819_clstrd_SA_divs':'all',
                                    'HPC_BGR1920_clstrd_SA_divs':'all',
                                    'HPC_BGR2021_clstrd_SA_divs':'all',
                                    'HPC_BGR2122_clstrd_SA_divs':'all',
                                    }
}

# An array of all the values of salinity to use as divisions between clusters
# These are the salinity values at the minimums of the pdf of salinity
#     of all non-noise points for every period, found by taking a rolling
#     average of the pdf, taking the difference, finding where that difference
#     changed signs, then finding the salinity values of the minimums
#     of the pdf between those points (the troughs of the pdf)
BGR_HPC_SA_divs = [34.1590168476106, 34.179234064102346, 34.19485645866415, 34.21139781761194, 34.22334435462979, 34.24080467796357, 34.258265001297346, 34.27296843147316, 34.28307703971903, 34.29686150550886, 34.315240793228625, 34.33913386726432, 34.35383729744014, 34.371297620773916, 34.39151483726566, 34.405299303055486, 34.417245840073335, 34.43103030586316, 34.44297684288101, 34.45859923744281, 34.47605956077659, 34.48892506218043, 34.50362849235624, 34.523845708847986, 34.542224996567754, 34.55968531990153, 34.57346978569136, 34.587254251481184, 34.60839043235892, 34.625850755692696, 34.6414731502545, 34.65250072288636, 34.66904208183415, 34.68742136955392, 34.704881692887696, 34.72142305183549, 34.74072130394124, 34.767371271134905, 34.78299366569671, 34.79861606025851, 34.806886739732406, 34.82618499183816, 34.847321172715894, 34.862943567277696, 34.87397113990956, 34.88775560569938, 34.91072971534909, 34.92910900306886, 34.954840005876534]

# From manual relabeling:
# BGR_all_clstr_plt_ranges = {
#     6:{
#         'press_lims':[350,200],
#         'pcs_press_lims':[0,3],
#         'CT_lims':[-0.8,-1.4],
#         'pcs_CT_lims':[0,0.035],
#         'SA_lims':[34.588,34.570],
#         'pcs_SA_lims':[0,0.01],
#         'sig_lims':[32.4,32.36]
#         },
#     23:{
#         'press_lims':[350,200],
#         'pcs_press_lims':[0,3],
#         'CT_lims':[-0.6,-1.25],
#         'pcs_CT_lims':[0,0.035],
#         'SA_lims':[34.588,34.570],
#         'pcs_SA_lims':[0,0.01],
#         'sig_lims':[32.4,32.36]
#         },
#     35:{
#         'press_lims':[350,200],
#         'pcs_press_lims':[0,3],
#         'CT_lims':[-0.55,-1.05],
#         'pcs_CT_lims':[0,0.035],
#         'SA_lims':[34.588,34.570],
#         'pcs_SA_lims':[0,0.01],
#         'sig_lims':[32.4,32.36]
#         },
#     52:{
#         'press_lims':[350,200],
#         'pcs_press_lims':[0,3],
#         'CT_lims':[-0.41,-0.8],
#         'pcs_CT_lims':[0,0.035],
#         'SA_lims':[34.588,34.570],
#         'pcs_SA_lims':[0,0.01],
#         'sig_lims':[32.4,32.36]
#         },
#     63:{
#         'press_lims':[350,200],
#         'pcs_press_lims':[0,3],
#         'CT_lims':[-0.195,-0.51],
#         'pcs_CT_lims':[0,0.035],
#         'SA_lims':[34.588,34.570],
#         'pcs_SA_lims':[0,0.01],
#         'sig_lims':[32.4,32.36]
#         },
#     69:{
#         'press_lims':[350,200],
#         'pcs_press_lims':[0,3],
#         'CT_lims':[-0.12,-0.35],
#         'pcs_CT_lims':[0,0.035],
#         'SA_lims':[34.588,34.570],
#         'pcs_SA_lims':[0,0.01],
#         'sig_lims':[32.4,32.36]
#         },
#     79:{
#         'press_lims':[350,200],
#         'pcs_press_lims':[0,3],
#         'CT_lims':[0.15,-0.05],
#         'pcs_CT_lims':[0,0.035],
#         'SA_lims':[34.588,34.570],
#         'pcs_SA_lims':[0,0.01],
#         'sig_lims':[32.4,32.36]
#         },
#     90:{
#         'press_lims':[350,200],
#         'pcs_press_lims':[0,3],
#         'CT_lims':[0.5,0.2],
#         'pcs_CT_lims':[0,0.035],
#         'SA_lims':[34.588,34.570],
#         'pcs_SA_lims':[0,0.01],
#         'sig_lims':[32.4,32.36]
#         },
#     95:{
#         'press_lims':[350,200],
#         'pcs_press_lims':[0,3],
#         'CT_lims':[0.63,0.35],
#         'pcs_CT_lims':[0,0.035],
#         'SA_lims':[34.588,34.570],
#         'pcs_SA_lims':[0,0.01],
#         'sig_lims':[32.4,32.36]
#         },
#     96:{
#         'press_lims':[350,200],
#         'pcs_press_lims':[0,3],
#         'CT_lims':[0.63,0.35],
#         'pcs_CT_lims':[0,0.035],
#         'SA_lims':[34.588,34.570],
#         'pcs_SA_lims':[0,0.01],
#         'sig_lims':[32.4,32.36]
#         },
# }

BGR_HPC_SA_divs = [34.1590168476106, 34.179234064102346, 34.19485645866415, 34.21139781761194, 34.22334435462979, 34.24080467796357, 34.258265001297346, 34.27296843147316, 34.28307703971903, 34.29686150550886, 34.315240793228625, 34.33913386726432, 34.35383729744014, 34.371297620773916, 34.39151483726566, 34.405299303055486, 34.417245840073335, 34.43103030586316, 34.44297684288101, 34.45859923744281, 34.47605956077659, 34.48892506218043, 34.50362849235624, 34.523845708847986, 34.542224996567754, 34.55968531990153, 34.57346978569136, 34.587254251481184, 34.60839043235892, 34.625850755692696, 34.6414731502545, 34.65250072288636, 34.66904208183415, 34.68742136955392, 34.704881692887696, 34.72142305183549, 34.74072130394124, 34.767371271134905, 34.78299366569671, 34.79861606025851, 34.806886739732406, 34.82618499183816, 34.847321172715894, 34.862943567277696, 34.87397113990956, 34.88775560569938, 34.91072971534909, 34.92910900306886, 34.954840005876534]

# From SA divs:
BGR_all_clstr_plt_ranges = {
    3:{
        'press_lims':[300,155],
        'pcs_press_lims':[0,3],
        'CT_lims':[-0.72,-1.38],
        'pcs_CT_lims':[0,0.035],
        'SA_lims':[34.21139781761194, 34.19485645866415],
        'pcs_SA_lims':[0,0.01],
        'sig_lims':[32.4,32.36]
        },
    5:{
        'press_lims':[296,160],
        'pcs_press_lims':[0,3],
        'CT_lims':[-0.68,-1.35],
        'pcs_CT_lims':[0,0.035],
        'SA_lims':[34.24080467796357, 34.22334435462979],
        'pcs_SA_lims':[0,0.01],
        'sig_lims':[32.4,32.36]
        },
    9:{
        'press_lims':[300,165],
        'pcs_press_lims':[0,3],
        'CT_lims':[-0.62,-1.28],
        'pcs_CT_lims':[0,0.035],
        'SA_lims':[34.29686150550886, 34.28307703971903],
        'pcs_SA_lims':[0,0.01],
        'sig_lims':[32.4,32.36]
        },
    10:{
        'press_lims':[305,166],
        'pcs_press_lims':[0,3],
        'CT_lims':[-0.59,-1.25],
        'pcs_CT_lims':[0,0.035],
        'SA_lims':[34.315240793228625, 34.27296843147316],
        'pcs_SA_lims':[0,0.01],
        'sig_lims':[32.4,32.36]
        },
    18:{
        'press_lims':[320,180],
        'pcs_press_lims':[0,3],
        'CT_lims':[-0.43,-0.95],
        'pcs_CT_lims':[0,0.035],
        'SA_lims':[34.44297684288101, 34.43103030586316],
        'pcs_SA_lims':[0,0.01],
        'sig_lims':[32.4,32.36]
        },
    20:{
        'press_lims':[320,185],
        'pcs_press_lims':[0,3],
        'CT_lims':[-0.37,-0.85],
        'pcs_CT_lims':[0,0.035],
        'SA_lims':[34.47605956077659, 34.45859923744281],
        'pcs_SA_lims':[0,0.01],
        'sig_lims':[32.4,32.36]
        },
    22:{
        'press_lims':[330,190],
        'pcs_press_lims':[0,3],
        'CT_lims':[-0.38,-0.75],
        'pcs_CT_lims':[0,0.035],
        'SA_lims':[34.50362849235624, 34.48892506218043],
        'pcs_SA_lims':[0,0.01],
        'sig_lims':[32.4,32.36]
        },
    25:{
        'press_lims':[328,198],
        'pcs_press_lims':[0,3],
        'CT_lims':[-0.23,-0.63],
        'pcs_CT_lims':[0,0.035],
        'SA_lims':[34.55968531990153, 34.542224996567754],
        'pcs_SA_lims':[0,0.01],
        'sig_lims':[32.4,32.36]
        },
    27:{
        'press_lims':[340,200],
        'pcs_press_lims':[0,3],
        'CT_lims':[-0.20,-0.5],
        'pcs_CT_lims':[0,0.035],
        'SA_lims':[34.587254251481184, 34.57346978569136],
        'pcs_SA_lims':[0,0.01],
        'sig_lims':[32.4,32.36]
        },
    35:{
        'press_lims':[370,230],
        'pcs_press_lims':[0,3],
        'CT_lims':[0.1,-0.1],
        'pcs_CT_lims':[0,0.035],
        'SA_lims':[34.72142305183549, 34.704881692887696],
        'pcs_SA_lims':[0,0.01],
        'sig_lims':[32.4,32.36]
        },
    36:{
        'press_lims':[360,235],
        'pcs_press_lims':[0,3],
        'CT_lims':[0.16,-0.07],
        'pcs_CT_lims':[0,0.035],
        'SA_lims':[34.74072130394124, 34.72142305183549],
        'pcs_SA_lims':[0,0.01],
        'sig_lims':[32.4,32.36]
        },
    39:{
        'press_lims':[370,250],
        'pcs_press_lims':[0,3],
        'CT_lims':[0.33,0.07],
        'pcs_CT_lims':[0,0.035],
        'SA_lims':[34.79861606025851, 34.78299366569671],
        'pcs_SA_lims':[0,0.01],
        'sig_lims':[32.4,32.36]
        },
    40:{
        'press_lims':[385,248],
        'pcs_press_lims':[0,3],
        'CT_lims':[0.37,0.14],
        'pcs_CT_lims':[0,0.035],
        'SA_lims':[34.806886739732406, 34.79861606025851],
        'pcs_SA_lims':[0,0.01],
        'sig_lims':[32.4,32.36]
        },
    42:{
        'press_lims':[380,260],
        'pcs_press_lims':[0,3],
        'CT_lims':[0.48,0.18],
        'pcs_CT_lims':[0,0.035],
        'SA_lims':[34.847321172715894, 34.82618499183816],
        'pcs_SA_lims':[0,0.01],
        'sig_lims':[32.4,32.36]
        },
    45:{
        'press_lims':[420,280],
        'pcs_press_lims':[0,3],
        'CT_lims':[0.56,0.36],
        'pcs_CT_lims':[0,0.035],
        'SA_lims':[34.88775560569938, 34.87397113990956],
        'pcs_SA_lims':[0,0.01],
        'sig_lims':[32.4,32.36]
        },
    47:{
        'press_lims':[440,275],
        'pcs_press_lims':[0,3],
        'CT_lims':[0.7,0.4],
        'pcs_CT_lims':[0,0.035],
        'SA_lims':[34.92910900306886, 34.91072971534909],
        'pcs_SA_lims':[0,0.01],
        'sig_lims':[32.4,32.36]
        },
}

# Add fit ranges to BGR_all_clstr_plt_ranges
var_lims = ['press_lims', 'CT_lims', 'SA_lims', 'sig_lims']
var_fit_lims = ['press-fit_lims', 'CT-fit_lims', 'SA-fit_lims', 'sig-fit_lims']
for key in BGR_all_clstr_plt_ranges.keys():
    for i in range(len(var_lims)):
        # Find range of var lims
        var_lim_range = abs(BGR_all_clstr_plt_ranges[key][var_lims[i]][1] - BGR_all_clstr_plt_ranges[key][var_lims[i]][0])
        BGR_all_clstr_plt_ranges[key][var_fit_lims[i]] = [var_lim_range/2, -var_lim_range/2]









# by year
# BGR04   = {'BGR04':'all'}
# BGR0506 = {'BGR0506':'all'}
# BGR0607 = {'BGR0607':'all'}
# BGR0708 = {'BGR0708':'all'}
# With minimal variables to reduce size
BGR04   = {'minimal_BGR04':'all'}
BGR0506 = {'minimal_BGR0506':'all'}
BGR0607 = {'minimal_BGR0607':'all'}
BGR0708 = {'minimal_BGR0708':'all'}
BGR0809 = {'minimal_BGR0809':'all'}
BGR0910 = {'minimal_BGR0910':'all'}
BGR1011 = {'minimal_BGR1011':'all'}

# With m_pts fixed manually
# BGR04   = {'BGRm250_04':'all'}
# BGR0506 = {'BGRm350_0506':'all'}
# BGR0607 = {'BGRm350_0607':'all'}
# BGR0708 = {'BGRm350_0708':'all'}
# BGR05060708 = {'BGRm350_0506':'all', 'BGRm350_0607':'all', 'BGRm350_0708':'all'}
# BGR_all = {'BGRm350_0506':'all','BGRm350_0607':'all','BGRm350_0708':'all'}

# With m_pts fixed manually
# BGR04   = {'BGR04_clstrd':'all'}
# BGR04_clstrs_456 = {'BGR04_clstrs_456':'all'}
# BGR0506 = {'BGR0506_clstrd':'all'}
# BGR0607 = {'BGR0607_clstrd':'all'}
# BGR0708 = {'BGR0708_clstrd':'all'}
# BGR0809 = {'BGR0809_clstrd':'all'}
# BGR0910 = {'BGR0910_clstrd':'all'}
# BGR1011 = {'BGR1011_clstrd':'all'}

BGR0508 = {'BGR0508':'all'}
BGR050607 = {'BGR0506_clstrd':'all', 'BGR0607_clstrd':'all'}
BGR05060708 = {'BGR0506_clstrd':'all', 'BGR0607_clstrd':'all', 'BGR0708_clstrd':'all'}
BGR05060708_clstrs_456 = {'BGR05060708_clstrs_456':'all'}
BGR_all = {
            # 'BGR04_clstrd':'all',
            'BGR0506_clstrd':'all',
            'BGR0607_clstrd':'all',
            'BGR0708_clstrd':'all',
            'BGR0809_clstrd':'all',
            'BGR0910_clstrd':'all',
            'BGR1011_clstrd':'all',
        }

# With m_pts fixed automatically
BGR_auto_mpts_dict = {  'BGR04':{'auto_mpts_BGR04_clstrd':'all'},
                        'BGR0506':{'auto_mpts_BGR0506_clstrd':'all'},
                        'BGR0607':{'auto_mpts_BGR0607_clstrd':'all'},
                        'BGR0708':{'auto_mpts_BGR0708_clstrd':'all'},
                        'BGR0809':{'auto_mpts_BGR0809_clstrd':'all'},
                        'BGR0910':{'auto_mpts_BGR0910_clstrd':'all'},
                        'BGR1011':{'auto_mpts_BGR1011_clstrd':'all'},
                        'BGR1112':{'auto_mpts_BGR1112_clstrd':'all'},
                        'BGR1213':{'auto_mpts_BGR1213_clstrd':'all'},
                        'BGR1314':{'auto_mpts_BGR1314_clstrd':'all'},
                        'BGR1415':{'auto_mpts_BGR1415_clstrd':'all'},
                        'BGR1516':{'auto_mpts_BGR1516_clstrd':'all'},
                        'BGR1617':{'auto_mpts_BGR1617_clstrd':'all'},
                        'BGR1718':{'auto_mpts_BGR1718_clstrd':'all'},
                        'BGR1819':{'auto_mpts_BGR1819_clstrd':'all'},
                        'BGR1920':{'auto_mpts_BGR1920_clstrd':'all'},
                        'BGR2021':{'auto_mpts_BGR2021_clstrd':'all'},
                        'BGR2122':{'auto_mpts_BGR2122_clstrd':'all'},
                        'BGR2223':{'auto_mpts_BGR2223_clstrd':'all'},
                        'BGR_all':{'auto_mpts_BGR0506_clstrd':'all','auto_mpts_BGR0607_clstrd':'all','auto_mpts_BGR0708_clstrd':'all','auto_mpts_BGR0809_clstrd':'all','auto_mpts_BGR0910_clstrd':'all','auto_mpts_BGR1011_clstrd':'all','auto_mpts_BGR1112_clstrd':'all','auto_mpts_BGR1213_clstrd':'all','auto_mpts_BGR1314_clstrd':'all','auto_mpts_BGR1415_clstrd':'all','auto_mpts_BGR1516_clstrd':'all','auto_mpts_BGR1617_clstrd':'all','auto_mpts_BGR1718_clstrd':'all','auto_mpts_BGR1819_clstrd':'all','auto_mpts_BGR1920_clstrd':'all','auto_mpts_BGR2021_clstrd':'all','auto_mpts_BGR2122_clstrd':'all','auto_mpts_BGR2223_clstrd':'all'},
                    }
# BGR0506 = {'mpts_auto_BGR0506_clstrd':'all'}
# BGR0607 = {'mpts_auto_BGR0607_clstrd':'all'}
# BGR0708 = {'mpts_auto_BGR0708_clstrd':'all'}
# BGR0809 = {'mpts_auto_BGR0809_clstrd':'all'}
# BGR0910 = {'mpts_auto_BGR0910_clstrd':'all'}
# BGR1011 = {'mpts_auto_BGR1011_clstrd':'all'}
# BGR_all = {
#             'mpts_auto_BGR0506_clstrd':'all',
#             'mpts_auto_BGR0607_clstrd':'all',
#             'mpts_auto_BGR0708_clstrd':'all',
#             'mpts_auto_BGR0809_clstrd':'all',
#             'mpts_auto_BGR0910_clstrd':'all',
#             'mpts_auto_BGR1011_clstrd':'all',
#         }

# # All profiles from certain ITPs
ITP2_all  = {'ITP_2':'all'}
ITP002_all  = {'ITP_002':'all'}
ITP003_all  = {'ITP_003':'all'}
# ITP33_all = {'ITP_33':'all'}
# ITP34_all = {'ITP_34':'all'}
# ITP35_all = {'ITP_35':'all'}
# ITP41_all = {'ITP_41':'all'}
# ITP42_all = {'ITP_42':'all'}
# ITP43_all = {'ITP_43':'all'}