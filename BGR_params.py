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
S_range_LHW_AW = [34.1, 35.03]
test_S_range = [34.4, 34.6]

# Beaufort Gyre Region (BGR), see Shibley2022
lon_BGR = [-160,-130]
lat_BGR = [73,81.5]

# Example area, choosen to have a few profiles from all time periods
lon_ex_area = [-135.5,-135]
lat_ex_area = [77,77.4]

# For this study, I'm only using profiles that extend down to at least 400 dbar
this_min_press = 400

################################################################################
# Make dictionaries for what data to load in and analyze
################################################################################

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



## Subsets of ITPs in the BGR
BGRITPs0508 = {**BGRITPs0506, **BGRITPs0607, **BGRITPs0708}
BGRITPs0511  = {**BGRITPs0506, **BGRITPs0607, **BGRITPs0708, **BGRITPs0809, **BGRITPs0910, **BGRITPs1011}

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
                        'BGR_all':{ 'HPC_BGR0506_clstrd':'all',
                                    'HPC_BGR0607_clstrd':'all',
                                    'HPC_BGR0708_clstrd':'all',
                                    'HPC_BGR0809_clstrd':'all',
                                    'HPC_BGR0910_clstrd':'all',
                                    'HPC_BGR1011_clstrd':'all',}
}

## Relabeling clusters
# From HPC
BGR_HPC_relab_dict  = { 'BGR04':{},
                        'BGR0506':{32:31, 35:36, 38:39, 42:43, 48:49, 51:100, 52:101, 53:102, 54:103, 55:104, 56:105, 57:106, 58:107, 59:108, 60:109, 61:110, 62:111, 63:112, 64:113, 65:114, 66:115, 67:116, 68:117, 69:118, 70:119, 71:120, 72:121, 73:122, 74:123, 75:124, 76:125, 77:126, 78:127},
                        'BGR0607':{32:33, 33:34, 34:39, 35:39, 36:40, 37:41, 38:44, 39:46, 40:46, 41:47, 42:49, 43:50, 44:100, 45:101, 46:102, 47:103, 48:104, 49:105, 50:106, 51:107, 52:108, 53:109, 54:110, 55:111, 56:112, 57:113, 58:114, 59:115, 60:116, 61:117, 62:118, 63:119, 64:120, 65:121, 66:122, 67:123, 68:124},
                        'BGR0708':{0:66, 1:65, 2:64, 3:8, 4:62, 5:9, 6:10, 7:11, 8:12, 9:13, 10:15, 11:60, 12:16, 13:17, 14:55, 15:56, 16:19, 17:20, 18:21, 19:22, 20:57, 21:58, 22:24, 23:25, 24:26, 25:27, 26:28, 27:29, 28:30, 29:31, 30:33, 31:34, 32:39, 33:39, 34:40, 35:41, 36:59, 37:44, 38:46, 39:46, 40:47, 41:49, 42:50, 43:51, 44:100, 45:101, 46:102, 47:103, 48:104, 49:105, 50:106, 51:107, 52:108, 53:109, 54:110, 55:111, 56:112, 57:113},
                        'BGR0809':{0:66, 1:65, 2:7, 3:8, 4:62, 5:10, 6:61, 7:14, 8:60, 9:16, 10:17, 11:55, 12:56, 13:19, 14:20, 15:21, 16:22, 17:23, 18:24, 19:25, 20:26, 21:27, 22:28, 23:29, 24:30, 25:31, 26:33, 27:34, 28:39, 29:40, 30:41, 31:59, 32:44, 33:46, 34:47, 35:47, 36:49, 37:50, 38:51, 39:51, 40:100, 41:101, 42:102, 43:103, 44:104, 45:105, 46:106, 47:107, 48:108, 49:109, 50:110, 51:111, 52:112, 53:113, 54:114, 55:115, 56:116, 57:117, 58:118, 59:119, 60:120, 61:121, 62:122, 63:123, 64:124},
                        'BGR0910':{48:100, 49:101, 50:102, 51:103, 52:104, 53:105, 54:106, 55:107, 56:108, 57:109, 58:110, 59:111, 60:112, 61:113, 62:114, 63:115, 64:116, 65:117, 66:118, 67:119, 68:120, 69:121, 70:122, 71:123, 72:124, 73:125, 74:126, 75:127, 76:128, 77:129, 78:130, 79:131, 80:132, 81:133, 82:134, 83:135, 84:136},
                        'BGR1011':{53:100, 54:101, 55:102, 56:103, 57:104, 58:105, 59:106, 60:107, 61:108, 62:109, 63:110, 64:111, 65:112, 66:113, 67:114, 68:115, 69:116, 70:117, 71:118, 72:119, 73:120, 74:121, 75:122, 76:123, 77:124, 78:125, 79:126, 80:127, 81:128, 82:129},
                        'BGR1112':{},
                        'BGR1213':{},
                        'BGR1314':{},
                        'BGR1415':{},
                        'BGR1516':{},
                        'BGR1617':{},
                        'BGR1718':{},
                        'BGR1819':{},
                        'BGR1920':{},
                        'BGR2021':{},
                        'BGR2122':{},
                        'BGR2223':{},
                        'BGR_all':{}
}


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