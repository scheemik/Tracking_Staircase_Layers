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
test_S_range = [34.4, 34.6]

# Beaufort Gyre Region (BGR), see Shibley2022
lon_BGR = [-160,-130]
lat_BGR = [73,81.5]

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

## Subsets of ITPs in the BGR
BGRITPs0508 = {**BGRITPs0506, **BGRITPs0607, **BGRITPs0708}
BGRITPs0511  = {**BGRITPs0506, **BGRITPs0607, **BGRITPs0708, **BGRITPs0809, **BGRITPs0910, **BGRITPs1011}

## Pre-clustered files
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