"""
Author: Mikhail Schee
Created: 2022-08-18

This script is set up to make figures of Arctic Ocean profile data that has been
formatted into netcdfs by the `make_netcdf` function.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions in source code must retain the accompanying copyright notice, this list of conditions, and the following disclaimer.
    2. Redistributions in binary form must reproduce the accompanying copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Names of the copyright holders must not be used to endorse or promote products derived from this software without prior written permission from the copyright holders.
    4. If any files are modified, you must cause the modified files to carry prominent notices stating that you changed the files and the date of any change.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""
Note: You unfortunately cannot pickle plots with parasite axes. So, the pickling
functionality won't work if you try to make a plot with the box and whisker plot
in an inset.

The Jupyter notebook Create_Figures.ipynb offers detailed explanations of each 
plot created by this script.

NOTE: BEFORE YOU RUN THIS SCRIPT

This script expects that the following files exist:
netcdfs/ITP_2.nc
netcdfs/ITP_3.nc

and that they have been created by running the following scripts in this order:
make_netcdf.py
take_moving_average.py
cluster_data.py
"""

# For custom analysis functions
import analysis_helper_functions as ahf

# BGOS_S_range = [34.1, 34.76]
BGOS_S_range = [34.366, 34.9992]
LHW_S_range = [34.366, 35.5]
AIDJEX_S_range = [34.366, 35.0223]

# Axis limits
x_ax_lims_SA = {'x_lims':[LHW_S_range[0], 35.01]}

### Filters for reproducing plots from Timmermans et al. 2008
ITP2_p_range = [185,300]
ITP2_S_range = [34.05,34.75]
ITP2_m_pts = 170
# Timmermans 2008 Figure 4 depth range
T2008_fig4_y_lims = {'y_lims':[260,220]}
# Timmermans 2008 Figure 4 shows profile 185
T2008_fig4_pfs = [183, 185, 187]
# Filters used in Timmermans 2008 T-S and aT-BS plots
T2008_p_range = [180,300]
T2008_fig5a_x_lims  = {'x_lims':[34.05,34.75]}
T2008_fig5a_ax_lims = {'x_lims':[34.05,34.75], 'y_lims':[-1.3,0.5]}
T2008_fig6a_ax_lims = {'x_lims':[0.027002,0.027042], 'y_lims':[-13e-6,3e-6]}
# The actual limits are above, but need to adjust the x lims for some reason
T2008_fig6a_ax_lims = {'x_lims':[0.026838,0.026878], 'y_lims':[-13e-6,3e-6]}

# A list of many profiles to plot from ITP2
start_pf = 1
import numpy as np
n_pfs_to_plot = 10
ITP2_some_pfs = list(np.arange(start_pf, start_pf+(n_pfs_to_plot*2), 2))

# For showing multiple layers grouped into one cluster
ITP2_some_pfs_0 = [87, 89, 95, 97, 99, 101, 103, 105, 109, 111]
ITP2_some_pfs_ax_lims_0 = {'y_lims':[245,220]}
# For showing one layer split into multiple clusters
ITP2_some_pfs_1 = [67, 69, 73, 75, 81, 83, 91, 93, 97, 99]
ITP2_some_pfs_ax_lims_1 = {'y_lims':[295,270]}

# For showing examples of shallow profiles
ITP35_some_pfs0 = [7,15,23,47,55,63,71]
# For showing examples of middle incomplete profiles
ITP35_some_pfs1 = [3,11,19,27,51,59,67]
# For showing examples of full profiles
ITP35_some_pfs2 = [1,9,17,25,49,57,65,73]

### Filters for reproducing plots from Lu et al. 2022
Lu2022_p_range = [200,355]
Lu2022_T_range = [-1.0,0.9]
Lu2022_S_range = [34.21,34.82]
Lu2022_m_pts = 580

################################################################################
# Make dictionaries for what data to load in and analyze
################################################################################

# All profiles from all sources in this study
# all_sources = {'AIDJEX_BigBear':'all','AIDJEX_BlueFox':'all','AIDJEX_Caribou':'all','AIDJEX_Snowbird':'all','ITP_2':'all','ITP_3':'all'}
all_sources = {'AIDJEX_BigBear':'all','AIDJEX_BlueFox':'all','AIDJEX_Caribou':'all','AIDJEX_Snowbird':'all','ITP_33':'all','ITP_34':'all','ITP_35':'all','ITP_41':'all','ITP_42':'all','ITP_43':'all','SHEBA_Seacat':'all'}

# All profiles from all ITPs in this study
all_ITPs = {'ITP_2':'all','ITP_3':'all','ITP_35':'all','ITP_41':'all','ITP_42':'all','ITP_43':'all'}
all_ITPs = {'ITP_001':'all',
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
            # 'ITP_013':'all',
            # 'ITP_014':'all',
            # 'ITP_015':'all',
            # 'ITP_016':'all',
            # 'ITP_017':'all',
            # 'ITP_018':'all',
            # 'ITP_019':'all',
            # 'ITP_021':'all',
            # 'ITP_022':'all',
            # 'ITP_023':'all',
            # 'ITP_024':'all',
            # 'ITP_025':'all',
            # 'ITP_026':'all',
            # 'ITP_027':'all',
            # 'ITP_028':'all',
            # 'ITP_029':'all',
            # 'ITP_030':'all',
            # 'ITP_032':'all',
            # 'ITP_033':'all',
            # 'ITP_034':'all',
            # 'ITP_035':'all',
            # 'ITP_036':'all',
            # 'ITP_037':'all',
            # 'ITP_038':'all',
            # 'ITP_041':'all',
            # 'ITP_042':'all',
            # 'ITP_043':'all',
            # 'ITP_047':'all',
            # 'ITP_048':'all',
            # 'ITP_049':'all',
            # 'ITP_051':'all',
            # 'ITP_052':'all',
            # 'ITP_053':'all',
            # 'ITP_054':'all',
            # 'ITP_055':'all',
            # 'ITP_056':'all',
            # 'ITP_057':'all',
            # 'ITP_058':'all',
            # 'ITP_059':'all',
            # 'ITP_060':'all',
            # 'ITP_061':'all',
            # 'ITP_062':'all',
            # 'ITP_063':'all',
            # 'ITP_064':'all',
            # 'ITP_065':'all',
            # 'ITP_068':'all',
            # 'ITP_069':'all',
            # 'ITP_070':'all',
            # 'ITP_072':'all',
            # 'ITP_073':'all',
            # 'ITP_074':'all',
            # 'ITP_075':'all',
            # 'ITP_076':'all',
            # 'ITP_077':'all',
            # 'ITP_078':'all',
            # 'ITP_079':'all',
            # 'ITP_080':'all',
            # 'ITP_081':'all',
            # 'ITP_082':'all',
            # 'ITP_083':'all',
            # 'ITP_084':'all',
            # 'ITP_085':'all',
            # 'ITP_086':'all',
            # 'ITP_087':'all',
            # 'ITP_088':'all',
            # 'ITP_089':'all',
            # 'ITP_090':'all',
            # 'ITP_091':'all',
            # 'ITP_092':'all',
            # 'ITP_094':'all',
            # 'ITP_095':'all',
            # 'ITP_097':'all',
            # 'ITP_098':'all',
            # 'ITP_099':'all',
            # 'ITP_100':'all',
            # 'ITP_101':'all',
            # 'ITP_102':'all',
            # 'ITP_103':'all',
            # 'ITP_104':'all',
            # 'ITP_105':'all',
            # 'ITP_107':'all',
            # 'ITP_108':'all',
            # 'ITP_109':'all',
            # 'ITP_110':'all',
            # 'ITP_111':'all',
            # 'ITP_113':'all',
            # 'ITP_114':'all',
            # 'ITP_116':'all'
            }
all_BGOS = {'ITP_33':'all','ITP_34':'all','ITP_35':'all','ITP_41':'all','ITP_42':'all','ITP_43':'all'}

# Sets of ITPs within the CB that appear within certain time ranges
## 2004-08-20 00:00:01 to 2004-09-29 00:00:05
CB_ITPs_0a = {'ITP_002':'all'}
## 2005-08-16 06:00:01 to 
CB_ITPs_0b = {'ITP_001':'all',
            #   'ITP_002':'all', # Not in this time range
              'ITP_003':'all',
              'ITP_004':'all',
              'ITP_005':'all',
              'ITP_006':'all',
            #   'ITP_007':'all', # Not in CB
              'ITP_008':'all',
            #   'ITP_009':'all', # Not in CB
            #   'ITP_010':'all', # Not in CB
              'ITP_011':'all', # can't time slice? but only sometimes
            #   'ITP_012':'all', # Not in CB
            #   'ITP_013':'all',  # something weird is going on here
            #   'ITP_014':'all', # Not in CB
            #   'ITP_015':'all', # Not in CB
            #   'ITP_016':'all', # Not in CB
            #   'ITP_017':'all', # Not in CB
              'ITP_018':'all',
            #   'ITP_019':'all', # Not in CB
              'ITP_021':'all',
              'ITP_022':'all',
              'ITP_023':'all',
            #   'ITP_024':'all', # Not in CB
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
              }

# Sets of ITPs within the BGR that appear within certain time ranges
## 2004-08-20 00:00:01 to 2004-09-29 00:00:05, duration: 41 days
BGR_ITPs_0a = {'ITP_002':'all'}
## 2005-08-16 06:00:01 to 2009-08-31 00:00:08, gap: 320 days, duration: 1477 days
BGR_ITPs_0b = { 'ITP_001':'all',
                # 'ITP_002':'all', # Not in this time range
                'ITP_003':'all',
                'ITP_004':'all',
                'ITP_005':'all',
                'ITP_006':'all',
                # 'ITP_007':'all', # Not in BGR
                'ITP_008':'all',
                # 'ITP_009':'all', # Not in BGR
                # 'ITP_010':'all', # Not in BGR
                'ITP_011':'all', # can't time slice? but only sometimes
                # 'ITP_012':'all', # Not in BGR
                'ITP_013':'all',  # something weird is going on here
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
## 2009-10-04 06:00:02 to 2010-06-08 00:00:07, gap: 33 days, duration: 248 days
BGR_ITPs_0c = { 'ITP_032':'all',
                'ITP_033':'all',
                'ITP_034':'all',
                'ITP_035':'all',
                }
## 2010-06-16 00:00:06 to 2013-08-12 00:00:06, gap: 8 days, duration: 1154 days
BGR_ITPs_0d = { 'ITP_033':'all',
                # 'ITP_036':'all', # Not in BGR
                # 'ITP_037':'all', # Not in BGR
                # 'ITP_038':'all', # Not in BGR
                'ITP_041':'all',
                'ITP_042':'all',
                'ITP_043':'all',
                # 'ITP_047':'all',
                # 'ITP_048':'all',
                # 'ITP_049':'all',
                # 'ITP_051':'all',
                'ITP_052':'all',
                'ITP_053':'all',
                'ITP_054':'all',
                'ITP_055':'all',
                # 'ITP_056':'all',
                # 'ITP_057':'all',
                # 'ITP_058':'all',
                # 'ITP_059':'all',
                # 'ITP_060':'all',
                # 'ITP_061':'all',
                'ITP_062':'all',
                # 'ITP_063':'all',
                'ITP_064':'all',
                'ITP_065':'all',
                }
## 2013-08-22 00:02:02 to 2016-05-02 00:02:01, gap: 9 days, duration: 985 days
BGR_ITPs_0e = { 'ITP_064':'all',
                # 'ITP_065':'all', # Doesn't appear in this time period
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
                'ITP_081':'all',
                'ITP_082':'all',
                # 'ITP_083':'all', # Not in BGR
                'ITP_084':'all',
                'ITP_085':'all',
                'ITP_086':'all',
                'ITP_087':'all',
                'ITP_088':'all',
                'ITP_089':'all',
                # 'ITP_090':'all', # Not in BGR
                # 'ITP_091':'all', # Not in BGR
                # 'ITP_092':'all', # Not in BGR
                # 'ITP_094':'all', # Not in BGR
                # 'ITP_095':'all', # Not in BGR
                }
## 2016-10-03 00:02:02 to 2017-08-04 00:02:01, gap: 153 days, duration: 306 days
BGR_ITPs_0f = { 'ITP_097':'all',
                # 'ITP_098':'all', # Just barely outside BGR
                'ITP_099':'all',
                }
## 2017-09-17 00:02:02 to 2018-07-21 00:02:02, gap: 43 days, duration: 308 days
BGR_ITPs_0g = { 'ITP_097':'all',
                'ITP_100':'all',
                'ITP_101':'all',
                # 'ITP_102':'all', # Not in BGR
                # 'ITP_103':'all', # Not in this time period
                # 'ITP_104':'all', # Not in this time period
                # 'ITP_105':'all', # Not in this time period
                # 'ITP_107':'all', # Not in this time period
                'ITP_108':'all',
                }
## 2018-09-19 00:02:03 to 2019-04-14 00:02:02, gap: 59 days, duration: 208 days
BGR_ITPs_0h = { 'ITP_103':'all',
                'ITP_104':'all',
                'ITP_105':'all',
                'ITP_107':'all',
                'ITP_108':'all', # Not in this time period
                'ITP_109':'all',
                'ITP_110':'all',
                }
## 
BGR_ITPs_0i = { 'ITP_103':'all',
                'ITP_104':'all',
                'ITP_105':'all',
                'ITP_107':'all',
                'ITP_108':'all', # Not in this time period
                'ITP_109':'all',
                'ITP_110':'all',
                # 'ITP_111':'all', # Not in BGR
                'ITP_113':'all',
                'ITP_114':'all',
                # 'ITP_116':'all', # Not in BGR
                'ITP_117':'all',
                'ITP_118':'all',
                'ITP_120':'all',
                'ITP_121':'all',
                'ITP_122':'all',
                'ITP_123':'all',
                'ITP_125':'all',
                'ITP_128':'all',
                }

## All
BGR_ITPs_all = {**BGR_ITPs_0a, **BGR_ITPs_0b, **BGR_ITPs_0c, **BGR_ITPs_0d, **BGR_ITPs_0e, **BGR_ITPs_0f, **BGR_ITPs_0g, **BGR_ITPs_0h, **BGR_ITPs_0i}

## Pre-clustered files
# Single time periods
BGRa_m110 = {'BGRa_mpts_110':'all'}
BGRf_m410 = {'BGRf_mpts_410':'all'}
BGRg_m240 = {'BGRg_mpts_240':'all'}
BGRh_m390 = {'BGRh_mpts_390':'all'}
# Comparing time periods
BGRfg = {'BGRf_mpts_410':'all','BGRg_mpts_240':'all'}
BGRgh = {'BGRg_mpts_240':'all','BGRh_mpts_390':'all'}
BGRfgh = {'BGRf_mpts_410':'all','BGRg_mpts_240':'all','BGRh_mpts_390':'all'}

# # All profiles from all AIDJEX camps
# all_AIDJEX = {'AIDJEX_BigBear':'all','AIDJEX_BlueFox':'all','AIDJEX_Caribou':'all','AIDJEX_Snowbird':'all'}
# 
# # Pre-clustered files
# all_AIDJEX_clstrd = {'AIDJEX_all':'all'}
# #   ell = 10
# AIDJEX_m500_e010 = {'AIDJEX_mpts_500_ell_010':'all'}
# BGOS_m440_e010 = {'BGOS_mpts_440_ell_010':'all'}
# BGOSss_m260_e010 = {'BGOSss_mpts_260_ell_010':'all'}
# #   ell = 10
# AIDJEX_m490_e050 = {'AIDJEX_mpts_490_ell_050':'all'}
# BGOS_m280_e050 = {'BGOS_mpts_280_ell_050':'all'}
# BGOSss_m340_e050 = {'BGOSss_mpts_340_ell_050':'all'}
# #   ell = 100
# AIDJEX_m300_e100 = {'AIDJEX_mpts_300_ell_100':'all'}
# BGOS_m360_e100 = {'BGOS_mpts_360_ell_100':'all'}
# BGOSss_m220_e100 = {'BGOSss_mpts_220_ell_100':'all'}
# 
# # All profiles from certain ITPs
# ITP2_all  = {'ITP_2':'all'}
# ITP3_all  = {'ITP_3':'all'}
# ITP33_all = {'ITP_33':'all'}
# ITP34_all = {'ITP_34':'all'}
# ITP35_all = {'ITP_35':'all'}
# ITP41_all = {'ITP_41':'all'}
# ITP42_all = {'ITP_42':'all'}
# ITP43_all = {'ITP_43':'all'}

# A list of many profiles to plot from ITP3
start_pf = 1331
import numpy as np
n_pfs_to_plot = 5
ITP3_some_pfs_1 = list(np.arange(start_pf, start_pf+(n_pfs_to_plot*2), 2))
ITP3_pfs1  = {'ITP_3':ITP3_some_pfs_1}
ITP35_pfs0 = {'ITP_35':ITP35_some_pfs0}
ITP35_pfs1 = {'ITP_35':ITP35_some_pfs1}
ITP35_pfs2 = {'ITP_35':ITP35_some_pfs2}

# Example profiles
# Coincident_pfs0 = {'AIDJEX_Snowbird':[138,140,142], 'SHEBA_Seacat':['SH36200'], 'ITP_33':[779, 781, 783]}
# ex_pfs1 = {'AIDJEX_BigBear':[389], 'ITP_33':[313], 'SHEBA_Seacat':['SH31200']}
# 
# ## AIDJEX
# 
# AIDJEX_BigBear_all = {'AIDJEX_BigBear':'all'}
# AIDJEX_BlueFox_all = {'AIDJEX_BlueFox':'all'}
# AIDJEX_Caribou_all = {'AIDJEX_Caribou':'all'}
# AIDJEX_Snowbird_all = {'AIDJEX_Snowbird':'all'}
# 
# AIDJEX_BigBear_blacklist = {'AIDJEX_BigBear':[531, 535, 537, 539, 541, 543, 545, 547, 549]}
# AIDJEX_BlueFox_blacklist = {'AIDJEX_BlueFox':[94, 308, 310]}
# AIDJEX_Snowbird_blacklist = {'AIDJEX_Snowbird':[443]}
# 
# AIDJEX_missing_ll = {'AIDJEX_BigBear':[4, 5, 6, 7, 8, 9, 10, 13, 14, 22]}#, 28, 30, 32]}
# 
# AIDJEX_ex_stairs = {'AIDJEX_Snowbird':[1]}
# 
# ## SHEBA
# 
# all_SHEBA = {'SHEBA_Seacat':'all'}
# SHEBA_Seacat_blacklist = {'SHEBA_Seacat':['SH15200.UP', 'SH03200', 'SH30500', 'SH34100']}
# # SHEBA_example_profile = {'SHEBA_Seacat':['CT165654']}
# SHEBA_example_profile = {'SHEBA_Seacat':['SH31200','SH31900']}
# 
# ## filtered out by p_max > 400 dbar
# 
# pmax_lt_400_AIDJEX_BigBear = {'AIDJEX_BigBear':[14]}
# pmax_lt_400_AIDJEX_BlueFox = {'AIDJEX_BlueFox':[30, 144, 168, 262, 431]}
# pmax_lt_400_AIDJEX_Caribou = {'AIDJEX_Caribou':[39, 47, 55, 63, 71, 79, 83, 91, 96, 98, 102, 104, 105, 113, 117, 123, 125, 128, 136, 144, 152, 159, 163, 169, 177, 185, 231, 440, 552, 610, 687, 694, 700, 704, 722, 727, 852]}
# pmax_lt_400_AIDJEX_Snowbird = {'AIDJEX_Snowbird':[320, 369, 404, 412, 414]}
# # ITP 33 had none filtered out
# # ITP 34 had none filtered out
# pmax_lt_400_ITP_35 = {'ITP_35':
#     [3, 4, 6, 7, 11, 12, 14, 15, 19, 22, 23, 27, 30, 31, 35, 36, 38, 43, 44, 46, 47, 51, 52, 54, 55, 59, 60, 62, 63, 67, 70, 71, 75, 76, 78, 79, 83, 84, 86, 87, 91, 92, 94, 99, 100, 102, 103, 107, 108, 110, 111, 115, 116, 118, 119, 123, 124, 126, 127, 131, 132, 134, 135, 139, 140, 142, 143, 147, 150, 151, 155, 156, 158, 159, 163, 164, 166, 167, 171, 172, 174, 175, 179, 180, 182, 187, 188, 190, 195, 196, 198, 203, 204, 206, 207, 211, 214, 219, 220, 222, 223, 227, 228, 230, 231, 235, 236, 238, 239, 243, 244, 246, 247, 251, 252, 254, 255, 259, 260, 262, 267, 268, 270, 271, 275, 276, 278, 279, 283, 284, 286, 287, 291, 292, 294, 295, 299, 300, 302, 307, 308, 310, 311, 315, 318, 323, 324, 326, 330, 331, 333, 334, 338, 339, 341, 342, 346, 347, 349, 350, 354, 355, 357, 358, 362, 363, 365, 366, 370, 372, 377, 378, 380, 381, 385, 386, 388, 389, 393, 394, 396, 397, 401, 402, 404, 405, 409, 410, 412, 413, 417, 420, 425, 426, 428, 429, 433, 434, 436, 441, 442, 444, 445, 449, 452, 457, 460, 465, 466, 468, 469, 473, 474, 476, 477, 481, 482, 484, 485, 489, 492, 493, 497, 498, 500, 505, 506, 508, 509, 513, 514, 516, 521, 522, 524, 525, 529, 532, 533, 537, 538, 540, 541, 545, 546, 548, 549, 553, 554, 556, 561, 564, 565, 569, 570, 572, 577, 578, 580, 585, 586, 588, 589, 593, 594, 596, 597, 601, 602, 604, 605, 609, 610, 612, 617, 618, 620, 621, 625, 626, 628, 629, 633, 634, 636, 637, 641, 642, 644, 645, 649, 650, 652, 653, 657, 658, 660, 661, 665, 666, 668, 669, 672, 673, 675, 680, 681, 683, 684, 688, 691, 696, 697, 699, 700, 704, 707, 708, 712, 713, 715, 716, 720, 722, 723, 727, 728, 730, 735, 738, 739, 743, 744, 746, 747, 749, 752, 753, 757, 758, 760, 761, 765, 766, 768, 769, 773, 774, 776, 777, 781, 782, 783, 788, 789, 791, 792, 796, 797, 799, 800, 804, 805, 807, 808, 812, 813, 815, 816, 820, 821, 823, 824, 828, 829, 831, 832, 834, 837, 841, 845, 846, 848, 849, 853, 854, 856, 857, 861, 862, 864, 865, 869, 871, 872, 876, 877, 879, 884, 885, 887, 888, 892, 893, 895, 896, 900, 901, 903, 904, 908, 909, 911, 916, 917, 919, 920, 924, 925, 927, 928, 932, 933, 935, 936, 940, 941, 943, 944, 948, 951, 952, 956, 957, 961, 962, 964, 965, 969, 971, 972, 976, 977, 979, 980, 984, 985, 987, 988, 992, 993, 995, 1000, 1001, 1003, 1004, 1008, 1009, 1011, 1012, 1016, 1017, 1019, 1024, 1025, 1027, 1033, 1034, 1039, 1040, 1042, 1043, 1047, 1048, 1050, 1051, 1055, 1056, 1058, 1059, 1063, 1064, 1066, 1067, 1071, 1072, 1076, 1077, 1078, 1079, 1083, 1086, 1087, 1091, 1092, 1094, 1095, 1099, 1100, 1102, 1103, 1107, 1108, 1110, 1111, 1115, 1116, 1118, 1119, 1123, 1124, 1126, 1127, 1131, 1132, 1134, 1135, 1139, 1140, 1142, 1143, 1147, 1148, 1150, 1151, 1155, 1156, 1158, 1159, 1163, 1166, 1171, 1172, 1174, 1175, 1179, 1180, 1182, 1183, 1187, 1190, 1191, 1195, 1196, 1198, 1199, 1203, 1204, 1206, 1207, 1211, 1214, 1215, 1219, 1220, 1222, 1223, 1227, 1228, 1230, 1235, 1238, 1239, 1243, 1244, 1246, 1247, 1251, 1252, 1254, 1258, 1261, 1266, 1267, 1269, 1270, 1274, 1275, 1277, 1278, 1282, 1283, 1285, 1286, 1290, 1293, 1298, 1299, 1301, 1305, 1307, 1308, 1312, 1313, 1314, 1315, 1319, 1322, 1326, 1327, 1329, 1330, 1334, 1337, 1342, 1343, 1345, 1350, 1351, 1353]}
# pmax_lt_400_ITP_41 = {'ITP_41':[1393, 1395, 1397, 1399, 1401, 1403, 1405, 1407, 1409, 1411, 1413, 1417, 1419, 1423, 1425, 1427, 1429, 1431, 1433, 1435, 1439, 1441, 1443, 1445, 1447, 1449, 1451, 1453, 1457, 1459, 1461, 1463, 1465, 1467, 1469, 1471, 1473, 1477, 1479]}
# pmax_lt_400_ITP_42 = {'ITP_42':[203, 204, 205, 206, 207, 208, 209]}
# # ITP 43 had none filtered out
# pmax_lt_400_SHEBA_Seacat = {'SHEBA_Seacat':['CT205515', 'SH12200', 'SH14400', 'SH17400', 'SH17701', 'SH12900']}

################################################################################
# Create data filtering objects
print('- Creating data filtering objects')
################################################################################

dfs_all = ahf.Data_Filters(keep_black_list=True, cast_direction='any')
dfs0 = ahf.Data_Filters()
dfs1 = ahf.Data_Filters(min_press=400)
dfs2 = ahf.Data_Filters(min_press_CT_max=400)

dfs_CB = ahf.Data_Filters(geo_extent='CB', keep_black_list=True, cast_direction='any')
this_min_press = 400
dfs1_CB_0a = ahf.Data_Filters(geo_extent='CB', min_press=this_min_press)
dfs1_CB_0b = ahf.Data_Filters(geo_extent='CB', min_press=this_min_press)#, date_range=['2005/08/15 00:00:00','2010/07/22 00:00:00'])

# Beaufort Gyre Region
dfs1_BGR_0c = ahf.Data_Filters(min_press=this_min_press, date_range=['2009/08/31 00:00:08','2010/06/11 00:00:00'])
dfs1_BGR_0d = ahf.Data_Filters(min_press=this_min_press, date_range=['2010/06/11 00:00:00','2013/08/17 00:00:00'])
dfs1_BGR_0e = ahf.Data_Filters(min_press=this_min_press, date_range=['2013/08/17 00:00:00','2016/07/10 00:00:00'])
dfs1_BGR_0f = ahf.Data_Filters(min_press=this_min_press, date_range=['2016/07/10 00:00:00','2017/08/29 00:00:00'])
dfs1_BGR_0g = ahf.Data_Filters(min_press=this_min_press, date_range=['2017/08/29 00:00:00','2018/08/17 00:00:00'])
dfs1_BGR_0h = ahf.Data_Filters(min_press=this_min_press, date_range=['2018/08/17 00:00:00','2019/04/23 00:00:00'])
dfs1_BGR_0i = ahf.Data_Filters(min_press=this_min_press, date_range=['2019/04/23 00:00:00','2024/08/17 00:00:00'])

################################################################################
# Create data sets by combining filters and the data to load in
print('- Creating data sets')
################################################################################

# ds_all_sources_all  = ahf.Data_Set(all_sources, dfs_all)
# ds_all_sources_up   = ahf.Data_Set(all_sources, dfs0)
# ds_all_sources_pmin = ahf.Data_Set(all_sources, dfs1)

## Example profiles
# ds_all_sources_ex_pfs = ahf.Data_Set(Coincident_pfs0, dfs_all)
# ds_all_sources_ex_pfs = ahf.Data_Set(ex_pfs1, dfs1)

## ITP

# ds_all_BGOS = ahf.Data_Set(all_BGOS, dfs_all)
# ds_all_BGOS = ahf.Data_Set(all_BGOS, dfs0)
# ds_BGOS = ahf.Data_Set(all_BGOS, dfs1)

# ds_all_ITP = ahf.Data_Set(all_ITPs, dfs_all)
# ds_all_ITP = ahf.Data_Set(all_ITPs, dfs_CB)

# ds_ITP = ahf.Data_Set(all_ITPs, dfs1_CB)

# ds_ITP2_all = ahf.Data_Set(ITP2_all, dfs0)

# ds_ITP_test = ahf.Data_Set({'ITP_098':'all'}, dfs_all)

## CB ITP datasets, by time period
# ds_CB_ITPs_0a = ahf.Data_Set(CB_ITPs_0a, dfs1_CB_0a)
# ds_CB_ITPs_0b = ahf.Data_Set(CB_ITPs_0b, dfs1_CB_0b)
# ds_CB_ITPs_0b = ahf.Data_Set(CB_ITPs_0a, dfs1_CB_0b)

## BGR ITP datasets, by time period
ds_BGR_ITPs_all = ahf.Data_Set(BGR_ITPs_all, dfs1)
# ds_BGR_ITPs_0a = ahf.Data_Set(BGR_ITPs_0a, dfs1)
# ds_BGR_ITPs_0b = ahf.Data_Set(BGR_ITPs_0b, dfs1)
# ds_BGR_ITPs_0c = ahf.Data_Set(BGR_ITPs_0c, dfs1_BGR_0c)
# ds_BGR_ITPs_0d = ahf.Data_Set(BGR_ITPs_0d, dfs1_BGR_0d)
# ds_BGR_ITPs_0e = ahf.Data_Set(BGR_ITPs_0e, dfs1_BGR_0e)
# ds_BGR_ITPs_0f = ahf.Data_Set(BGR_ITPs_0f, dfs1_BGR_0f)
# ds_BGR_ITPs_0g = ahf.Data_Set(BGR_ITPs_0g, dfs1_BGR_0g)
# ds_BGR_ITPs_0h = ahf.Data_Set(BGR_ITPs_0h, dfs1_BGR_0h)
# ds_BGR_ITPs_0i = ahf.Data_Set(BGR_ITPs_0i, dfs1_BGR_0i)

# Data Sets without filtering based on CT_max
# ds_ITP22_all  = ahf.Data_Set(ITP22_all, dfs_all)
# ds_ITP23_all  = ahf.Data_Set(ITP23_all, dfs_all)
# ds_ITP32_all  = ahf.Data_Set(ITP32_all, dfs_all)
# ds_ITP33_all  = ahf.Data_Set(ITP33_all, dfs_all)
# ds_ITP34_all  = ahf.Data_Set(ITP34_all, dfs_all)
# ds_ITP35_all  = ahf.Data_Set(ITP35_all, dfs_all)
# ds_ITP41_all  = ahf.Data_Set(ITP41_all, dfs_all)
# ds_ITP42_all  = ahf.Data_Set(ITP42_all, dfs_all)
# ds_ITP43_all  = ahf.Data_Set(ITP43_all, dfs_all)

# Data Sets filtered based on CT_max
# ds_ITP2 = ahf.Data_Set(ITP2_all, dfs1)

# ds_ITP33 = ahf.Data_Set(ITP33_all, dfs1)
# ds_ITP34 = ahf.Data_Set(ITP34_all, dfs1)
# ds_ITP35 = ahf.Data_Set(ITP35_all, dfs1)
# ds_ITP41 = ahf.Data_Set(ITP41_all, dfs1)
# ds_ITP42 = ahf.Data_Set(ITP42_all, dfs1)
# ds_ITP43 = ahf.Data_Set(ITP43_all, dfs1)
# 
# ds_ITP35_some_pfs0 = ahf.Data_Set(ITP35_pfs0, dfs0)
# ds_ITP35_some_pfs1 = ahf.Data_Set(ITP35_pfs1, dfs0)
# ds_ITP35_some_pfs2 = ahf.Data_Set(ITP35_pfs2, dfs0)

## AIDJEX

# ds_all_AIDJEX = ahf.Data_Set(all_AIDJEX, dfs_all)
# ds_AIDJEX = ahf.Data_Set(all_AIDJEX, dfs1)

# ds_AIDJEX_BigBear  = ahf.Data_Set(AIDJEX_BigBear_all, dfs1)
# ds_AIDJEX_BlueFox  = ahf.Data_Set(AIDJEX_BlueFox_all, dfs1)
# ds_AIDJEX_Caribou  = ahf.Data_Set(AIDJEX_Caribou_all, dfs1)
# ds_AIDJEX_Snowbird = ahf.Data_Set(AIDJEX_Snowbird_all, dfs1)

# ds_AIDJEX_BigBear_blacklist = ahf.Data_Set(AIDJEX_BigBear_blacklist, dfs_all)
# ds_AIDJEX_BlueFox_blacklist = ahf.Data_Set(AIDJEX_BlueFox_blacklist, dfs_all)
# ds_AIDJEX_Snowbird_blacklist = ahf.Data_Set(AIDJEX_Snowbird_blacklist, dfs_all)

# ds_AIDJEX_missing_ll = ahf.Data_Set(AIDJEX_missing_ll, dfs_all)

# ds_AIDJEX_ex_stairs = ahf.Data_Set(AIDJEX_ex_stairs, dfs_all)

## SHEBA

# ds_all_SHEBA = ahf.Data_Set(all_SHEBA, dfs_all)
# ds_SHEBA = ahf.Data_Set(all_SHEBA, dfs1)
# ds_SHEBA_Seacat_blacklist = ahf.Data_Set(SHEBA_Seacat_blacklist, dfs_all)
# ds_SHEBA_example_profile = ahf.Data_Set(SHEBA_example_profile, dfs1)

## filtered out by p_max > 400 dbar
# ds_pmax_lt_400 = ahf.Data_Set(pmax_lt_400_AIDJEX_BigBear, dfs0)
# ds_pmax_lt_400 = ahf.Data_Set(pmax_lt_400_AIDJEX_BlueFox, dfs0)
# ds_pmax_lt_400 = ahf.Data_Set(pmax_lt_400_AIDJEX_Caribou, dfs0)
# ds_pmax_lt_400 = ahf.Data_Set(pmax_lt_400_AIDJEX_Snowbird, dfs0)
# ds_pmax_lt_400 = ahf.Data_Set(pmax_lt_400_ITP_35, dfs0)
# ds_pmax_lt_400 = ahf.Data_Set(pmax_lt_400_ITP_41, dfs0)
# ds_pmax_lt_400 = ahf.Data_Set(pmax_lt_400_ITP_42, dfs0)
# ds_pmax_lt_400 = ahf.Data_Set(pmax_lt_400_SHEBA_Seacat, dfs0)

# Pre-clustered
#   ell = 10
# ds_AIDJEX_m500_e010 = ahf.Data_Set(AIDJEX_m500_e010, dfs_all)
# ds_BGOS_m440_e010 = ahf.Data_Set(BGOS_m440_e010, dfs_all)
# ds_BGOSss_m260_e010 = ahf.Data_Set(BGOSss_m260_e010, dfs_all)
#   ell = 50
# ds_AIDJEX_m490_e050 = ahf.Data_Set(AIDJEX_m490_e050, dfs_all)
# ds_BGOS_m280_e050 = ahf.Data_Set(BGOS_m280_e050, dfs_all)
# ds_BGOSss_m340_e050 = ahf.Data_Set(BGOSss_m340_e050, dfs_all)
#   ell = 100
# ds_AIDJEX_m300_e100 = ahf.Data_Set(AIDJEX_m300_e100, dfs_all)
# ds_BGOS_m360_e100 = ahf.Data_Set(BGOS_m360_e100, dfs_all)
# ds_BGOSss_m220_e100 = ahf.Data_Set(BGOSss_m220_e100, dfs_all)

## Pre-clustered
# Single time periods
# ds_BGRa_m110 = ahf.Data_Set(BGRa_m110, dfs_all)
# ds_BGRf_m410 = ahf.Data_Set(BGRf_m410, dfs_all)
# ds_BGRg_m240 = ahf.Data_Set(BGRg_m240, dfs_all)
# ds_BGRh_m390 = ahf.Data_Set(BGRh_m390, dfs_all)
# Comparing time periods
# ds_BGRfg = ahf.Data_Set(BGRfg, dfs_all)
# ds_BGRgh = ahf.Data_Set(BGRgh, dfs_all)
# ds_BGRfgh = ahf.Data_Set(BGRfgh, dfs_all)

################################################################################
# Create profile filtering objects
print('- Creating profile filtering objects')
################################################################################

pfs_0 = ahf.Profile_Filters()
pfs_1 = ahf.Profile_Filters(SA_range=ITP2_S_range)
pfs_2 = ahf.Profile_Filters(p_range=[400,225])

# # AIDJEX Operation Area (AOA)
# lon_AOA = [-152.9,-133.7]
# lat_AOA = [72.6,77.4]
# pfs_AOA = ahf.Profile_Filters(lon_range=lon_AOA,lat_range=lat_AOA)
# pfs_AOA1 = ahf.Profile_Filters(lon_range=lon_AOA,lat_range=lat_AOA, p_range=[1000,5])
# Canada Basin (CB), see Peralta-Ferriz2015
lon_CB = [-155,-130]
lat_CB = [72,84]
pfs_CB = ahf.Profile_Filters(lon_range=lon_CB,lat_range=lat_CB)
pfs_CB1 = ahf.Profile_Filters(lon_range=lon_CB,lat_range=lat_CB, p_range=[1000,5])
# Beaufort Gyre Region (BGR), see Shibley2022
lon_BGR = [-160,-130]
lat_BGR = [73,81.5]
pfs_BGR = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR)
pfs_BGR1 = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR, p_range=[1000,5], SA_range=LHW_S_range, lt_pCT_max=True)

# Finding coincident profiles
# lon_coin = [-148.9,-147.8]
# lat_coin = [74.9,75.5]
# pfs_coin = ahf.Profile_Filters(lon_range=lon_coin,lat_range=lat_coin)
# pfs_coin_fltrd = ahf.Profile_Filters(lon_range=lon_coin,lat_range=lat_coin, SA_range=LHW_S_range, lt_pCT_max=True)

# Finding close example profiles
# lon_ex_pfs = [-145.2,-144.53]
# lat_ex_pfs = [75.962,76.06]
# pfs_ex_pfs = ahf.Profile_Filters(lon_range=lon_ex_pfs, lat_range=lat_ex_pfs)
# pfs_ex_pfs_fltrd = ahf.Profile_Filters(lon_range=lon_ex_pfs, lat_range=lat_ex_pfs, SA_range=LHW_S_range, lt_pCT_max=True)
# 
# pfs_ITP2  = ahf.Profile_Filters(SA_range=ITP2_S_range)
# pfs_ITP3  = ahf.Profile_Filters(SA_range=Lu2022_S_range)
# pfs_BGOS  = ahf.Profile_Filters(lon_range=lon_AOA,lat_range=lat_AOA,SA_range=BGOS_S_range)
# pfs_fltrd = ahf.Profile_Filters(lon_range=lon_AOA, lat_range=lat_AOA, SA_range=LHW_S_range, lt_pCT_max=True)
# pfs_fltrd_ss = ahf.Profile_Filters(lon_range=lon_AOA, lat_range=lat_AOA, SA_range=LHW_S_range, lt_pCT_max=True, subsample=True)
# pfs_fltrd = ahf.Profile_Filters(SA_range=LHW_S_range)#, lt_pCT_max=True)
# 
# pfs_subs = ahf.Profile_Filters(SA_range=ITP2_S_range, subsample=True)
# pfs_regrid = ahf.Profile_Filters(SA_range=ITP2_S_range, regrid_TS=['CT',0.01,'SP',0.005])
# pfs_ss_rg = ahf.Profile_Filters(SA_range=ITP2_S_range, subsample=True, regrid_TS=['CT',0.01,'SP',0.005])
# 
# pfs_AIDJEX = ahf.Profile_Filters(SA_range=AIDJEX_S_range)

################################################################################
# Create plotting parameter objects

# print('- Creating plotting parameter objects')
################################################################################

### Test plots

## Number of up-going profiles
# Make the Plot Parameters
# pp_test = ahf.Plot_Parameters()
# pp_test = ahf.Plot_Parameters(x_vars=['SA','CT'], y_vars=['press'], plot_type='profiles')
# Make the Analysis Group
# group_AIDJEX_missing_ll = ahf.Analysis_Group(ds_AIDJEX_missing_ll, pfs_0, pp_test, plot_title='BigBear lat-lon errors')
# group_AIDJEX_missing_ll = ahf.Analysis_Group(ds_ITP35_some_pfs2, pfs_0, pp_test, plot_title='ITP35 test pfs')
# group_test = ahf.Analysis_Group(ds_all_AIDJEX, pfs_0, pp_test)
# Make the figure
# ahf.make_figure([group_AIDJEX_missing_ll])
# exit(0)


# Use these things
pfs_this_BGR = pfs_BGR1
ds_this_BGR = ds_BGR_ITPs_all
# ds_this_BGR = ds_BGR_ITPs_0a
# ds_this_BGR = ds_BGR_ITPs_0b
# ds_this_BGR = ds_BGR_ITPs_0c
# ds_this_BGR = ds_BGR_ITPs_0d
# ds_this_BGR = ds_BGR_ITPs_0e
# ds_this_BGR = ds_BGR_ITPs_0f
# ds_this_BGR = ds_BGR_ITPs_0g
# ds_this_BGR = ds_BGR_ITPs_0h


# Output summary
if False:
    # Make the Plot Parameters
    pp_test = ahf.Plot_Parameters(extra_args={'extra_vars_to_keep':['dt_start']})
    # Make the Analysis Group
    group_test = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_test)
    ahf.txt_summary([group_test])

################################################################################
### Figures

### Figure 1
## Maps
pp_map = ahf.Plot_Parameters(plot_type='map', clr_map='clr_by_instrmt', extra_args={'map_extent':'Western_Arctic'})
pp_map_full_Arctic = ahf.Plot_Parameters(plot_type='map', clr_map='clr_by_instrmt', extra_args={'map_extent':'Full_Arctic'})#, legend=False, add_grid=False)
pp_map_by_date = ahf.Plot_Parameters(plot_type='map', clr_map='dt_start', extra_args={'map_extent':'Western_Arctic'})
## Map of all profiles for all sources
if False:
    print('')
    print('- Creating a map of all profiles from all sources')
    # Make the subplot groups
    group_map_full_Arctic = ahf.Analysis_Group(ds_all_ITP, pfs_0, pp_map_full_Arctic, plot_title='')
    group_map = ahf.Analysis_Group(ds_all_ITP, pfs_0, pp_map, plot_title='')
    # Make the figure
    ahf.make_figure([group_map_full_Arctic, group_map], use_same_x_axis=False, use_same_y_axis=False, filename='Figure_1.pickle')
## Map of just in the Beaufort Gyre Region
if True:
    print('')
    print('- Creating a map of profiles in the Beaufort Gyre Region')
    # Make the subplot groups
    # group_map = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_map, plot_title='')
    group_map = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_map_by_date, plot_title='')
    # Make the figure
    ahf.make_figure([group_map], use_same_x_axis=False, use_same_y_axis=False, filename='Figure_1.pickle')
## Map of all upgoing profiles for all sources
if False:
    print('')
    print('- Creating a map of all upgoing profiles from all sources')
    # Make the subplot groups
    group_map_full_Arctic = ahf.Analysis_Group(ds_all_sources_up, pfs_0, pp_map_full_Arctic, plot_title='')
    group_map = ahf.Analysis_Group(ds_all_sources_up, pfs_0, pp_map, plot_title='')
    # Make the figure
    ahf.make_figure([group_map_full_Arctic, group_map], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_1.pickle')
## Map of all upgoing profiles for all sources with a max pressure > 400 dbar
if False:
    print('')
    print('- Creating a map of all upgoing profiles from all sources that go below 400 dbar')
    # Make the subplot groups
    group_map_full_Arctic = ahf.Analysis_Group(ds_all_sources_pmin, pfs_0, pp_map_full_Arctic, plot_title='')
    group_map = ahf.Analysis_Group(ds_all_sources_pmin, pfs_0, pp_map, plot_title='')
    # Make the figure
    ahf.make_figure([group_map_full_Arctic, group_map], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_1.pickle') 
## Map of select profiles from all sources within AOA
if False:
    print('')
    print('- Creating a map of select profiles from all sources within AOA')
    # Make the subplot groups
    group_map_full_Arctic = ahf.Analysis_Group(ds_all_sources_pmin, pfs_AOA, pp_map_full_Arctic, plot_title='')
    group_map = ahf.Analysis_Group(ds_all_sources_pmin, pfs_AOA, pp_map, plot_title='')
    # Make the figure
    ahf.make_figure([group_map_full_Arctic, group_map], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_1.pickle')
## Map of select profiles from all sources within AOA within selected ranges
if False:
    print('')
    print('- Creating a map of select profiles from all sources within AOA filtered')
    # Make plot parameter objects
    pp_map = ahf.Plot_Parameters(plot_type='map', clr_map='clr_by_instrmt', extra_args={'map_extent':'AIDJEX_focus', 'extra_vars_to_keep':['SA','press']})
    pp_map_full_Arctic = ahf.Plot_Parameters(plot_type='map', clr_map='clr_by_source', extra_args={'map_extent':'Full_Arctic', 'extra_vars_to_keep':['SA','press']}, legend=False, add_grid=False)
    # Make the subplot groups
    group_map_full_Arctic = ahf.Analysis_Group(ds_all_sources_pmin, pfs_fltrd, pp_map_full_Arctic, plot_title='')
    group_map = ahf.Analysis_Group(ds_all_sources_pmin, pfs_fltrd, pp_map, plot_title='')
    # Make the figure
    ahf.make_figure([group_map_full_Arctic, group_map], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_1.pickle')
## Maps comparing available profiles to selected profiles
if False:
    print('')
    print('- Creating maps comparing available vs. selected profiles')
    # Make the subplot groups
    group_all      = ahf.Analysis_Group(ds_all_sources_all, pfs_0, pp_map, plot_title='Profiles available')
    group_selected = ahf.Analysis_Group(ds_all_sources_pmin, pfs_AOA, pp_map, plot_title='Profiles selected')
    # Make the figure
    ahf.make_figure([group_all, group_selected], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_1.pickle')
    # Find the maximum distance between any two profiles for each data set in the group
    # ahf.find_max_distance([group_all])
    # ahf.find_max_distance([group_selected])
## Map to find coincident profiles (or as close as I can get)
if False:
    print('')
    print('- Creating map to find example profiles')
    pp_map = ahf.Plot_Parameters(plot_type='map', clr_map='clr_by_source', extra_args={'map_extent':'AIDJEX_focus'})
    # Make the subplot groups
    # group_selected = ahf.Analysis_Group(ds_all_sources_pmin, pfs_coin, pp_map, plot_title='Profiles used in study, filtered to small area')
    group_selected = ahf.Analysis_Group(ds_all_sources_ex_pfs, pfs_ex_pfs, pp_map, plot_title='Example profiles')
    # Make the figure
    ahf.make_figure([group_selected], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_1.pickle')
### Date and Distance spans
## Maps of BGOS ITPs by date
if False:
    print('')
    print('- Creating maps for each BGOS ITP')
    # Make the subplot groups
    group_ITP33 = ahf.Analysis_Group(ds_ITP33, pfs_AOA, pp_map_by_date)
    group_ITP34 = ahf.Analysis_Group(ds_ITP34, pfs_AOA, pp_map_by_date)
    group_ITP35 = ahf.Analysis_Group(ds_ITP35, pfs_AOA, pp_map_by_date)
    group_ITP41 = ahf.Analysis_Group(ds_ITP41, pfs_AOA, pp_map_by_date)
    group_ITP42 = ahf.Analysis_Group(ds_ITP42, pfs_AOA, pp_map_by_date)
    group_ITP43 = ahf.Analysis_Group(ds_ITP43, pfs_AOA, pp_map_by_date)
    # Make the figure
    # ahf.make_figure([group_ITP33, group_ITP34, group_ITP35, group_ITP41, group_ITP42, group_ITP43], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_1.pickle')
    # Find the maximum distance between any two profiles for each data set in the group
    ahf.find_max_distance([group_ITP33, group_ITP34, group_ITP35, group_ITP41, group_ITP42, group_ITP43])
## Maps of AIDJEX stations by date
if False:
    print('')
    print('- Creating maps for each AIDJEX station')
    # Make the subplot groups
    group_BigBear  = ahf.Analysis_Group(ds_AIDJEX_BigBear,  pfs_AOA, pp_map_by_date)
    group_BlueFox  = ahf.Analysis_Group(ds_AIDJEX_BlueFox,  pfs_AOA, pp_map_by_date)
    group_Caribou  = ahf.Analysis_Group(ds_AIDJEX_Caribou,  pfs_AOA, pp_map_by_date)
    group_Snowbird = ahf.Analysis_Group(ds_AIDJEX_Snowbird, pfs_AOA, pp_map_by_date)
    # Make the figure
    # ahf.make_figure([group_BigBear, group_BlueFox, group_Caribou, group_Snowbird], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_1.pickle')
    # Find the maximum distance between any two profiles for each data set in the group
    ahf.find_max_distance([group_BigBear, group_BlueFox, group_Caribou, group_Snowbird])
## Maps of AIDJEX and BGOS ITPs by date
if False:
    print('')
    print('- Creating maps by date for AIDJEX and BGOS ITPs')
    # Make the subplot groups
    group_AIDJEX    = ahf.Analysis_Group(ds_AIDJEX, pfs_AOA, pp_map_by_date, plot_title='AIDJEX')
    group_BGOS_ITP  = ahf.Analysis_Group(ds_BGOS,   pfs_AOA, pp_map_by_date, plot_title='BGOS ITPs')
    # Make the figure
    # ahf.make_figure([group_AIDJEX, group_BGOS_ITP], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_1.pickle')
    # Find the maximum distance between any two profiles for each data set in the group
    ahf.find_max_distance([group_AIDJEX, group_BGOS_ITP])

## la_CT-SA vs. la_CT-SA-dt_start plots
if False:
    print('')
    print('- Creating TS and TS-time plots')
    # Make the Plot Parameters
    pp_CT_SA = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['SA'], clr_map='clr_by_instrmt')
    # pp_CT_SA = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':280, 'b_a_w_plt':False})
    # pp_CT_SA_3d = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['dt_start'], z_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'dt_start', 'cl_z_var':'la_CT', 'm_pts':280, 'b_a_w_plt':False})
    # Make the Analysis Group pfs_fltrd pfs_BGR1
    group_CT_SA_plot = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_CT_SA)
    # group_CT_SA_plot = ahf.Analysis_Group(ds_ITP_test, pfs_BGR1, pp_CT_SA)
    # group_CT_SA_3d_plot = ahf.Analysis_Group(ds_BGOS, pfs_fltrd, pp_CT_SA_3d)
    # Make the figure
    ahf.make_figure([group_CT_SA_plot])#, group_CT_SA_3d_plot])

# TS plot
if False:
    print('')
    print('- Creating TS plots')
    # Make the Plot Parameters
    pp_TS = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['iT'], clr_map='clr_all_same')
    # Make the Analysis Group
    group_TS_plot = ahf.Analysis_Group(ds_ITP_test, pfs_fltrd, pp_TS)
    # Make the figure
    ahf.make_figure([group_TS_plot])

## AIDJEX blacklisted profiles TS plots
if False:
    print('')
    print('- Creating figure for AIDJEX blacklisted profiles')
    # Make the Plot Parameters
    pp_all = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['iT'], clr_map='clr_by_instrmt', legend=True)
    pp_one = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['iT'], clr_map='prof_no', legend=True)
    # Make the Analysis Group
    group_AIDJEX__blacklist = ahf.Analysis_Group(ds_all_AIDJEX, pfs_0, pp_all, plot_title='All AIDJEX Profiles')
    group_BigBear_blacklist = ahf.Analysis_Group(ds_AIDJEX_BigBear_blacklist, pfs_0, pp_one, plot_title='BigBear Blacklist')
    group_BlueFox_blacklist = ahf.Analysis_Group(ds_AIDJEX_BlueFox_blacklist, pfs_0, pp_one, plot_title='BlueFox Blacklist')
    group_Snowbird_blacklist = ahf.Analysis_Group(ds_AIDJEX_Snowbird_blacklist, pfs_0, pp_one, plot_title='Snowbird Blacklist')
    # Make the figure
    ahf.make_figure([group_AIDJEX__blacklist, group_BigBear_blacklist, group_BlueFox_blacklist, group_Snowbird_blacklist], use_same_x_axis=False, use_same_y_axis=False)
## SHEBA blacklisted profiles plots
if False:
    print('')
    print('- Creating figure for SHEBA blacklisted profiles')
    # Make the Plot Parameters
    # pp_pfs = ahf.Plot_Parameters(x_vars=['SP','iT'], y_vars=['press'], plot_type='profiles')
    pp_pfs = ahf.Plot_Parameters(x_vars=['SP'], y_vars=['iT'])
    # Make the Analysis Group
    group_SHEBA_blacklist = ahf.Analysis_Group(ds_SHEBA_Seacat_blacklist, pfs_0, pp_pfs, plot_title='SHEBA Blacklisted Profiles')
    # Make the figure
    ahf.make_figure([group_SHEBA_blacklist])
## AIDJEX BigBear profiles missing latitude and longitude values
if False:
    print('')
    print('- Creating figure for BigBear profiles missing latitude and longitude values')
    # Make the Plot Parameters
    pp_pfs = ahf.Plot_Parameters(x_vars=['SA','CT'], y_vars=['press'], plot_type='profiles')
    # Make the Analysis Group
    group_AIDJEX_missing_ll = ahf.Analysis_Group(ds_AIDJEX_missing_ll, pfs_0, pp_pfs, plot_title='BigBear lat-lon errors')
    # Make the figure
    ahf.make_figure([group_AIDJEX_missing_ll])
## Profiles filtered out by p_max > 400 dbar
if False:
    print('')
    print('- Creating figure for profiles eliminated by p_max > 400 dbar filter')
    # Make the Plot Parameters
    # pp_pfs = ahf.Plot_Parameters(x_vars=['SA','CT'], y_vars=['press'], plot_type='profiles')
    pp_pfs = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='press')
    # Make the Analysis Group
    group_pmax_lt_400 = ahf.Analysis_Group(ds_pmax_lt_400, pfs_0, pp_pfs, plot_title='Profiles with p_max < 400 dbar')
    # Make the figure
    ahf.make_figure([group_pmax_lt_400])

## SHEBA TS plot
if False:
    print('')
    print('- Creating TS plot for SHEBA profiles')
    # Make the Plot Parameters
    pp_TS = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='dt_start')#, extra_args={'extra_vars_to_keep':['SA','press']})
    # pp_map = ahf.Plot_Parameters(plot_type='map', clr_map='clr_by_source', extra_args={'map_extent':'AIDJEX_focus', 'extra_vars_to_keep':['SA','press']})
    # Make the Analysis Group pfs_fltrd pfs_AOA
    group_SHEBA_TS = ahf.Analysis_Group(ds_SHEBA, pfs_AOA, pp_TS)
    group_SHEBA_TS1 = ahf.Analysis_Group(ds_SHEBA, pfs_AOA1, pp_TS)
    # group_SHEBA_map = ahf.Analysis_Group(ds_SHEBA, pfs_fltrd, pp_map)
    # Make the figure
    # ahf.make_figure([group_SHEBA_map, group_SHEBA_TS])
    ahf.make_figure([group_SHEBA_TS, group_SHEBA_TS1], use_same_x_axis=False, use_same_y_axis=False)
## TS plot of all sources
if False:
    print('')
    print('- Creating TS plot of all sources')
    # Make the Plot Parameters
    pp_CT_SA = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='clr_by_source')
    # Make the Analysis Group pfs_fltrd pfs_AOA1
    group_CT_SA_plot = ahf.Analysis_Group(ds_all_sources_pmin, pfs_AOA1, pp_CT_SA)
    # Make the figure
    ahf.make_figure([group_CT_SA_plot])
## Example profiles from all sources
if False:
    print('')
    print('- Creating figure of example profiles')
    # Make the Plot Parameters
    zoom_range = [425,300]
    pp_pfs_CT_ = ahf.Plot_Parameters(x_vars=['CT'], y_vars=['press'], plot_type='profiles', extra_args={'shift_pfs':False})
    pp_pfs_SA_ = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['press'], plot_type='profiles', extra_args={'shift_pfs':False})
    pp_pfs_CTz = ahf.Plot_Parameters(x_vars=['CT'], y_vars=['press'], plot_type='profiles', extra_args={'shift_pfs':False}, ax_lims={'y_lims':zoom_range})
    pp_pfs_SAz = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['press'], plot_type='profiles', extra_args={'shift_pfs':False}, ax_lims={'y_lims':zoom_range})
    # Make the Analysis Group
    this_ds = ds_all_sources_ex_pfs
    group_example_profiles1 = ahf.Analysis_Group(this_ds, pfs_0, pp_pfs_CT_, plot_title='')
    group_example_profiles2 = ahf.Analysis_Group(this_ds, pfs_0, pp_pfs_SA_, plot_title='')
    group_example_profiles3 = ahf.Analysis_Group(this_ds, pfs_0, pp_pfs_CTz, plot_title='')
    group_example_profiles4 = ahf.Analysis_Group(this_ds, pfs_0, pp_pfs_SAz, plot_title='')
    # Make the figure
    # ahf.make_figure([group_example_profiles1], use_same_y_axis=False)
    ahf.make_figure([group_example_profiles1, group_example_profiles2, group_example_profiles3, group_example_profiles4], use_same_y_axis=False)
## Example profile plot
if False:
    print('')
    print('- Creating figure of an example profile')
    # Make the Plot Parameters
    zoom_range = [360,310]
    pp_pfs_full = ahf.Plot_Parameters(x_vars=['iT','SP'], y_vars=['depth'], plot_type='profiles')
    pp_pfs_zoom = ahf.Plot_Parameters(x_vars=['SP','iT'], y_vars=['depth'], plot_type='profiles', ax_lims={'y_lims':zoom_range}, add_grid=False)
    # Make the Analysis Group
    # this_ds = ds_all_sources_ex_pfs
    this_ds = ds_AIDJEX_ex_stairs
    group_example_profiles1 = ahf.Analysis_Group(this_ds, pfs_0, pp_pfs_full, plot_title='')
    group_example_profiles2 = ahf.Analysis_Group(this_ds, pfs_0, pp_pfs_zoom, plot_title='')
    # Make the figure
    # ahf.make_figure([group_example_profiles2], use_same_y_axis=False)
    ahf.make_figure([group_example_profiles1, group_example_profiles2], use_same_y_axis=False, use_same_x_axis=False)
## Example profiles from SHEBA
if False:
    print('')
    print('- Creating figure of example profiles')
    # Make the Plot Parameters
    pp_pfs = ahf.Plot_Parameters(x_vars=['SA','CT'], y_vars=['press'], plot_type='profiles')
    # Make the Analysis Group
    group_SHEBA_profile = ahf.Analysis_Group(ds_SHEBA_example_profile, pfs_0, pp_pfs)
    # Make the figure
    ahf.make_figure([group_SHEBA_profile])
## Histograms of pressure max
if False:
    print('')
    print('- Creating histograms of pressure max in each profile')
    # Make the Plot Parameters
    pp_max_press_hist = ahf.Plot_Parameters(plot_scale='by_pf', x_vars=['max_press'], y_vars=['hist'], clr_map='clr_all_same')
    # Make the subplot groups
    group_BGOS_max_press_hist = ahf.Analysis_Group(ds_all_BGOS, pfs_0, pp_max_press_hist, plot_title=r'BGOS ITPs')
    group_BGOS_mp_max_press_hist = ahf.Analysis_Group(ds_all_BGOS_mp, pfs_0, pp_max_press_hist, plot_title=r'BGOS ITPs, min press filtered')
    # group_AIDJEX_max_press_hist = ahf.Analysis_Group(ds_all_AIDJEX, pfs_0, pp_max_press_hist, plot_title=r'AIDJEX')
    # Make the figure
    ahf.make_figure([group_BGOS_max_press_hist, group_BGOS_mp_max_press_hist])
## Histograms of CT_max data
if False:
    print('')
    print('- Creating histograms of pressure at temperature max in each profile')
    # Make the Plot Parameters
    pp_press_CT_max_hist = ahf.Plot_Parameters(x_vars=['press_CT_max'], y_vars=['hist'], clr_map='clr_all_same')
    # Make the subplot groups
    group_BGOS_press_CT_max_hist = ahf.Analysis_Group(ds_all_BGOS, pfs_AOA, pp_press_CT_max_hist, plot_title=r'BGOS ITPs')
    group_AIDJEX_press_CT_max_hist = ahf.Analysis_Group(ds_all_AIDJEX, pfs_AOA, pp_press_CT_max_hist, plot_title=r'AIDJEX')
    # # Make the figure
    ahf.make_figure([group_BGOS_press_CT_max_hist, group_AIDJEX_press_CT_max_hist])
## Plots of SA_CT_max vs press_CT_max
if False:
    print('')
    print('- Creating plots of the info about CT_max')
    # Make the Plot Parameters
    pp_SA_CT_max = ahf.Plot_Parameters(x_vars=['SA_CT_max'], y_vars=['press_CT_max'], clr_map='CT_max')
    # Make the subplot groups
    group_AIDJEX = ahf.Analysis_Group(ds_AIDJEX, pfs_AOA, pp_SA_CT_max, plot_title=r'AIDJEX')
    group_SHEBA  = ahf.Analysis_Group(ds_SHEBA, pfs_AOA, pp_SA_CT_max, plot_title=r'SHEBA')
    group_BGOS   = ahf.Analysis_Group(ds_BGOS, pfs_AOA, pp_SA_CT_max, plot_title=r'BGOS')
    # # Make the figure
    ahf.make_figure([group_AIDJEX, group_SHEBA, group_BGOS])

# Plots in la_CT vs. SA space with different values of ell
if False:
    print('')
    print('- Creating plots in la_CT--SA space with different values of ell')
    pfs_fltrd1 = ahf.Profile_Filters(lon_range=lon_AOA, lat_range=lat_AOA, SA_range=LHW_S_range, lt_pCT_max=True, m_avg_win=10)
    pfs_fltrd2 = ahf.Profile_Filters(lon_range=lon_AOA, lat_range=lat_AOA, SA_range=LHW_S_range, lt_pCT_max=True, m_avg_win=50)
    pfs_fltrd3 = ahf.Profile_Filters(lon_range=lon_AOA, lat_range=lat_AOA, SA_range=LHW_S_range, lt_pCT_max=True, m_avg_win=100)
    pfs_fltrd4 = ahf.Profile_Filters(lon_range=lon_AOA, lat_range=lat_AOA, SA_range=LHW_S_range, lt_pCT_max=True, m_avg_win=150)
    # Make the Plot Parameters
    pp_test = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='clr_by_instrmt')
    # Make the subplot groups
    group_test1 = ahf.Analysis_Group(ds_BGOS, pfs_fltrd1, pp_test, plot_title=r'ell = 10')
    group_test2 = ahf.Analysis_Group(ds_BGOS, pfs_fltrd2, pp_test, plot_title=r'ell = 50')
    group_test3 = ahf.Analysis_Group(ds_BGOS, pfs_fltrd3, pp_test, plot_title=r'ell = 100')
    group_test4 = ahf.Analysis_Group(ds_BGOS, pfs_fltrd4, pp_test, plot_title=r'ell = 150')
    # # Make the figure
    ahf.make_figure([group_test1, group_test2, group_test3, group_test4], use_same_y_axis=False)
# Plots in la_CT vs. SA space for ell=10 and ell=100
if False:
    print('')
    print('- Creating plots in la_CT--SA space with different values of ell')
    # Make profile filters
    pfs_ell_010 = ahf.Profile_Filters(lon_range=lon_AOA, lat_range=lat_AOA, SA_range=LHW_S_range, lt_pCT_max=True, m_avg_win=10)
    pfs_ell_100 = ahf.Profile_Filters(lon_range=lon_AOA, lat_range=lat_AOA, SA_range=LHW_S_range, lt_pCT_max=True, m_avg_win=100)
    #   Subsampled versions
    pfs_ell_010ss = ahf.Profile_Filters(lon_range=lon_AOA, lat_range=lat_AOA, SA_range=LHW_S_range, lt_pCT_max=True, m_avg_win=10)
    pfs_ell_100ss = ahf.Profile_Filters(lon_range=lon_AOA, lat_range=lat_AOA, SA_range=LHW_S_range, lt_pCT_max=True, m_avg_win=100, subsample=True)
    # Make the Plot Parameters
    pp_test = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='clr_by_instrmt')
    # Make the subplot groups
    group_AIDJEX_ell_010 = ahf.Analysis_Group(ds_AIDJEX, pfs_ell_010, pp_test, plot_title=r'AIDJEX ell = 10')
    group_AIDJEX_ell_100 = ahf.Analysis_Group(ds_AIDJEX, pfs_ell_100, pp_test, plot_title=r'AIDJEX ell = 100')
    group_BGOS_ell_010 = ahf.Analysis_Group(ds_BGOS, pfs_ell_010, pp_test, plot_title=r'BGOS ell = 10')
    group_BGOS_ell_100 = ahf.Analysis_Group(ds_BGOS, pfs_ell_100, pp_test, plot_title=r'BGOS ell = 100')
    group_BGOSss_ell_010 = ahf.Analysis_Group(ds_BGOS, pfs_ell_010ss, pp_test, plot_title=r'BGOSss ell = 10')
    group_BGOSss_ell_100 = ahf.Analysis_Group(ds_BGOS, pfs_ell_100ss, pp_test, plot_title=r'BGOSss ell = 100')
    # # Make the figure
    ahf.make_figure([group_AIDJEX_ell_010, group_BGOS_ell_010, group_BGOSss_ell_010, group_AIDJEX_ell_100, group_BGOS_ell_100, group_BGOSss_ell_100], use_same_x_axis=False, use_same_y_axis=False, filename='AIDJEX_BGOS-ss_la_CT_vs_SA_ell_010_and_100.png')

## Subsampling ITP data
# Histograms of the first differences for data sets
if False:
    print('')
    print('- Creating histograms of first differences in pressures')
    # Make the Plot Parameters
    pp_first_diff_press = ahf.Plot_Parameters(x_vars=['press'], y_vars=['hist'], clr_map='clr_all_same', first_dfs=[True,False])
    # Make the subplot groups
    # group_AIDJEX_press_hist = ahf.Analysis_Group(ds_AIDJEX, pfs_fltrd, pp_first_diff_press, plot_title=r'AIDJEX')
    group_AIDJEX_press_hist = ahf.Analysis_Group(ds_AIDJEX, pfs_BGOS, pp_first_diff_press, plot_title=r'AIDJEX')
    # # Make the figure
    ahf.make_figure([group_AIDJEX_press_hist])#, use_same_x_axis=False, use_same_y_axis=False)
# Comparing BGOS to ssBGOS
if False:
    print('')
    print('- Creating TS plots to compare full profiles vs. subsampled')
    # Make the Plot Parameters
    pp_TS = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='clr_all_same')
    # Make the subplot groups
    group_BGOS_full = ahf.Analysis_Group(ds_BGOS, pfs_fltrd, pp_TS, plot_title=r'BGOS ITPs full')
    group_BGOS_ss = ahf.Analysis_Group(ds_BGOS, pfs_fltrd_ss, pp_TS, plot_title=r'BGOS ITPs subsampled, every 4')
    # # Make the figure
    ahf.make_figure([group_BGOS_full, group_BGOS_ss])

## Plotting salinity vs time
# 
if False:
    print('')
    print('- Creating plots of salinity vs time')
    # Make the Plot Parameters
    pp_SA_vs_dt = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['dt_start'], clr_map='cluster', extra_args={'plt_noise':True})
    # Make the subplot groups ds_BGOS_m280_e050 ds_BGOS
    group_BGOS = ahf.Analysis_Group(ds_BGOS_m280_e050, pfs_fltrd, pp_SA_vs_dt, plot_title=r'BGOS ITPs')
    # Make the figure
    ahf.make_figure([group_BGOS], use_same_x_axis=False, use_same_y_axis=False)

    # Make the subplot groups
    # group_AIDJEX = ahf.Analysis_Group(ds_AIDJEX, pfs_fltrd, pp_SA_vs_dt, plot_title=r'BGOS ITPs')
    # # Make the figure
    # ahf.make_figure([group_AIDJEX], use_same_x_axis=False, use_same_y_axis=False)

    # Make the subplot groups
    # group_SHEBA = ahf.Analysis_Group(ds_SHEBA, pfs_fltrd, pp_SA_vs_dt, plot_title=r'BGOS ITPs')
    # # Make the figure
    # ahf.make_figure([group_SHEBA], use_same_x_axis=False, use_same_y_axis=False)

## TS diagrams
# ITP TS diagrams
if False:
    print('')
    print('- Creating TS plots to compare full profiles vs. filtered to p<pCT_max')
    # Make the Plot Parameters
    pp_TS = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='clr_all_same')
    # Make the subplot groups
    group_BGOS_full_pfs = ahf.Analysis_Group(ds_all_BGOS, pfs_AOA, pp_TS, plot_title=r'BGOS ITPs full')
    # group_BGOS_filtered = ahf.Analysis_Group(ds_all_BGOS, pfs_BGOS, pp_TS, plot_title=r'BGOS ITPs SA: 34.366 - 35.5')
    group_BGOS_filtered = ahf.Analysis_Group(ds_all_BGOS, pfs_BGOS_fltrd, pp_TS, plot_title=r'BGOS ITPs filtered')
    # # Make the figure
    ahf.make_figure([group_BGOS_full_pfs, group_BGOS_filtered], use_same_x_axis=False, use_same_y_axis=False)
# TS diagram for AIDJEX, SHEBA, and BGOS datasets on same plot
if False:
    print('')
    print('- Creating test TS plots')
    # Make the Plot Parameters
    pp_TS = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='clr_by_source')
    # Make the subplot groups pfs_fltrd
    group_test0 = ahf.Analysis_Group(ds_all_sources_ex_pfs, pfs_AOA, pp_TS)
    # # Make the figure
    ahf.make_figure([group_test0], use_same_x_axis=False, use_same_y_axis=False)
# TS diagrams for AIDJEX, SHEBA, and BGOS datasets separately
if False:
    print('')
    print('- Creating TS plots of all datasets')
    # Make the Plot Parameters
    pp_TS = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='lon')
    # pp_TS = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='density_hist', extra_args={'clr_min':0, 'clr_max':100, 'clr_ext':'max', 'xy_bins':250})
    # Make the subplot groups pfs_fltrd pfs_AOA
    group_AIDJEX_TS = ahf.Analysis_Group(ds_AIDJEX, pfs_AOA, pp_TS, plot_title=r'AIDJEX')
    group_SHEBA_TS  = ahf.Analysis_Group(ds_SHEBA, pfs_AOA, pp_TS, plot_title=r'SHEBA')
    group_BGOS_TS   = ahf.Analysis_Group(ds_BGOS, pfs_AOA, pp_TS, plot_title=r'BGOS ITPs')
    # # Make the figure
    ahf.make_figure([group_AIDJEX_TS, group_SHEBA_TS, group_BGOS_TS])#, use_same_x_axis=False, use_same_y_axis=False)
    # ahf.make_figure([group_AIDJEX_TS, group_BGOS_TS])#, use_same_x_axis=False, use_same_y_axis=False)

# test clustering
if False:
    print('')
    print('- Creating clustering plot')
    # Make the Plot Parameters
    # pp_live_clstr = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='clr_all_same')
    pp_live_clstr = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':660, 'b_a_w_plt':True})
    # Make the subplot groups
    # group_clstrd = ahf.Analysis_Group(ds_BGR_ITPs_0a, pfs_BGR1, pp_live_clstr)
    group_clstrd = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_live_clstr)
    # # Make the figure
    ahf.make_figure([group_clstrd])

## Clustering parameter sweeps
# ## Parameter sweep for BGR ITP data
if False:
    print('')
    print('- Creating clustering parameter sweep for BGR ITP data')
    test_mpts = 360
    # Make the Plot Parameters
    pp_mpts_param_sweep = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'cl_ps_tuple':[10,721,10]}) #[10,711,20]
    # pp_ell_param_sweep  = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'cl_ps_tuple':[10,271,10]}) 
    # Make the subplot groups
    group_mpts_param_sweep = ahf.Analysis_Group(ds_this_BGR, pfs_this_BGR, pp_mpts_param_sweep)
    # group_ell_param_sweep  = ahf.Analysis_Group(ds_BGOS, pfs_fltrd, pp_ell_param_sweep, plot_title='BGOS')
    # # Make the figure
    # ahf.make_figure([group_mpts_param_sweep, group_ell_param_sweep], filename='test_param_sweep_BGOS.pickle')
    ahf.make_figure([group_mpts_param_sweep], filename='test_param_sweep_BGR.pickle')
## Parameter sweep for BGOS ITP data
if False:
    print('')
    print('- Creating clustering parameter sweep for BGOS ITP data')
    test_mpts = 360
    # Make the Plot Parameters
    pp_mpts_param_sweep = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'cl_ps_tuple':[10,721,10]}) #[50,711,20]
    # pp_ell_param_sweep  = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'cl_ps_tuple':[10,271,10]}) 
    # Make the subplot groups
    group_mpts_param_sweep = ahf.Analysis_Group(ds_BGOS, pfs_fltrd, pp_mpts_param_sweep, plot_title='BGOS')
    # group_ell_param_sweep  = ahf.Analysis_Group(ds_BGOS, pfs_fltrd, pp_ell_param_sweep, plot_title='BGOS')
    # # Make the figure
    # ahf.make_figure([group_mpts_param_sweep, group_ell_param_sweep], filename='test_param_sweep_BGOS.pickle')
    ahf.make_figure([group_mpts_param_sweep], filename='test_param_sweep_BGOS.pickle')
## Parameter sweep for BGOSss ITP data
if False:
    print('')
    print('- Creating clustering parameter sweep for BGOSss ITP data')
    test_mpts = 220
    # Make the Plot Parameters
    pp_mpts_param_sweep = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'cl_ps_tuple':[50,721,10]}) #[50,711,20]
    pp_ell_param_sweep  = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'cl_ps_tuple':[10,271,10]}) 
    # Make the subplot groups
    group_mpts_param_sweep = ahf.Analysis_Group(ds_BGOS, pfs_fltrd_ss, pp_mpts_param_sweep, plot_title='BGOSss')
    # group_ell_param_sweep  = ahf.Analysis_Group(ds_BGOS, pfs_fltrd_ss, pp_ell_param_sweep, plot_title='BGOSss')
    # # Make the figure
    ahf.make_figure([group_mpts_param_sweep], filename='test_param_sweep_BGOSss.pickle')
## Parameter sweep for AIDJEX data
if False:
    print('')
    print('- Creating clustering parameter sweep for AIDJEX data')
    test_mpts = 300
    # Make the Plot Parameters
    pp_mpts_param_sweep = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'cl_ps_tuple':[10,721,10]}) #[50,711,20]
    # pp_ell_param_sweep  = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'cl_ps_tuple':[10,271,10]}) 
    # Make the subplot groups
    group_mpts_param_sweep = ahf.Analysis_Group(ds_AIDJEX, pfs_fltrd, pp_mpts_param_sweep, plot_title='AIDJEX')
    # group_ell_param_sweep  = ahf.Analysis_Group(ds_AIDJEX, pfs_fltrd, pp_ell_param_sweep, plot_title='AIDJEX')
    # # Make the figure
    ahf.make_figure([group_mpts_param_sweep], filename='test_param_sweep_AIDJEX.pickle')

## Clustering Results
# AIDJEX clustering
if False:
    print('')
    print('- Creating clustering plot of AIDJEX data')
    pfs_fltrd1 = ahf.Profile_Filters(lon_range=lon_AOA, lat_range=lat_AOA, SA_range=LHW_S_range, lt_pCT_max=True, m_avg_win=100)
    test_mpts = 470
    # Make the Plot Parameters
    pp_live_clstr = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':100, 'b_a_w_plt':True})
    pp_live_clstr2 = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':500, 'b_a_w_plt':True})
    pp_pre_clstrd = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'b_a_w_plt':True})
    # Make the subplot groups
    # group_AIDJEX_BB_clstrs = ahf.Analysis_Group(ds_AIDJEX_BigBear, pfs_fltrd, pp_clstrs)
    # group_AIDJEX_live_clstr = ahf.Analysis_Group(ds_AIDJEX, pfs_fltrd1, pp_live_clstr, plot_title=r'AIDJEX ell = 10')
    # group_AIDJEX_live_clstr2 = ahf.Analysis_Group(ds_AIDJEX, pfs_fltrd, pp_live_clstr2, plot_title=r'AIDJEX ell = 10')
    group_AIDJEX_pre_clstrd = ahf.Analysis_Group(ds_AIDJEX_m500_e010, pfs_0, pp_pre_clstrd, plot_title=r'AIDJEX clusters from file')
    # # Make the figure
    ahf.make_figure([group_AIDJEX_pre_clstrd])
    # ahf.make_figure([group_AIDJEX_live_clstr, group_AIDJEX_live_clstr2])
# BGOS clustering
if False:
    print('')
    print('- Creating clustering plot of BGOS data')
    # Make the Plot Parameters
    pp_live_clstr = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':440, 'b_a_w_plt':True})
    pp_live_clstr2 = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':640, 'b_a_w_plt':True})
    # pp_pre_clstrd = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'b_a_w_plt':True})
    # Make the subplot groups
    group_BGOS_live_clstr = ahf.Analysis_Group(ds_BGOS, pfs_fltrd, pp_live_clstr, plot_title=r'BGOS ell = 10')
    group_BGOS_live_clstr2 = ahf.Analysis_Group(ds_BGOS, pfs_fltrd, pp_live_clstr2, plot_title=r'BGOS ell = 10')
    # group_BGOS_pre_clstrd = ahf.Analysis_Group(ds_BGOS_clstrd, pfs_fltrd, pp_pre_clstrd, plot_title=r'BGOS clusters from file')
    # # Make the figure
    ahf.make_figure([group_BGOS_live_clstr, group_BGOS_live_clstr2])
# BGOS clustering 2
if False:
    print('')
    print('- Creating clustering plot of BGOS data')
    pfs_fltrd1 = ahf.Profile_Filters(lon_range=lon_AOA, lat_range=lat_AOA, SA_range=LHW_S_range, lt_pCT_max=True, m_avg_win=100)
    # Make the Plot Parameters
    pp_live_clstr = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':360, 'b_a_w_plt':True})
    pp_live_clstr2 = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':360, 'b_a_w_plt':True})
    # pp_pre_clstrd = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'b_a_w_plt':True})
    # Make the subplot groups
    group_BGOS_live_clstr = ahf.Analysis_Group(ds_BGOS, pfs_fltrd1, pp_live_clstr, plot_title=r'BGOS ell = 100')
    group_BGOS_live_clstr2 = ahf.Analysis_Group(ds_BGOS, pfs_fltrd, pp_live_clstr2, plot_title=r'BGOS m_pts = 200')
    # group_BGOS_pre_clstrd = ahf.Analysis_Group(ds_BGOS_clstrd, pfs_fltrd, pp_pre_clstrd, plot_title=r'BGOS clusters from file')
    # # Make the figure
    ahf.make_figure([group_BGOS_live_clstr, group_BGOS_live_clstr2])
# BGOSss clustering
if False:
    print('')
    print('- Creating clustering plot of BGOSss data')
    test_mpts = 260
    # Make the Plot Parameters
    pp_live_clstr = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':140, 'b_a_w_plt':True})
    pp_live_clstr2 = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':470, 'b_a_w_plt':True})
    # pp_pre_clstrd = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'b_a_w_plt':True})
    # Make the subplot groups
    group_BGOS_live_clstr = ahf.Analysis_Group(ds_BGOS, pfs_fltrd_ss, pp_live_clstr, plot_title=r'BGOSss ell = 10')
    # group_BGOS_live_clstr2 = ahf.Analysis_Group(ds_BGOS, pfs_fltrd_ss, pp_live_clstr2, plot_title=r'BGOSss ell = 10')
    # group_BGOS_pre_clstrd = ahf.Analysis_Group(ds_BGOS_clstrd, pfs_fltrd, pp_pre_clstrd, plot_title=r'BGOS clusters from file')
    # # Make the figure
    ahf.make_figure([group_BGOS_live_clstr])

### Pre-clustered files
## la_CT vs. SA plots
# Make the Plot Parameters
pp_pre_clstrd = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'b_a_w_plt':True})
pp_nir_SA = ahf.Plot_Parameters(x_vars=['nir_SA'], y_vars=['ca_press'], clr_map='cluster', extra_args={'b_a_w_plt':False, 'plot_noise':False})
pp_cRL = ahf.Plot_Parameters(x_vars=['cRL'], y_vars=['ca_press'], clr_map='cluster', extra_args={'b_a_w_plt':False, 'plot_noise':False, 'plot_slopes':True})
# BGR ITP clustering
if False:
    print('')
    print('- Creating plots of pre-clustered BGR ITP data')
    # this_ds = ds_BGRa_m110
    # this_ds = ds_BGRf_m410
    # this_ds = ds_BGRg_m240
    this_ds = ds_BGRh_m390
    # Make the subplot groups
    group_pre_clstrd = ahf.Analysis_Group(this_ds, pfs_0, pp_pre_clstrd)
    # group_AIDJEX_nir_SA = ahf.Analysis_Group(this_ds, pfs_0, pp_nir_SA, plot_title=r'AIDJEX')
    # group_AIDJEX_cRL = ahf.Analysis_Group(this_ds, pfs_0, pp_cRL, plot_title=r'AIDJEX')
    # # Make the figure
    # ahf.make_figure([group_AIDJEX_pre_clstrd, group_AIDJEX_nir_SA, group_AIDJEX_cRL], use_same_y_axis=False)
    ahf.make_figure([group_pre_clstrd], use_same_y_axis=False)

# AIDJEX clustering
if False:
    print('')
    print('- Creating plots of pre-clustered AIDJEX data')
    # this_ds = ds_AIDJEX_m500_e010
    this_ds = ds_AIDJEX_m490_e050
    # this_ds = ds_AIDJEX_m300_e100
    # Make the subplot groups
    group_AIDJEX_pre_clstrd = ahf.Analysis_Group(this_ds, pfs_0, pp_pre_clstrd, plot_title=r'AIDJEX clusters from file')
    # group_AIDJEX_nir_SA = ahf.Analysis_Group(this_ds, pfs_0, pp_nir_SA, plot_title=r'AIDJEX')
    # group_AIDJEX_cRL = ahf.Analysis_Group(this_ds, pfs_0, pp_cRL, plot_title=r'AIDJEX')
    # # Make the figure
    # ahf.make_figure([group_AIDJEX_pre_clstrd, group_AIDJEX_nir_SA, group_AIDJEX_cRL], use_same_y_axis=False)
    ahf.make_figure([group_AIDJEX_pre_clstrd], use_same_y_axis=False)
# BGOS clustering
if False:
    print('')
    print('- Creating plots of pre-clustered BGOS data')
    # this_ds = ds_BGOS_m440_e010
    this_ds = ds_BGOS_m280_e050
    # this_ds = ds_BGOS_m360_e100
    # Make the subplot groups
    group_BGOS_pre_clstrd = ahf.Analysis_Group(this_ds, pfs_0, pp_pre_clstrd, plot_title=r'BGOS clusters from file')
    group_BGOS_nir_SA = ahf.Analysis_Group(this_ds, pfs_0, pp_nir_SA, plot_title=r'BGOS')
    group_BGOS_cRL = ahf.Analysis_Group(this_ds, pfs_0, pp_cRL, plot_title=r'BGOS')
    # # Make the figure
    ahf.make_figure([group_BGOS_pre_clstrd, group_BGOS_nir_SA, group_BGOS_cRL])
# BGOSss clustering
if False:
    print('')
    print('- Creating plot of pre-clustered BGOSss data')
    # this_ds = ds_BGOSss_m260_e010
    this_ds = ds_BGOSss_m340_e050
    # this_ds = ds_BGOSss_m220_e100
    # Make the subplot groups
    group_BGOSss_pre_clstrd = ahf.Analysis_Group(this_ds, pfs_0, pp_pre_clstrd, plot_title=r'BGOSss clusters from file')
    group_BGOSss_nir_SA = ahf.Analysis_Group(this_ds, pfs_0, pp_nir_SA, plot_title=r'BGOSss')
    group_BGOSss_cRL = ahf.Analysis_Group(this_ds, pfs_0, pp_cRL, plot_title=r'BGOSss')
    # # Make the figure
    ahf.make_figure([group_BGOSss_pre_clstrd, group_BGOSss_nir_SA, group_BGOSss_cRL])
#

## Comparing two BGR ITP time period clusterings
# BGR ITP clustering
if False:
    print('')
    print('- Creating plots to compare pre-clustered BGR ITP data')
    # this_ds = ds_BGRa_m110
    # this_ds = ds_BGRf_m410
    # this_ds = ds_BGRg_m240
    # this_ds = ds_BGRfg
    # this_ds = ds_BGRgh
    this_ds = ds_BGRfgh
    # Make the Plot Parameters
    pp_comp_clstrs = ahf.Plot_Parameters(x_vars=['ca_SA'], y_vars=['ca_CT'], clr_map='clr_by_dataset', extra_args={'extra_vars_to_keep':['cluster']}) 
    # pp_comp_clstrs = ahf.Plot_Parameters(x_vars=['ca_SA'], y_vars=['ca_CT'], clr_map='cluster')# clr_map='clr_by_dataset', extra_args={'extra_vars_to_keep':['cluster']}) 
    # Make the subplot groups
    group_comp_clstrs = ahf.Analysis_Group(this_ds, pfs_0, pp_comp_clstrs)
    # print('done making analysis group')
    # # Make the figure
    ahf.make_figure([group_comp_clstrs])
# BGR ITP clustering, comparing with histograms
if False:
    print('')
    print('- Creating plots to compare pre-clustered BGR ITP data')
    # this_ds = ds_BGRa_m110
    # this_ds = ds_BGRf_m410
    # this_ds = ds_BGRg_m240
    # this_ds = ds_BGRfg
    # Make the Plot Parameters
    pp_comp_clstrs = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['hist'], clr_map='cluster', extra_args={'plt_noise':False}, ax_lims=x_ax_lims_SA, legend=False) 
    # Make the subplot groups
    group_comp_clstrs0 = ahf.Analysis_Group(ds_BGRf_m410, pfs_0, pp_comp_clstrs)
    group_comp_clstrs1 = ahf.Analysis_Group(ds_BGRg_m240, pfs_0, pp_comp_clstrs)
    group_comp_clstrs2 = ahf.Analysis_Group(ds_BGRh_m390, pfs_0, pp_comp_clstrs)
    # print('done making analysis group')
    # # Make the figure
    # ahf.make_figure([group_comp_clstrs0, group_comp_clstrs1], row_col_list=[2,1, 0.8, 1.25])
    ahf.make_figure([group_comp_clstrs0, group_comp_clstrs1, group_comp_clstrs2], row_col_list=[3,1, 0.4, 2.0])


exit(0)





### Figure 2
## Plotting a TS diagram of both ITP and AIDJEX data
# TS diagram of all sources, colored by source
# pp_TS_clr_all_same = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='clr_all_same')
# pp_TS_by_source = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='clr_by_source')
# pp_TS_by_instrmt = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='clr_by_instrmt')
# 
# pp_LA_TS = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='clr_all_same')
# 
# pp_CT_max = ahf.Plot_Parameters(x_vars=['CT_max'], y_vars=['press_CT_max'], clr_map='prof_no')
# pp_SA_CT_max = ahf.Plot_Parameters(x_vars=['SA_CT_max'], y_vars=['press_CT_max'], clr_map='CT_max')
# pp_SA_CT_max_hist = ahf.Plot_Parameters(x_vars=['SA_CT_max'], y_vars=['hist'], clr_map='clr_all_same')
# 
# test_mpts = 65
# pp_ITP2_clstr = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'b_a_w_plt':True})
# pp_ITP2_clstr_og = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':test_mpts, 'b_a_w_plt':True})
# 
# BGOS_mpts = 120
# pp_BGOS_clstrd = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':BGOS_mpts, 'b_a_w_plt':True})
# pp_BGOS_clstr_TS = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['la_CT'], clr_map='cluster', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':BGOS_mpts, 'b_a_w_plt':True})
# 
# ### Figure 3
# ## Parameter sweep across \ell and m_pts for ITP2 subsampled
# ITP2_ss_mpts = 65
# pp_ITP2_ss_mpts = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':ITP2_ss_mpts, 'cl_ps_tuple':[10,220,5]})
# pp_ITP2_ss_ell  = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':ITP2_ss_mpts, 'cl_ps_tuple':[10,271,10]})
# ## Parameter sweep across \ell and m_pts for BGOS
# BGOS_mpts = 170
# pp_BGOS_mpts = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':BGOS_mpts, 'cl_ps_tuple':[10,440,10]})
# pp_BGOS_ell  = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':BGOS_mpts, 'cl_ps_tuple':[10,271,10]})
# ## Parameter sweep across \ell and m_pts for AIDJEX
# AIDJEX_mpts = 500
# pp_AIDJEX_mpts = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':AIDJEX_mpts, 'cl_ps_tuple':[300,700,20]})
# pp_AIDJEX_ell  = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':AIDJEX_mpts, 'cl_ps_tuple':[10,210,10]})
# 
# ## Parameter sweep across \ell and m_pts for ITP3, Lu et al. 2022
# pp_ITP3_ps_m_pts = ahf.Plot_Parameters(x_vars=['m_pts'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':Lu2022_m_pts, 'cl_ps_tuple':[800,1001,10]}, legend=False, add_grid=False)
# pp_ITP3_ps_ell  = ahf.Plot_Parameters(x_vars=['ell_size'], y_vars=['n_clusters','DBCV'], clr_map='clr_all_same', extra_args={'cl_x_var':'SA', 'cl_y_var':'la_CT', 'm_pts':Lu2022_m_pts, 'cl_ps_tuple':[10,210,10]}, legend=False, add_grid=False)
# 
# ### Figure 4
# ## Evaluating clusterings with the lateral density ratio and the normalized inter-cluster range
# pp_salt_R_L = ahf.Plot_Parameters(x_vars=['cRL'], y_vars=['ca_press'], clr_map='cluster', extra_args={'b_a_w_plt':False, 'plt_noise':False, 'plot_slopes':True}, legend=False)
# pp_salt_nir = ahf.Plot_Parameters(x_vars=['nir_SA'], y_vars=['ca_press'], clr_map='cluster', extra_args={'b_a_w_plt':False, 'plt_noise':False}, legend=False)
# 
# ### Figure 5
# ## Tracking clusters across a subset of profiles
# # For ITP2
# pp_ITP2_some_pfs_0  = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['press'], plot_type='profiles', clr_map='cluster', extra_args={'pfs_to_plot':ITP2_some_pfs_0, 'plt_noise':True}, legend=False, ax_lims=ITP2_some_pfs_ax_lims_0)
# pp_ITP2_some_pfs_1  = ahf.Plot_Parameters(x_vars=['SA'], y_vars=['press'], plot_type='profiles', clr_map='cluster', extra_args={'pfs_to_plot':ITP2_some_pfs_1, 'plt_noise':True}, legend=False, ax_lims=ITP2_some_pfs_ax_lims_1)
# 
# ### Figure 6
# ## Tracking clusters across profiles, reproducing Lu et al. 2022 Figure 3
# pp_Lu2022_fig3a = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['pca_press'], clr_map='cluster', extra_args={'b_a_w_plt':False}, legend=False)
# pp_Lu2022_fig3b = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['pca_CT'], clr_map='cluster', extra_args={'b_a_w_plt':False}, legend=False)
# pp_Lu2022_fig3c = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['pca_SA'], clr_map='cluster', extra_args={'b_a_w_plt':False}, legend=False)
# 
# ### Figure 8
# ## Tracking clusters across a subset of profiles
# # For ITP35
# pp_ITP35_some_pfs = ahf.Plot_Parameters(x_vars=['CT'], y_vars=['press'], plot_type='profiles')

################################################################################
# Create analysis group objects
print('- Creating analysis group objects')
################################################################################

## Test Analysis Groups
# my_group0 = ahf.Analysis_Group(ds_ITP2_all, pfs_1, pp_clstr_test)
# my_group1 = ahf.Analysis_Group(ds_AIDJEX_BigBear_all, pfs_1, pp_clstr_test)
# ahf.make_figure([my_group0, my_group1])#, filename='test.pickle')
## Test figures

################################################################################
### Figures for paper

### Figure 1
## Map of drifts for all sources
if False:
    print('')
    print('- Creating Figure 1')
    # Make the subplot groups pfs_AOA
    group_map_full_Arctic = ahf.Analysis_Group(ds_all_sources, pfs_AOA, pp_map_full_Arctic, plot_title='')
    group_map = ahf.Analysis_Group(ds_all_sources, pfs_AOA, pp_map, plot_title='')
    # Make the figure
    ahf.make_figure([group_map_full_Arctic, group_map], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_1.pickle')
    # Find the maximum distance between any two profiles for each data set in the group
    # ahf.find_max_distance([group_map])
## Map of drifts for all AIDJEX stations
if False:
    print('')
    print('- Creating Figure 1')
    # Make the subplot groups
    group_map = ahf.Analysis_Group(ds_all_AIDJEX, pfs_0, pp_map, plot_title='')
    group_map_full_Arctic = ahf.Analysis_Group(ds_all_AIDJEX, pfs_0, pp_map_full_Arctic, plot_title='')
    # Make the figure
    ahf.make_figure([group_map], use_same_x_axis=False, use_same_y_axis=False, filename='Figure_1.pickle')
    # Find the maximum distance between any two profiles for each data set in the group
    # ahf.find_max_distance([group_map])
## Map of drifts for selected ITPs
if False:
    print('')
    print('- Creating Figure 1')
    # Make the subplot groups
    group_map_full_Arctic = ahf.Analysis_Group(ds_all_BGOS, pfs_AOA, pp_map_full_Arctic, plot_title='')
    group_map = ahf.Analysis_Group(ds_all_BGOS, pfs_AOA, pp_map, plot_title='')
    # Make the figure
    ahf.make_figure([group_map_full_Arctic, group_map], use_same_x_axis=False, use_same_y_axis=False)
## Map of drifts colored by date
if False:
    print('')
    print('- Creating Figure 1')
    # Make the subplot groups
    group_AIDJEX_map = ahf.Analysis_Group(ds_all_AIDJEX, pfs_0, pp_map_by_date, plot_title='')
    group_ITPs_map = ahf.Analysis_Group(ds_all_BGOS, pfs_AOA, pp_map_by_date, plot_title='')
    # Make the figure
    ahf.make_figure([group_AIDJEX_map, group_ITPs_map], use_same_x_axis=False, use_same_y_axis=False)

### Figure 2
## Plotting a TS diagram of both ITP and AIDJEX data
if False:
    print('')
    print('- Creating Figure 2')
    # Make the subplot groups
    # group_TS_all_sources = ahf.Analysis_Group(ds_all_sources, pfs_AIDJEX, pp_TS_by_source, plot_title='')
    group_TS_all_BGOS = ahf.Analysis_Group(ds_all_BGOS, pfs_0, pp_TS_by_instrmt, plot_title='BGOS ITPs')
    group_TS_all_AIDJEX  = ahf.Analysis_Group(ds_all_AIDJEX, pfs_0, pp_TS_by_instrmt, plot_title='AIDJEX')
    # Make the figure
    ahf.make_figure([group_TS_all_AIDJEX, group_TS_all_BGOS])#, filename='Figure_2.png')
    # ahf.make_figure([group_TS_all_sources, group_TS_all_AIDJEX], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_2.png')
## Plotting a LA_TS diagram of both ITP and AIDJEX data
if False:
    print('')
    print('- Creating Figure 2')
    # Make the subplot groups
    group_TS_all_AIDJEX = ahf.Analysis_Group(ds_all_AIDJEX, pfs_AIDJEX, pp_LA_TS, plot_title='AIDJEX')
    # group_TS_all_BGOS = ahf.Analysis_Group(ds_all_BGOS, pfs_BGOS, pp_TS_by_instrmt, plot_title='')
    # # Make the figure
    ahf.make_figure([group_TS_all_AIDJEX])#, filename='Figure_2.png')
    # ahf.make_figure([group_TS_all_sources, group_TS_all_AIDJEX], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_2.png')
    # group_TS_all_BGOS = ahf.Analysis_Group(ds_all_BGOS, pfs_BGOS, pp_LA_TS, plot_title='BGOS')
    # Make the figure
    # ahf.make_figure([group_TS_all_BGOS])
## Plotting clusters of both ITP and AIDJEX data
if False:
    print('')
    print('- Creating Figure 2')
    # Make the subplot groups
    # group_TS_all_AIDJEX = ahf.Analysis_Group(ds_all_AIDJEX, pfs_ITP2, pp_TS_by_instrmt, plot_title='')
    # group_TS_all_BGOS = ahf.Analysis_Group(ds_all_BGOS, pfs_BGOS, pp_TS_by_instrmt, plot_title='')
    # # Make the figure
    # ahf.make_figure([group_TS_all_AIDJEX, group_TS_all_BGOS])#, filename='Figure_2.png')
    # ahf.make_figure([group_TS_all_sources, group_TS_all_AIDJEX], use_same_x_axis=False, use_same_y_axis=False)#, filename='Figure_2.png')
    group_BGOS_clstrd = ahf.Analysis_Group(ds_all_BGOS, pfs_BGOS, pp_BGOS_clstrd, plot_title='')
    group_BGOS_clstr_TS = ahf.Analysis_Group(ds_all_BGOS, pfs_BGOS, pp_BGOS_clstr_TS, plot_title='')
    # Make the figure
    ahf.make_figure([group_BGOS_clstrd, group_BGOS_clstr_TS])
## Plotting CT_max data with min_press_CT_max limit
if False:
    print('')
    print('- Creating Figure 2')
    # # Make the subplot groups
    group_ITP35_CT_max = ahf.Analysis_Group(ds_ITP35_dfs1, pfs_AOA, pp_SA_CT_max)
    group_ITP41_CT_max = ahf.Analysis_Group(ds_ITP41_dfs1, pfs_AOA, pp_SA_CT_max)
    group_ITP42_CT_max = ahf.Analysis_Group(ds_ITP42_dfs1, pfs_AOA, pp_SA_CT_max)
    group_ITP43_CT_max = ahf.Analysis_Group(ds_ITP43_dfs1, pfs_AOA, pp_SA_CT_max)
    # # Make the figure
    ahf.make_figure([group_ITP35_CT_max, group_ITP41_CT_max, group_ITP42_CT_max, group_ITP43_CT_max])
    # Make the subplot groups
    # group_BB_CT_max = ahf.Analysis_Group(ds_AIDJEX_BigBear_all, pfs_0, pp_SA_CT_max)
    # group_BF_CT_max = ahf.Analysis_Group(ds_AIDJEX_BlueFox_all, pfs_0, pp_SA_CT_max)
    # group_Ca_CT_max = ahf.Analysis_Group(ds_AIDJEX_Caribou_all, pfs_0, pp_SA_CT_max)
    # group_Sn_CT_max = ahf.Analysis_Group(ds_AIDJEX_Snowbird_all, pfs_0, pp_SA_CT_max)
    # Make the figure
    # ahf.make_figure([group_BB_CT_max, group_BF_CT_max, group_Ca_CT_max, group_Sn_CT_max])

### Figure 3
## Parameter sweep across \ell and m_pts
# For ITP2 ss
if False:
    print('')
    print('- Creating Figure 3')
    # Make the subplot groups
    group_ITP2_ss_mpts = ahf.Analysis_Group(ds_ITP2_all, pfs_subs, pp_ITP2_ss_mpts, plot_title='ITP2 Subsampled')
    group_ITP2_ss_ell  = ahf.Analysis_Group(ds_ITP2_all, pfs_subs, pp_ITP2_ss_ell, plot_title='ITP2 Subsampled')
    # Make the figure
    ahf.make_figure([group_ITP2_ss_mpts, group_ITP2_ss_ell], filename='Figure_3.pickle')
# For BGOS
if False:
    print('')
    print('- Creating Figure 3')
    # Make the subplot groups
    group_BGOS_mpts = ahf.Analysis_Group(ds_all_BGOS, pfs_BGOS, pp_BGOS_mpts, plot_title='BGOS')
    group_BGOS_ell  = ahf.Analysis_Group(ds_all_BGOS, pfs_BGOS, pp_BGOS_ell, plot_title='BGOS')
    # Make the figure
    ahf.make_figure([group_BGOS_mpts, group_BGOS_ell], filename='Figure_3.pickle')
# For AIDJEX
if False:
    print('')
    print('- Creating Figure 3')
    # Make the subplot groups
    group_AIDJEX_mpts = ahf.Analysis_Group(ds_all_AIDJEX, pfs_ITP2, pp_AIDJEX_mpts, plot_title='AIDJEX')
    group_AIDJEX_ell  = ahf.Analysis_Group(ds_all_AIDJEX, pfs_ITP2, pp_AIDJEX_ell, plot_title='AIDJEX')
    # Make the figure
    ahf.make_figure([group_AIDJEX_mpts, group_AIDJEX_ell], filename='Figure_3.pickle')

### Figure 3-1
## Plotting test clusterings
if False:
    print('')
    print('- Creating Figure 3')
    # Make the subplot groups
    # group_ITP2_og = ahf.Analysis_Group(ds_ITP2_all, pfs_ITP2, pp_ITP2_clstr, plot_title='Original')
    group_ITP2_ss = ahf.Analysis_Group(ds_ITP2_all, pfs_subs, pp_ITP2_clstr, plot_title='Subsampled')
    group_ITP2_ss2 = ahf.Analysis_Group(ds_ITP2_all, pfs_subs, pp_ITP2_clstr_og, plot_title='Subsampled')
    # group_ITP2_rg = ahf.Analysis_Group(ds_ITP2_all, pfs_regrid, pp_ITP2_clstr, plot_title='Regridded')
    # group_ITP2_ss_rg = ahf.Analysis_Group(ds_ITP2_all, pfs_ss_rg, pp_ITP2_clstr, plot_title='Subsampled & Regridded')
    # Make the figure
    # ahf.make_figure([group_ITP2_og, group_ITP2_ss, group_ITP2_rg])#, group_ITP2_ss_rg])#, filename='Figure_3.png')
    ahf.make_figure([group_ITP2_ss, group_ITP2_ss2])#, group_ITP2_ss_rg])#, filename='Figure_3.png')

### Figure 4
## Evaluating clusterings with the overlap ratio and lateral density ratio
# For the reproduction of Timmermans et al. 2008
if False:
    print('')
    print('- Creating Figure 4')
    # Make the subplot groups
    group_salt_R_L = ahf.Analysis_Group(ds_ITP2_all, pfs_ITP2, pp_salt_R_L, plot_title='')
    group_salt_nir = ahf.Analysis_Group(ds_ITP2_all, pfs_ITP2, pp_salt_nir, plot_title='')
    # Make the figure
    ahf.make_figure([group_salt_R_L, group_salt_nir], filename='Figure_4.pickle')
# For the reproduction of Lu et al. 2022
if False:
    print('')
    print('- Creating Figure 4, for ITP3')
    group_salt_R_L = ahf.Analysis_Group(ds_ITP3_all, pfs_ITP3, pp_salt_R_L)
    group_salt_nir = ahf.Analysis_Group(ds_ITP3_all, pfs_ITP3, pp_salt_nir)
    # Make the figure
    #   Confirmed
    ahf.make_figure([group_salt_R_L, group_salt_nir])

### Figure 5
## Tracking clusters across a subset of profiles
# For ITP2
if False:
    print('')
    print('- Creating Figure 5')
    # Make the subplot groups
    group_ITP2_some_pfs_0 = ahf.Analysis_Group(ds_ITP2_all, pfs_ITP2, pp_ITP2_some_pfs_0, plot_title='')
    group_ITP2_some_pfs_1 = ahf.Analysis_Group(ds_ITP2_all, pfs_ITP2, pp_ITP2_some_pfs_1, plot_title='')
    # Make the figure
    ahf.make_figure([group_ITP2_some_pfs_0, group_ITP2_some_pfs_1], use_same_x_axis=False, use_same_y_axis=False, row_col_list=[2,1, 0.3, 1.70], filename='Figure_5.pickle')

### Figure 6
## Tracking clusters across profiles, reproducing Lu et al. 2022 Figure 3
if False:
    print('')
    print('- Creating Figure 6')
    # Make the subplot groups
    group_Lu2022_fig3a = ahf.Analysis_Group(ds_ITP3_all, pfs_ITP3, pp_Lu2022_fig3a, plot_title='')
    group_Lu2022_fig3b = ahf.Analysis_Group(ds_ITP3_all, pfs_ITP3, pp_Lu2022_fig3b, plot_title='')
    group_Lu2022_fig3c = ahf.Analysis_Group(ds_ITP3_all, pfs_ITP3, pp_Lu2022_fig3c, plot_title='')
    # Make the figure
    #   Confirmed that it will pull from netcdf. Takes a long time to run still
    ahf.make_figure([group_Lu2022_fig3a, group_Lu2022_fig3b, group_Lu2022_fig3c], filename='Figure_6.pickle')
# For ITP2
if False:
    print('')
    print('- Creating Figure 6, for ITP2')
    # Make the subplot groups
    group_Lu2022_fig3a = ahf.Analysis_Group(ds_ITP2_all, pfs_ITP2, pp_Lu2022_fig3a)
    group_Lu2022_fig3b = ahf.Analysis_Group(ds_ITP2_all, pfs_ITP2, pp_Lu2022_fig3b, plot_title='')
    group_Lu2022_fig3c = ahf.Analysis_Group(ds_ITP2_all, pfs_ITP2, pp_Lu2022_fig3c, plot_title='')
    # Make the figure
    ahf.make_figure([group_Lu2022_fig3a, group_Lu2022_fig3b, group_Lu2022_fig3c])

### Figure 8
## Tracking clusters across a subset of profiles
# For ITP3
if False:
    print('')
    print('- Creating Figure 8')
    # Make the subplot group
    group_BGOS_some_pfs0 = ahf.Analysis_Group(ds_ITP35_some_pfs0, pfs_AOA, pp_ITP35_some_pfs)
    group_BGOS_some_pfs1 = ahf.Analysis_Group(ds_ITP35_some_pfs1, pfs_AOA, pp_ITP35_some_pfs, plot_title='')
    group_BGOS_some_pfs2 = ahf.Analysis_Group(ds_ITP35_some_pfs2, pfs_AOA, pp_ITP35_some_pfs, plot_title='')
    # Make the figure
    ahf.make_figure([group_BGOS_some_pfs0, group_BGOS_some_pfs1, group_BGOS_some_pfs2], use_same_x_axis=False, use_same_y_axis=False, row_col_list=[3,1, 0.3, 1.70])
    # ahf.make_figure([group_ITP3_some_pfs_2], row_col_list=[1,1, 0.27, 0.90], filename='Figure_8.pickle')

################################################################################