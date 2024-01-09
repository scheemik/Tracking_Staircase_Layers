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

# Data filters
dfs0 = ahf.Data_Filters()
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

# Beaufort Gyre Region (BGR), see Shibley2022
lon_BGR = [-160,-130]
lat_BGR = [73,81.5]
LHW_S_range = [34.366, 35.5]
test_S_range = [34.4, 34.6]
pfs_BGR1 = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR, p_range=[1000,5], SA_range=LHW_S_range, lt_pCT_max=True)
pfs_test = ahf.Profile_Filters(lon_range=lon_BGR,lat_range=lat_BGR, p_range=[1000,5], SA_range=test_S_range, lt_pCT_max=True)
pfs_0 = ahf.Profile_Filters()
pfs_these_clstrs = ahf.Profile_Filters(clstrs_to_plot=[4,5,6])

################################################################################
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

################################################################################
### By year
################################################################################
## Clustered
test_cls = 1000
# BGR ITPs 04
BGR04_clstr_dict = {'netcdf_file':'netcdfs/BGR04_clstrd.nc',
                   'sources_dict':{'ITP_002':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                #    'm_pts':'None',
                #    'm_pts':190,
                #    'm_pts':250,
                   }
# BGR ITPs 0506
BGR0506_clstr_dict = {'netcdf_file':'netcdfs/BGR0506_clstrd.nc',
                   'sources_dict':{'ITP_001':'all','ITP_003':'all'},
                   'data_filters':dfs1_BGR0506,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':680,
                #    'm_pts':'auto',
                   'm_cls':'auto',
                #    'm_pts':200,
                #    'm_cls':1000,
                   }
# BGR ITPs 0607
BGR0607_clstr_dict = {'netcdf_file':'netcdfs/BGR0607_clstrd.nc',
                   'sources_dict':{'ITP_001':'all','ITP_003':'all','ITP_004':'all','ITP_005':'all','ITP_006':'all','ITP_008':'all','ITP_013':'all'},
                   'data_filters':dfs1_BGR0607,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':220,
                #    'm_pts':'auto',
                   'm_cls':'auto',
                #    'm_pts':200,
                #    'm_cls':1000,
                   }
# BGR ITPs 0708
BGR0708_clstr_dict = {'netcdf_file':'netcdfs/BGR0708_clstrd.nc',
                   'sources_dict':{'ITP_004':'all','ITP_005':'all','ITP_006':'all','ITP_008':'all','ITP_013':'all','ITP_018':'all','ITP_021':'all','ITP_030':'all'},
                   'data_filters':dfs1_BGR0708,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':550,
                #    'm_pts':'auto',
                   'm_cls':'auto',
                #    'm_pts':200,
                #    'm_cls':1000,
                   }
# BGR ITPs 0809
BGR0809_clstr_dict = {'netcdf_file':'netcdfs/BGR0809_clstrd.nc',
                   'sources_dict':{'ITP_008':'all','ITP_011':'all','ITP_013':'all','ITP_018':'all','ITP_021':'all','ITP_025':'all','ITP_030':'all'},
                   'data_filters':dfs1_BGR0809,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':760,
                #    'm_pts':'auto',
                   'm_cls':'auto',
                #    'm_pts':200,
                #    'm_cls':1000,
                   }
# BGR ITPs 0910
BGR0910_clstr_dict = {'netcdf_file':'netcdfs/BGR0910_clstrd.nc',
                   'sources_dict':{'ITP_021':'all','ITP_032':'all','ITP_033':'all','ITP_034':'all','ITP_035':'all'},
                   'data_filters':dfs1_BGR0910,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':140,
                #    'm_pts':'auto',
                   'm_cls':'auto',
                #    'm_pts':200,
                #    'm_cls':1000,
                   }
# BGR ITPs 1011
BGR1011_clstr_dict = {'netcdf_file':'netcdfs/BGR1011_clstrd.nc',
                   'sources_dict':{'ITP_033':'all','ITP_041':'all','ITP_042':'all','ITP_043':'all'},
                   'data_filters':dfs1_BGR1011,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':100,
                #    'm_pts':'auto',
                   'm_cls':'auto',
                #    'm_pts':200,
                #    'm_cls':1000,
                   }

################################################################################
## Clustered, auto mpts selection
# BGR ITPs 04
BGR04_clstr_dict = {'netcdf_file':'netcdfs/BGR04_clstrd.nc',
                   'sources_dict':{'ITP_002':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                #    'm_pts':'None',
                #    'm_pts':190,
                #    'm_pts':250,
                   }
# BGR ITPs 0506
BGR0506_clstr_dict = {'netcdf_file':'netcdfs/mpts_auto_BGR0506_clstrd.nc',
                   'sources_dict':{'ITP_001':'all','ITP_003':'all'},
                   'data_filters':dfs1_BGR0506,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                   'm_cls':'auto',
                   'relab_these':{2:1},
                   }
# BGR ITPs 0607
BGR0607_clstr_dict = {'netcdf_file':'netcdfs/mpts_auto_BGR0607_clstrd.nc',
                   'sources_dict':{'ITP_001':'all','ITP_003':'all','ITP_004':'all','ITP_005':'all','ITP_006':'all','ITP_008':'all','ITP_013':'all'},
                   'data_filters':dfs1_BGR0607,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                   'm_cls':'auto',
                   'relab_these':{2:3,3:4,4:5,5:6,6:7,7:8,8:9,9:10,10:11,11:12,12:13,13:14},
                   }
# BGR ITPs 0708
BGR0708_clstr_dict = {'netcdf_file':'netcdfs/mpts_auto_BGR0708_clstrd.nc',
                   'sources_dict':{'ITP_004':'all','ITP_005':'all','ITP_006':'all','ITP_008':'all','ITP_013':'all','ITP_018':'all','ITP_021':'all','ITP_030':'all'},
                   'data_filters':dfs1_BGR0708,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                   'm_cls':'auto',
                   'relab_these':{},
                   }
# BGR ITPs 0809
BGR0809_clstr_dict = {'netcdf_file':'netcdfs/mpts_auto_BGR0809_clstrd.nc',
                   'sources_dict':{'ITP_008':'all','ITP_011':'all','ITP_013':'all','ITP_018':'all','ITP_021':'all','ITP_025':'all','ITP_030':'all'},
                   'data_filters':dfs1_BGR0809,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                   'm_cls':'auto',
                   'relab_these':{3:4,4:5,5:6,6:7,7:8,8:9,9:10,10:11,11:12,12:13,13:14},
                   }
# BGR ITPs 0910
BGR0910_clstr_dict = {'netcdf_file':'netcdfs/mpts_auto_BGR0910_clstrd.nc',
                   'sources_dict':{'ITP_021':'all','ITP_032':'all','ITP_033':'all','ITP_034':'all','ITP_035':'all'},
                   'data_filters':dfs1_BGR0910,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                   'm_cls':'auto',
                   'relab_these':{3:4,4:5,5:6,6:7,7:8,8:9,9:15,12:11,13:12,14:13,15:14},
                   }
# BGR ITPs 1011
BGR1011_clstr_dict = {'netcdf_file':'netcdfs/mpts_auto_BGR1011_clstrd.nc',
                   'sources_dict':{'ITP_033':'all','ITP_041':'all','ITP_042':'all','ITP_043':'all'},
                   'data_filters':dfs1_BGR1011,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                   'm_cls':'auto',
                   'relab_these':{3:4,10:15,11:10,12:11,13:11,14:12,15:13,16:14},
                   }
# BGR ITPs 1112
BGR1112_clstr_dict = {'netcdf_file':'netcdfs/mpts_auto_BGR1112_clstrd.nc',
                   'sources_dict':{'ITP_041':'all','ITP_052':'all','ITP_053':'all','ITP_054':'all','ITP_055':'all'},
                   'data_filters':dfs1_BGR1112,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                   'm_cls':'auto',
                   'relab_these':{}#3:4,10:15,11:10,12:11,13:11,14:12,15:13,16:14},
                   }
# BGR ITPs 1213
BGR1213_clstr_dict = {'netcdf_file':'netcdfs/mpts_auto_BGR1213_clstrd.nc',
                   'sources_dict':{'ITP_041':'all','ITP_062':'all','ITP_064':'all','ITP_065':'all'},
                   'data_filters':dfs1_BGR1213,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                   'm_cls':'auto',
                   'relab_these':{}#3:4,10:15,11:10,12:11,13:11,14:12,15:13,16:14},
                   }
# BGR ITPs 1314
BGR1314_clstr_dict = {'netcdf_file':'netcdfs/mpts_auto_BGR1314_clstrd.nc',
                   'sources_dict':{'ITP_064':'all','ITP_068':'all','ITP_069':'all','ITP_070':'all','ITP_077':'all','ITP_078':'all','ITP_079':'all','ITP_080':'all'},
                   'data_filters':dfs1_BGR1314,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                   'm_cls':'auto',
                   'relab_these':{}#3:4,10:15,11:10,12:11,13:11,14:12,15:13,16:14},
                   }
# BGR ITPs 1415
BGR1415_clstr_dict = {'netcdf_file':'netcdfs/mpts_auto_BGR1415_clstrd.nc',
                   'sources_dict':{'ITP_077':'all','ITP_079':'all','ITP_080':'all','ITP_081':'all','ITP_082':'all','ITP_084':'all','ITP_085':'all','ITP_086':'all','ITP_087':'all'},
                   'data_filters':dfs1_BGR1415,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                   'm_cls':'auto',
                   'relab_these':{}#3:4,10:15,11:10,12:11,13:11,14:12,15:13,16:14},
                   }
# BGR ITPs 1516
BGR1516_clstr_dict = {'netcdf_file':'netcdfs/mpts_auto_BGR1516_clstrd.nc',
                   'sources_dict':{'ITP_082':'all','ITP_085':'all','ITP_086':'all','ITP_088':'all','ITP_089':'all'},
                   'data_filters':dfs1_BGR1516,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                   'm_cls':'auto',
                   'relab_these':{}#3:4,10:15,11:10,12:11,13:11,14:12,15:13,16:14},
                   }
# BGR ITPs 1617
BGR1617_clstr_dict = {'netcdf_file':'netcdfs/mpts_auto_BGR1617_clstrd.nc',
                   'sources_dict':{'ITP_097':'all','ITP_098':'all'},
                   'data_filters':dfs1_BGR1617,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                   'm_cls':'auto',
                   'relab_these':{}#3:4,10:15,11:10,12:11,13:11,14:12,15:13,16:14},
                   }
# BGR ITPs 1718
BGR1718_clstr_dict = {'netcdf_file':'netcdfs/mpts_auto_BGR1718_clstrd.nc',
                   'sources_dict':{'ITP_097':'all','ITP_100':'all','ITP_101':'all','ITP_108':'all'},
                   'data_filters':dfs1_BGR1718,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                   'm_cls':'auto',
                   'relab_these':{}#3:4,10:15,11:10,12:11,13:11,14:12,15:13,16:14},
                   }
# BGR ITPs 1819
BGR1819_clstr_dict = {'netcdf_file':'netcdfs/mpts_auto_BGR1819_clstrd.nc',
                   'sources_dict':{'ITP_103':'all','ITP_104':'all','ITP_105':'all','ITP_107':'all','ITP_109':'all','ITP_110':'all'},
                   'data_filters':dfs1_BGR1819,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                   'm_cls':'auto',
                   'relab_these':{}#3:4,10:15,11:10,12:11,13:11,14:12,15:13,16:14},
                   }
# BGR ITPs 1920
BGR1920_clstr_dict = {'netcdf_file':'netcdfs/mpts_auto_BGR1920_clstrd.nc',
                   'sources_dict':{'ITP_105':'all','ITP_113':'all','ITP_114':'all','ITP_117':'all','ITP_118':'all'},
                   'data_filters':dfs1_BGR1920,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                   'm_cls':'auto',
                   'relab_these':{}#3:4,10:15,11:10,12:11,13:11,14:12,15:13,16:14},
                   }
# BGR ITPs 2021
BGR2021_clstr_dict = {'netcdf_file':'netcdfs/mpts_auto_BGR2021_clstrd.nc',
                   'sources_dict':{'ITP_113':'all','ITP_114':'all','ITP_120':'all','ITP_121':'all'},
                   'data_filters':dfs1_BGR2021,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                   'm_cls':'auto',
                   'relab_these':{}#3:4,10:15,11:10,12:11,13:11,14:12,15:13,16:14},
                   }
# BGR ITPs 2122
BGR2122_clstr_dict = {'netcdf_file':'netcdfs/mpts_auto_BGR2122_clstrd.nc',
                   'sources_dict':{'ITP_120':'all','ITP_121':'all','ITP_122':'all','ITP_123':'all'},
                   'data_filters':dfs1_BGR2122,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                   'm_cls':'auto',
                   'relab_these':{}#3:4,10:15,11:10,12:11,13:11,14:12,15:13,16:14},
                   }
# BGR ITPs 2223
BGR2223_clstr_dict = {'netcdf_file':'netcdfs/mpts_auto_BGR2223_clstrd.nc',
                   'sources_dict':{'ITP_122':'all'},
                   'data_filters':dfs1_BGR2223,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'auto',
                   'm_cls':'auto',
                   'relab_these':{}#3:4,10:15,11:10,12:11,13:11,14:12,15:13,16:14},
                   }

################################################################################
## Paired down, minimal variables, just to run parameter sweeps on HPC
# BGR ITPs 04
BGR04_minimal   = {'netcdf_file':'netcdfs/minimal_BGR04.nc',
                   'sources_dict':{'ITP_002':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'None'
                   }
# BGR ITPs 0506
BGR0506_minimal = {'netcdf_file':'netcdfs/minimal_BGR0506.nc',
                   'sources_dict':{'ITP_001':'all','ITP_003':'all'},
                   'data_filters':dfs1_BGR0506,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'None'
                   }
# BGR ITPs 0607
BGR0607_minimal = {'netcdf_file':'netcdfs/minimal_BGR0607.nc',
                   'sources_dict':{'ITP_001':'all','ITP_003':'all','ITP_004':'all','ITP_005':'all','ITP_006':'all','ITP_008':'all','ITP_013':'all'},
                   'data_filters':dfs1_BGR0607,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'None'
                   }
# BGR ITPs 0708
BGR0708_minimal = {'netcdf_file':'netcdfs/minimal_BGR0708.nc',
                   'sources_dict':{'ITP_004':'all','ITP_005':'all','ITP_006':'all','ITP_008':'all','ITP_013':'all','ITP_018':'all','ITP_021':'all','ITP_030':'all'},
                   'data_filters':dfs1_BGR0708,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'None'
                   }
# BGR ITPs 0809
BGR0809_minimal = {'netcdf_file':'netcdfs/minimal_BGR0809.nc',
                   'sources_dict':{'ITP_008':'all','ITP_011':'all','ITP_013':'all','ITP_018':'all','ITP_021':'all','ITP_025':'all','ITP_030':'all'},
                   'data_filters':dfs1_BGR0809,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'None'
                   }
# BGR ITPs 0910
BGR0910_minimal = {'netcdf_file':'netcdfs/minimal_BGR0910.nc',
                   'sources_dict':{'ITP_021':'all','ITP_032':'all','ITP_033':'all','ITP_034':'all','ITP_035':'all'},
                   'data_filters':dfs1_BGR0910,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'None'
                   }
# BGR ITPs 1011
BGR1011_minimal = {'netcdf_file':'netcdfs/minimal_BGR1011.nc',
                   'sources_dict':{'ITP_033':'all','ITP_041':'all','ITP_042':'all','ITP_043':'all'},
                   'data_filters':dfs1_BGR1011,
                   'pfs_object':pfs_test,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'None'
                   }
################################################################################

### Just certain clusters
## BGR ITPs 04 cluster 5
BGR04_clstr_456 = {'netcdf_file':'netcdfs/BGR04_clstrs_456.nc',
                   'sources_dict':{'BGR04_clstrd':'all'},
                   'data_filters':dfs0,
                   'pfs_object':pfs_these_clstrs,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'second',
                   }
## BGR ITPs 0508
BGR05060708_clstrs_456 = {'netcdf_file':'netcdfs/BGR05060708_clstrs_456.nc',
                   'sources_dict':{'BGR0506_clstrd':'all','BGR0607_clstrd':'all','BGR0708_clstrd':'all'},
                   'data_filters':dfs1,
                   'pfs_object':pfs_these_clstrs,
                   'cl_x_var':'SA',
                   'cl_y_var':'la_CT',
                   'cl_z_var':'None',
                   'm_pts':'second',
                   }

xr.set_options(display_max_rows=44)

# for clstr_dict in [BGR05060708_clstrs_456]:
# for clstr_dict in [BGR0607_clstr_dict, BGR0708_clstr_dict]:
for clstr_dict in [BGR0506_clstr_dict, BGR0607_clstr_dict, BGR0708_clstr_dict, BGR0809_clstr_dict, BGR0910_clstr_dict, BGR1011_clstr_dict]:
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
        # print('Before:')
        # Copy down the global attributes to put into the new netcdf later
        #   Loop through each dataset (one for each instrument)
        for ds in ds_object.arr_of_ds:
            # Loop through the variables to get the information
            # print(ds.variables)
            # print(ds)
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
            keep_these_vars = ['source', 'instrmt', 'entry', 'prof_no', 'SA', 'CT', 'ma_CT']
            # keep_these_vars = ['entry', 'prof_no', 'BL_yn', 'dt_start', 'dt_end', 'lon', 'lat', 'region', 'up_cast', 'press_max', 'CT_max', 'press_CT_max', 'SA_CT_max', 'R_rho', 'press', 'depth', 'iT', 'CT', 'PT', 'SP', 'SA', 'sigma', 'alpha', 'beta', 'aiT', 'aCT', 'aPT', 'BSP', 'BSA', 'ss_mask', 'ma_iT', 'ma_CT', 'ma_PT', 'ma_SP', 'ma_SA', 'ma_sigma', 'la_iT', 'la_CT', 'la_PT', 'la_SP', 'la_SA', 'la_sigma']
            pp_clstr = ahf.Plot_Parameters(x_vars=[clstr_dict['cl_x_var']], y_vars=[clstr_dict['cl_y_var']], clr_map='instrmt', extra_args={'extra_vars_to_keep':keep_these_vars, 'relab_these':clstr_dict['relab_these']}, legend=True)
        elif clstr_dict['m_pts'] == 'second':
            keep_these_vars = ['entry', 'prof_no', 'BL_yn', 'dt_start', 'dt_end', 'lon', 'lat', 'region', 'up_cast', 'press_max', 'CT_max', 'press_CT_max', 'SA_CT_max', 'R_rho', 'press', 'depth', 'iT', 'CT', 'PT', 'SP', 'SA', 'sigma', 'alpha', 'beta', 'aCT', 'BSA', 'ss_mask', 'ma_iT', 'ma_CT', 'ma_PT', 'ma_SP', 'ma_SA', 'ma_sigma', 'la_iT', 'la_CT', 'la_PT', 'la_SP', 'la_SA', 'la_sigma', 'cluster', 'clst_prob']
            pp_clstr = ahf.Plot_Parameters(x_vars=[clstr_dict['cl_x_var']], y_vars=[clstr_dict['cl_y_var']], clr_map='instrmt', extra_args={'extra_vars_to_keep':keep_these_vars}, legend=True)
        else:
            # keep_these_vars = ['la_CT', 'entry', 'prof_no', 'CT', 'SA']
            keep_these_vars = ['entry', 'prof_no', 'BL_yn', 'dt_start', 'dt_end', 'lon', 'lat', 'region', 'up_cast', 'press_max', 'CT_max', 'press_CT_max', 'SA_CT_max', 'R_rho', 'press', 'depth', 'iT', 'CT', 'PT', 'SP', 'SA', 'sigma', 'alpha', 'beta', 'aCT', 'BSA', 'ss_mask', 'ma_iT', 'ma_CT', 'ma_PT', 'ma_SP', 'ma_SA', 'ma_sigma', 'la_iT', 'la_CT', 'la_PT', 'la_SP', 'la_SA', 'la_sigma']
            pp_clstr = ahf.Plot_Parameters(x_vars=[clstr_dict['cl_x_var']], y_vars=[clstr_dict['cl_y_var']], clr_map='cluster', extra_args={'b_a_w_plt':True, 'cl_x_var':clstr_dict['cl_x_var'], 'cl_y_var':clstr_dict['cl_y_var'], 'm_pts':clstr_dict['m_pts'], 'm_cls':clstr_dict['m_cls'], 'extra_vars_to_keep':keep_these_vars, 'relab_these':clstr_dict['relab_these']}, legend=True)
            # pp_clstr = ahf.Plot_Parameters(x_vars=['dt_start'], y_vars=['SA'], clr_map='cluster', extra_args={'b_a_w_plt':True, 'cl_x_var':clstr_dict['cl_x_var'], 'cl_y_var':clstr_dict['cl_y_var'], 'm_pts':clstr_dict['m_pts'], 'm_cls':clstr_dict['m_cls'], 'extra_vars_to_keep':keep_these_vars, 'relab_these':clstr_dict['relab_these']}, legend=True)
        # Create analysis group
        print('Creating analysis group')
        group_test_clstr = ahf.Analysis_Group(ds_object, pfs_object, pp_clstr)
        # Make a figure to run clustering algorithm and check results
        print('making figure')
        ahf.make_figure([group_test_clstr])#, filename='test.txt')

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
        new_df = pd.concat(dfs)
        if clstr_dict['m_pts'] == 'None':
            # Remove unnecessary variables
            print('variables in new_df:',list(new_df))
            only_need_these_vars = keep_these_vars #['entry','prof_no','CT','ma_CT','SA',]
            for var in list(new_df):
                if not var in only_need_these_vars:
                    new_df = new_df.drop(var, axis=1)
        # print('after, vars in new_df:',list(new_df))
        # print('Duplicated indices')
        # print(new_df.index[new_df.index.duplicated()].unique())
        # exit(0)
        # Convert back to xarray dataset
        ds = new_df.to_xarray()
        # print('Before:')
        # print(ds)
        # print('')
        # ds = ds.where(np.isnan(ds.entry), drop=True)
        pf_vars = ahf.pf_vars + ['instrmt', 'source', 'notes', 'SA_CT_max', 'press_max']
        # Get the values of the dimensions
        time_vals, vert_vals = ds.indexes.values()
        # Loop across the Time dimension
        for this_time in time_vals:
            for var in ds.variables:
                if var in pf_vars:
                    # print('Adjusting this var:',var)
                    # Get the array of values for this var for this Time
                    these_vals = ds[var].sel(Time=this_time)
                    if var in ['dt_start', 'dt_end', 'region', 'instrmt', 'source', 'notes']:
                        these_vals1 = [x for x in np.array(these_vals) if not isinstance(x, float)]
                    else:
                        these_vals1 = np.unique(these_vals[~np.isnan(these_vals)])
                    if len(these_vals1) > 0:
                        this_val = these_vals1[0]
                    else:
                        this_val = np.nan
                    # print('\tthis_val:',this_val)
                    # Copy this value to all `Vertical` entries for this variable
                    ds[var].loc[dict(Time=this_time)] = this_val
                #
            #
        # Drop the Vertical dimension for variables which vary in Time only
        for var in ds.variables:
            if var in pf_vars:
                # print('Dropping Vertical dimension for this var:',var)
                ds[var] = ds[var].isel(Vertical=0, drop=True)
        # print('After:')
        # print(ds)
        # exit(0)
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
        ds.attrs['Clustering m_cls'] = group_test_clstr.data_set.arr_of_ds[0].attrs['Clustering m_cls']
        ds.attrs['Clustering m_pts'] = group_test_clstr.data_set.arr_of_ds[0].attrs['Clustering m_pts']
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


