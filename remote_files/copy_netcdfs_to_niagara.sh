#!/bin/bash
# Author: Mikhail Schee


# Run this script to use secure copy to transfer all the files in netcdf to Niagara

# Add the public key to allow file transfer
ssh-add ~/.ssh/DA_23_id_rsa

# Where the netcdfs are located
NC_LOCATION="/Users/Grey/Documents/Research/PhD_Projects/Project_2-AIDJEX_staircases/Staircase_Clustering_Detection_Algorithm_v2/netcdfs/"

# List the files to transfer:
files_to_transfer=(
    # "ITP_001.nc"
    # "ITP_002.nc"
    # "ITP_003.nc"
    # "ITP_004.nc"
    # "ITP_005.nc"
    # "ITP_006.nc"
    # "ITP_008.nc"
    # "ITP_011.nc"
    # "ITP_013.nc"
    # "ITP_018.nc"
    # "ITP_021.nc"
    # "ITP_025.nc"
    # "ITP_030.nc"
    # "ITP_032.nc"
    # "ITP_033.nc"
    # "ITP_034.nc"
    # "ITP_035.nc"
    # "ITP_041.nc"
    # "ITP_042.nc"
    # "ITP_043.nc"
    # "ITP_052.nc"
    # "ITP_053.nc"
    # "ITP_054.nc"
    # "ITP_055.nc"
    # "ITP_062.nc"
    # "ITP_064.nc"
    # "ITP_065.nc"
    # "ITP_068.nc"
    # "ITP_069.nc"
    # "ITP_070.nc"
    # "ITP_077.nc"
    # "ITP_078.nc"
    # "ITP_079.nc"
    # "ITP_080.nc"
    # "ITP_081.nc"
    # "ITP_082.nc"
    # "ITP_084.nc"
    # "ITP_085.nc"
    # "ITP_086.nc"
    # "ITP_087.nc"
    # "ITP_088.nc"
    # "ITP_089.nc"
    # "ITP_097.nc"
    # "ITP_100.nc"
    # "ITP_101.nc"
    # "ITP_103.nc"
    # "ITP_104.nc"
    # "ITP_105.nc"
    # "ITP_107.nc"
    # "ITP_108.nc"
    # "ITP_109.nc"
    # "ITP_110.nc"
    # "ITP_113.nc"
    # "ITP_114.nc"
    # "ITP_117.nc"
    # "ITP_118.nc"
    # "ITP_120.nc"
    "ITP_121.nc"
    "ITP_122.nc"
    "ITP_123.nc"
    "ITP_125.nc"
    "ITP_128.nc"
)

file_list=""

for f in "${files_to_transfer[@]}"; do 
    file_list="${file_list}${NC_LOCATION}${f} "
done

# echo "$file_list"

scp ${file_list} mschee@niagara.scinet.utoronto.ca:/home/n/ngrisoua/mschee/Clustering/Staircase_Clustering_Detection_Algorithm_v2/netcdfs/