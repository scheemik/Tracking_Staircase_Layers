#!/bin/bash
# Author: Mikhail Schee


# Run this script to use secure copy to transfer netcdf files to Niagara
# Takes in optional arguments:
#	$ bash HPC_scp_to_Niagara.sh -f <file name> 				 

# Having a ":" after a flag means an option is required to invoke that flag
while getopts f: option
do
	case "${option}"
		in
		f) FILENAME=${OPTARG};;
	esac
done

# check to see if arguments were passed
if [ -z "$FILENAME" ]
then
	echo "-f, No file specified, aborting script"
    exit 0
else
  echo "-f, Transfering FILENAME=$FILENAME"
fi

# Add the public key to allow file transfer
ssh-add ~/.ssh/DA_23_id_rsa

scp ${FILENAME} mschee@niagara.scinet.utoronto.ca:/scratch/n/ngrisoua/mschee/Clustering/Staircase_Clustering_Detection_Algorithm_v2/netcdfs/
