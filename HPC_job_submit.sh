#!/bin/bash
# Author: Mikhail Schee
# 2023-10-03

# Run this script to submit a job to Niagara. Example:
# $ sbatch HPC_job_submit.sh -j my_latest_run
# Takes in optional arguments:
#	$ sh HPC_job_submit.sh -j <job name> 				 Default: current datetime

# Current datetime
DATETIME=`date +"%Y-%m-%d_%Hh%M"`

# Having a ":" after a flag means an option is required to invoke that flag
while getopts j: option
do
	case "${option}"
		in
		j) JOBNAME=${OPTARG};;
	esac
done

# check to see if arguments were passed
if [ -z "$JOBNAME" ]
then
	JOBNAME=$DATETIME
	echo "-j, No name specified, using JOBNAME=$JOBNAME"
else
  echo "-j, Name specified, using JOBNAME=$JOBNAME"
fi

###############################################################################
# Name of the lanceur file
LANCEUR="HPC_lanceur.slrm"

# Pull the most recent changes from git
git pull

###############################################################################
# Submit job to queue
sbatch --job-name=$JOBNAME $LANCEUR -j $JOBNAME
