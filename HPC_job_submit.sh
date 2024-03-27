#!/bin/bash
# Author: Mikhail Schee
# 2023-10-03

# Run this script to submit a job to Niagara. 
# Takes in optional arguments:
#	$ bash HPC_job_submit.sh -j <job name> 			Default: current datetime
#							 -c <cluster data>		Default: don't cluster data, run a parameter sweep

# Current datetime
DATETIME=`date +"%Y-%m-%d_%Hh%M"`

# Having a ":" after a flag means an option is required to invoke that flag
while getopts j:c: option
do
	case "${option}"
		in
		j) JOBNAME=${OPTARG};;
		c) CLS=${OPTARG};;
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
if [ -z "$CLS" ]
then
	CLS=false
else
	echo "-c, Clustering data instead of parameter sweep with m_pts=$CLS"
	FULLJOBNAME="${JOBNAME}_${CLS}"
	echo "Using FULLJOBNAME=$FULLJOBNAME"
fi

###############################################################################
# Name of the lanceur file
if [ "$CLS" = false ]
then
	LANCEUR="HPC_lanceur.slrm"
else
	LANCEUR="HPC_launcher.slrm"
fi

# Pull the most recent changes from git
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_GH_23
git fetch --all
git reset --hard origin
git pull

###############################################################################
# Submit job to queue
sbatch --job-name=$FULLJOBNAME $LANCEUR -j $FULLJOBNAME -b $JOBNAME -c $CLS