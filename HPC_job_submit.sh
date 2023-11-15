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
while getopts j:c option
do
	case "${option}"
		in
		j) JOBNAME=${OPTARG};;
	c) CLS=c;;
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
if [ "$CLS" = c ]
then
	CLS=true
	echo "-c, Clustering data instead of parameter sweep"
else
	CLS=false
fi

###############################################################################
# Name of the lanceur file
if [ "$CLS" = true ]
then
	LANCEUR="HPC_launcher.slrm"
else
	LANCEUR="HPC_lanceur.slrm"
fi

# Pull the most recent changes from git
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_GH_23
git fetch --all
git reset --hard origin
git pull

###############################################################################
# Submit job to queue
sbatch --job-name=$JOBNAME $LANCEUR -j $JOBNAME