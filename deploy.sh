#!/bin/bash

set -x

# add app to python path                                                                                                                                     \
export PYTHONPATH=/home1/l/lingxiao/xiao/neural-chatbot

# $HOME/sample_applications/sample_sge_script_1.sh
# wallclock time reservation (format is hours:minutes:seconds).
# man 5 complex
#$ -l h_rt=100:15:0

# request x gigabyte of RAM 
# man 5 complex
#$ -l mem=20G

# name of job
# man 1 qsub
#$ -N deploy
# working directory (check for specific requirements for your research group)
# man 1 qsub

# make sure I set my $CWD (current working directory)
cd $HOME/xiao/neural-chatbot

# You must set the error and output locations, as the default log output
# location is the directory you qsub'ed from. For instance, say I cd to
# "/home/whaun/blargh" and qsub from there, not specifying "-e" or "-o." The
# output location for the error and output logs is going to default to
# "/home/whaun/blargh." To make it easier you may add -e and -o to your job
# script so you do not have to specify them every qsub call. Ex.:
#$ -o $PWD/logs
#$ -e $PWD/logs

# when am I running
#/bin/date

# where am I running
#/bin/hostname

# what environment variables are available to this job script, e.g. $JOB_ID
#/usr/bin/env
#echo $JOB_ID $SGE_STDOUT_PATH 


source /home1/l/lingxiao/xiao/tensorflow-1.0.1/bin/activate

# run my scripts
python $HOME/xiao/neural-chatbot/models/ts_hred/top.py

