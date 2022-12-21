#!/bin/bash
nvidia-smi
source /nfs-share/aai30/miniconda3/bin/activate  /nfs-share/aai30/miniconda3/envs/Opera

/nfs-share/aai30/miniconda3/envs/Opera/bin/python3.10 /home/aai30/nfs-share/projects/Opera_Fed/fed_main.py --config-path $1 strategy=$2
