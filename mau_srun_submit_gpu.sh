#!/bin/bash

srun -w mauao -u --job-name "Opera" -c $3 --gres=gpu:$2 $1 $4 $5

