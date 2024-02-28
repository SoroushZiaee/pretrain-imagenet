#!/bin/bash

# Request resources:
# -N: Number of nodes
# -n: Number of tasks
# -c: Number of CPUs per task
# --mem: Memory per node (in MB)
# -t: Time limit (hours:minutes:seconds)
# Replace the values according to your requirements

salloc -N 1 \
       -n 10 \
       -c 1 \
       --mem=32GB \
       -t 02:00:00 \
       --gres=gpu:a100:1 \
       --exclusive
