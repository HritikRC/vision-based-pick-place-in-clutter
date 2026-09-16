#!/bin/bash

# This is the path where Isaac Sim is installed which contains the python.sh script
ISAAC_SIM_PATH="/home/ubuntu/IsaacSim"

## Go to location of the SDG script
cd ../ur5e_sdg
SCRIPT_PATH="${PWD}/ur5e_sdg_script.py"
OUTPUT_DATA="${PWD}/training_data"


## Go to Isaac Sim location for running with ./python.sh
cd $ISAAC_SIM_PATH

echo "Starting Data Generation"  

./python.sh $SCRIPT_PATH --height 960 --width 1280 --num_frames 1000 --data_dir $OUTPUT_DATA


