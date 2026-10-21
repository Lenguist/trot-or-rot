#!/bin/bash

# Script to run variable_friction_hillclimber_learner.py with different friction parameters

# Array of friction parameters
friction_params=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Loop through each friction parameter and execute the Python script
for friction in "${friction_params[@]}"; do
    echo "Running simulation with floor_friction=${friction}"
    # Use echo and pipe to skip note entry
    (echo "" && sleep 1) | python3 variable_friction_hillclimber_learner.py --floor_friction "$friction"
done
