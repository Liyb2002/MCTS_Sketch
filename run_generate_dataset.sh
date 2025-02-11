#!/bin/bash

# Loop until the Python script runs successfully
while true; do
    echo "Running generate_dataset.py..."
    
    # Run the Python script
    python generate_CAD_program.py
    
    # Check the exit status of the Python script
    if [ $? -eq 0 ]; then
        echo "Script completed successfully."
        break  # Exit the loop if the script runs without error
    else
        echo "Script failed. Retrying..."
    fi
done
