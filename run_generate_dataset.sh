#!/bin/bash

while true; do
    echo "Running generate_CAD_program.py with a 5-minute limit..."

    # Run the Python script with a 5-minute timeout
    timeout 300 python generate_CAD_program.py

    # Check exit status
    if [ $? -eq 0 ]; then
        echo "Script completed successfully."
        break  # Exit loop if script finishes successfully
    else
        echo "Script timed out or failed. Restarting..."
    fi

    sleep 2  # Optional: Pause for 2 seconds before restarting
done
