#!/bin/bash


# python sketch_prediction.py

python extrude_prediction.py

python fillet_prediction.py

python chamfer_prediction.py

python operation_prediction.py



# while true; do
#     echo "Running generate_dataset.py..."
    
#     # Run the Python script
#     python generate_CAD_program.py
    
#     # Check the exit status of the Python script
#     if [ $? -eq 0 ]; then
#         echo "Script completed successfully."
#         break  # Exit the loop if the script runs without error
#     else
#         echo "Script failed. Retrying..."
#     fi
# done
