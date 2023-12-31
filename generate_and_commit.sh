#!/bin/bash

# Navigate to the project directory
cd /path/to/your/project

# Run the data generation script
python generate_data.py

# Add the new data to DVC
dvc add data

# Commit the changes to DVC
git add .
git commit -m "Update data"
dvc commit