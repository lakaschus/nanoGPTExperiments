import logging
import os
import subprocess
import numpy as np
import json
import argparse
import train

import itertools

def hyperparameter_grid(hyperparameter_dict: dict):
    """
    Generates a hyperparameter grid for both integer and float parameters.
    
    Args:
    hyperparameter_dict (dict): A dictionary containing hyperparameters information.

    Returns:
    List[dict]: A list of dictionaries, each representing a unique combination of hyperparameters.
    """
    # Extract the hyperparameter tuning information
    tuning_info = hyperparameter_dict.get("hyperparameter_tuning", [])
    
    # Lists to store parameter names and their corresponding range of values
    parameter_names = []
    parameter_values = []

    # Populate the lists with parameter names and their ranges
    for parameter_info in tuning_info:
        name = parameter_info.get("parameter")
        param_type = parameter_info.get("type", "int")
        start, end = parameter_info.get("range")
        number_of_trials = parameter_info.get("number_of_trials", 2)

        if param_type == "float":
            values = np.linspace(start, end, number_of_trials)
        else:  # default to int
            values = range(start, end + 1, int((end + 1 - start) / number_of_trials))

        parameter_names.append(name)
        parameter_values.append(values)

    # Use itertools.product to compute the Cartesian product of parameter values
    grid = list(itertools.product(*parameter_values))

    # Convert the grid to a list of dictionaries
    grid_dicts = [dict(zip(parameter_names, values)) for values in grid]

    return grid_dicts

def train_models(hyperparameter_grid, base_train_params):
    """
    Trains models for each combination of hyperparameters in the hyperparameter grid, 
    while retaining the base training parameters.

    Args:
    hyperparameter_grid (List[dict]): A list of dictionaries, each representing a unique combination of hyperparameters.
    base_train_params (dict): Base training parameters.
    """
    # Set up a logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Iterate over each configuration in the hyperparameter grid; If the grid is empty, go into this loop nevertheless
    for config in hyperparameter_grid or [None]:
        # Copy the base training parameters and update with the current configuration
        train_params = base_train_params.copy()
        train_params.update(config)

        # Log and print the command
        logger.info(f"TRAINING MODEL with: {train_params}")

        # Execute the training command
        train.main(train_params)

# Parse arg "config_file" from command line
parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="./configs/config.json")
config_file = parser.parse_args().config_file

with open(config_file) as f:
    params = json.load(f)
    train_params = params['train_params']
    sample_params = params['sample_params']
    pipeline_params = params['pipeline_params']

sample_cmd = "python sample.py"
for key, value in sample_params.items():
    sample_cmd += f" --{key}={value}"

print(sample_cmd)
    
# Step 1: Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Step 2: Start training
hyperparameter_grid = hyperparameter_grid(pipeline_params)
train_models(hyperparameter_grid, train_params)


# Step 3: Create sample
logger.info("CREATING SAMPLE")
print("CREATING SAMPLE")
subprocess.run(sample_cmd, shell=True, check=True)