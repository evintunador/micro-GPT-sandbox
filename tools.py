import torch

###########################################################
################ LOADING DATA #############################
###########################################################
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

class TinyStoriesDataset(Dataset):
    def __init__(self, split):
        # Load the dataset
        self.dataset = load_dataset("noanabeshima/TinyStoriesV2", split=split)
        
    def __len__(self):
        # Return the size of the dataset
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Fetch one item from the dataset
        return self.dataset[idx]['text']

def get_data_loader(batch_size=32, shuffle=True, split='train', num_workers=0):
    # Create the dataset
    dataset = TinyStoriesDataset(split)
    # Create the DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def torcherize_batch(
    tokenizer, 
    batch, 
    max_seq_len = 512, 
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> (torch.Tensor, torch.Tensor):
    b = torch.zeros(len(batch), max_seq_len+1)
    for i, s in enumerate(batch):
        b[i] = torch.tensor(
            tokenizer.encode(s, bos=True, eos=True, pad=max_seq_len+1), 
            device=device
        )
    x, y = b[:,:max_seq_len], b[:, 1:]
    return x.to(torch.long), y.to(torch.long)

# this might be a faster alternative but idk how it works (other than "threads") and i couldn't measure a noticeable performance improvement
#from concurrent.futures import ThreadPoolExecutor
#def torcherize_batch(tokenizer, batch, max_seq_len=512, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
#    with ThreadPoolExecutor() as executor:
#        # Encode each text in parallel using ThreadPoolExecutor
#        encoded_batch = list(executor.map(
#            lambda text: tokenizer.encode(text, bos=True, eos=True, pad=max_seq_len + 1), 
#            batch
#        ))

#    # Create the torch tensor from the encoded batch
#    b = torch.tensor(encoded_batch, device=device)
#
#    # Extract input (x) and target (y) sequences
#    x, y = b[:, :max_seq_len], b[:, 1:]
#
#    # Ensure data types are correct
#    return x.to(torch.long), y.to(torch.long)

###########################################################
###################### DYNAMIC IMPORTING ##################
###########################################################

# allows us to import functions specific to a given model project, meaning you can change those functions in your project & stuff still works
import importlib
def import_from_nested_path(folders, file, items):
    try:
        # Construct the module path from a list of folders
        module_path = ".".join(folders) + "." + file
        
        # Dynamically import the module
        module = importlib.import_module(module_path)
        
        # Extract specific items (functions, classes, etc.)
        imported_items = {}
        for item in items:
            if hasattr(module, item):
                imported_items[item] = getattr(module, item)
            else:
                print(f"{item} is not available in {module_path}")
        return imported_items
                
    except ImportError as e:
        print(f"Failed to import module: {e}")

# a wrapper to force a given function to behave using a specified working directory rather than the current working directory
import os
def run_in_directory(func, path, *args, **kwargs):
    original_dir = os.getcwd()  # Save the current directory
    os.chdir(path)  # Change to the target directory
    try:
        result = func(*args, **kwargs)  # Execute the function
    finally:
        os.chdir(original_dir)  # Change back to the original directory
    return result

# Example usage
#def example_function():
    #print("Current Working Directory:", os.getcwd())

# Calling the function with a custom directory
#run_in_directory(example_function, "models/customGPT/")

###########################################################
#################### LOAD MODELS ##########################
###########################################################
import os
import json
from dataclasses import asdict
import time
import csv

def load_model(
    name: str, # the filepath to the model. ex: 'models/customGPT/trained/customGPT_1m_tall_and_skinny'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    path_parts = name.split('/')

    # grabbing the config and Model class from the correct directories
    def internal():
        from modules.model import Model
        from config import ModelConfig
        return Model, ModelConfig
    Model, ModelConfig = run_in_directory(internal, os.path.join(path_parts[0], path_parts[1]))

    # Deserialize the JSON file back to a dictionary
    with open(f'{name}/model_config.json', 'r') as f:
        config_dict = json.load(f)
    
    # Convert the dictionary back to a Config object
    cfg = ModelConfig(**config_dict)
    cfg.device = device
    
    # grabbing the get_tokenizer function from the correct directory
    imported_objects = import_from_nested_path(
        [path_parts[0], path_parts[1], 'tokenizers', cfg.tokenizer], 
        'tokenizer', 
        ['get_tokenizer']
    )
    get_tokenizer = imported_objects.get('get_tokenizer')
    
    # defining the tokenizer
    vocab_size = cfg.vocab_len - 3
    tokenizer = run_in_directory(get_tokenizer, os.path.join(path_parts[0], path_parts[1]), vocab_size)
    
    # Initialize a blank model
    model = Model(cfg).to(cfg.device) 
    
    # Load the saved model parameters
    model_path = os.path.join(path_parts[2], path_parts[3], 'model.pth')
    run_in_directory(
        lambda: model.load_state_dict(
            torch.load(
                model_path, 
                map_location=cfg.device
            )
        ), 
        os.path.join(path_parts[0], path_parts[1])
    )
    
    print(sum(p.numel() for p in model.parameters())/1e3, 'K parameters', '\n', cfg)

    return model, tokenizer, cfg

###########################################################
############# MODEL COMPARISONS ###########################
###########################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_column_from_csv(models, x_column, y_column, log_x=False, log_y=False, trim_percentage=0):
    """
    Reads multiple CSV files, extracts specific columns, and plots a graph comparing their values
    with options to use logarithmic scales on the axes and trim initial data.

    Parameters:
    - models: List of strings, where each string is a name of a model residing in models/
    - x_column: String, the name of the column to use as the x-axis.
    - y_column: String, the name of the column to use as the y-axis.
    - log_x: Boolean, whether to use a logarithmic scale on the x-axis (default is False).
    - log_y: Boolean, whether to use a logarithmic scale on the y-axis (default is False).
    - trim_percentage: Integer, percentage of the initial data to exclude from the plot (default is 0).
    """
    if not models or not x_column or not y_column:
        raise ValueError("Paths list and column names must be provided and not empty.")

    plt.figure(figsize=(10, 6))
    
    for model in models:
        path = f'{model}/log_data.csv'
        try:
            data = pd.read_csv(path)
            if x_column not in data.columns or y_column not in data.columns:
                raise ValueError(f"Columns {x_column} or {y_column} not found in {path}")

            # Calculate the index to start slicing from
            start_index = int(len(data) * (trim_percentage / 100))

            # Slice data from the calculated start index
            data = data.iloc[start_index:]
            plt.plot(data[x_column], data[y_column], label=f'{model}')
        except Exception as e:
            print(f"Failed to process {path}: {str(e)}")
    
    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')

    plt.title(f'{y_column} Over Training {x_column}s')
    plt.xlabel(x_column + (' (log scale)' if log_x else ''))
    plt.ylabel(y_column + (' (log scale)' if log_y else ''))
    plt.legend(title="Model")
    plt.grid(True)
    plt.show()