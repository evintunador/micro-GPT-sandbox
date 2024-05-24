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
import importlib
import sys
import os

# allows us to import functions specific to a given model project, meaning you can change those functions in your project & stuff still works
def import_from_nested_path(folders, file, items):
    try:
        # Construct the module path from a list of folders
        module_path = ".".join(folders) + "." + file
        print(f"Trying to import from module path: {module_path}")
        
        # Ensure the base directory is in sys.path
        base_dir = os.path.abspath(os.path.join(*folders))
        if base_dir not in sys.path:
            sys.path.append(base_dir)
        
        # Remove the module from sys.modules if it's already imported
        if module_path in sys.modules:
            del sys.modules[module_path]
        
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
        return None

# a wrapper to force a given function to behave using a specified working directory rather than the current working directory
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
import json
from dataclasses import asdict
import time
import csv

def load_model(
    name: str, # the filepath to the model. ex: 'models/templateGPT/trained/templateGPT_1m_tall_and_skinny'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    path_parts = name.split('/')

    imported_items = import_from_nested_path(
        [path_parts[0], path_parts[1]], 
        'config', 
        ['ModelConfig']
    )
    if imported_items is None:
        raise ImportError("ModelConfig not found in the specified path.")
    ModelConfig = imported_items['ModelConfig']

    # import the model class
    #def internal():
    #    from modules.model import Model
    #    return Model
    #Model = run_in_directory(internal, os.path.join(path_parts[0], path_parts[1]))
    # alternative that doesn't seem to be working. whenever I un-comment it insists that there is no 'modules'
    imported_items = import_from_nested_path(
        [path_parts[0], path_parts[1], 'modules'], 
        'model', 
        ['Model']
    )
    if imported_items is None:
        raise ImportError("Model not found in the specified path.")
    Model = imported_items['Model']

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
    if imported_objects is None:
        raise ImportError("get_tokenizer not found in the specified path.")
    get_tokenizer = imported_objects.get('get_tokenizer')
    
    # defining the tokenizer
    tokenizer = run_in_directory(get_tokenizer, os.path.join(path_parts[0], path_parts[1]), cfg.vocab_len)
    
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