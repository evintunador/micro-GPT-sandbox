import torch
from torch import nn

###########################################################
########### DEBUGGING / DEMONSTRATION #####################
###########################################################
import functools
import inspect

# using this is way cleaner than cluttering up our code with print statements
def log_io(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check if logging is enabled globally and for the specific function
        if not self.logging_enabled or func.__name__ in self.disabled_logging_functions:
            return func(self, *args, **kwargs)
        #if not self.logging_enabled:
            #return func(self, *args, **kwargs)

        def log_item(item, name, level=0, is_root=False):
            indent = "    " * level
            if isinstance(item, torch.Tensor):
                print(f"{indent}Tensor '{name}' shape: {item.shape}")
            elif isinstance(item, tuple):
                if is_root and level == 0:
                    # Root level tuple, don't print it as a tuple unless it's a "true" tuple
                    for idx, sub_item in enumerate(item):
                        log_item(sub_item, f"{name}[{idx}]", level)
                else:
                    print(f"{indent}Tuple '{name}':")
                    for idx, sub_item in enumerate(item):
                        log_item(sub_item, f"{name}[{idx}]", level + 1)
            elif isinstance(item, int):
                print(f"{indent}Integer '{name}': Value={item}")
            elif isinstance(item, float):
                print(f"{indent}Float '{name}': Value={item}")
            else:
                print(f"{indent}Other-type '{name}': Type={type(item).__name__}, Value={item}")

        print(f"\n{'='*10}Entering {self.__class__.__name__}.{func.__name__}{'='*10}")
        print("Inputs:")
        arg_names = inspect.getfullargspec(func).args[1:]  # Excluding 'self'
        arg_values = args + tuple(kwargs.values())
        for name, value in zip(arg_names, arg_values):
            log_item(value, name)

        result = func(self, *args, **kwargs)
        print("\nOutputs:")
        if isinstance(result, tuple):
            log_item(result, "output", is_root=True)
        else:
            log_item(result, "output")

        print(f"{'='*10}Exiting {self.__class__.__name__}.{func.__name__}{'='*10}")
        return result
    return wrapper

class LoggingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.logging_enabled = False
        self.disabled_logging_functions = set()

    def enable_logging(self):
        self.logging_enabled = True

    def disable_logging(self):
        self.logging_enabled = False

    def disable_function_logging(self, func_name):
        self.disabled_logging_functions.add(func_name)

    def enable_function_logging(self, func_name):
        self.disabled_logging_functions.discard(func_name)

###########################################################
############# SAVE / LOAD MODELS ##########################
###########################################################
import os
import json
from dataclasses import asdict
import time
import csv

def save_model(model, cfg, tcfg, log_data = None, checkpoint = False):
    if checkpoint == False:
        path = f'models/{tcfg.model_name}'
    else: 
        path = f'models/{tcfg.model_name}/checkpoint-{time.strftime("%Y-%m-%d|%H-%M-%S")}'
    os.makedirs(path, exist_ok=True)
    
    if log_data is not None:
        # Save training data to CSV
        with open(f'{path}/log_data.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Step', 
                'Learning Rate', 
                'Train Loss', 
                'Validation Loss', 
                'Perplexity', 
                'Time Elapsed'
            ])
            writer.writerows(log_data)
    
    # saving model
    torch.save(model.state_dict(), f'{path}/model.pth')
    
    # saving configs
    cfg_dict = asdict(cfg)
    with open(f'{path}/model_config.json', 'w') as f:
        json.dump(cfg_dict, f)
    tcfg_dict = asdict(tcfg)
    with open(f'{path}/train_config.json', 'w') as f:
        json.dump(tcfg_dict, f)

def load_model(
    name: str, 
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    from config import ModelConfig
    from tokenizer import get_tokenizer
    from model import customGPT

    model_name = f'models/{name}'

    # Deserialize the JSON file back to a dictionary
    with open(f'{model_name}/model_config.json', 'r') as f:
        config_dict = json.load(f)
    
    # Convert the dictionary back to a Config object
    cfg = ModelConfig(**config_dict)
    cfg.device = device
    
    # tokenizer
    path = f'./tokenizers/tiny_stories_tokenizer_{cfg.vocab_len-3}.model'
    tokenizer = get_tokenizer(path) 
    
    # Initialize a blank model
    model = customGPT(cfg).to(cfg.device) 
    
    # Load the saved state dictionary
    path = f'{model_name}/model.pth'
    model.load_state_dict(torch.load(path)) 
    
    print(cfg, '\n\n', sum(p.numel() for p in model.parameters())/1e3, 'K parameters')

    return model, tokenizer, cfg