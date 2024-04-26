import functools
import inspect
import torch
from torch import nn

# this function will be used throughout for debugging/demonstration purposes
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

############# stuff for training & evaluation
# dataloader
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

def torcherize_batch(tokenizer, batch, max_seq_len, device):
    b = torch.zeros(len(batch), max_seq_len+1)
    for i, s in enumerate(batch):
        b[i] = torch.tensor(
            tokenizer.encode(s, bos=True, eos=True, pad=max_seq_len+1), 
            device=device
        )
    x, y = b[:,:max_seq_len], b[:, 1:]
    return x.to(torch.long), y.to(torch.long)

@torch.no_grad()
def estimate_loss(model, tokenizer, dataloader, eval_iters = 3): # to estimate loss during the training loop
    out = {}
    model.eval() # sets model to eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            batch = next(iter(dataloader))
            X, Y = torcherize_batch(tokenizer, batch, model.max_seq_len, model.device)
            logits, loss = model(X, target_token_ids=Y)
            losses[k] = loss.item()
        out[split] = losses
    model.train() # just resets to training mode
    return out