import torch
from torch import nn

# dataloader
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# training loop
import time
import csv

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

def scheduler_lambda(current_iter, debug=False):
    T_i = T_0
    if current_iter < warmup_iters:
        # Linear warmup
        lr = lr_min + (lr_max - lr_min) * (current_iter / warmup_iters)
    elif current_iter < max_iters - final_flat_iters:
        # Cosine annealing with warm restarts
        cycle_iter = current_iter - warmup_iters
        while cycle_iter >= T_i:
            cycle_iter -= T_i
            T_i *= T_mult
        if anneal_type == 'lin': 
            lr = lr_max - (lr_max - lr_min) * (cycle_iter / T_i)
        else:
            # defaults to 'cos' learning rate annealing
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + torch.cos(torch.pi * torch.tensor(cycle_iter / T_i)))
    else:
        # Constant learning rate
        lr = lr_min
    return lr

def train(
    model, 
    tokenizer, 
    cfg, 
    optimizer,
    scheduler,
    train_cfg, 
    train_data_loader,
    test_data_loader,
    eval_interval = 50, 
    log_data: list = None, 
    checkpoint_interval = None, # currently doesn't do anything
    detect_anomoly = False # use if you're getting crazy errors about a the gradient being broken
):
    if log_data is None:
        log_data = []
    
    # Enable anomaly detection. useful for really deep issues in the model where the gradient breaks
    if detect_anomoly: torch.autograd.set_detect_anomaly(True)
    
    start_time = time.time()
    
    for i in range(train_cfg.max_iters):
    
        # sample a batch of data
        batch = next(iter(train_data_loader))
        x,y = torcherize_batch(tokenizer, batch, cfg.max_seq_len, cfg.device)
        
        # train
        logits, loss = model(x, target_token_ids=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward() # find the gradients
        optimizer.step() # edit parameters
        scheduler.step() # Update the learning rate
        
        # every once in a while evaluate the loss on train and val sets
        if i % eval_interval == 0 or i == max_iters - 1:
            elapsed_time = time.time() - start_time
            losses = estimate_loss(model, tokenizer, test_data_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Collect data for CSV
            log_data.append([
                i,
                current_lr,
                losses['train'].mean().item(),
                losses['val'].mean().item(),
                torch.exp(losses['val']).mean().item(),
                elapsed_time,
                batch_size,
                weight_decay
            ])
            print(
                f"step {i:04d}: lr {current_lr:.6f}, train loss {losses['train'].mean().item():.4f}, "
                f"val loss {losses['val'].mean().item():.4f}, ppl {torch.exp(losses['val']).mean().item():.0f}, "
                f"time elapsed: {elapsed_time:.2f} seconds"
            )
    
    # Disable anomaly detection after the training loop
    if detect_anomoly: torch.autograd.set_detect_anomaly(False)

    return model, optimizer, log_data # do i need to send out train & test dataloader?