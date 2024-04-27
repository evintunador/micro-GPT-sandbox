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
def estimate_loss(model, tokenizer, train_data_loader, test_data_loader, eval_samples = 3): # to estimate loss during the training loop
    out = {} # dictionary to record & separate train loss from val loss
    model.eval() # sets model to eval mode so we're not keeping track of gradients
    for split in ['train', 'val']:
        losses = torch.zeros(eval_samples)
        for k in range(eval_samples):
            # grab a list of strings from either the train or test set
            batch = next(iter(train_data_loader)) if split == 'train' else next(iter(test_data_loader))
            # turn list of strings into tensor of token indices
            X, Y = torcherize_batch(tokenizer, batch, model.max_seq_len, model.device) 
            # run the model to get loss
            logits, loss = model(X, target_token_ids=Y)
            losses[k] = loss.item()
        out[split] = losses
    model.train() # just resets the model to training mode
    return out

def scheduler_lambda(current_iter):
    from config import TrainConfig
    tcfg = TrainConfig()
    T_i = tcfg.T_0()
    if current_iter < tcfg.warmup_iters:
        # Linear warmup
        lr = tcfg.lr_min + (tcfg.lr_max - tcfg.lr_min) * (current_iter / tcfg.warmup_iters)
    elif current_iter < tcfg.max_iters - tcfg.final_flat_iters:
        # Cosine annealing with warm restarts
        cycle_iter = current_iter - tcfg.warmup_iters
        while cycle_iter >= T_i:
            cycle_iter -= T_i
            T_i *= tcfg.T_mult
        if tcfg.anneal_type == 'lin': 
            lr = tcfg.lr_max - (tcfg.lr_max - tcfg.lr_min) * (cycle_iter / T_i)
        else:
            # defaults to 'cos' learning rate annealing
            lr = tcfg.lr_min + 0.5 * (tcfg.lr_max - tcfg.lr_min) * (1 + torch.cos(torch.pi * torch.tensor(cycle_iter / T_i)))
    else:
        # Constant learning rate
        lr = tcfg.lr_min
    return lr

def train(
    model, 
    tokenizer, 
    cfg, 
    optimizer,
    scheduler,
    tcfg, 
    train_data_loader,
    test_data_loader,
    log_data: list = None, 
    detect_anomoly: bool = False # use if you're getting crazy errors about a the gradient being broken
):
    from tools import save_model # for checkpoints
    
    if log_data is None: # for recording loss/ppl curves
        log_data = []
    
    # Enable anomaly detection. useful for really deep issues in the model where the gradient breaks
    if detect_anomoly: torch.autograd.set_detect_anomaly(True)
    
    start_time = time.time()
    
    for i in range(tcfg.max_iters):
    
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
        if (i % tcfg.eval_interval) == 0 or (i == tcfg.max_iters - 1):
            elapsed_time = time.time() - start_time
            losses = estimate_loss(model, tokenizer, train_data_loader, test_data_loader, eval_samples = tcfg.eval_samples)
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr is torch.Tensor:
                current_lr = current_lr.item()
            
            # Collect data for CSV & print it
            log_data.append([
                i,
                current_lr,
                losses['train'].mean().item(),
                losses['val'].mean().item(),
                torch.exp(losses['val']).mean().item(),
                elapsed_time,
            ])
            print(
                f"step {i:04d}: lr {current_lr:.6f}, train loss {losses['train'].mean().item():.4f}, "
                f"val loss {losses['val'].mean().item():.4f}, ppl {torch.exp(losses['val']).mean().item():.0f}, "
                f"time elapsed: {elapsed_time:.2f} seconds"
            )

        # every once in awhile save a checkpoint of the model
        if tcfg.checkpoint_interval is not None:
            if (i % tcfg.checkpoint_interval == 0) or i == 0:
                save_model(model, cfg, tcfg, log_data, checkpoint=True)
    
    # Disable anomaly detection after the training loop
    if detect_anomoly: torch.autograd.set_detect_anomaly(False)

    return model, optimizer, log_data # do i need to send out train & test dataloader?