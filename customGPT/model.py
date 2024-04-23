import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from tools import *
from config import *

###########################################################
#################### NORM #######################################
###########################################################
class Norm(LoggingModule):
    def __init__(self, dim, cfg: Config):
        super().__init__()
        self.eps = cfg.eps

        # We start with ones for weight to keep the original scale initially, and zeros for bias.
        self.affine = cfg.norm_affine
        self.bias = cfg.norm_bias
        if cfg.norm_affine:
            self.w = nn.Parameter(torch.ones(cfg.dim))
            if cfg.norm_bias:
                self.b = nn.Parameter(torch.zeros(cfg.dim))

        # Mapping norm types to their respective methods
        self.type = cfg.norm_type
        self.norm_methods = {
            "CosineNorm": self.CosineNorm,
            "LayerNorm": self.LayerNorm,
            "RMSNorm": self.RMSNorm}
        # Ensure the specified norm type exists, default to RMSNorm if not found
        if self.type not in self.norm_methods:
            print(f'norm type {self.type} not found. defaulting to RMSNorm')
            self.type = "RMSNorm"

    @log_io
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_method = self.norm_methods[self.type]
        x = norm_method(x)

        # Optionally apply the affine transformation
        if self.affine: 
            x = x * self.w
        if self.bias:
            x = x + self.b
            
        return x

    @log_io
    def CosineNorm(self, x):
        return x / torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=self.eps)

    @log_io
    def LayerNorm(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)

    @log_io
    def RMSNorm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

###########################################################
#################### ATTENTION #######################################
###########################################################
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

class MQSA(LoggingModule): # multi-head self-attention
    def __init__(self, cfg: Config):
        super().__init__()
        self.num_q_heads = cfg.num_q_heads
        self.num_kv_heads = cfg.num_q_heads if cfg.num_kv_heads is None else cfg.num_kv_heads
        self.num_q_per_kv = self.num_q_heads // self.num_kv_heads
        self.head_dim = cfg.dim // cfg.num_q_heads if cfg.head_dim is None else cfg.head_dim
        self.dropout_rate = cfg.dropout_rate

        self.Wq = nn.Linear(cfg.dim, cfg.num_q_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(cfg.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(cfg.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.Wo = nn.Linear(cfg.num_q_heads * self.head_dim, cfg.dim, bias=False)

        self.cache_k = torch.zeros(
            (cfg.max_batch_size, cfg.max_seq_len, self.num_kv_heads, self.head_dim),
            requires_grad = False).to(cfg.device)
        self.cache_v = torch.zeros(
            (cfg.max_batch_size, cfg.max_seq_len, self.num_kv_heads, self.head_dim),
            requires_grad = False).to(cfg.device)
    
    @log_io
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache_len: int = None,
        training: bool = False,
    ):
        batch_size, seqlen, _ = x.shape
        xq, xk, xv = self.Wq(x), self.Wk(x), self.Wv(x)

        xq = xq.view(batch_size, seqlen, self.num_q_heads, self.head_dim)
        xk = xk.view(batch_size, seqlen, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seqlen, self.num_kv_heads, self.head_dim)

        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis)

        if cache_len is not None: # if we're performing inference and using kv caching. it'll init at 0
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:batch_size, cache_len : cache_len + seqlen] = xk
            self.cache_v[:batch_size, cache_len : cache_len + seqlen] = xv

            keys = self.cache_k[:batch_size, : cache_len + seqlen]
            values = self.cache_v[:batch_size, : cache_len + seqlen]
        else: 
            # if we're training, do full sequence length
            keys, values = xk, xv
        queries = xq # for sake of keeping the naming scheme consistent

        # adjusts keys and values to match the query heads count.
        if self.num_kv_heads != self.num_q_heads:
            keys, values = self.match_headcount(keys, values) # (batch_sizes, cache_len + seqlen, num_q_heads, head_dim)

        queries = queries.transpose(1, 2)  # (bs, num_q_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, num_q_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, num_q_heads, cache_len + seqlen, head_dim)
        
        logits = self.attend(queries, keys, training)
        if mask is not None:
            logits = logits + mask  # (bs, num_q_heads, seqlen, cache_len + seqlen)
        output = self.calc_output(logits, values, training) 
        
        return F.dropout(self.Wo(output), p=self.dropout_rate, training=training)
    
    @log_io
    def apply_rotary_emb(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = self.reshape_for_broadcast(freqs_cis.to(xq.device), xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    @log_io
    def reshape_for_broadcast(
        self,
        freqs_cis: torch.Tensor, 
        x: torch.Tensor
    ):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f'freqs_cis.shape {freqs_cis.shape} != (x.shape[1], x.shape[-1]) {(x.shape[1], x.shape[-1])}'
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @log_io
    def match_headcount(self, keys, values):
        keys = torch.repeat_interleave(keys, self.num_q_per_kv, dim=2)
        values = torch.repeat_interleave(values, self.num_q_per_kv, dim=2)
        return keys, values

    @log_io
    def attend(self, queries, keys, training):
        return torch.matmul(queries, keys.transpose(2, 3)) * (self.head_dim ** -0.5)
    
    @log_io
    def calc_output(
        self, 
        logits, 
        values, 
        training
    ):
        batch_size, _, seqlen, _ = logits.shape
        scores = F.softmax(logits, dim=-1)
        scores = F.dropout(scores, p=self.dropout_rate, training=training)
        output = scores @ values # [batch_size, n_heads, seqlen, head_dim]
        return output.transpose(1, 2).contiguous().view(batch_size, seqlen, -1) # [batch_size, seqlen, n_heads * head_dim]

###################################################
################# MLP ##################################
###################################################
class MLP(LoggingModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        nonlinearity: str = 'GeLU',
        gated: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = input_dim * cfg.mlp_hidden_mult
        self.dropout_rate = cfg.dropout_rate

        # the gate, up and down projections
        self.gated = gated
        if gated: self.gate_proj = nn.Linear(input_dim, self.hidden_size, bias)
        self.up_proj = nn.Linear(input_dim, self.hidden_size, bias)
        self.down_proj = nn.Linear(self.hidden_size, output_dim, bias)

        # Mapping norm types to their respective methods
        self.nonlinearities = {
            "GeLU": nn.GeLU(),
            "SiLU": nn.SiLU(),
            "ReLU": nn.ReLU()}
        # Ensure the specified norm type exists, default to GeLU if not found
        if nonlinearity not in self.nonlinearities:
            self.nonlinearity = nn.GeLU
            print(f'nonlinearity {nonlinearity} not found. defaulting to GeLU')
        else:
            self.nonlinearity = self.nonlinearities[nonlinearity]
        
    @log_io
    def forward(self, x: torch.Tensor, training: bool = False ) -> torch.Tensor:
        if self.gated:
            hidden_neurons = self.nonlinearity(self.gate_proj(x)) * self.up_proj(x)
        else:
            hidden_neurons = self.nonlinearity(self.up_proj(x))
        return self.down_proj(F.dropout(hidden_neurons, p=self.dropout_rate, training=training))


class TransformerBlock(LoggingModule):
    def __init__(self, layer_id: int, cfg: Config):
        super().__init__()
        self.num_q_heads = cfg.num_q_heads
        self.dim = cfg.dim
        self.head_dim = cfg.dim // cfg.num_q_heads
        self.attention = MQSA(cfg)
        self.feed_forward = MLP(
            dim=cfg.dim,
            hidden_dim=4 * cfg.dim,
            multiple_of=cfg.multiple_of,
            ffn_dim_multiplier=cfg.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = Norm(cfg.dim, cfg)
        self.ffn_norm = Norm(cfg.dim, cfg)
        self.dropout_rate = cfg.dropout_rate

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache_len: int = None,
        training = False,
    ):
        h = x + F.dropout(self.attention(self.attention_norm(x), freqs_cis, mask, cache_len), p=self.dropout_rate, training=training)
        out = h + F.dropout(self.feed_forward(self.ffn_norm(h)), p=self.dropout_rate, training=training)
        return out


class Llama3(LoggingModule):
    def __init__(self, cfg: Config, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.num_layers = cfg.num_layers
        self.max_seq_len = cfg.max_seq_len
        self.tokenizer = tokenizer

        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(cfg.num_layers):
            self.layers.append(TransformerBlock(layer_id, cfg))

        self.norm = Norm(cfg.dim, cfg)
        self.output = nn.Linear(
            cfg.dim, 
            cfg.vocab_size, 
            bias=False)

        self.freqs_cis = precompute_freqs_cis(
            cfg.dim // cfg.num_q_heads,
            cfg.max_seq_len * 2,
            cfg.theta,).to(cfg.device)

        mask = torch.full((cfg.max_seq_len, cfg.max_seq_len), 
                          float("-inf"), 
                          device=cfg.device)
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, # specifically for training
                tokens: torch.Tensor, 
                targets: torch.Tensor):
        batch_size, seqlen = tokens.shape
        assert tokens.shape == targets.shape
        assert seqlen == self.max_seq_len
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]
        
        for layer in self.layers:
            h = layer(
                h, 
                freqs_cis, 
                self.mask, 
                cache_len = None, 
                training = True
            )
        h = self.norm(h)
        logits = self.output(h).float()

        loss = self.criterion(
            logits.view(batch_size * seqlen, self.vocab_size),
            targets.reshape(batch_size * seqlen))
        
        return logits, loss

    @torch.inference_mode()
    def forward_inference(self, 
                          tokens: torch.Tensor,
                          cache_len: int,
                          max_context_window: int,
                         ):
        _batch_size, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[cache_len : cache_len + seqlen]

        mask = self.mask[:seqlen, :seqlen]
        # When performing key-value caching, we compute the attention scores
        # only for the new sequence. Thus, the matrix of scores is of size
        # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
        # j > cache_len + i, since row i corresponds to token cache_len + i.
        mask = torch.hstack(
            [torch.zeros((seqlen, cache_len), device=tokens.device), mask]
        ).type_as(h)

        for layer in self.layers:
            h = layer(
                h, 
                freqs_cis, 
                mask, 
                cache_len = cache_len
            )
        h = self.norm(h)
        logits = self.output(h).float()
        return logits

    @torch.inference_mode() # no need to keep track of gradients during inference
    def Sampler(
        self,
        logits: torch.Tensor, # shape (batch_size, input_len, vocab_size)
        temperature: float, # controls how boring vs random the outputs should be
        top_p: float, # the maximum cumulative probability of output options we're willing to consider
        top_k: int, # the maximum number of output options we're willing to consider
    ) -> torch.Tensor:
        """
        The Sampler function is responsible for generating token predictions
        It supports temperature scaling, top-p (nucleus) sampling, and top-k sampling 
        """
        # Select the last element for each sequence.
        logits = logits[:,-1,:] # (batch_size, input_len, vocab_size) -> (batch_size, vocab_size)
        
        # Apply temperature scaling
        logits.div_(temperature) # (batch_size, vocab_size) / float -> (batch_size, vocab_size)

        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float) # dim=-1 is the vocab_size dimension that we calculate along

        # sort the probabilities to for use in top-p & top-k. both are (batch_size, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        ## calculating top-p
        # creates same-size tensor of cumulatve probabilities instead of indivdiual probs
        probs_sum = torch.cumsum(probs_sort, dim=-1) 
        # mask where 0's are top-p selections & 1's are to be excluded
        top_ps_mask = (probs_sum - probs_sort) > top_p
        # the original probabilities with excluded tokens changed to 0.0
        probs_sort = torch.where(top_ps_mask, 0, probs_sort) 

        ## calculating top_k
        # create a shape (vocab_size) tensor that just iterates up by 1's
        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device) 
        # expand our mask along the batch_size dimension to become size (batch_size, vocab_size)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        # top_ks is a list of integers. we keep whichever entries in top_ks_mask are greater than their corresponding entries in top_ks
        top_ks_mask = top_ks_mask >= top_k

        # we'll be combining top-p with top-k and using whichever gives us fewer tokens. a very conservative approach
        # this trims probs_sort to also fit within our top_k requirement
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

        # Re-normalization so that total probabilities add up to 1
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        
        # now we rearrange the modified probabilities in probs_sort back to their original order according to probs_idx
        probs = torch.gather(probs_sort, dim=-1, index=torch.cfgort(probs_idx, dim=-1))
        
        # samples from the distribution
        next_token_id = torch.multinomial(probs, num_samples=1)
        
        return next_token_id # returns the predicted token

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_gen_len: int = None,
        memory_saver_div: int = 1, # defaults to full max_seq_len**2 memory use. must be power of 2
        temperature: float = 0.6, # default value in meta's code
        top_p: float = 0.9, # default value in meta's code
        top_k: int = None, # meta's code doesn't bother with topk
    ) -> str: 
        """ Wrapper around sampler() that deals with manipulation of the sequence """
        assert ((memory_saver_div & (memory_saver_div-1)) == 0) & (memory_saver_div > 0), f'memory_saver_div {memory_saver_div} must be power of 2'
        max_context_window = self.max_seq_len // memory_saver_div
        if max_context_window < self.max_seq_len:
            print(f'maximum attention matrix size will be {max_context_window}x{self.max_seq_len} rather than {self.max_seq_len}x{self.max_seq_len}\n')
        if top_k is None:
            top_k = self.tokenizer.vocab_len
        
        # encoding the prompt into token indices
        tokens = self.tokenizer.encode(prompt)
        
        if max_gen_len is None:
            max_gen_len = self.max_seq_len - len(tokens)
        elif max_gen_len + len(tokens) > self.max_seq_len:
            print(f'capping max_gen_len at max_seq_len={self.max_seq_len} including input\n')
            max_gen_len = self.max_seq_len - len(tokens)

        # turning it into the right tensor shape
        tokens = torch.tensor(tokens, device=self.cfg.device)
        tokens = tokens.unsqueeze(0) if len(tokens.shape)==1 else tokens # jic we need to add a batch dimension
        
        cache_len = max(tokens.shape[1] - max_context_window, 0)
        
        for i in range(max_gen_len):
            # get the model's output logits and ignore the loss, which would be a NoneType object
            logits = self.forward_inference(
                tokens[:,-max_context_window:],
                cache_len = cache_len,
                max_context_window = max_context_window
            )
            
            next_token = self.Sampler(
                logits = logits,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k
            )

            # add our new token to the sequence
            tokens = torch.cat((tokens, next_token), dim=1)
            
            if tokens.shape[1] >= max_context_window:
                cache_len += 1

        # decode our list of tokens to an actual string
        output = self.tokenizer.decode(tokens.squeeze(0).tolist())

        return output