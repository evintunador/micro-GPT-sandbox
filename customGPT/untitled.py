class Model(LoggingModule):
    def __init__(self, config: Config, tokenizer: tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        ### hyperparameters
        self.max_seq_len = config.max_seq_len
        self.sa_head_dim = config.sa_head_dim
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.combine_factor = config.combine_factor
        self.levels = config.levels
        
        ### embedding
        # the embedding matrix. for converting tokens to the first residual state, and the last residual state to logits
        self.embedder = nn.Embedding(config.vocab_size, config.embed_dim)
        # the padding vector to get used when sequence length isn't perfect
        self.padding_vector = nn.Parameter(torch.zeros(config.embed_dim), requires_grad=True)
        
        # the function that combines embeddings into higher level concept residual states
        self.embedding_combiner = CombineEmbeddings(config, self.padding_vector)

        ### the actual bulk of the model
        self.body = Body(config, self.embedder.weight, self.embedding_combiner)
        
        ### the loss functions
        # lowest-level token model
        self.ce_loss_fn = nn.CrossEntropyLoss()
        # concept models
        self.concept_loss_fn = ConceptLoss(config)
        
    @log_io
    def forward(self,
                input_token_ids: torch.Tensor, # a shape (batch_size, input_seq_len) list of integer token ids to run forward pass on
                target_token_ids: torch.Tensor = None, # a shape (batch_size, input_seq_len + combo ** (levels-1)) list of token ids to train on
                cvec_topk: int = None,
                cvec_greedy: bool = False,
                cvec_temp: float = 1.0,
               ) -> torch.Tensor:

        # create the tuple of initial residual states to calculate on
        x0s = self.create_x0s(input_token_ids) # x0s are ordered token level -> highest concept level
        
        if target_token_ids is None: ### if we're doing inference
            # the body of the model that iterates through the decoder & cross-attention layers
            xfs = self.body(x0s, cvec_topk=cvec_topk, cvec_greedy=cvec_greedy, cvec_temp=cvec_temp) 

            # the actual token output logits we care about
            logits = xfs[-1]
            
            # if we're not training, then we don't need to calculate loss
            loss = None
        else: ### if we're training
            assert input_token_ids.shape[1] == target_token_ids.shape[1] - (self.combine_factor ** (self.levels - 1)), f'inputs:{input_token_ids.shape[1]} and targets:{target_token_ids.shape[1]} have unexpected shapes'

            # create the tuple of target embedding vectors
            targets = self.create_targets(target_token_ids, input_token_ids.shape[1]) # targets are ordered token level -> highest concept level

            # the body of the model that iterates through the decoder & cross-attention layers
            xfs = self.body(x0s, targets) # xfs are ordered highest concept level -> token level

            ### first up is regular CE token loss
            logits = xfs[-1]
            batch_size, input_len, vocab_size = logits.shape
            # splice target tokens to exclude the ones that were only to be used by concept levels
            target_token_ids_spliced = target_token_ids[:,:input_len]
            # we reshape our logits & targets before calculating cross-entropy loss
            ce_loss = self.ce_loss_fn(logits.view(batch_size*input_len, vocab_size),
                                      target_token_ids_spliced.reshape(batch_size*input_len))

            ### the new thing, a regression loss for all our concept-embedding layers
            concept_loss = self.concept_loss_fn(xfs, tuple(reversed(targets)))
            # adding it all together
            loss = ce_loss + concept_loss
        
        return logits, loss
        
    @log_io
    def create_x0s(self, input_token_ids: torch.Tensor) -> Tuple[torch.Tensor]:
        #print(f'input_token_ids: {input_token_ids.shape}\n{input_token_ids[0,:32]}')
        
        # turn the input tokens into the first residual state using the embedding matrix
        x0 = self.embedder(input_token_ids) # (batch_size, input_len, embed_dim)
        #print(f'x0: {x0.shape}\n{x0[0,0:(self.combine_factor**(config.levels-1))*2,:4]}')

        # finding the number of padding vectos we have to use at the token level to ensure the cross-attention predictive mask will line up
        remainder = x0.shape[1] % self.combine_factor
        padding_needed = 0 if remainder == 0 else self.combine_factor - remainder

        # do the actual padding for the token level
        # once i get a more complicatead tokenizer would i replace this with a <|bos|> token? Would that token be unique to each level?
        if padding_needed > 0:
            # Replicate the padding vector the necessary number of times
            padding = self.padding_vector.repeat(padding_needed, 1).unsqueeze(0).expand(x0.shape[0], -1, -1)
            #print(f'padding: {padding.shape}')
            
            x0 = torch.cat([padding, x0], dim=1)
            #print(f'x0 after padding: {x0.shape}\n{x0[0,0:(self.combine_factor**(config.levels-1))*2,:4]}')
        
        # instantiate the tuple that'll hold all the residual states
        x0s = (x0 * (self.embed_dim ** 0.5),) 
        
        ### iterating through levels to create each higher-level concept residual state
        for i in range(self.levels-1):
            # combine into smaller tensor by adding token (or lower level concept) embeddings together
            lvl_combo = self.combine_factor ** (i+1)
            x0c = self.embedding_combiner(x0, lvl_combo) # c stands for concept
            #print(f'x0c: {x0c.shape}\n{x0c[0,0:self.combine_factor,:4]}')
            
            # finally scale & add it to the tuple of residual states
            x0s += (x0c * (self.embed_dim ** 0.5),)
        
        return x0s

    @log_io
    def create_targets(self, target_token_ids: torch.Tensor, input_len: int) -> Tuple[torch.Tensor]:
        #print(f'target_token_ids: {target_token_ids.shape}\n{target_token_ids[0,:32]}')
        
        # turn the target tokens into the first residual state using the embedding matrix
        token_lvl_target_token_ids = target_token_ids[:,1:1+input_len]
        t0 = self.embedder(token_lvl_target_token_ids) # (batch_size, input_len, embed_dim)
        #print(f'token_lvl_target_token_ids: {token_lvl_target_token_ids.shape}\n{token_lvl_target_token_ids[0,:32]}')
        #print(f't0: {t0.shape}\n{t0[0,0:(self.combine_factor**(config.levels-1))*2,:4]}')
        
        # need to account for offsets in sequence length, which includes both an offset correction & padding vectors
        remainder = t0.shape[1] % self.combine_factor
        padding_needed = 0 if remainder == 0 else self.combine_factor - remainder
        if padding_needed == 1:
            t0 = torch.cat([self.embedder(target_token_ids[:,0].unsqueeze(1)), t0], dim=1)
            #print(f't0 after padding: {t0.shape}\n{t0[0,0:(self.combine_factor**(config.levels-1))*2,:4]}')
        elif padding_needed > 1:
            padding = self.padding_vector.repeat(padding_needed - 1, 1).unsqueeze(0).expand(t0.shape[0], -1, -1)
            #print(f'padding: {padding.shape}')
            t0 = torch.cat([padding, self.embedder(target_token_ids[:,0].unsqueeze(1)), t0], dim=1)
            #print(f't0 after padding: {t0.shape}\n{t0[0,0:(self.combine_factor**(config.levels-1))*2,:4]}')
        
        # instantiate the tuple that'll hold all the residual states
        targets = (t0,) 
        
        ### iterating through levels to create each higher-level concepts
        for i in range(1, self.levels):
            # calculate the correct combo factor for this level
            lvl_combo = self.combine_factor ** i

            # my subsetting here is all messy. doesn't properly take into account off-sequences & the padding token
            # i think maybe i can fix this in the predictive mask once i make that part

            # how many tokens off are we from a perfectly sized (multiple of lvl_combo) sequence, meaning how many padding vectors do we need?
            remainder = input_len % self.combine_factor # will only ever be self.combine_factor -1 at most
            offset = 0 if remainder == 0 else self.combine_factor - remainder
            
            # adjust input_len to ceiling the size necessary for this level
            #input_len_adj = input_len + lvl_combo

            # subset the currect targets to be predicted at this level
            concept_lvl_target_token_ids = target_token_ids[:, lvl_combo - offset:lvl_combo + input_len]# - offset]
            #print(f'concept_lvl_target_token_ids: {concept_lvl_target_token_ids.shape}\n{concept_lvl_target_token_ids[0,:32]}')

            # turn them into embeddings
            t0c = self.embedder(concept_lvl_target_token_ids)

            # combine the token embeddings into concepts
            t0c = self.embedding_combiner(t0c, lvl_combo)
            #print(f't0c: {t0c.shape}\n{t0c[0,0:self.combine_factor,:4]}')
            
            # append to tuple
            targets += (t0c,)
        
        return targets
        
    @log_io
    def generate(self,
                 prompt: str,
                 output_len: int = 1, # the model will output 1 token by default
                 temperature: float = 1.0, # 1.0 would be no effect
                 top_p: float = 1.0,
                 top_k: int = config.vocab_size,
                ) -> str: 
        """ Wrapper around sampler() that deals with manipulation of the sequence """
        # encoding the prompt into token indices
        tokens = self.tokenizer.encode(prompt)

        # turning it into the right tensor shape
        tokens = torch.tensor(tokens, device=config.device).unsqueeze(0)
        
        # we wouldn't want to go past the maximum context length we trained on
        if len(tokens) + output_len > self.config.max_seq_len:
            output_len = self.max_seq_len - len(tokens)
            print("capping output at maximum sequence length")

        for i in range(output_len):
            # get the model's output logits and ignore the loss, which would be a NoneType object
            logits, _ = self(tokens[:,:self.max_seq_len])
            
            next_token = self.Sampler(logits, temperature, top_p, top_k)

            # add our new token to the sequence
            tokens = torch.cat((tokens, next_token), dim=1)

        # resets this variable so that the corresponding warning in Body.concept_matchup can come up next time we perform inference
        global cvec_warning 
        cvec_warning = False

        # decode our list of tokens to an actual string
        return self.tokenizer.decode(tokens.squeeze(0).tolist())

    @torch.no_grad() # no need to keep track of gradients during inference
    @log_io
    def Sampler(
        self,
        logits: torch.Tensor, # shape (batch_size, input_len, vocab_size)
        temperature: float, # controls how boring vs random the outputs should be
        top_p: float, # the maximum cumulative probability of output options we're willing to consider
        top_k: int, # the maximum number of output options we're willing to consider
    ) -> torch.Tensor:
        
        # Select the last element for each sequence & apply temperature scaling
        logits = logits[:,-1,:].div_(temperature) # -> (batch_size, vocab_size)
        
        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float) # dim=-1 is the vocab_size dimension that we calculate along
        #print('first probs: ', probs)
        
        # sort the probabilities to for use in top-p & top-k. both are (batch_size, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        ### calculating top-p
        # creates same-size tensor of cumulatve probabilities instead of indivdiual probs
        probs_sum = torch.cumsum(probs_sort, dim=-1) 
        # mask where 0's are top-p selections & 1's are to be excluded
        top_ps_mask = (probs_sum - probs_sort) > top_p
        # the original probabilities with excluded tokens changed to 0.0
        probs_sort = torch.where(top_ps_mask, 0, probs_sort) 

        ### calculating top_k
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
        probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))
        #print('probs after topp & k: ', probs)
        
        # samples from the distribution
        next_token_id = torch.multinomial(probs, num_samples=1)
        
        return next_token_id # returns the predicted token














class customGPT(LoggingModule):

    def __init__(self,
        config: Config, # the hyperparameters
        tokenizer: tokenizer, # the tokenizer. we don't always store the tokenizer inside of the model, but it doesn't matter here
    ):
        super().__init__()
        self.config = config

        # the attention heads need to cleanly divide up the embed_dim of the model so that we can split it all apart & combine back together
        assert config.embed_dim % config.num_q_heads == 0

        self.max_seq_len = config.max_seq_len
        self.head_dim = config.head_dim
        self.vocab_size = config.vocab_size
        self.tokenizer = tokenizer

        # the embedding matrix. for converting tokens to the first residual state, and the last residual state to logits
        self.embedder = nn.Embedding(self.vocab_size+1, config.embed_dim)
        self.scaling = config.embed_dim ** 0.5 # for normalizing the first embedding
        
        # Initialize a sequence of DecoderLayer instances as specified by the number of hidden layers in the config
        self.layers = nn.ModuleList(Layer(config) for _ in range(config.num_hidden_layers))

        # Initialize a normalization layer to be applied after the last decoder layer, stabilizing the output
        self.final_norm = RMSNorm(config.embed_dim, use_scale=True)

        # the loss function
        self.criterion = nn.CrossEntropyLoss()

    @log_io
    def forward(
        self,
        input_token_ids: torch.Tensor, # a shape (batch_size, input_seq_len) list of integer token ids
        target_token_ids: torch.Tensor = None, # a shape (batch_size, input_seq_len) list of token ids to train on
        ) -> torch.Tensor:

        # turn the input tokens into the first resudial state using the embedding matrix
        x = self.embedder(input_token_ids) * self.scaling # (batch_size, input_len) & (vocab_size, embed_dim) -> (batch_size, input_len, embed_dim)

        # Iteratively process the input through each Layer
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
        
        # Apply normalization to the output of the final decoder layer
        x = self.final_norm(x)

        # grabbing the weights of the embedding matrix shape (vocab_size, hidden_dim) for use as the output layer
        embedder_weight = self.embedder.weight

        # the embedding matrix is also used as the output layer
        logits = torch.matmul(x, embedder_weight.t()) # (batch_size, input_len, embed_dim) @ (embed_dim, vocab_size) -> (batch_size, input_len, vocab_size)
        
        if target_token_ids is None: # if we're not training, then we don't need to calculate loss
            loss = None
        else:
            # if we are training
            batch_size, input_len, vocab_size = logits.shape
            # then we reshape our logits & targets before calculating cross-entropy loss
            loss = self.criterion(logits.view(batch_size*input_len, vocab_size), 
                                  target_token_ids.reshape(batch_size*input_len))
        
        return logits, loss

    @torch.no_grad() # no need to keep track of gradients during inference
    @log_io
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

        ### calculating top-p
        # creates same-size tensor of cumulatve probabilities instead of indivdiual probs
        probs_sum = torch.cumsum(probs_sort, dim=-1) 
        # mask where 0's are top-p selections & 1's are to be excluded
        top_ps_mask = (probs_sum - probs_sort) > top_p
        # the original probabilities with excluded tokens changed to 0.0
        probs_sort = torch.where(top_ps_mask, 0, probs_sort) 

        ### calculating top_k
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
        probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))
        
        # samples from the distribution
        next_token_id = torch.multinomial(probs, num_samples=1)
        
        return next_token_id # returns the predicted token
        
    @log_io
    def generate(
        self,
        prompt: str,
        output_len: int = None, 
        temperature: float = 1.0, # defaulting to 1.0 means we essentially don't use temperature
        top_p: float = 1.0, # defaulting to 1.0 means we essentially don't use top-p
        top_k: int = config.vocab_size, # setting top_k = vocab_size means we're effectively not using top_k at all
    ) -> str: 
        """ Wrapper around sampler() that deals with manipulation of the sequence """
        # encoding the prompt into token indices
        tokens = self.tokenizer.encode(prompt)
        
        if output_len is None:
            output_len = config.max_seq_len - len(tokens)

        # turning it into the right tensor shape
        tokens = torch.tensor(tokens, device=config.device).unsqueeze(0)
        
        # we wouldn't want to go past the maximum context length we trained on
        assert len(tokens) + output_len <= self.config.max_seq_len

        for i in range(output_len):
            # get the model's output logits and ignore the loss, which would be a NoneType object
            logits, _ = self(tokens[:,:self.max_seq_len])
            
            next_token = self.Sampler(
                logits = logits,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k
            )
            if next_token == config.vocab_size: break

            # add our new token to the sequence
            tokens = torch.cat((tokens, next_token), dim=1)

        # decode our list of tokens to an actual string
        output = self.tokenizer.decode(tokens.squeeze(0).tolist())

        return output