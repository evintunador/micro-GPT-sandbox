# my base model
This is the model I edit whenever I want to test a new architecture idea I have. It's designed to be flexible with many large architecture changes being tweakable from the config, easy to demonstrate what's happening in terms of the progression of tensor shapes, and easy to read/edit the code. The default config values are most similar to [minLlama3](https://github.com/evintunador/minLlama3) but include options for tweaks from other models such as the post-attention and post-MLP norm from [minGrok](https://github.com/evintunador/minGrok). Feel free to toy around and build off of it

Notice that even though the models we're training here are very small (0.5m to 5m parameters), they are actually reasonable proxies for how well a scaled up version may do on real text data because of our use of the [TinyStories](https://arxiv.org/abs/2305.07759) dataset. Basically, because this dataset is so high quality and narrow in scope, these tiny models actually have a fighting chance of picking up on real linguistic relationships. Somewhere in the 1 to 3m parameter range, a GPT2-inspired architecture is capable of understanding relationships like how the token 'apple' is something that the main character of the tiny story 'Tim' would like to 'eat'; it can actually pick up on the relationships in this text which are an isomorphic subset of the ones that an actual LLM would see when training on the entire internet. When the TinyStories paper came out it was big news for researchers with limited compute; this basic idea is the backbone behind microsoft's Phi family of models, originally described in the paper [Textbooks Are All You Need](https://arxiv.org/pdf/2306.11644). I hope this repo can be of help to anyone who wants to get into designing & building novel architectures but doesn't have the compute to test a larger model; i'm literally training these on the CPU of a 2019 iMac with 8gb of ram (running linux mint so that ram is actually available to be fair).

# File Structure
- `inference.ipynb`: open this notebook if you just want to load a model and perform inference
- `train.ipynb`: open this notebook to train a model
- `build_tokenizer.ipynb`: open this notebook to train a Character-Pair Encoding tokenizer (like Byte-Pair Encoding except I used the 95 unique characters that show up in the TinyStories dataset instead of actual bytes)
- `testing_modules.ipynb`: creates easy printouts that allow you to follow the progression of tensor shapes for demonstration & debugging purposes of all the modules in `model.py`. If you're building new modules for a novel architecture idea you have then this notebook will be of extreme value to you
- `model_evaluation.ipynb`: open this notebook to compare different models against each other. includes loss curve plots and topk teacher-forcing accuracy rate
- `hyperparameter_search.ipynb`: eventually in here i'd like to build an automated system that tests different hyperparameter configurations & skips past ones that go over the ram limit, but rn it's empty
- `config.py`: all of the editable model and training settings
- `inference.py`: functions for performing inference, used in `inference.ipynb`
- `train.py`: functions for training a model, used in `train.ipynb`
- `model.py`: the actual pytorch modules like RMSNorm, MLP, MQA, ResidualLayer, etc
- `tokenizer.py`: an overly-simplistic and annoyingly inefficient tokenizer with bos & eos tokens and post-sequence padding
- `tools.py`: A variety of functions & classes that don't fit elsewhere and/or are used by more than one of the jupyter notebooks. Of note is the `LoggingModule`, a wrapper for pytorch's `nn.module` that makes for very pretty & easy to follow printing of the progression of tensor shapes over in `testing_modules.ipynb`
- `model_code/`
    - `baseGPT.py`
    - `baseGPT_config.py`
    - `modules/`
        - `attentions.py`
        - `norms.py`
        - `feedforward.py`
- `tokenizers/`
    - `bpe_tokenizer_{95, 128, 256, 512, 1024, 2048}.model`: the 95 one is character-wise tokenization
- `trained_models/`
    - `0.5m_{short_and_thick, 5foot11_and_skinnyfat, tall_and_skinny}/`: a series of 0.5m parameter models designed to be compared against one another
        - `log_data.csv`: record of loss & perplexity data over the course of training
        - `model_config.json`: hyperparameters of the model
        - `model.pth`: weights of the model
        - `train_config.json`: hyperparameters of the training loop used to train the model
- `requirements.txt` - I should probably change this to only include the packages that are actually necessary and not be so strict on versions. The command I used to get this list is `pip freeze | grep -v " @ file://" > requirements.txt`, lmk if you know of a better method

# how to contribute
Other than the below TODO lists, appreciated contributions include bug fixes, adding more verbose comment explanations of what the code is doing, general readability edits, efficiency edits, taking better advantage of the `LoggingModule`, etc. Because I'm not super knowledgeable on how collaborating on git projects works and I tend to edit directly on the main branch, please reach out and communicate with me about any edits you plan to make so that I can avoid editing the same files. [Click here to join my discord server](https://discord.gg/hTYQyDPpr9)

# definite TODOs
- [x] setup to use TinyStories dataset by default
- [x] rewrite `tokenizer.py` with features like bos, eos and padding
- [x] record loss values more permanently & setup a way to compare loss curves between models
- [x] add options for different learning rate schedules
- [x] periodically save model checkpoints
- [x] add a loss mask to prevent from training on the padding tokens
- [x] rebuild the tokenizer to use more complicated pairing rules
    - [x] build tokenization models with larger vocabulary sizes
- [x] build a simple notebook for comparison bw diff ppl graphs & model outputs
- [ ] rearrange file structure such that every future experiment i do can be a submodule git repo. this outermost repo should only include the base model and all shared files
- [ ] create a hyperparameter search loop that knows to cancel a run if it's going over your available vram usage
- [ ] fix & enable batched inference
    - [ ] update `model_evaluation.ipynb`'s teacher-forcing topk analysis to get more accurate %'s using batches

# potential future TODOs
- [ ] setup .py files to be runnable in terminal rather than in the .ipynb files
- [ ] make tokenizer & function that turns list of token indices into tensors more efficient
- [ ] add option to continually train pre-existing models & update its training data/hyperparameters accordingly
- [ ] add automated model comparison analysis by GPT4 like in the [TinyStories](https://arxiv.org/abs/2305.07759) paper
- [ ] add an option to use a pre-built tokenizer from huggingface like GPT2's
- [ ] add sparse/local attention mask options
- [ ] make training parallelizable over multiple GPUs with fairscale
- [ ] build an easy way to design blocks in residual networks using lists of strings in the config. for example, the parallel MoE from [Snowflake](https://www.snowflake.com/en/) would be
```Python
[
'Norm->Attn->+->Norm->MLP->+',
'Norm->MoE->+'
]
```
- [ ] different architectures/modules to incorporate
    - [ ] cross-attention as a standalone module bc idk maybe it'll be useful at some point
    - [ ] [Mixture of Experts]()
    - [ ] [DiffuSeq](https://arxiv.org/abs/2210.08933)
    - [ ] whatever these new RNN-like transformers have going on such as [Gemma 1.1](https://arxiv.org/abs/2402.19427) or [Megawhatever]()
    - [ ] [Mamba](https://arxiv.org/abs/2312.00752)

# check me out
- guides on how to build miniature versions of popular models from scratch: [minGemma](https://github.com/evintunador/minGemma), [minGrok](https://github.com/evintunador/minGrok), and [minLlama3](https://github.com/evintunador/minLlama3)
- [my YouTube channel](https://www.youtube.com/@Tunadorable)
- my [other links](https://linktr.ee/tunadorable)