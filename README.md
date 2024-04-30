# my base model
This is the model I edit whenever I want to test a new architecture idea I have. It's designed to be very flexible with many large architecture changes being tweakable from the config, easy to demonstrate what's happening, and easy to read/edit the code. The default config values are most similar to [minLlama3](https://github.com/evintunador/minLlama3) but include options for tweaks from other models such as the post-attention and post-MLP norm from [minGrok](https://github.com/evintunador/minGrok). Feel free to toy around with it

# File Structure
- `config.py` - all of the editable model and training settings. the defaults are pretty similar to [minLlama3](https://github.com/evintunador/minLlama3)
- `inference.ipynb` - if you just want to load a model and perform inference
- `train.ipynb` - trains the models
- `testing_modules.ipynb` - creates easy prints that allow you to follow the progression of tensor shapes for demonstration & debugging purposes
- `inference.py` - functions for loading trained models and performing inference
- `train.py` - functions for loading batches and training a model
- `model.py` - the actual modules like RMSNorm, MLP, MQA, ResidualLayer, etc
- `tokenizer.py` - an overly-simplistic and annoyingly inefficient tokenizer with post-sequence padding (should i change to pre-sequence padding?)
- `tools.py` - A variety of functions & classes that don't fit elsewhere. Of note is the `LoggingModule`, a wrapper for pytorch's `nn.module` that makes for very pretty & easy to follow printing of the progression of tensor shapes
- `hyperparameter_search.ipynb` - eventually in here i'd like to build an automated system that tests hyperparameters & skips past ones that go over the ram limit, but rn it's empty
- `tokenizers/`
    - `build_tokenizer_TinyStories.ipynb` - need to redo this with a better regex
    - `tiny_stories_tokenizer_{128, 256, 512, 1024}.model` - need to make more of these with larger vocab sizes
- `models/`
    - to be trained
- `requirements.txt` - I should probably change this to only include the packages that are actually necessary and not be so strict on versions. The command I used to get this list is `pip freeze | grep -v " @ file://" > requirements.txt`, lmk if you know of a better method

# definite TODOs
- [x] setup to use TinyStories dataset by default
- [x] rewrite `tokenizer.py` with features like bos, eos and padding
- [x] record loss values more permanently & setup a way to compare loss curves between models
- [x] add option to continually train pre-existing model & update its training data/hyperparameters accordingly
- [x] add options for different learning rate schedules
- [x] periodically save model checkpoints
- [x] add a loss mask to prevent from training on the padding tokens
- [ ] rebuild the tokenizer to use a more complicated regex
    - [ ] build tokenization models with larger vocabulary sizes
- [ ] build a simple notebook for comparison bw diff ppl graphs & model outputs
    - [ ] automated analysis by GPT3.5 or 4?
- [ ] create a hyperparameter search loop

# potential future TODOs
- [ ] setup .py files to be runnable in terminal rather than in the .ipynb files
- [ ] fix & enable batched inference
- [ ] add an option to use a pre-built tokenizer from huggingface like GPT2's?
- [ ] add sparse/local attention mask options
- [ ] make training parallelizable over multiple GPUs with fairscale
- [ ] add in TinyStories' methodology for automated story grading by GPT4
- [ ] different architectures/modules to incorporate
    - [ ] cross-attention bc idk maybe it'll be useful at some point
    - [ ] [Mixture of Experts]()
    - [ ] [DiffuSeq](https://arxiv.org/abs/2210.08933)
    - [ ] whatever these new RNN-like transformers have going on such as [Gemma 1.1](https://arxiv.org/abs/2402.19427) or [Megawhatever]()
    - [ ] [Mamba](https://arxiv.org/abs/2312.00752)

# check me out
- guides on how to build miniature versions of popular models from scratch: [minGemma](https://github.com/evintunador/minGemma), [minGrok](https://github.com/evintunador/minGrok), and [minLlama3](https://github.com/evintunador/minLlama3)
- [my YouTube channel](https://www.youtube.com/@Tunadorable)
- my [other links](https://linktr.ee/tunadorable)