# my base model
This is the model I edit whenever I want to test a new architecture idea I have. It's designed to be very flexible with many large architecture changes being tweakable from the config, easy to demonstrate what's happening, and easy to read/edit the code. The default config values are most similar to [minLlama3](https://github.com/evintunador/minLlama3) but include options for tweaks from other models such as the post-attention and post-MLP norm from [minGrok](https://github.com/evintunador/minGrok). Feel free to toy around with it

# File Structure
- `requirements.txt` - I should probably change this to use `>=` instead of `==` and only the packages that are actually necessary, which i think are just pandas, datasets and torch. The command I used to get this list is `pip freeze | grep -v " @ file://" > requirements.txt`
- `config.py` - all of the settings. the defaults are pretty similar to [minLlama3](https://github.com/evintunador/minLlama3)
- `inference.py` - functions for performing inference. maybe at some point i'll make it support batched inference
- `model.py`
- `tokenizer.py` - a simple and inefficient tokenizer class with post-sequence padding
- `tools.py` - of note is the `LoggingModule`, a wrapper for pytorch's `nn.module` that makes for very pretty & easy to follow printing of the progression of tensor shapes
- `inference.ipynb` - if you just want to load a model and perform inference
- `testing_modules.ipynb` - allows you to follow the progression of tensor shapes
- `training.ipynb` - sets up stuff like the data loader and trains the model
- `tokenizers/`
    - `build_tokenizer_TinyStories.ipynb` - need to redo this with better regex
    - `tiny_stories_tokenizer_{128, 256, 512, 1024}.model` - need to make larger vocab sizes
- `models/`
    - to be filled out

# TODOs
- [x] setup to use TinyStories dataset by default
    - [x] rewrite `tokenizer.py` with features like bos, eos and padding
- [x] periodically save model checkpoints
- [ ] create a hyperparameter search loop
- [x] record loss values more permanently & setup a way to compare loss curves between models
- [ ] setup .py files to be runnable in terminal rather than in the .ipynb files
- [ ] rebuild the tokenizer to use a more complicated regex and also be bigger
- [ ] add a loss mask to prevent from training on the padding tokens

# potential future TODOs
- [ ] fix & enable batched inference
- [ ] add an option to use a pre-built tokenizer like GPT2's?
- [ ] add an option to train on attention chunks? maybe these new RNN-transformers that google is marketing as infinte attention? 
- [x] add options for different learning rate schedules
- [ ] add option to continually train pre-existing model & update its training data/hyperparameters accordingly
- [ ] add in an MoE option. likely won't do this until i stumble upon multiple GPUs to run in parallel
- [ ] add sparse/local attention mask options
- [ ] make it parallelizable over multiple GPUs with fairscale
- [ ] refactor everything to fit with better programming practices that i'm not aware of

# check me out
- guides on how to build miniature versions of popular models from scratch: [minGemma](https://github.com/evintunador/minGemma), [minGrok](https://github.com/evintunador/minGrok), and [minLlama3](https://github.com/evintunador/minLlama3)
- [my YouTube channel](https://www.youtube.com/@Tunadorable)
- my [other links](https://linktr.ee/tunadorable)