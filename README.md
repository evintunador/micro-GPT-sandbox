# base model
This is the model I edit whenever I want to test a new architecture idea I have. It's designed to be very flexible with many large architecture changes being tweakable from the config, easy to demonstrate what's happening, and easy to read/edit the code. The default config values are most similar to [minLlama3](https://github.com/evintunador/minLlama3) but include options for tweaks from other models such as the post-attention and post-MLP norm from [minGrok](https://github.com/evintunador/minGrok). Feel free to toy around with it

# File Structure
- `tokenizers/`
    - `build_tokenizer_TinyShakespeare.ipynb` - need to redo this with better regex
    - `build_tokenizer_TinyStories.ipynb` - need to redo this with better regex
    - `tokenizer_TinyShakespeare.py`
    - `tokenizer_TinyStories.py` - need to redo this with features like bos, eos and padding
    - `tiny_shakespeare_tokenizer_{128, 256, 512, 1024}.model`
    - `tiny_stories_tokenizer_{128, 256, 512, 1024}.model`
- `requirements.txt` - I should probably change this to use `>=` instead of `==` and only the packages that are actually necessary, which i think are just pandas, datasets and torch. The command I used to get this list is `pip freeze | grep -v " @ file://" > requirements.txt`
- `data_TinyShakespeare.txt` - what it sounds like. i need to make this fetch from the internet with a dataloader instead of being loaded locally & entirely into ram
- `config.py` - all of the settings. the defaults are pretty similar to [minLlama3](https://github.com/evintunador/minLlama3)
- `inference.py` - functions for performing inference. maybe at some point i'll make it support batched inference
- `model.py`
- `tools.py` - of note is the `LoggingModule`, a wrapper for pytorch's `nn.module` that makes for very pretty & easy to follow printing of the progression of tensor shapes
- `dataloader.ipynb` - this will either get turned into its own .py file or added to `training.ipynb`
- `inference.ipynb` - if you just want to load a model and perform inference
- `testing_modules.ipynb` - allows you to follow the progression of tensor shapes
- `training.ipynb` - should i turn this into a .py file?

# TODOs
- [ ] setup to use TinyStories dataset by default
    - [ ] rewrite `tokenizer_TinyStories.py` with features like bos, eos and padding
    - [ ] setup to use TinyShakespeare as an option in either `config.py` or `training.ipynb` and use a dataloader instead of loading the entire dataset into ram
- [ ] build a much better tokenizer (bos & eos tokens, better regex, etc) or maybe just use a pre-existing one
- [ ] fix & enable batched inference

# potential future TODOs
- [ ] add options for different learning rate schedules
- [ ] add option to continually train pre-existing model & update its training data/hyperparameters accordingly
- [ ] create a hyperparameter search for loop
- [ ] record loss values more permanently & setup a way to compare loss curves between models
- [ ] add in an MoE option
- [ ] add sparse/local attention mask options
- [ ] make it parallelizable over multiple GPUs with fairscale
- [ ] add mamba block? ugh probably not, mamba's code is not fun
- [ ] add a DiffuSeq alternative instead of NTP? ugh probably not, that'd be a whole different repo

# Check out my other links/resources
- guides on how to build miniature versions of popular models from scratch: [minGemma](https://github.com/evintunador/minGemma), [minGrok](https://github.com/evintunador/minGrok), and [minLlama3](https://github.com/evintunador/minLlama3)
- [my YouTube channel](https://www.youtube.com/@Tunadorable)
- my [other links](https://linktr.ee/tunadorable)