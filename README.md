# micro GPT sandbox
## about
This repo is means as a hub for all of my novel GPT architecture experiments. Of note is [templateGPT](https://github.com/evintunador/templateGPT), the template that I always start with. If you would like to experiment with ideas you have for how to change GPTs, join me!

The idea behind the files in this repo is that they allow for easily comparing the performance of different experimental models and keeping each experiment in its own git repo with the use of submodules. To start with we've only got templateGPT and [fractal-head-attention](https://github.com/evintunador/fractal-head-attention) which is just a quick little example I made to demonstrate how you might edit templateGPT, but in the future as I do more experiments I'll be adding them as submodules to this repo. 

## installation
clone the repo. if you'd like to clone it along with every single available submodule right from the beginning, then do 
```
git clone --recurse-submodules <parent-repository-URL>
```
or if you'd like to be selective about your submodules, then just clone them indivudually into `models/`

## how to use
1. when starting new experiments make sure to use templateGPT as your base and not change the fundamental behavior of any functions or classes referenced here in this repo, because if you do then they won't be compatible with this repo. If people actually gain any interest in this stuff then i'll get around to giving a more in-depth explanation as to what explicitly should not be changed. 
2. once you've finished with an experiment, add it to `models/` and see how it compares to the baselines trained in templateGPT

## file structure
- `models/`: the directory containing all submodule experiments
    - `templateGPT/`: the template that I always build my new experiments off of
    - `fractal-head-attention/`: the first little example experiment
- `model_comparison.py`: holds the functions used in `model_comparison.ipynb`
- `tools.py`: holds functions used for both `model_comparison.ipynb` and `inference.ipynb`
- `inference.ipynb`: for looking at the output of a specific model from one of the submodules
- `model_comparison.ipynb` is for comparing the loss curves and top-k teacher forcing accuracy of multiple models side-by-side. by default it's set to compare `models/templateGPT/trained/templateGPT_1m_short_and_thicc` against `models/fractal-head-attention/trained/FHA_1m_short_and_thicc`. As you can see, this fractal head attention idea doesn't perform as well, although to be fair it uses 1 fewer attention heads per model and around 20k fewer parameters. *this is currently broken. topk accuracy is bugged & misleading; only use the loss curve comparison*

## definite TODOs
- [ ] fix the model topk comparison
    - [ ] fix dynamic importing. i think i can just list off all the files in modules/ and delete all of them from cache and that should fix the problem
- [ ] add a better guide on how to use submodules

## potential future TODOs
- [ ] build a script that calls the OpenAI API and has GPT4 rate & compare the outputs of different models

## how to contribute
Other than the above TODO lists, appreciated contributions include:
- bug fixes
- adding more detailed comment explanations of what the code is doing
- general readability edits
- efficiency edits
- editing the code in `modules/` to take better advantage of the `LoggingModule`. This means splitting up each class into more and tinier functions
- training more models (especially if they're bigger than what's already here!)

Because I'm not super knowledgeable on how collaborating on git projects works and I tend to edit directly on the main branch, please reach out and communicate with me about any edits you plan to make so that I can avoid editing the same files. [Click here to join my discord server](https://discord.gg/hTYQyDPpr9)

## check me out
- guides on how to build miniature versions of popular models from scratch: [minGemma](https://github.com/evintunador/minGemma), [minGrok](https://github.com/evintunador/minGrok), and [minLlama3](https://github.com/evintunador/minLlama3)
- [my YouTube channel](https://www.youtube.com/@Tunadorable)
- my [other links](https://linktr.ee/tunadorable)
