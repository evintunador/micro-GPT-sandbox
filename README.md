These are the base models I edit whenever I want to test a new architecture idea I have. Feel free to use them
- `input.txt` is just [TinyShakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
- `tokenizer.ipynb` is loosely based on [andrej karpathy's guide to building a tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE). Instead of encoding based on bytes I just use the 65 individual characters that appear in the dataset. Total is 128 tokens since I didn't want to make the datset size too small. The only regex-style rule i've enforced is that a token can only be made up of letters or non-letters, but not both.
- `minGPT.ipynb` is closely based on [andrej karpathy's minGPT](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5014s). I'm no longer really using this one since it's so outdated
- `minGemma.ipynb` is based on [my own guide to Gemma](https://youtu.be/WW7ZxaC3OtA?si=BheH1zSakFxXoHP4) which itself uses a little bit of karpathy's code but mostly [google's open-sourced Gemma inference code](https://github.com/google/gemma_pytorch)
