This is the base model I edit whenever I want to test a new architecture idea I have. Feel free to use it

*cleaned up filestructure & description of each file to come in future*

potential future TODOs
- build a much better tokenizer (bos & eos tokens, better regex, etc)
    - fix & enable batched inference
- add in an MoE option
- add sparse/local attention mask options
- make it parallelizable over multiple GPUs with fairscale
- add mamba block?
- add a DiffuSeq alternative instead of NTP?