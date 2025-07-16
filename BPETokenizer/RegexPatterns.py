RegexPatterns = {
    "GPT2RegexPattern" : r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s", # GPT2 Tokenizer: https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
}