from .BaseBPETokenizer import BaseBPETokenizer
import json
import os

class BPETokenizer(BaseBPETokenizer):
    def __init__(self, merges : dict, vocab : dict, pattern : str = "GPT2RegexPattern"):
        super().__init__(pattern)
        self.merges = merges
        self.vocab = vocab
    
    def decode(self, ids : list):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors = "replace")
        return text
    
    def encode(self, text : str):
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key = lambda p : self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break
            
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
            
        return tokens

    @classmethod
    def from_dir(cls, load_dir : str):
        """Carga un tokenizador entrenado desde un directorio."""

        with open(os.path.join(load_dir, "vocab.json"), "r", encoding="utf-8") as f:
            inverted_vocab = json.load(f)
        vocab = {int(i): token.encode('utf-8') for token, i in inverted_vocab.items()}

        merges = {}
        with open(os.path.join(load_dir, "merges.txt"), "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                id1, id2 = line.strip().split()
                pair = (int(id1), int(id2))
                idx = 256 + i # Las fusiones se asignan secuencialmente a partir de 256
                merges[pair] = idx
        
        with open(os.path.join(load_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)

        return cls(
            pattern = tokenizer_data["pattern"],
            vocab = vocab,
            merges = merges
        )