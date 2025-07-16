from .BaseBPETokenizer import BaseBPETokenizer
import json
import os

class BPETrainer(BaseBPETokenizer):
    def __init__(self, pattern : str = "GPT2RegexPattern") -> None:
        super().__init__(pattern)
        self.merges = dict()
        self.vocab = dict()
    
    def train(self, text : str, vocab_size : int, verbose : bool = True):
        num_merges = vocab_size - 256
        
        text_chunks = self.split(text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        self.vocab = {idx : bytes([idx]) for idx in range(256)}
        
        for i in range(num_merges):
            stats = {}

            for chunk_ids in ids:
                self.get_stats(chunk_ids, stats)

            pair = max(stats, key=stats.get)
            idx = 256 + i

            ids = [self.merge(chunk_ids, pair, idx) for chunk_ids in ids]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({self.vocab[idx]}) had {stats[pair]} occurrences")
    
    def save(self, save_dir : str):
        assert self.merges != {} and self.vocab, "Entrena el tokenizador primero"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(save_dir, "merges.txt"), "w", encoding="utf-8") as f:
            for pair, idx in self.merges.items():
                f.write(f"{pair[0]} {pair[1]}\n")

        inverted_vocab = {token.decode('utf-8', errors='replace'): i for i, token in self.vocab.items()}
        
        with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(inverted_vocab, f, ensure_ascii=False, indent=2)
        
        tokenizer_data = dict(
            base = "GPT2Tokenizer",
            pattern = self.pattern
        )
        
        with open(os.path.join(save_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)