from .RegexPatterns import RegexPatterns
import regex as re

class BaseBPETokenizer:
    def __init__(self, pattern : str = "GPT2RegexPattern"):
        self.pattern = self._load_pattern(pattern)
        self.regex = re.compile(self.pattern, re.IGNORECASE)

    def split(self, text):
        return re.findall(self.pattern, text)
    
    def _load_pattern(self, pattern : str):
        try:
            return RegexPatterns[pattern]

        except:
            return pattern

    @staticmethod
    def get_stats(ids, counts=None):
        """
        Given a list of integers, return a dictionary of counts of consecutive pairs
        Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        Optionally allows to update an existing dictionary of counts
        """
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def merge(ids, pair, idx):
        newids = list()
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids