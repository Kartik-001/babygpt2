import unicodedata
import re
import pickle

class MyBPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab  # token_id → bytes
        self.merges = merges  # list of (token_a, token_b)
        self.vocab_reverse = {v: k for k, v in vocab.items()}  # bytes → token_id
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or {}

        if "eot" not in self.special_tokens:
                self.add_special_token("eot", "<|endoftext|>")
    
    def _preprocess_text(self, text):
        norm_text = unicodedata.normalize('NFKC', text.lower())
        pre_tokens = re.findall(r"\s+|\w+|[^\s\w]+", norm_text)
        byte_chunks = [chunk.encode('utf-8') for chunk in pre_tokens]
        return byte_chunks

    def _byte_encode(self, text):
        byte_chunks = self._preprocess_text(text)
        token_chunks = []
        for chunk in byte_chunks:
            token_chunks.extend(list(chunk))  # split each byte
        return token_chunks

    def _get_pairs(self, tokens):
        return {(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)}

    def tokenize(self, text):
        tokens = self._byte_encode(text)
        tokens = [t for t in tokens]  # Start with raw bytes

        while True:
            pairs = self._get_pairs(tokens)
            if not pairs:
                break

            best = None
            min_rank = float('inf')
            for pair in pairs:
                if pair in self.merge_ranks and self.merge_ranks[pair] < min_rank:
                    best = pair
                    min_rank = self.merge_ranks[pair]

            if best is None:
                break

            new_token = max(self.vocab.keys()) + 1
            self.vocab[new_token] = self.vocab[best[0]] + self.vocab[best[1]]
            self.vocab_reverse[self.vocab[new_token]] = new_token

            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == best:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def decode(self, tokens):
        byte_seq = b''.join([self.vocab[tok] for tok in tokens])
        return byte_seq.decode('utf-8', 'replace')

    def add_special_token(self, name, token_str):
        token_bytes = token_str.encode("utf-8")
        token_id = max(self.vocab.keys()) + 1
        self.vocab[token_id] = token_bytes
        self.vocab_reverse[token_bytes] = token_id
        self.special_tokens[name] = token_id
        return token_id

    def save(self, path="tokenizer.pkl"):
        with open(path, "wb") as f:
            pickle.dump((self.vocab, self.merges, self.special_tokens), f)

    @staticmethod
    def load(path="tokenizer.pkl"):
        with open(path, "rb") as f:
            vocab, merges, special_tokens = pickle.load(f)
        return MyBPETokenizer(vocab, merges, special_tokens)
