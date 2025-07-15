from MyBPETokenizer import MyBPETokenizer

# Load the entire tokenizer (vocab, merges, special tokens) in one call
tokenizer = MyBPETokenizer.load("tokenizer.pkl")

# Now you can immediately use it
tokens = tokenizer.tokenize("why is my is tokens list greater than my actual text length?")
print("Tokens:", tokens)
print("Decoded:", tokenizer.decode(tokens))