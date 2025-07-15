import json
import regex as re
import unicodedata
from collections import Counter
import os
import pickle # Import pickle for saving/loading
import requests  # Added for downloading data

# Function to pre-tokenize text
def pre_tokenize(text):
    text = text.lower()
    tokens = re.findall(
        r''''s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+''',
        text
    )
    return tokens

# --- Data Loading and Preparation ---

# Assuming tiny_shakespeare.txt is available in the same directory as the script
# If not, the script will attempt to download it.
data_file_path = "tiny_shakespeare.txt"

def download_shakespeare_data():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = data_file_path
    print(f"Attempting to download {filename} from {url} ...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print("Download successful.")
    else:
        print("Download failed.")
        exit()

try:
    # Check if the file exists locally first
    if not os.path.exists(data_file_path):
        download_shakespeare_data()

    with open(data_file_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    print(f"Loaded {len(text_data):,} characters from {data_file_path}.")
except FileNotFoundError:
    print(f"Error: {data_file_path} not found after attempted download. Exiting.")
    exit() # Exit the script if data loading fails even after trying to download


# Convert the string to raw bytes
byte_data = text_data.encode("utf-8")
print(f"Total bytes: {len(byte_data):,}")

# Apply pre-tokenization
chunks = pre_tokenize(text_data)
byte_chunks = [chunk.encode('utf-8') for chunk in chunks]
# Re-join the byte chunks to get a single byte sequence for initial BPE tokens
byte_data_pretokenized = b''.join(byte_chunks)


# --- BPE Training ---

# Initialize vocab and merges
# Start with raw bytes from the pre-tokenized data
tokens = list(byte_data_pretokenized)
vocab = {i: bytes([i]) for i in range(256)} # Initial vocab with byte tokens
merges = {}
next_token_id = 256 # First new token ID
target_vocab_size = 50257 # Final vocab size (example, can be adjusted)

print(f"Starting BPE training with target vocabulary size: {target_vocab_size:,}")

# Initialize pair frequencies once from the initial tokens
pair_freqs = Counter()
for i in range(len(tokens) - 1):
    pair = (tokens[i], tokens[i+1])
    pair_freqs[pair] += 1

merges_this_run = 0 # Track merges in this training session

while len(vocab) < target_vocab_size:
    # Step 2: Find most frequent pair
    # Find the most frequent pair that *can* be merged (has frequency > 1)
    best_pair = None
    max_freq = -1
    # Find the most frequent pair that has a frequency > 1
    # Iterating through pair_freqs.most_common(1) is efficient
    most_common_items = pair_freqs.most_common(1)
    if not most_common_items:
         print("No more pairs to merge.")
         break

    best_pair, max_freq = most_common_items[0]

    if max_freq < 2: # Stop if the most frequent pair only appears once
        print("Most frequent pair has frequency less than 2. Stopping merges.")
        break


    b1, b2 = best_pair

    # Step 3: Merge and assign new token ID
    # Check if this pair has already been merged in a previous run (if loading from saved state)
    # In this script, we assume we are training from scratch or continuing from a saved state
    # The logic below assumes we are assigning a *new* token ID for each merge in this run
    # If you were loading and continuing training, you'd check if best_pair is in 'merges'
    # and use the existing new_id if it is. For this script conversion, we'll train from scratch.

    new_token = next_token_id
    vocab[new_token] = vocab[b1] + vocab[b2]
    merges[best_pair] = new_token # Store the merge rule
    next_token_id += 1
    merges_this_run += 1


    # Step 4: Replace in token stream and update pair frequencies incrementally
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
            # Merging (b1, b2) into new_token
            new_tokens.append(new_token)

            # Update pair frequencies around the merge
            # Decrement count of the merged pair (b1, b2)
            pair_freqs[best_pair] -= 1
            if pair_freqs[best_pair] == 0:
                 del pair_freqs[best_pair]

            # Check pairs before and after the merged pair in the original tokens
            if i > 0:
                prev_token = tokens[i-1]
                # Decrement count of (prev_token, b1)
                pair_freqs[(prev_token, b1)] -= 1
                if pair_freqs[(prev_token, b1)] == 0:
                    del pair_freqs[(prev_token, b1)]
                # Increment count of the new pair (prev_token, new_token)
                pair_freqs[(prev_token, new_token)] += 1

            if i < len(tokens) - 2:
                next_token = tokens[i+2]
                # Decrement count of (b2, next_token)
                pair_freqs[(b2, next_token)] -= 1
                if pair_freqs[(b2, next_token)] == 0:
                    del pair_freqs[(b2, next_token)]
                # Increment count of the new pair (new_token, next_token)
                pair_freqs[(new_token, next_token)] += 1

            i += 2 # Skip both tokens that were merged
        else:
            new_tokens.append(tokens[i])
            i += 1

    tokens = new_tokens # Update the tokens list for the next iteration

    if len(vocab) % 1000 == 0:
        print(f"Vocab size: {len(vocab):,} | Token count: {len(tokens):,}")

print(f"Final vocab size after training: {len(vocab):,}")
print(f"Final token count after training: {len(tokens):,}")
print(f"Number of new merges added in this run: {merges_this_run:,}")


# --- Save the Vocabulary and Merges using pickle ---

tokenizer_output_file = "tokenizer.pkl"
try:
    with open(tokenizer_output_file, "wb") as f:
        pickle.dump((vocab, merges), f)
    print(f"Tokenizer saved to {tokenizer_output_file}")
except Exception as e:
    print(f"An error occurred while saving the tokenizer: {e}")

# You might want to add functions for encoding and decoding here as well,
# so the script can be imported and used as a tokenizer module.
# For example:

# def encode(text, vocab, merges, special_tokens=None):
#     # ... (your encode function code)
#     pass # Replace with actual encode logic

# def decode(tokens, vocab):
#     # ... (your decode function code)
#     pass # Replace with actual decode logic

