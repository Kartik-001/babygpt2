import torch
import tiktoken
import logging
from .model_architecture import GPT
from .model_architecture import GPTConfig

"""
Model loading and text generation utilities for BabyGPT-2.
"""

# 1) Prepare device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2) Load tokenizer
enc = tiktoken.get_encoding("gpt2")


# 3) Load model configuration & weights
cfg = GPTConfig(
    vocab_size=enc.n_vocab,
    block_size=128,
    d_model=256,
    n_heads=4,
    n_layers=4,
    dropout=0.1
)
model = GPT(cfg).to(device)
model_state_path = "C:\\Users\\Asus\\Learning\\Project\\AI\\baby gpt-2\\babygpt-api\\babygpt2_model_final.pt"

# Set up logging
logging.basicConfig(level=logging.INFO)

try:
    model.load_state_dict(torch.load(model_state_path, map_location=device))
    logging.info(f"Model state dictionary loaded successfully from {model_state_path}")
except FileNotFoundError:
    logging.error(f"Error: Model state file not found at {model_state_path}. Please check the path.")
except Exception as e:
    logging.error(f"An error occurred while loading the model state dictionary: {e}")

model.eval()


# 4) Generation function
@torch.no_grad()
def generate_response(prompt: str, max_new_tokens: int = 50, eos_token: int = None) -> str:
    """
    Generate a response from the model given a prompt.

    Args:
        prompt (str): The input prompt string.
        max_new_tokens (int): Maximum number of tokens to generate.
        eos_token (int, optional): End-of-sequence token id. If provided, generation will stop when this token is produced.

    Returns:
        str: The generated text.
    """
    ids = enc.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(input_ids[:, -cfg.block_size :])
        next_id = torch.argmax(logits[0, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, next_id.unsqueeze(0)), dim=1)
        if eos_token is not None and next_id.item() == eos_token:
            break
    out_ids = input_ids[0].tolist()
    return enc.decode(out_ids)