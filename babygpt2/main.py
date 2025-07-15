import requests

# Download the tiny Shakespeare dataset if not already present
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
filename = "tiny_shakespeare.txt"

try:
    with open(filename, "rb"):
        print("Dataset already exists.")
except FileNotFoundError:
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print("Download successful.")
    else:
        print("Download failed.")

# ...existing code...