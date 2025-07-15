# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .model import generate_response

app = FastAPI(title="BabyGPT API")

# ── Add this block ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://babygpt2.vercel.app"],           # or ["https://babygpt2.vercel.app"] to lock down
    allow_methods=["*"],           # GET, POST, etc.
    allow_headers=["*"],           # Content‑Type, Authorization, etc.
)
# ────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "ok"}

class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50

class GenResponse(BaseModel):
    text: str

@app.post("/generate", response_model=GenResponse)
def generate(req: GenRequest):
    output = generate_response(req.prompt, req.max_new_tokens)
    return GenResponse(text=output)
