```markdown
# BabyGPT2

An end‑to‑end “ChatGPT‑style” demo:  
- **Train & explore** a miniature GPT‑2 “BabyGPT” in Colab  
- **Serve** the trained model via a FastAPI + Docker API  
- **Interact** through a Next.js web chat UI  

---

## 🚀 Repository Structure

```

baby gpt2/
├── babygpt2/
│   └── baby gpt-2.ipynb             ← Notebook for training & experimentation
│
├── babygpt2-api/
│   ├── app/
│   │   ├── main.py                  ← FastAPI app (+ CORS & health‑check)
│   │   ├── model.py                 ← load model & `generate_response()`
│   │   └── model\_architecture.py    ← GPT2Model definition
│   │
│   ├── babygpt2\_model\_final.pt      ← Saved PyTorch model weights
│   ├── Dockerfile                   ← Build instructions for the API
│   └── requirements.txt             ← fastapi, uvicorn, torch, tiktoken
│
├── babygpt2-web/
│   ├── pages/
│   │   └── index.js                 ← Next.js chat UI (axios → `/generate`)
│   ├── package.json                 ← next, react, axios, etc.
│   └── …                            ← other Next.js config (public/, styles/, etc.)
│
└── README.md                        ← You are here

````

---

## 📦 What’s Inside

1. **`babygpt2/`**  
   - A Colab notebook to **train** or fine‑tune your BabyGPT model on a small dataset.

2. **`babygpt2-api/`**  
   - **Inference server** powered by FastAPI.  
   - **Tokenizes** inputs with `tiktoken.get_encoding("gpt2")`.  
   - **Generates** text with your saved `babygpt2_model_final.pt`.  
   - **Dockerized** for easy deployment.

3. **`babygpt2-web/`**  
   - **Next.js** front‑end that renders a chat box.  
   - **Calls** your `/generate` API endpoint and displays responses.  
   - Ready to deploy on Vercel, Netlify, or any static host.

---

## 🛠️ Quickstart

### 1. Run the API Locally

```bash
cd babygpt2-api
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
````

* **Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health‑check**: [http://localhost:8000/](http://localhost:8000/)
* **Generate**: `POST /generate` with JSON

  ```json
  { "prompt": "Hello, world!", "max_new_tokens": 50 }
  ```

---

### 2. Dockerize & Deploy

```bash
cd babygpt2-api
# Build
docker build -t yourhubuser/babygpt2-api:latest .
# Run locally
docker run --rm -d -p 8000:8000 babygpt2-api:latest
# Push to Docker Hub
docker tag babygpt2-api:latest yourhubuser/babygpt2-api:latest
docker push yourhubuser/babygpt2-api:latest
```

Then connect `yourhubuser/babygpt2-api:latest` to **Render.com** (free Docker service) or any container host.

---

### 3. Run the Front‑End

```bash
cd babygpt2-web
npm install
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000), type a prompt, and click **Generate** to see BabyGPT’s reply.

To deploy, use Vercel:

```bash
cd babygpt2-web
vercel login
vercel --prod
```

---

## 🌐 Custom Domains & HTTPS

* **API**: add `api.yourdomain.com` in Render → set CNAME → auto‑SSL
* **UI**: add `yourdomain.com` in Vercel → set CNAME → auto‑SSL

---

## 🤝 Contributing

1. Fork this repo
2. Create a branch: `git checkout -b feat/your-feature`
3. Commit & push: `git push origin feat/your-feature`
4. Open a Pull Request

---

## 📄 License

This project is released under the **MIT License**. Feel free to use, modify, and extend!

---
