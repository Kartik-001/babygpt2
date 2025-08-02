# 🍼 BabyGPT2
🧪 Try it live on [Streamlit](https://babygpt2.streamlit.app/)
An end-to-end "ChatGPT-style" demo project built from scratch:

- 🧠 **Train & experiment** with a miniature GPT-2 model (`BabyGPT`) using Colab
- ⚙️ **Serve** the trained model via a FastAPI + Docker API
- 💬 **Interact** with it through a modern Next.js web chat UI

---

## 🚀 Repository Structure

```

baby-gpt2/
├── babygpt2/
│   └── baby gpt-2.ipynb           # Notebook to train and test the model
│
├── babygpt2-api/
│   ├── app/
│   │   ├── main.py                # FastAPI app with CORS and health check
│   │   ├── model.py               # Model loading and response generation
│   │   └── model\_architecture.py  # GPT2Model class definition
│   ├── babygpt2\_model\_final.pt    # Trained model weights (PyTorch)
│   ├── Dockerfile                 # Docker build file for the API
│   └── requirements.txt           # fastapi, uvicorn, torch, tiktoken
│
├── babygpt2-web/
│   ├── pages/
│   │   └── index.js               # Frontend UI with prompt + axios calls
│   ├── package.json               # Next.js, React, and frontend deps
│   └── ...                        # Other frontend assets and config
│
└── README.md                      # This file

````

---

## 📦 What's Inside

### 1. `babygpt2/`
- A Jupyter notebook (`.ipynb`) for training the BabyGPT model from scratch or fine-tuning on custom data.

### 2. `babygpt2-api/`
- FastAPI server that loads and serves the trained model.
- Accepts prompts via `/generate` and returns model responses.
- Docker-ready for deployment on services like Render, Railway, or Heroku.

### 3. `babygpt2-web/`
- A modern web-based chat interface built with Next.js and React.
- Connects to the API using `axios` to send user prompts and render model responses.
- Can be deployed using Vercel, Netlify, or any frontend host.

---

## 🛠️ Quickstart

### 1. Run the API Locally

```bash
cd babygpt2-api
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
````

* Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* Health Check: [http://localhost:8000/](http://localhost:8000/)
* Test Generation:

```json
POST /generate
{
  "prompt": "Once upon a time",
  "max_new_tokens": 50
}
```

---

### 2. Dockerize & Deploy the API

```bash
cd babygpt2-api

# Build Docker image
docker build -t yourhubuser/babygpt2-api:latest .

# Run locally
docker run --rm -d -p 8000:8000 babygpt2-api:latest

# Push to DockerHub (optional)
docker tag babygpt2-api:latest yourhubuser/babygpt2-api:latest
docker push yourhubuser/babygpt2-api:latest
```

You can then deploy this container on:

* 🔁 [Render.com](https://render.com) (Free container service)
* 📦 Railway, Heroku, Fly.io, etc.

---

### 3. Run the Frontend Locally

```bash
cd babygpt2-web
npm install
npm run dev
```

Visit: [http://localhost:3000](http://localhost:3000)

---

### 4. Deploy the Web UI

```bash
cd babygpt2-web
vercel login
vercel --prod
```

You can also use Netlify, Surge, or any static frontend hosting.

---

## 🌐 Custom Domains & HTTPS

These steps are for **users deploying their own version**:

* **API**: Add `api.yourdomain.com` to Render, point DNS, and enable auto‑SSL
* **Web UI**: Add `yourdomain.com` to Vercel/Netlify and connect DNS

---

## 🤝 Contributing

Feel free to contribute and improve the project!

1. Fork [this repository](https://github.com/Kartik-001/babygpt2)
2. Create a feature branch:

   ```bash
   git checkout -b feat/your-feature
   ```
3. Commit and push:

   ```bash
   git push origin feat/your-feature
   ```
4. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — meaning you're free to use, copy, modify, and distribute it for personal or commercial use. Just don’t hold me liable if it breaks 😉

---

## 🙌 Acknowledgements

Inspired by:

* GPT-2 architecture from OpenAI
* FastAPI + Docker for backend deployment
* Next.js for rapid frontend UI development

---
