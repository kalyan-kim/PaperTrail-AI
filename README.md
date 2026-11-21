# 📚 PaperTrail-AI | Multi-Document RAG Assistant

A Streamlit application for uploading documents, chunking and indexing their contents into a vector database, and querying them using **Groq** or **Hugging Face** language models with grounded source citations.

---

## 🌟 Key Features

- **Multi-Format Ingestion**: Supports `.pdf`, `.docx`, `.txt`, `.md`, `.csv`, and `.xlsx`.
- **Intelligent Chunking & Metadata**: Documents are split using recursive character chunking with preserved source filename, page numbers, and chunk IDs.
- **Local FAISS Vector Index**: Uses `sentence-transformers/all-MiniLM-L6-v2` embeddings for fast semantic search.
- **Dual LLM Providers**:
  - **Groq (Default & Recommended)**: Fast inference with `qwen/qwen3.8-27b`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, etc.
  - **Hugging Face**: Connects via Hugging Face Router for models like `meta-llama/Llama-3.1-8B-Instruct`.
- **Source Traceability**: Every answer includes an expandable breakdown showing the exact retrieved chunks, similarity distance, and source pages.
- **Automatic Fallback**: If an API provider encounters a quota or credit limit, the app automatically falls back to extractive synthesis so your RAG workflow never fails.

---

## 🚀 Local Setup & Installation

### 1. Clone the repository and navigate to the project:
```powershell
cd c:\Users\hp\OneDrive\Desktop\rag
```

### 2. Activate the virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install required dependencies:
```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables:
Create or edit `.env` in the root directory:
```dotenv
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

> **Note**: You can generate a free Groq API key instantly at [console.groq.com/keys](https://console.groq.com/keys).

### 5. Run the application:
```powershell
python -m streamlit run app.py
```

---

## ☁️ How to Redeploy / Update on Streamlit Cloud

If your app is already deployed on **Streamlit Community Cloud**, follow these steps to deploy the latest changes:

### Step 1: Commit and Push Your Changes to GitHub
Open your terminal in the project directory and run:

```bash
git add app.py requirements.txt README.md .env
git commit -m "Update RAG app: add Groq support, metadata chunking, and remove API key sidebar"
git push origin main
```

*(Note: Never commit sensitive `.env` files to public repositories. If your repository is public, add `.env` to `.gitignore` and configure secrets via Streamlit Cloud).*

---

### Step 2: Add Secrets to Streamlit Cloud Dashboard
Because the API key input has been removed from the UI for security, your deployed app reads keys directly from **Streamlit Secrets**:

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
2. Find your deployed app in your dashboard and click the **⋮ (three dots)** menu next to it -> select **Settings**.
3. In the left panel, click on **Secrets**.
4. Paste your API keys in the TOML editor:
   ```toml
   GROQ_API_KEY = "gsk_..."
   HF_TOKEN = "hf_..."
   ```
5. Click **Save**.

---

### Step 3: Trigger Redeploy / Reboot
- **Automatic Deployment**: Streamlit Cloud automatically detects the new git push on your `main` branch and begins rebuilding the app.
- **Manual Reboot**: If the update does not reflect immediately:
  1. Open your live app URL.
  2. Click the **Manage app** button in the bottom right corner (or the **⋮** menu in the top right).
  3. Click **Reboot App** or **Clear cache and deploy**.

---

## 📁 Project Structure

- `app.py` — Main Streamlit application, document ingestion, vectorstore indexing, and multi-provider RAG logic.
- `htmltemplate.py` — Custom styling and CSS layout.
- `requirements.txt` — Python package dependencies (including `groq`, `openai`, `langchain`, `faiss-cpu`, etc.).
- `.env` — Local environment variables for Groq and Hugging Face keys.
- `man.py` — Model and provider diagnostics utility script.
