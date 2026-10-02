# PDF AI Assistant

A Streamlit application for uploading documents, searching their contents, and asking questions using a Hugging Face language model.

## Supported Files

- PDF
- DOCX
- TXT
- Markdown
- CSV
- XLSX

## Setup

Create and activate a virtual environment on Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create a `.env` file in the project directory:

```dotenv
HF_TOKEN=your_huggingface_token
```

Never commit `.env` or share the token. If a token has been exposed, revoke it and create a replacement in Hugging Face settings.

## Run

```powershell
python -m streamlit run app.py
```

Open the local URL shown by Streamlit, upload one or more documents, click **Process Documents**, and then ask questions in the chat box.

## Project Files

- `app.py` - Streamlit application and document-processing logic
- `htmltemplate.py` - Chat layout and styling
- `requirements.txt` - Python dependencies
- `.env` - Local Hugging Face token configuration
