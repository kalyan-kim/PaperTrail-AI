import os
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from docx import Document as DocxDocument

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from groq import Groq
from openai import OpenAI
from htmltemplate import load_css

# Load environment variables from .env
load_dotenv(override=True)

# Page configuration
st.set_page_config(
    page_title="RAG AI Assistant | Multi-Document Chat",
    page_icon="📚",
    layout="wide"
)

# Apply custom CSS
load_css()


# ==============================================================================
# SECRETS & ENVIRONMENT RESOLVER
# ==============================================================================
def get_secret(key_name: str, default: str = "") -> str:
    """
    Retrieves secret from st.secrets (Streamlit Cloud) if present,
    otherwise falls back to os.getenv / .env file (Local).
    """
    try:
        if hasattr(st, "secrets") and key_name in st.secrets:
            return str(st.secrets[key_name]).strip()
    except Exception:
        pass
    return os.getenv(key_name, default).strip()


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files_summary" not in st.session_state:
    st.session_state.processed_files_summary = {}

if "total_chunks_count" not in st.session_state:
    st.session_state.total_chunks_count = 0


# ==============================================================================
# DOCUMENT PROCESSING & CHUNKING
# ==============================================================================
def clean_text(text: str) -> str:
    """Normalize whitespace and strip unnecessary characters."""
    if not text:
        return ""
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def extract_documents_from_files(files):
    """
    Extracts text from multiple file types (PDF, DOCX, TXT, MD, CSV, XLSX)
    and packages each segment into a LangChain Document with rich metadata.
    """
    documents = []
    file_summaries = {}

    for file in files:
        file_name = file.name
        file_name_lower = file_name.lower()
        file_docs = []

        try:
            # 1. PDF
            if file_name_lower.endswith(".pdf"):
                pdf_reader = PdfReader(file)
                total_pages = len(pdf_reader.pages)
                for page_idx, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        file_docs.append(Document(
                            page_content=clean_text(page_text),
                            metadata={
                                "source": file_name,
                                "page": page_idx + 1,
                                "total_pages": total_pages,
                                "type": "pdf"
                            }
                        ))

            # 2. DOCX
            elif file_name_lower.endswith(".docx"):
                doc = DocxDocument(file)
                body_parts = []
                for p in doc.paragraphs:
                    if p.text.strip():
                        body_parts.append(p.text.strip())

                for table in doc.tables:
                    for row in table.rows:
                        row_vals = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if row_vals:
                            body_parts.append(" | ".join(row_vals))

                full_docx = "\n\n".join(body_parts)
                if full_docx.strip():
                    file_docs.append(Document(
                        page_content=clean_text(full_docx),
                        metadata={
                            "source": file_name,
                            "page": 1,
                            "type": "docx"
                        }
                    ))

            # 3. Plain Text / Markdown
            elif file_name_lower.endswith((".txt", ".md")):
                text_content = file.getvalue().decode("utf-8", errors="ignore")
                if text_content.strip():
                    file_docs.append(Document(
                        page_content=clean_text(text_content),
                        metadata={
                            "source": file_name,
                            "page": 1,
                            "type": "text"
                        }
                    ))

            # 4. CSV
            elif file_name_lower.endswith(".csv"):
                df = pd.read_csv(file)
                csv_str = df.to_string(index=False)
                if csv_str.strip():
                    file_docs.append(Document(
                        page_content=clean_text(csv_str),
                        metadata={
                            "source": file_name,
                            "page": 1,
                            "rows": len(df),
                            "type": "csv"
                        }
                    ))

            # 5. Excel
            elif file_name_lower.endswith(".xlsx"):
                excel_file = pd.ExcelFile(file)
                for sheet in excel_file.sheet_names:
                    df = pd.read_excel(file, sheet_name=sheet)
                    sheet_str = df.to_string(index=False)
                    if sheet_str.strip():
                        file_docs.append(Document(
                            page_content=clean_text(f"Sheet: {sheet}\n{sheet_str}"),
                            metadata={
                                "source": file_name,
                                "sheet": sheet,
                                "page": 1,
                                "type": "xlsx"
                            }
                        ))

            file_summaries[file_name] = {
                "sections_extracted": len(file_docs),
                "status": "Success" if file_docs else "No readable text found"
            }
            documents.extend(file_docs)

        except Exception as e:
            file_summaries[file_name] = {
                "sections_extracted": 0,
                "status": f"Error: {e}"
            }

    return documents, file_summaries


def create_chunks(documents, chunk_size=800, chunk_overlap=150):
    """
    Splits documents into overlapping chunks with preserved metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    raw_chunks = text_splitter.split_documents(documents)
    valid_chunks = []

    for idx, chunk in enumerate(raw_chunks):
        cleaned_content = chunk.page_content.strip()
        if len(cleaned_content) >= 15:
            chunk.page_content = cleaned_content
            chunk.metadata["chunk_id"] = idx + 1
            valid_chunks.append(chunk)

    return valid_chunks


# ==============================================================================
# EMBEDDINGS & VECTOR STORE
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """
    Loads sentence-transformers embedding model with offline cache preference.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings


def build_vectorstore(chunks):
    """Creates a FAISS index from document chunks."""
    embeddings = get_embedding_model()
    return FAISS.from_documents(chunks, embeddings)


# ==============================================================================
# DIRECT EXTRACTIVE RAG SYNTHESIZER (FALLBACK)
# ==============================================================================
def synthesize_extractive_answer(question: str, retrieved_docs: list) -> str:
    """
    Synthesizes an informative, structured answer directly from retrieved chunks
    without requiring any external LLM API key.
    """
    if not retrieved_docs:
        return "I could not find any relevant information in the uploaded documents for this query."

    stopwords = {
        "what", "is", "the", "a", "an", "in", "on", "of", "and", "to", "for",
        "with", "about", "how", "why", "can", "you", "tell", "me", "are", "do",
        "does", "did", "from", "that", "this", "these", "those"
    }
    q_words = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_]+\b', question) if w.lower() not in stopwords]

    scored_sentences = []
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Document")
        page = doc.metadata.get("page", 1)

        sentences = re.split(r'(?<=[.!?])\s+', doc.page_content)
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 20:
                continue
            s_words = set(re.findall(r'\b[a-zA-Z0-9_]+\b', s_clean.lower()))
            overlap = sum(1 for w in q_words if w in s_words)
            if overlap > 0:
                scored_sentences.append((overlap, s_clean, source, page))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    lines = ["### 📄 Direct Document Retrieval Findings\n"]

    if scored_sentences:
        seen = set()
        count = 0
        lines.append("Here are the most relevant findings extracted from your documents:\n")
        for score, s, src, pg in scored_sentences:
            if s.lower() not in seen:
                seen.add(s.lower())
                lines.append(f"- **{s}**\n  *Source: `{src}` (Page {pg})*")
                count += 1
            if count >= 4:
                break
    else:
        top_doc = retrieved_docs[0]
        lines.append(
            f"Here is the most relevant section found in **`{top_doc.metadata.get('source', 'Document')}`**:\n\n"
            f"> {top_doc.page_content[:450]}..."
        )

    return "\n\n".join(lines)


# ==============================================================================
# LLM QUESTION ANSWERING (GROQ & HUGGING FACE ONLY)
# ==============================================================================
PROVIDER_MODELS = {
    "Groq": [
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "allam-2-7b"
    ],
    "Hugging Face": [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "deepseek-ai/DeepSeek-V4.1-Flash"
    ]
}


def generate_rag_answer(question: str, retrieved_docs_with_scores: list, provider: str, model_name: str, temperature: float = 0.2):
    """
    Generates an answer using either Groq or Hugging Face.
    API keys are automatically resolved from st.secrets or environment variables.
    """
    retrieved_docs = [doc for doc, score in retrieved_docs_with_scores]

    # Resolve API Key
    if provider == "Groq":
        api_key = get_secret("GROQ_API_KEY")
        if not api_key:
            warning = (
                "⚠️ **GROQ_API_KEY is not set.** Please configure `GROQ_API_KEY` in `.env` "
                "(for local run) or in Streamlit Cloud Secrets (for deployed app).\n\n"
                "**Falling back to Direct Extractive Synthesis below:**"
            )
            return f"{warning}\n\n{synthesize_extractive_answer(question, retrieved_docs)}"
    else:  # Hugging Face
        api_key = get_secret("HF_TOKEN")
        if not api_key:
            warning = (
                "⚠️ **HF_TOKEN is not set.** Please configure `HF_TOKEN` in `.env` "
                "(for local run) or in Streamlit Cloud Secrets (for deployed app).\n\n"
                "**Falling back to Direct Extractive Synthesis below:**"
            )
            return f"{warning}\n\n{synthesize_extractive_answer(question, retrieved_docs)}"

    # Build Context with Source Citation Headers
    context_blocks = []
    for idx, (doc, score) in enumerate(retrieved_docs_with_scores, 1):
        source = doc.metadata.get("source", "Document")
        page = doc.metadata.get("page", 1)
        content = doc.page_content
        context_blocks.append(
            f"--- [Document: {source} | Page: {page} | Chunk #{idx}] ---\n{content}"
        )
    formatted_context = "\n\n".join(context_blocks)

    # Format Chat History (last 6 turns)
    history_messages = []
    for msg in st.session_state.messages[-6:-1]:
        role = "user" if msg["role"] == "user" else "assistant"
        history_messages.append({"role": role, "content": msg["content"]})

    system_instruction = (
        "You are an expert AI Document Assistant. Your goal is to answer the user's questions "
        "accurately and concisely using ONLY the provided document context.\n\n"
        "RULES:\n"
        "1. Strictly base your answer on the provided document excerpts.\n"
        "2. Always cite the source document name and page number when referencing facts (e.g., '[Source: file.pdf, Page 1]').\n"
        "3. If the answer cannot be found in the context, state clearly: "
        "'I could not find information about that in the uploaded documents.'\n"
        "4. Use bullet points and clear headings for readability.\n"
        "5. Do not make up facts or extrapolate beyond what is given."
    )

    user_prompt = (
        f"DOCUMENT CONTEXT:\n{formatted_context}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        "Please provide a comprehensive, well-structured answer citing sources from the context above."
    )

    messages = [{"role": "system", "content": system_instruction}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_prompt})

    try:
        if provider == "Groq":
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=600
            )
        else:
            client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=api_key
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=600
            )

        answer = response.choices[0].message.content
        if not answer or not answer.strip():
            raise ValueError("LLM returned an empty response.")

        return answer

    except Exception as e:
        err_str = str(e)
        notice = (
            f"⚠️ **{provider} Generation Notice**:\n"
            f"`{err_str[:250]}`\n\n"
            f"*Tip: If using Hugging Face, Inference Providers may require pre-paid credits. "
            f"You can switch to **Groq** in the sidebar for fast free generation.*\n\n"
            f"**Fell back to Direct Extractive Synthesis below:**"
        )
        extractive = synthesize_extractive_answer(question, retrieved_docs)
        return f"{notice}\n\n{extractive}"


# ==============================================================================
# MAIN APPLICATION INTERFACE
# ==============================================================================
def main():
    # Header Area
    st.markdown('<div class="app-title">📚 RAG Document AI Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">'
        'Upload your documents, inspect chunking, and chat with AI grounded in your files'
        '</div>',
        unsafe_allow_html=True
    )

    # ==========================================================================
    # SIDEBAR CONFIGURATION
    # ==========================================================================
    with st.sidebar:
        st.header("⚙️ Configuration")

        # ----------------------------------------------------------------------
        # Model Selection (Only Groq and Hugging Face)
        # ----------------------------------------------------------------------
        st.subheader("🤖 AI Model")

        selected_provider = st.radio(
            "Select Provider",
            options=["Groq", "Hugging Face"],
            index=0,
            help="Groq provides ultra-fast inference with your configured API key."
        )

        available_models = PROVIDER_MODELS[selected_provider]
        selected_model = st.selectbox(
            "Model",
            options=available_models,
            index=0
        )

        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        st.divider()

        # ----------------------------------------------------------------------
        # Document Upload & Chunking Parameters
        # ----------------------------------------------------------------------
        st.subheader("📁 Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload files to ingest",
            type=["pdf", "docx", "txt", "md", "csv", "xlsx"],
            accept_multiple_files=True,
            help="Supports PDF, DOCX, TXT, MD, CSV, XLSX"
        )

        with st.expander("🛠️ Chunking & Retrieval Settings", expanded=False):
            chunk_size = st.slider("Chunk Size (characters)", min_value=300, max_value=2000, value=800, step=50)
            chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=50, max_value=400, value=150, step=25)
            top_k = st.slider("Top Chunks to Retrieve (k)", min_value=1, max_value=10, value=4, step=1)

        # Process Documents Button
        if st.button("⚡ Process & Index Documents", use_container_width=True, type="primary"):
            if not uploaded_files:
                st.warning("Please upload at least one document first.")
            else:
                with st.spinner("📄 Extracting text and chunking documents..."):
                    try:
                        raw_docs, summaries = extract_documents_from_files(uploaded_files)

                        if not raw_docs:
                            st.error("No readable text found in the uploaded documents.")
                        else:
                            chunks = create_chunks(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                            if not chunks:
                                st.error("No valid chunks were generated from the documents.")
                            else:
                                with st.spinner("🧠 Embedding chunks into FAISS vector database..."):
                                    vectorstore = build_vectorstore(chunks)

                                st.session_state.vectorstore = vectorstore
                                st.session_state.processed_files_summary = summaries
                                st.session_state.total_chunks_count = len(chunks)
                                st.session_state.messages = []

                                st.success(f"Successfully indexed {len(chunks)} chunks across {len(uploaded_files)} files!")

                    except Exception as e:
                        st.error(f"Processing failed: {e}")

        # Summary of Processed Files
        if st.session_state.processed_files_summary:
            st.divider()
            st.subheader("📋 Document Index Status")
            st.metric("Total Chunks in Vectorstore", st.session_state.total_chunks_count)
            for fname, info in st.session_state.processed_files_summary.items():
                status_icon = "✅" if info["status"] == "Success" else "⚠️"
                st.write(f"{status_icon} **{fname}** ({info['sections_extracted']} sections)")

        st.divider()

        # Action Buttons
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        with col_c2:
            if st.button("🔄 Reset All", use_container_width=True):
                st.session_state.vectorstore = None
                st.session_state.processed_files_summary = {}
                st.session_state.total_chunks_count = 0
                st.session_state.messages = []
                st.rerun()

    # ==========================================================================
    # MAIN CONVERSATION AREA
    # ==========================================================================
    # Status Banner
    col_stat1, col_stat2, col_stat3 = st.columns([2, 1, 1])
    with col_stat1:
        if st.session_state.vectorstore:
            st.success(f"🟢 **RAG Ready**: {st.session_state.total_chunks_count} chunks indexed across {len(st.session_state.processed_files_summary)} files.")
        else:
            st.info("ℹ️ Upload and process documents from the sidebar to begin asking questions.")

    with col_stat2:
        st.caption(f"**Provider**: {selected_provider}")
    with col_stat3:
        st.caption(f"**Model**: `{selected_model}`")

    st.divider()

    # Display Chat History with Rich Formatting and Source Accordions
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"🔍 View {len(msg['sources'])} Retrieved Sources & Chunks"):
                    for s_idx, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk #{s_idx}** — *{src['source']}* (Page {src['page']}) | Distance: `{src['score']}`")
                        st.info(src["content"])

    # User Input
    user_question = st.chat_input("Ask a question about your documents...")

    if user_question:
        if st.session_state.vectorstore is None:
            st.warning("⚠️ Please upload and process your documents before asking questions.")
        else:
            # 1. Append User Message
            st.session_state.messages.append({
                "role": "user",
                "content": user_question
            })

            with st.chat_message("user"):
                st.markdown(user_question)

            # 2. Retrieve Relevant Chunks
            with st.spinner("🔎 Searching vectorstore for relevant chunks..."):
                try:
                    retrieved_docs_with_scores = st.session_state.vectorstore.similarity_search_with_score(
                        user_question,
                        k=top_k
                    )
                except Exception as e:
                    st.error(f"Vector search failed: {e}")
                    retrieved_docs_with_scores = []

            # 3. Generate Answer
            with st.chat_message("assistant"):
                with st.spinner(f"🤖 Generating answer with {selected_provider} ({selected_model})..."):
                    if not retrieved_docs_with_scores:
                        answer = "I could not find any matching documents or chunks for your query."
                        sources_list = []
                    else:
                        sources_list = [
                            {
                                "source": doc.metadata.get("source", "Document"),
                                "page": doc.metadata.get("page", 1),
                                "score": round(float(score), 4),
                                "content": doc.page_content
                            }
                            for doc, score in retrieved_docs_with_scores
                        ]

                        answer = generate_rag_answer(
                            question=user_question,
                            retrieved_docs_with_scores=retrieved_docs_with_scores,
                            provider=selected_provider,
                            model_name=selected_model,
                            temperature=temperature
                        )

                    st.markdown(answer)

                    if sources_list:
                        with st.expander(f"🔍 View {len(sources_list)} Retrieved Sources & Chunks"):
                            for s_idx, src in enumerate(sources_list, 1):
                                st.markdown(f"**Chunk #{s_idx}** — *{src['source']}* (Page {src['page']}) | Distance: `{src['score']}`")
                                st.info(src["content"])

            # 4. Save Assistant Message to State
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources_list
            })


if __name__ == "__main__":
    main()
