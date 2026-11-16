import os
import streamlit as st

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage

from htmltemplate import load_css, display_chat_history


load_dotenv()

st.set_page_config(
    page_title="Chat With Multiple PDFs",
    page_icon="📚",
    layout="wide"
)

load_css()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

def get_text_from_files(files):
    raw_text = ""

    for file in files:
        file_name = file.name.lower()

        if file_name.endswith(".pdf"):
            pdf_reader = PdfReader(file)

            for page in pdf_reader.pages:
                text = page.extract_text()

                if text:
                    raw_text += text + "\n"

        elif file_name.endswith(".docx"):
            document = Document(file)

            for paragraph in document.paragraphs:
                if paragraph.text.strip():
                    raw_text += paragraph.text + "\n"

            for table in document.tables:
                for row in table.rows:
                    row_text = " | ".join(
                        cell.text for cell in row.cells
                    )
                    raw_text += row_text + "\n"

        elif file_name.endswith(".txt") or file_name.endswith(".md"):
            text = file.getvalue().decode("utf-8", errors="ignore")
            raw_text += text + "\n"

        elif file_name.endswith(".csv"):
            dataframe = pd.read_csv(file)
            raw_text += dataframe.to_string(index=False) + "\n"

        elif file_name.endswith(".xlsx"):
            dataframe = pd.read_excel(file)
            raw_text += dataframe.to_string(index=False) + "\n"

    return raw_text

def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    text_chunks = text_splitter.split_text(raw_text)

    return text_chunks


@st.cache_resource
def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return embeddings


def get_vectorstore(text_chunks):
    embeddings = get_embedding_model()

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore


def get_llm():
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            "HF_TOKEN is missing. Add HF_TOKEN=your_token to a .env file."
        )

    endpoint = HuggingFaceEndpoint(
       repo_id="Qwen/Qwen3-8B-GGUF",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.2,
        huggingfacehub_api_token=hf_token
    )

    llm = ChatHuggingFace(llm=endpoint)

    return llm


def get_answer(question):
    vectorstore = st.session_state.vectorstore

    documents = vectorstore.similarity_search(
        question,
        k=4
    )

    context = "\n\n".join(
        document.page_content for document in documents
    )

    history = ""

    for message in st.session_state.messages[-8:]:
        role = message["role"].upper()
        history += f"{role}: {message['content']}\n"

    prompt = f"""
You are a helpful PDF assistant.

Answer the user's question using the provided PDF context.

Rules:
1. Give accurate and clear answers.
2. Do not invent information that is not in the context.
3. If the answer is not available in the PDFs, say so.
4. Use previous conversation only when it helps answer the current question.
5. Format answers clearly.

Previous conversation:
{history}

PDF context:
{context}

User question:
{question}
"""

    llm = get_llm()

    response = llm.invoke([
        SystemMessage(
            content="You are a helpful document question-answering assistant."
        ),
        HumanMessage(content=prompt)
    ])

    return response.content


def main():
    st.markdown(
        '<div class="app-title">📚 PDF AI Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="app-subtitle">'
        'Upload your documents and chat with them using AI'
        '</div>',
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.title("📁 Your Documents")

        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=["pdf", "docx", "txt", "md", "csv", "xlsx"],
            accept_multiple_files=True
        )

        if st.button("⚡ Process Documents", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload at least one document.")
            else:
                with st.spinner("Processing documents..."):
                    try:
                        raw_text = get_text_from_files(uploaded_files)

                        if not raw_text.strip():
                            st.error(
                                "No readable text found. "
                                "Your documents may be scanned images."
                            )
                        else:
                            text_chunks = get_text_chunks(raw_text)

                            if not text_chunks:
                                st.error("No text chunks generated.")
                            else:
                                vectorstore = get_vectorstore(text_chunks)
                                st.session_state.vectorstore = vectorstore
                                st.session_state.processed_files = [
                                    file.name for file in uploaded_files
                                ]
                                st.session_state.messages = []
                                st.success("Documents processed successfully!")
                                st.write("Total chunks:", len(text_chunks))

                    except Exception as e:
                        st.error(f"Processing failed: {e}")

        st.divider()

        st.subheader("📄 Processed Files")

        if st.session_state.processed_files:
            for filename in st.session_state.processed_files:
                st.write(f"📕 {filename}")
        else:
            st.caption("No documents processed yet.")

        st.divider()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("🔄 Reset Documents", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.processed_files = []
            st.session_state.messages = []
            st.rerun()

    if not st.session_state.messages:
        st.info(
            "👋 Welcome! Upload your PDFs from the sidebar, "
            "process them, and ask your first question."
        )

    display_chat_history(st.session_state.messages)

    user_question = st.chat_input("Ask anything about your PDFs...")

    if user_question:
        if st.session_state.vectorstore is None:
            st.warning("Please upload and process your PDFs first.")

        else:
            st.session_state.messages.append({
                "role": "user",
                "content": user_question
            })

            with st.spinner("🤖 Generating answer..."):
                try:
                    answer = get_answer(user_question)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Sorry, I encountered an error: {e}"
                    })

            st.rerun()


if __name__ == "__main__":
    main()
