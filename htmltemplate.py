import streamlit as st
from html import escape


def load_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f7f8fc;
    }

    .chat-container {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin: 15px 0;
        width: 100%;
    }

    .bot-container {
        justify-content: flex-start;
    }

    .user-container {
        justify-content: flex-end;
    }

    .chat-avatar {
        width: 38px;
        height: 38px;
        min-width: 38px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }

    .bot-avatar {
        background: #e0e7ff;
    }

    .user-avatar {
        background: #d1fae5;
    }

    .chat-bubble {
        padding: 14px 18px;
        border-radius: 16px;
        max-width: 80%;
        line-height: 1.65;
        font-size: 15px;
        overflow-wrap: anywhere;
        white-space: pre-wrap;
    }

    .bot-bubble {
        background: white;
        color: #202124;
        border: 1px solid #e5e7eb;
        border-top-left-radius: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .user-bubble {
        background: #4f46e5;
        color: white;
        border-top-right-radius: 4px;
    }

    .app-title {
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        color: #4f46e5;
        margin-bottom: 5px;
    }

    .app-subtitle {
        text-align: center;
        color: #777;
        margin-bottom: 25px;
    }

    </style>
    """, unsafe_allow_html=True)


def bot_template(message):
    safe_message = escape(str(message))

    return f"""
    <div class="chat-container bot-container">
        <div class="chat-avatar bot-avatar">🤖</div>
        <div class="chat-bubble bot-bubble">{safe_message}</div>
    </div>
    """


def user_template(message):
    safe_message = escape(str(message))

    return f"""
    <div class="chat-container user-container">
        <div class="chat-bubble user-bubble">{safe_message}</div>
        <div class="chat-avatar user-avatar">👤</div>
    </div>
    """


def display_chat_history(messages):
    for message in messages:
        if message["role"] == "user":
            st.markdown(
                user_template(message["content"]),
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                bot_template(message["content"]),
                unsafe_allow_html=True
            )