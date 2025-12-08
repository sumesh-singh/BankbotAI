import streamlit as st
import pandas as pd
import requests
import time
import random
import sqlite3
import json
import re
from datetime import datetime

st.set_page_config(
    page_title="BankBot AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #64748B; text-align: center; margin-bottom: 2rem; }
    /* Style for the sidebar history buttons */
    .stButton button {
        text-align: left;
        padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  title TEXT, 
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  session_id INTEGER, 
                  role TEXT, 
                  content TEXT, 
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(session_id) REFERENCES sessions(id))''')
    conn.commit()
    conn.close()

def create_session(first_message):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    title = first_message[:30] + "..." if len(first_message) > 30 else first_message
    c.execute("INSERT INTO sessions (title) VALUES (?)", (title,))
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return session_id

def save_message(session_id, role, content):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    if isinstance(content, dict):
        content = json.dumps(content)
    c.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", 
              (session_id, role, content))
    conn.commit()
    conn.close()

def load_messages(session_id):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp", (session_id,))
    rows = c.fetchall()
    conn.close()
    
    messages = []
    for role, content_str in rows:
        try:
            content = json.loads(content_str)
        except (json.JSONDecodeError, TypeError):
            content = content_str
        messages.append({"role": role, "content": content})
    return messages

def get_all_sessions():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("SELECT id, title FROM sessions ORDER BY id DESC") 
    sessions = c.fetchall()
    conn.close()
    return sessions

def delete_session(session_id):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()

init_db()

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def query_ollama(prompt, model="llama3:latest"):
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            return f"Ollama Error ({response.status_code}): {response.text}"
        return response.json().get("response", "Error: No response from Llama 3.")
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

def get_bank_response(user_query):
    user_query = user_query.lower()
    if "credit card" in user_query or "compare" in user_query:
        # NOTE: We use simple dictionaries that can be serialized to JSON
        return {
            "type": "table",
            "text": "Here is a comparison of our top credit card options:",
            "data": {
                "Card Name": ["Silver Checking", "Gold Travel", "Platinum Business"],
                "APR %": ["14.5%", "18.2%", "12.0%"],
                "Fee": ["$0", "$95", "$250"]
            }
        }
    elif "balance" in user_query:
        return {"type": "text", "text": "Your current balance is **$4,250.00**."}
    else:
        return {"type": "text", "text": query_ollama(user_query)}

def stream_text(text):
    # Split by whitespace but keep the delimiters to preserve formatting
    tokens = re.split(r'(\s+)', text)
    for token in tokens:
        yield token
        time.sleep(0.02)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2534/2534204.png", width=50)
    st.title("MyBank Services")
    
    if st.button("+ New Chat", use_container_width=True, type="primary"):
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("### Recent Chats")
    
    sessions = get_all_sessions()
    
    for sess_id, title in sessions:
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(f"💬 {title}", key=f"sess_{sess_id}", use_container_width=True):
                st.session_state.current_session_id = sess_id
                st.session_state.messages = load_messages(sess_id)
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{sess_id}"):
                delete_session(sess_id)
                if st.session_state.current_session_id == sess_id:
                    st.session_state.current_session_id = None
                    st.session_state.messages = []
                st.rerun()

if not st.session_state.messages:
    st.markdown('<div class="main-header">🤖 BankBot AI Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">How can I help with your finances today?</div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    initial_prompt = None
    if c1.button("Check Balance", use_container_width=True): initial_prompt = "What is my balance?"
    if c2.button("Credit Cards", use_container_width=True): initial_prompt = "Compare credit cards"
    if c3.button("Find ATM", use_container_width=True): initial_prompt = "Find nearest ATM"

    if initial_prompt:
        new_id = create_session(initial_prompt)
        st.session_state.current_session_id = new_id
        st.session_state.messages.append({"role": "user", "content": initial_prompt})
        save_message(new_id, "user", initial_prompt)
        
        bot_res = get_bank_response(initial_prompt)
        st.session_state.messages.append({"role": "assistant", "content": bot_res})
        save_message(new_id, "assistant", bot_res)
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        if isinstance(content, dict) and content.get("type") == "table":
            st.write(content["text"])
            st.dataframe(pd.DataFrame(content["data"]), hide_index=True)
        elif isinstance(content, dict):
             st.write(content["text"])
        else:
            st.write(content)

if prompt := st.chat_input("Ask a question..."):
    if st.session_state.current_session_id is None:
        st.session_state.current_session_id = create_session(prompt)
        
    session_id = st.session_state.current_session_id
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(session_id, "user", prompt)
    with st.chat_message("user"):
        st.write(prompt)
    
    response_data = get_bank_response(prompt)
    
    save_message(session_id, "assistant", response_data)
    st.session_state.messages.append({"role": "assistant", "content": response_data})
    
    with st.chat_message("assistant"):
        if response_data["type"] == "table":
            st.write(response_data["text"])
            st.dataframe(pd.DataFrame(response_data["data"]), hide_index=True)
        else:
            if hasattr(st, "write_stream"):
                st.write_stream(stream_text(response_data["text"]))
            else:
                st.write(response_data["text"])