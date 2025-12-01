import streamlit as st
import pandas as pd
import time
import random
import requests

st.set_page_config(
    page_title="BankBot AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Style the header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A; /* Dark Blue */
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

def stream_text(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.05)

def get_bank_response(user_query):
    url = "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "sk-default-Yq5Cz1mhH2FcgDMbzAf2wvCbAR6mgHRF"
    }
    payload = {
        "user_id": "sumesh13055@gmail.com",
        "agent_id": "69272769cebc7452a28f8d0f",
        "session_id": "69272769cebc7452a28f8d0f-v34ci244ny",
        "message": user_query
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        agent_response = data.get('response', str(data))
        return {
            "type": "text",
            "text": agent_response
        }
    except Exception as e:
        return {
            "type": "text",
            "text": f"Error connecting to agent: {str(e)}"
        }

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2534/2534204.png", width=50)
    st.title("MyBank Services")
    st.markdown("---")
    
    if st.button("+ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    

if len(st.session_state.messages) == 0:
    st.markdown('<div class="main-header">🤖 BankBot AI Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Hello! I\'m BankBot. How can I help with your finances today?</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("What's my balance?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What's my balance?"})
            st.rerun()
    with col2:
        if st.button("Compare credit cards", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Compare credit cards"})
            st.rerun()
    with col3:
        if st.button("Reset my PIN", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "How do I reset my PIN?"})
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict) and message["role"] == "assistant":
            st.write(message["content"]["text"])
            if message["content"]["type"] == "table":
                st.dataframe(message["content"]["data"], hide_index=True)
                st.markdown("[Apply Now on our secure portal >](https://google.com)")
        else:
            st.write(message["content"])

if prompt := st.chat_input("Ask a question about your account..."):
    with st.chat_message("user"):
        st.write(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    response_data = get_bank_response(prompt)
    
    with st.chat_message("assistant"):
        if response_data["type"] == "table":
            st.write(response_data["text"])
            st.dataframe(response_data["data"], hide_index=True)
            st.markdown("[Apply Now on our secure portal >](https://google.com)")
            
            st.session_state.messages.append({"role": "assistant", "content": response_data})
            
        else:
            response_container = st.empty()
            streamed_text = ""
            for word in stream_text(response_data["text"]):
                streamed_text += word
                response_container.write(streamed_text)
            
            st.session_state.messages.append({"role": "assistant", "content": response_data["text"]})