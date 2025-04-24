import streamlit as st
import os
import json
import tempfile
import asyncio
import httpx
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional

# PDF, Markdown, and text file handling
import PyPDF2
import markdown
import base64

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM

# Set page configuration
st.set_page_config(
    page_title="Enhanced Ollama Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.bot {
        background-color: #e3f2fd;
    }
    .chat-message .avatar {
        width: 40px;
        min-width: 40px;
    }
    .chat-message .message {
        margin-left: 1rem;
        width: 100%;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "file_content" not in st.session_state:
    st.session_state.file_content = None
if "model_params" not in st.session_state:
    st.session_state.model_params = {
        "model": "gemma3:12b",
        "temperature": 0.7,
        "max_tokens": 2000
    }
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful assistant. Answer the user's questions thoroughly and accurately."

# Function to get available Ollama models
async def get_available_models():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            else:
                return ["gemma3:12b", "deepseek-r1:8b", "llama3:8b"]
    except Exception as e:
        st.warning(f"Error connecting to Ollama server: {e}")
        return ["gemma3:12b", "deepseek-r1:8b", "llama3:8b"]  # Default models if service is unavailable

# Custom function to create OllamaLLM with parameters
def create_ollama_chain(model_name, temperature, max_tokens, system_prompt):
    # Create LLM with specified parameters
    llm = OllamaLLM(
        model=model_name,
        temperature=temperature,
        num_predict=max_tokens,
    )
    
    # Create chat prompt template with system message and conversation history
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    # Return the chain
    return prompt | llm

# Function to extract text from PDF file
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to extract text from Markdown file
def extract_text_from_markdown(md_file):
    md_content = md_file.read().decode("utf-8")
    return md_content

# Function to extract text from txt file
def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

# Function to process uploaded file
def process_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None
    
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
    try:
        if file_extension == "pdf":
            with open(temp_file_path, "rb") as f:
                return extract_text_from_pdf(f)
        elif file_extension == "md":
            with open(temp_file_path, "rb") as f:
                return extract_text_from_markdown(f)
        elif file_extension == "txt":
            with open(temp_file_path, "rb") as f:
                return extract_text_from_txt(f)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# Function to convert chat history to the format expected by LangChain
def format_chat_history_for_langchain(history):
    formatted_history = []
    for message in history:
        if message["role"] == "user":
            formatted_history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            formatted_history.append(AIMessage(content=message["content"]))
    return formatted_history

# Function to save conversation to file
def save_conversation(conversation, format="json", include_metadata=True):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not os.path.exists("_output"):
        os.makedirs("_output")
    
    if format == "json":
        filename = f"_output/conversation_{timestamp}.json"
        
        output_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "conversation": conversation
        }
        
        if include_metadata:
            output_data["metadata"] = {
                "model": st.session_state.model_params["model"],
                "temperature": st.session_state.model_params["temperature"],
                "max_tokens": st.session_state.model_params["max_tokens"],
                "system_prompt": st.session_state.system_prompt
            }
            
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
    elif format == "markdown":
        filename = f"_output/conversation_{timestamp}.md"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# Conversation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if include_metadata:
                f.write("## Metadata\n\n")
                f.write(f"- **Model**: {st.session_state.model_params['model']}\n")
                f.write(f"- **Temperature**: {st.session_state.model_params['temperature']}\n")
                f.write(f"- **Max Tokens**: {st.session_state.model_params['max_tokens']}\n")
                f.write(f"- **System Prompt**: {st.session_state.system_prompt}\n\n")
            
            f.write("## Conversation\n\n")
            
            for message in conversation:
                if message["role"] == "user":
                    f.write(f"### User\n\n{message['content']}\n\n")
                elif message["role"] == "assistant":
                    f.write(f"### Assistant\n\n{message['content']}\n\n")
    
    return filename

# Function to handle chat interaction
async def chat(question, chat_history, model_params, system_prompt, file_content=None):
    # Create a new instance of the chain with current parameters
    chain = create_ollama_chain(
        model_params["model"],
        model_params["temperature"],
        model_params["max_tokens"],
        system_prompt
    )
    
    # Format chat history for LangChain
    formatted_history = format_chat_history_for_langchain(chat_history)
    
    # If file content is present, add it to the question
    if file_content:
        question = f"Please analyze the following content and then answer my question:\n\n{file_content}\n\nMy question is: {question}"
    
    # Get response from the model
    response = await chain.ainvoke({
        "chat_history": formatted_history,
        "question": question
    })
    
    return response

# Sidebar for model settings and file upload
with st.sidebar:
    st.title("Settings")
    
    # Model selection section
    st.subheader("Model Selection")
    
    # Run get_available_models in asyncio
    available_models = asyncio.run(get_available_models())
    
    selected_model = st.selectbox(
        "Select Ollama Model",
        options=available_models,
        index=available_models.index(st.session_state.model_params["model"]) if st.session_state.model_params["model"] in available_models else 0
    )
    
    # Model parameters
    st.session_state.model_params["temperature"] = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.model_params["temperature"],
        step=0.1
    )
    
    st.session_state.model_params["max_tokens"] = st.slider(
        "Max Tokens",
        min_value=100,
        max_value=4096,
        value=st.session_state.model_params["max_tokens"],
        step=100
    )
    
    st.session_state.model_params["model"] = selected_model
    
    # System prompt
    st.subheader("System Prompt")
    st.session_state.system_prompt = st.text_area(
        "Enter system prompt",
        value=st.session_state.system_prompt,
        height=150
    )
    
    # File upload section
    st.subheader("File Upload")
    uploaded_file = st.file_uploader(
        "Upload PDF, Markdown, or TXT file",
        type=["pdf", "md", "txt"]
    )
    
    if uploaded_file:
        with st.spinner("Processing file..."):
            file_content = process_uploaded_file(uploaded_file)
            if file_content:
                st.session_state.file_content = file_content
                st.success(f"File processed: {uploaded_file.name}")
                
                # Display a preview of the extracted content
                with st.expander("File Content Preview"):
                    st.text(file_content[:500] + "..." if len(file_content) > 500 else file_content)
            else:
                st.error("Failed to process file")
    
    # Clear file button
    if st.session_state.file_content is not None:
        if st.button("Clear File"):
            st.session_state.file_content = None
    
    # Save conversation section
    st.subheader("Save Conversation")
    save_format = st.radio(
        "Save Format",
        ["JSON", "Markdown"]
    )
    
    include_metadata = st.checkbox("Include Metadata", value=True)
    
    if st.button("Save Conversation"):
        if st.session_state.conversation_history:
            filename = save_conversation(
                st.session_state.conversation_history,
                format=save_format.lower(),
                include_metadata=include_metadata
            )
            st.success(f"Conversation saved to {filename}")
        else:
            st.warning("No conversation to save")
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.file_content = None
        st.success("Conversation cleared")

# Main chat interface
st.title("Enhanced Ollama Chatbot")

# Information about current settings
with st.expander("Current Settings", expanded=False):
    st.write(f"**Model**: {st.session_state.model_params['model']}")
    st.write(f"**Temperature**: {st.session_state.model_params['temperature']}")
    st.write(f"**Max Tokens**: {st.session_state.model_params['max_tokens']}")
    st.write(f"**File Loaded**: {'Yes' if st.session_state.file_content else 'No'}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask something...")

# Handle chat input
if prompt:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.conversation_history.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Get response from model
            response = asyncio.run(chat(
                prompt,
                st.session_state.conversation_history,
                st.session_state.model_params,
                st.session_state.system_prompt,
                st.session_state.file_content
            ))
            
            # Display response
            message_placeholder.markdown(response)
            
            # Save assistant message to state
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.conversation_history.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.session_state.conversation_history.append({"role": "assistant", "content": error_message})

# Footer
st.markdown("---")
st.markdown("Enhanced Ollama Chatbot | Built with Streamlit and LangChain")
