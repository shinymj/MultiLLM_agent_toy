# Enhanced Ollama Chatbot

An interactive web application for conversing with Ollama models through an intuitive interface.

## Features

### Conversation Management
- Multi-turn conversation support
- Save and export conversations in JSON or Markdown formats
- Clear conversation history when needed

### Model Configuration
- Dynamic model selection from available Ollama models
- Adjustable parameters (temperature, max tokens)
- Customizable system prompts

### File Processing
- Upload and process PDF, Markdown, and TXT files
- Extracted content is automatically provided to the AI model
- Preview file content before processing

### User Interface
- Clean, intuitive Streamlit interface
- Mobile-friendly design
- Real-time interaction with the AI model

## Installation

1. Ensure you have [Ollama](https://ollama.com) installed and running on your system
2. Install Python 3.8 or newer
3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:

```bash
streamlit run ollama_streamlit_app.py
```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Select your preferred model and settings in the sidebar

4. Start chatting with the AI in the main panel

## Saving Conversations

Conversations can be saved in either JSON or Markdown format with optional metadata:

- **JSON**: Structured format ideal for further processing or analysis
- **Markdown**: Human-readable format perfect for documentation or sharing

## Customizing System Prompts

The system prompt sets the behavior and context for the AI model. Examples:

- "You are a helpful assistant. Answer the user's questions thoroughly and accurately."
- "You are an expert programmer. Provide detailed code explanations and examples."
- "You are a creative writing assistant. Help users craft engaging stories and narratives."

## File Upload Guidelines

- **PDF**: Text content will be extracted from all pages
- **Markdown**: Content will be processed as plain text while preserving structure
- **TXT**: Plain text files will be processed directly

## Troubleshooting

- Ensure Ollama is running with the selected model pulled
- Check terminal for any error messages
- Verify network connection if using a remote Ollama instance

---

Created with Streamlit and LangChain.
