# Medical RAG Chatbot

An AI-powered medical information chatbot built using Retrieval-Augmented Generation (RAG).

## Features

- Upload medical documents
- PDF text extraction
- Document chunking
- Hugging Face embeddings
- FAISS vector database
- Semantic document retrieval
- LLM-based question answering
- Source/page display
- Streamlit chatbot interface

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Hugging Face
- Groq
- Llama
- Sentence Transformers

## Architecture

PDF
↓
Text Extraction
↓
Text Chunking
↓
Hugging Face Embeddings
↓
FAISS Vector Database
↓
Retriever
↓
LLM
↓
Answer

## Installation

```bash
git clone YOUR_REPOSITORY_URL
cd Langchain

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt