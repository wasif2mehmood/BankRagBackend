from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from openai import OpenAI

class ChatBot:
    def __init__(self):
        self.api_key = None
        # Custom bot for RAG responses
        self.llm = None
        
        # OpenAI client for memory operations
        self.openai_client = None
        
        # Embeddings for vector store
        self.embeddings = None
        self.vector_store = None

bot = ChatBot()