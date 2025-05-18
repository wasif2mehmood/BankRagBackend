from fastapi import FastAPI, Request, UploadFile, File
from openai import OpenAI
from pydantic import BaseModel
from utils import load_chat_history
from l_graph import graph_builder
from chat_bot import bot
from langchain_chroma import Chroma
import os
import tempfile
import shutil
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI()
graph = graph_builder.compile()

# Define a persistent directory for the vector store
PERSISTENCE_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
os.makedirs(PERSISTENCE_DIRECTORY, exist_ok=True)

# Path for session chat memory
CHAT_MEMORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_memory.json")

class APIKeyRequest(BaseModel):
    api_key: str

class InputMessageRequest(BaseModel):
    message: str

@app.post("/set_openai_api_key/")
def set_openai_api_key(request: APIKeyRequest):
    global bot
    bot.api_key = request.api_key
    
    # Initialize the custom model client
    bot.llm = OpenAI(
        api_key=os.environ.get("RUNPOD_API_KEY") or request.api_key,
        base_url=f"https://api.runpod.ai/v2/{os.environ.get('RUNPOD_ENDPOINT_ID')}/openai/v1" if os.environ.get("RUNPOD_ENDPOINT_ID") else None,
    )
    
    # Initialize the OpenAI client for memory operations
    bot.openai_client = OpenAI(
        api_key=request.api_key
    )
    
    # Initialize embeddings
    bot.embeddings = OpenAIEmbeddings(api_key=request.api_key)
    os.environ["OPENAI_API_KEY"] = request.api_key
    
    # Try to load existing vector store if it exists
    try:
        if os.path.exists(PERSISTENCE_DIRECTORY):
            bot.vector_store = Chroma(
                persist_directory=PERSISTENCE_DIRECTORY,
                embedding_function=bot.embeddings
            )
            return {"message": "OpenAI API key set successfully and existing documents loaded"}
    except Exception as e:
        print(f"Error loading existing vector store: {str(e)}")
    
    return {"message": "OpenAI API key set successfully"}

@app.post("/upload_document/")
async def upload_document(file: UploadFile = File(...)):
    global bot
    
    # Create a temporary file to save the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_file.write(await file.read())
        file_path = temp_file.name
    
    try:
        # Choose appropriate loader based on file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.csv':
            loader = CSVLoader(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            loader = UnstructuredExcelLoader(file_path)
        elif file_extension in ['.doc', '.docx']:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            os.unlink(file_path)
            return {"error": "Unsupported file format. Please upload PDF, CSV, Excel, or Word documents."}
        
        # Load and split documents
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        all_splits = text_splitter.split_documents(documents)
        
        # Add document source metadata
        for doc in all_splits:
            doc.metadata["source"] = file.filename
        
        # Create or update vector store with persistence
        # Note: In newer versions of langchain_chroma, persistence is automatic 
        # when persist_directory is provided
        if bot.vector_store is None:
            bot.vector_store = Chroma.from_documents(
                documents=all_splits,
                embedding=bot.embeddings,
                persist_directory=PERSISTENCE_DIRECTORY
            )
            # No need to call persist() - it's automatic with persist_directory
        else:
            bot.vector_store.add_documents(all_splits)
            # No need to call persist() - it's automatic with persist_directory
        
        os.unlink(file_path)  # Remove the temporary file
        return {"message": f"Document '{file.filename}' processed successfully and saved for future use"}
    
    except Exception as e:
        os.unlink(file_path)  # Ensure temp file is removed even if processing fails
        return {"error": f"Error processing document: {str(e)}"}

@app.post("/process_input_message/")
def process_input_message_endpoint(request: InputMessageRequest):
    input_message = request.message
    
    messages, chat_summary = load_chat_history()
    print(f"Loaded chat history: {len(messages) if messages else 0} messages, summary length: {len(chat_summary) if chat_summary else 0}")
    
    # Create initial state with history and new user message
    initial_state = {
        "messages": (messages or []) + [{"role": "user", "content": input_message}],
        "chat_summary": chat_summary or "",
        "use_memory": False,
        "is_safe": True,
        "safety_issue": ""
    }
    # Process through the graph
    result = graph.invoke(initial_state)
    
    # Extract the assistant's response from the result
    # The format has changed with LangGraph - we need to get the last assistant message
    messages = result.get("messages", [])
    assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
    
    if assistant_messages:
        response_message = assistant_messages[-1]["content"]
    else:
        response_message = "No response generated."
    
    return {"message": response_message}

# New endpoint to list all documents in the vector store
@app.get("/list_documents/")
def list_documents():
    if bot.vector_store is None:
        return {"documents": []}
    
    try:
        # Get unique document sources from metadata
        all_docs = bot.vector_store.get()
        sources = set()
        if all_docs and "metadatas" in all_docs and all_docs["metadatas"]:
            for metadata in all_docs["metadatas"]:
                if "source" in metadata:
                    sources.add(metadata["source"])
        
        return {"documents": list(sources)}
    except Exception as e:
        return {"error": f"Error listing documents: {str(e)}", "documents": []}

# New endpoint to clear all documents (optional)
@app.post("/clear_documents/")
def clear_documents():
    global bot
    try:
        if bot.vector_store:
            bot.vector_store = None
            # Delete the persistence directory and recreate it
            if os.path.exists(PERSISTENCE_DIRECTORY):
                shutil.rmtree(PERSISTENCE_DIRECTORY)
                os.makedirs(PERSISTENCE_DIRECTORY, exist_ok=True)
        return {"message": "All documents cleared successfully"}
    except Exception as e:
        return {"error": f"Error clearing documents: {str(e)}"}

# New endpoint to clear chat memory
@app.post("/clear_chat_memory/")
def clear_chat_memory():
    try:
        if os.path.exists(CHAT_MEMORY_PATH):
            os.remove(CHAT_MEMORY_PATH)
        return {"message": "Chat memory cleared successfully"}
    except Exception as e:
        return {"error": f"Error clearing chat memory: {str(e)}"}

@app.get("/")
def greet_json():
    return {"Hello": "World!"}