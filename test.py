import json
import csv
import os
import sys
import random
from tqdm import tqdm
import logging
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Add the project root to the path so we can import the utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BackEnd.utils import format_docs

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('rag_evaluation')

# Path to your existing Chroma DB
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.strip().startswith('//'):
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse line: {line}")
                        continue
        logger.info(f"Loaded {len(data)} items from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        return []

def load_existing_chroma_db(api_key):
    """Load the existing Chroma database."""
    try:
        if not os.path.exists(CHROMA_DB_PATH):
            logger.error(f"Chroma DB not found at {CHROMA_DB_PATH}")
            return None

        # Create embedding function
        embeddings = OpenAIEmbeddings(api_key=api_key)
        
        # Load the existing database
        logger.info(f"Loading existing Chroma DB from {CHROMA_DB_PATH}")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        
        # Check if there are documents
        collection = vector_store._collection
        doc_count = collection.count()
        logger.info(f"Successfully loaded Chroma DB with {doc_count} documents")
        
        return vector_store
    except Exception as e:
        logger.error(f"Error loading Chroma DB: {str(e)}")
        return None

def direct_rag_process(query, vector_store, api_key=None):
    """Directly process a query through RAG with OpenAI API."""
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Retrieve similar documents
        docs = vector_store.similarity_search(query, k=3)
        
        # Extract raw contexts for saving
        raw_contexts = [doc.page_content for doc in docs]
        
        # Format context for the model
        context = format_docs(docs)
        
        # Construct prompt with context
        prompt = f"""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If the retrieved context doesn't contain relevant information to answer the question:
1. If you have general knowledge about the topic, provide a brief general answer but clearly indicate it's not specific to NUST Bank.
2. If you don't have sufficient general knowledge about the topic, respond with: "Please visit the NUST Bank website for more information on this topic."

Use three sentences maximum and keep the answer concise.

Context:
{context}

Question: {query}
Answer:"""

        # Generate response using OpenAI API directly
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use OpenAI model as fallback
            messages=[
                {"role": "system", "content": "You are a helpful banking assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1000,
        )
        
        return {
            "content": response.choices[0].message.content,
            "contexts": raw_contexts
        }
    except Exception as e:
        logger.error(f"Error in direct RAG processing: {str(e)}")
        return {
            "content": f"Error: {str(e)}",
            "contexts": []
        }

def main():
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        logger.error("Example: set OPENAI_API_KEY=your-api-key-here")
        return
    
    # Load the existing Chroma DB
    vector_store = load_existing_chroma_db(api_key)
    if not vector_store:
        logger.error("Failed to load existing Chroma DB. Exiting.")
        return
    
    # Use the absolute file path you provided
    input_path = r"C:\Users\A.C\Documents\GitHub\Bank-Assistant-RAG\data\preprocessing\data.jsonl"
    output_path = os.path.join("output", "rag_evaluation_random10.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load data for questions and ground truth
    logger.info(f"Loading questions and ground truth from {input_path}")
    data = load_jsonl(input_path)
    
    if not data:
        logger.error("No data loaded. Exiting.")
        return
    
    # Select 10 random items or all if less than 10
    sample_size = min(10, len(data))
    random_sample = random.sample(data, sample_size)
    logger.info(f"Selected {sample_size} random questions for evaluation")
    
    # Create CSV file
    logger.info(f"Processing {sample_size} queries and writing results to {output_path}")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'answer', 'contexts', 'ground_truths'])
        
        # Process each query in the random sample
        for item in tqdm(random_sample, desc="Processing queries"):
            query = item.get("input", "")
            ground_truth = item.get("output", "")
            
            # Process through direct RAG implementation with API key
            rag_result = direct_rag_process(query, vector_store, api_key)
            rag_output = rag_result["content"]
            contexts = rag_result["contexts"]
            
            # Format contexts as a string representation of a list for RAGAS
            contexts_str = json.dumps(contexts)
            ground_truths_str = json.dumps([ground_truth])
            
            # Write to CSV
            writer.writerow([query, rag_output, contexts_str, ground_truths_str])
    
    logger.info(f"Evaluation complete. Results saved to {output_path}")
    
    # Print the questions that were selected
    logger.info("Selected questions:")
    for i, item in enumerate(random_sample):
        logger.info(f"{i+1}. {item.get('input', '')}")

if __name__ == "__main__":
    main()