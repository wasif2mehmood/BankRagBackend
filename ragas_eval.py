import pandas as pd
from datasets import Dataset
import os
import sys
import glob
import logging
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness, context_recall, context_precision

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ragas_evaluation')

def find_latest_evaluation_file():
    """Find the most recent RAG evaluation CSV file."""
    # Define possible locations
    possible_paths = [
        os.path.join(os.getcwd(), "output", "*.csv"),
        os.path.join(os.path.dirname(os.getcwd()), "output", "*.csv"),
        os.path.join(os.getcwd(), "*.csv")
    ]
    
    all_files = []
    for path_pattern in possible_paths:
        all_files.extend(glob.glob(path_pattern))
    
    # Filter out RAGAS result files
    evaluation_files = [f for f in all_files if "ragas_results" not in f]
    
    if not evaluation_files:
        logger.error("No evaluation CSV files found.")
        return None
    
    # Get the most recently modified file
    latest_file = max(evaluation_files, key=os.path.getmtime)
    return latest_file

def run_ragas_evaluation(csv_path=None):
    """Run RAGAS evaluation on RAG outputs."""
    # Get the CSV path if not provided
    if not csv_path:
        csv_path = find_latest_evaluation_file()
        if not csv_path:
            logger.error("Could not find any evaluation files.")
            return
    
    # Check if OpenAI API key is set in environment
    if "OPENAI_API_KEY" not in os.environ:
        logger.error("OpenAI API key not found in environment variables.")
        logger.error("Please set the OPENAI_API_KEY environment variable.")
        return
    
    logger.info(f"Loading RAG evaluation data from {csv_path}")
    
    # Load the CSV file into a DataFrame
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records for evaluation")
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        return
    
    # Inspect the DataFrame structure
    logger.info("DataFrame structure:")
    logger.info(df.dtypes)
    
    # Check required columns
    required_cols = ['question', 'answer', 'contexts']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        logger.error(f"Available columns: {df.columns.tolist()}")
        return
    
    # Convert string representations of lists to actual lists
    logger.info("Converting string representations to lists...")
    
    # Process contexts column - keep as list
    if 'contexts' in df.columns:
        df['contexts'] = df['contexts'].apply(
            lambda x: eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else [x]
        )
    
    # Process ground_truths column - extract first item as string for reference
    if 'ground_truths' in df.columns:
        # First parse the JSON string to a list
        ground_truths = df['ground_truths'].apply(
            lambda x: eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else [x]
        )
        
        # Then extract the first item as a string for the reference
        df['reference'] = ground_truths.apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else ""
        )
        
        logger.info("Sample reference: " + str(df['reference'].iloc[0]))
    
    # Convert the DataFrame to a dictionary and create a Dataset
    data_samples = df.to_dict(orient='list')
    dataset = Dataset.from_dict(data_samples)
    
    # Define the metrics to evaluate
    metrics = [
        faithfulness,        # Measures if the answer contains hallucinations
        answer_correctness,  # Measures the correctness of the answer compared to ground truth
        context_precision,   # Measures if the retrieved contexts are relevant to the question
        context_recall       # Measures if the contexts contain the information needed to answer
    ]
    
    logger.info("Running RAGAS evaluation...")
    logger.info("Dataset columns: " + str(dataset.column_names))
    
    # Run the evaluation
    try:
        score = evaluate(dataset, metrics=metrics)
        
        # Convert the score to a pandas DataFrame and print it
        score_df = score.to_pandas()
        logger.info("\nRAGAS Evaluation Results:")
        logger.info(score_df)
        
        # Save the results to a CSV file
        output_path = os.path.splitext(csv_path)[0] + "_ragas_results.csv"
        score_df.to_csv(output_path)
        logger.info(f"Evaluation results saved to {output_path}")
        
        # Add mean scores for easy reference
        mean_scores = score_df.mean().to_frame().T
        mean_scores.index = ['Mean Scores']
        logger.info("\nAverage Scores:")
        logger.info(mean_scores)
        
        return score_df
    except Exception as e:
        logger.error(f"Error during RAGAS evaluation: {str(e)}")
        return None

def main():
    run_ragas_evaluation()

if __name__ == "__main__":
    main()