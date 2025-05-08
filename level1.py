"""
GAIA Question Answering Automation

This script automates the process of answering questions from the GAIA dataset:
1. Loads the dataset
2. For each example:
   - Extracts the question
   - Loads associated file content if available
   - Performs web search for relevant information
   - Summarizes both file and search content
   - Generates an answer using a specified model
   - Verifies the answer against the expected answer
   - Reports results and statistics
"""

import os
import json
import time
from datasets import load_dataset
import ollama
from tqdm import tqdm

# Import utility functions
from utils.search_capability import search_and_parse
from utils.summarize import summarize_text
from utils.file_reader import read_file

def process_gaia_dataset(levels=["2023_level1"], loader_path="./GAIA.py", split="validation"):
    """
    Process the GAIA dataset for the specified levels and generate answers.
    
    Args:
        levels: List of dataset levels to process
        loader_path: Path to the GAIA loader script
        split: Dataset split to use
    
    Returns:
        Dictionary with results and statistics
    """
    results = {
        "total_examples": 0,
        "correct_answers": 0,
        "incorrect_answers": 0,
        "examples": []
    }
    
    print(f"Starting GAIA question answering process for levels: {levels}")
    
    for level in levels:
        print(f"\nProcessing level: {level}")
        
        # Load the dataset for the current level
        try:
            dataset = load_dataset(loader_path, name=level, split=split)
            print(f"Loaded {len(dataset)} examples from {level}")
        except Exception as e:
            print(f"Error loading dataset {level}: {e}")
            continue
        
        # Process each example in the dataset
        for i, example in tqdm(enumerate(dataset), total=len(dataset)):
            try:
                example_result = process_single_example(example)
                results["examples"].append(example_result)
                
                # Update statistics
                results["total_examples"] += 1
                if example_result["is_correct"]:
                    results["correct_answers"] += 1
                else:
                    results["incorrect_answers"] += 1
                    
                # Print progress information
                print(f"\nExample {i+1} - Task ID: {example.get('task_id', 'N/A')}")
                print(f"Question: {example_result['question']}")
                print(f"Expected: {example_result['expected_answer']}")
                print(f"Generated: {example_result['generated_answer']}")
                print(f"Correct: {example_result['is_correct']}")
                
                # Optional: Add a small delay to avoid overwhelming APIs
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing example {i}: {e}")
    
    # Calculate accuracy
    if results["total_examples"] > 0:
        results["accuracy"] = results["correct_answers"] / results["total_examples"]
    else:
        results["accuracy"] = 0.0
        
    print(f"\nProcessing complete!")
    print(f"Total examples: {results['total_examples']}")
    print(f"Correct answers: {results['correct_answers']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    
    return results

def process_single_example(example):
    """
    Process a single example from the GAIA dataset.
    
    Args:
        example: Dictionary containing the example data
    
    Returns:
        Dictionary with processing results
    """
    # Extract information from the example
    question = example["Question"]
    expected_answer = example.get("Final answer", "")
    file_path = example.get("file_path", "")
    file_name = example.get("file_name", "")
    
    start_time = time.time()
    
    # Step 1: Get file content if available
    file_content = ""
    if file_path and os.path.exists(file_path):
        try:
            file_content = read_file(file_path)
            print(f"Read file content from {file_path}: {len(file_content)} characters")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    elif file_name and os.path.exists(file_name):
        try:
            file_content = read_file(file_name)
            print(f"Read file content from {file_name}: {len(file_content)} characters")
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
    
    # Step 2: Perform web search
    # Issue of too much data being returned from web search needs to be handled
    search_results = {}
    try:
        search_results = search_and_parse(question)
        print(f"Completed search with {len(search_results.get('parsed_content', ''))} characters of content")
    except Exception as e:
        print(f"Error during search: {e}")
    
    # Step 3: Summarize file content and search results
    summarized_file_content = ""
    if file_content:
        try:
            summarized_file_content = summarize_text(
                text=file_content,
                target_len=10000,
                chunk_size=10000,
                truncate=False,
                model="llama3:8b",
                temperature=0
            )
            print(f"Summarized file content: {len(summarized_file_content)} characters")
        except Exception as e:
            print(f"Error summarizing file content: {e}")
    
    summarized_search_content = ""
    if search_results and "parsed_content" in search_results:
        try:
            summarized_search_content = summarize_text(
                text=search_results.get("parsed_content", ""),
                target_len=10000,
                chunk_size=10000,
                truncate=False,
                model="llama3:8b",
                temperature=0
            )
            print(f"Summarized search content: {len(summarized_search_content)} characters")
        except Exception as e:
            print(f"Error summarizing search content: {e}")
    
    # Step 4: Combine contexts
    combined_context = ""
    if summarized_file_content:
        combined_context += f"## File Context:\n{summarized_file_content}\n\n"
    if summarized_search_content:
        combined_context += f"## Search Context:\n{summarized_search_content}"
    
    # Check if combined context exceeds 30,000 characters and re-summarize if needed
    if len(combined_context) > 30000:
        print(f"Combined context too large ({len(combined_context)} chars). Re-summarizing...")
        try:
            combined_context = summarize_text(
                text=combined_context,
                target_len=25000,  # Aim for less than 30,000 with some margin
                chunk_size=15000,
                truncate=True,
                model="llama3:8b",
                temperature=0
            )
            print(f"Re-summarized combined context: {len(combined_context)} characters")
        except Exception as e:
            print(f"Error re-summarizing combined context: {e}")
            # If re-summarization fails, truncate manually as a fallback
            combined_context = combined_context[:30000] + "... [content truncated]"
    # Step 5: Generate the first answer using a more powerful model
    raw_answer = ""
    try:
        response = ollama.chat(
            model='deepseek-r1:32b',
            messages=[
                {'role': 'user', 'content': f'##Question: {question}\n## Context: {combined_context}\n## Question: {question}\n## ANSWER:'}
            ]
        )
        raw_answer = response['message']['content']
        print(f"Generated initial answer: {len(raw_answer)} characters")
    except Exception as e:
        print(f"Error generating initial answer: {e}")
    
    # Step 6: Refine the answer to be concise using a smaller model
    final_answer = ""
    try:
        final_response = ollama.chat(
            model='llama3:8b',
            messages=[
                {'role': 'user', 'content': f"## QUESTION: {question}\n## UNSUMMARIZED ANSWER: {raw_answer}\n## TASK: Based on the given question format the above unsummarized answer into the least number of words or numbers required to answer the question. Return as JSON the\n## FINAL ANSWER:"}
            ]
        )
        final_answer = final_response['message']['content']
        
        # Clean up the final answer to extract just the text
        final_answer = clean_model_output(final_answer)
        print(f"Generated final answer: {final_answer}")
    except Exception as e:
        print(f"Error generating final answer: {e}")
    
    # Step 7: Check if the answer is correct
    is_correct = is_answer_correct(final_answer, expected_answer)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Return the results for this example
    return {
        "task_id": example.get("task_id", ""),
        "question": question,
        "expected_answer": expected_answer,
        "generated_answer": final_answer,
        "raw_answer": raw_answer,
        "is_correct": is_correct,
        "had_file_content": bool(file_content),
        "processing_time": processing_time
    }

def clean_model_output(text):
    """
    Clean the model output to extract just the answer.
    
    Args:
        text: Raw model output
    
    Returns:
        Cleaned answer text
    """
    # Try to extract JSON if it exists
    try:
        # Check if there's JSON in the text
        if "{" in text and "}" in text:
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            json_str = text[json_start:json_end]
            data = json.loads(json_str)
            if "FINAL ANSWER" in data:
                return data["FINAL ANSWER"]
            for key, value in data.items():
                if "answer" in key.lower():
                    return value
            # If no specific answer key, return the first value
            return next(iter(data.values()))
    except:
        pass
    
    # Try to extract based on common patterns
    common_indicators = [
        "FINAL ANSWER:", 
        "Final Answer:", 
        "Answer:", 
        "The answer is:"
    ]
    
    for indicator in common_indicators:
        if indicator in text:
            parts = text.split(indicator, 1)
            if len(parts) > 1:
                # Take the text after the indicator and before any other section
                answer = parts[1].strip()
                # Check if there are any other section headers
                for next_section in ["EXPLANATION:", "REASONING:", "##", "Note:"]:
                    if next_section in answer:
                        answer = answer.split(next_section, 1)[0].strip()
                return answer
    
    # If no structured format found, return the original text with minimal cleaning
    # Remove any markdown formatting and keep just the core text
    lines = text.strip().split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('```'):
            cleaned_lines.append(line)
    
    if cleaned_lines:
        return ' '.join(cleaned_lines)
    
    # If all else fails, return the original text
    return text.strip()

def is_answer_correct(generated_answer, expected_answer):
    """
    Check if the generated answer matches the expected answer.
    This function can be expanded with more sophisticated matching logic.
    
    Args:
        generated_answer: The answer generated by the model
        expected_answer: The expected correct answer
    
    Returns:
        Boolean indicating if the answer is correct
    """
    if not generated_answer or not expected_answer:
        return False
    
    # Clean both answers for comparison
    gen_clean = clean_text_for_comparison(generated_answer)
    exp_clean = clean_text_for_comparison(expected_answer)
    
    # Exact match
    if gen_clean == exp_clean:
        return True
    
    # Check if the expected answer is contained within the generated answer
    if exp_clean in gen_clean:
        return True
    
    # For numeric answers, check if the numbers match
    if gen_clean.isdigit() and exp_clean.isdigit():
        return int(gen_clean) == int(exp_clean)
    
    # For very short answers, allow slight variations
    if len(exp_clean) <= 5 and len(gen_clean) <= 5:
        return gen_clean.startswith(exp_clean) or exp_clean.startswith(gen_clean)
    
    # For longer answers, you might implement more sophisticated matching
    # such as semantic similarity, but that's beyond the scope of this example
    
    return False

def clean_text_for_comparison(text):
    """
    Clean text for comparison by removing punctuation, extra spaces, and lowercasing.
    
    Args:
        text: Text to clean
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Remove common prefixes that might be in model outputs
    prefixes = ["the answer is", "answer:", "final answer:", "the final answer is"]
    text_lower = text.lower()
    for prefix in prefixes:
        if text_lower.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    
    # Remove punctuation and normalize spacing
    import re
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    
    return text.lower()  # Convert to lowercase for case-insensitive comparison

def save_results(results, output_file="gaia_results.json"):
    """
    Save the results to a JSON file.
    
    Args:
        results: Results dictionary
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

def main():
    """
    Main function to run the GAIA question answering process.
    """
    # Configuration
    levels = ["2023_level1"]  # Add more levels as needed
    loader_path = "./GAIA.py"
    split = "validation"
    output_file = "gaia_results.json"
    
    # Process the dataset
    results = process_gaia_dataset(levels, loader_path, split)
    
    # Save the results
    save_results(results, output_file)
    
    # Print summary statistics
    print("\n=== SUMMARY ===")
    print(f"Total examples processed: {results['total_examples']}")
    print(f"Correct answers: {results['correct_answers']}")
    print(f"Incorrect answers: {results['incorrect_answers']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()