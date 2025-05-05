import ollama
import sys
import time
from tqdm import tqdm

def reduce_text(text_list, max_context_size=2500, model='deepseek-r1:32b', show_progress=True):
    """
    Takes a list of texts, summarizes each individually, then concatenates the summaries
    to fit under the specified maximum length.
    
    Args:
        text_list (list): List of text strings to be reduced
        max_context_size (int): Maximum desired length of the output text
        model (str): Ollama model to use for summarization
        show_progress (bool): Whether to display progress bars
    
    Returns:
        str: A single text containing the most important information from all inputs, under max_context_size
    """
    total_input_len = sum(len(text) for text in text_list)
    
    # Display initial information
    if show_progress:
        print(f"Input: {len(text_list)} articles, total {total_input_len} characters")
        print(f"Target: Single text under {max_context_size} characters")
        print(f"Compression ratio: {max_context_size/total_input_len:.2%}")
    
    # Check if summarization is needed
    if total_input_len <= max_context_size:
        if show_progress:
            print("No summarization needed - input is already under max context size")
        concatenated = "\n\n---\n\n".join(text_list)
        return concatenated[:max_context_size]
    
    # Calculate individual summary target lengths based on original proportions
    proportions = [len(text) / total_input_len for text in text_list]
    # Allow 90% of max_context for individual summaries, reserve 10% for final reduction if needed
    individual_targets = [int(p * max_context_size * 0.9) for p in proportions]
    
    # Summarize each text individually
    individual_summaries = []
    
    if show_progress:
        print("\nSummarizing individual texts...")
        main_progress = tqdm(total=len(text_list), desc="Files processed", unit="file")
    
    for i, (text, target_length) in enumerate(zip(text_list, individual_targets)):
        if show_progress:
            print(f"\nProcessing file {i+1}/{len(text_list)} ({len(text)} chars → target {target_length} chars)")
        
        # Skip summarization for short texts
        if len(text) <= target_length:
            individual_summaries.append(text)
            if show_progress:
                print(f"File {i+1} already under target length, skipping summarization")
                main_progress.update(1)
            continue
        
        # Prepare prompt for individual summarization
        prompt = f"""
        I need you to summarize the following text to approximately {target_length} characters.
        Focus on preserving:
        1. Important facts and statistics
        2. Key values and figures
        3. Critical context
        4. Essential points and conclusions
        
        Original text:
        {text}
        
        Provide ONLY the summarized text without any meta commentary.
        """
        
        # Create spinner animation
        if show_progress:
            print(f"Summarizing file {i+1}...", end='', flush=True)
            spinner_chars = "|/-\\"
            start_time = time.time()
        
        # Call the Ollama model
        try:
            # Do a simple spinner animation without threading
            if show_progress:
                counter = 0
                while True:
                    elapsed = time.time() - start_time
                    if elapsed > 0.1:  # Update every 0.1 seconds
                        sys.stdout.write("\rSummarizing file {0} {1} ".format(
                            i+1, spinner_chars[counter % len(spinner_chars)]))
                        sys.stdout.flush()
                        counter += 1
                        start_time = time.time()
                    # Check if we should break out for API call
                    if counter > 3:  # Do a few spins then make the call
                        break
            
            # Make the actual API call
            start_time = time.time()
            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            # Continue spinner while waiting
            if show_progress:
                counter = 0
                while response is None and counter < 1000:  # Safety limit
                    elapsed = time.time() - start_time
                    if elapsed > 0.1:
                        sys.stdout.write("\rSummarizing file {0} {1} ".format(
                            i+1, spinner_chars[counter % len(spinner_chars)]))
                        sys.stdout.flush()
                        counter += 1
                        start_time = time.time()
            
            summary = response['message']['content'].strip()
            
            # Ensure summary isn't longer than original text
            if len(summary) > len(text):
                summary = text[:target_length] + "..."
            
            individual_summaries.append(summary)
            
            if show_progress:
                processing_time = time.time() - start_time
                sys.stdout.write("\r" + " " * 30 + "\r")  # Clear the spinner line
                print(f"File {i+1} summarized in {processing_time:.2f}s: {len(summary)} chars ({len(summary)/len(text):.2%} of original)")
                main_progress.update(1)
                
        except Exception as e:
            if show_progress:
                sys.stdout.write("\r" + " " * 30 + "\r")  # Clear the spinner line
                print(f"Error summarizing file {i+1}: {e}")
                print(f"Falling back to truncation for this file")
                main_progress.update(1)
            
            # Fall back to simple truncation on error
            individual_summaries.append(text[:target_length] + "...")
    
    # Check if concatenated individual summaries fit within max context
    concatenated_summaries = "\n\n---\n\n".join(individual_summaries)
    
    if len(concatenated_summaries) <= max_context_size:
        if show_progress:
            print(f"\nAll individual summaries fit within max context ({len(concatenated_summaries)}/{max_context_size} chars)")
        return concatenated_summaries
    
    # If still too large, perform a final summarization on the concatenated summaries
    if show_progress:
        print(f"\nIndividual summaries still exceed max context ({len(concatenated_summaries)}/{max_context_size} chars)")
        print("Performing final summarization on combined summaries...")
    
    # Target a bit less than max to ensure we fit
    final_target = int(max_context_size * 0.95)
    
    # Final summarization prompt
    final_prompt = f"""
    I need you to summarize these already-summarized articles in approximately {final_target} characters.
    These are already condensed summaries, so focus on preserving the most important:
    1. Facts and statistics
    2. Key figures and values
    3. Critical conclusions
    
    The summaries are separated by "---" dividers.
    
    Summaries to condense:
    {concatenated_summaries}
    
    Provide ONLY the summarized text without any meta commentary.
    """
    
    # Create spinner for final summarization
    if show_progress:
        print("Final summarization in progress...", end='', flush=True)
        spinner_chars = "|/-\\"
        counter = 0
        start_time = time.time()
    
    try:
        # Simple spinner animation without threading
        if show_progress:
            while True:
                elapsed = time.time() - start_time
                if elapsed > 0.1:
                    sys.stdout.write("\rFinal summarization in progress... {0} ".format(
                        spinner_chars[counter % len(spinner_chars)]))
                    sys.stdout.flush()
                    counter += 1
                    start_time = time.time()
                # Break after a few spins
                if counter > 3:
                    break
        
        # Make the API call
        start_time = time.time()
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': final_prompt}]
        )
        
        final_summary = response['message']['content'].strip()
        
        # If still too long, truncate as last resort
        if len(final_summary) > max_context_size:
            if show_progress:
                sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the spinner line
                print(f"Final summary still too long ({len(final_summary)} chars), truncating to fit")
            final_summary = final_summary[:max_context_size - 3] + "..."
        
        if show_progress:
            sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the spinner line
            processing_time = time.time() - start_time
            print(f"Final summary complete in {processing_time:.2f}s: {len(final_summary)}/{max_context_size} chars")
            print(f"Total compression: {len(final_summary)/total_input_len:.2%} of original text")
        
        return final_summary
        
    except Exception as e:
        if show_progress:
            sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the spinner line
            print(f"Error in final summarization: {e}")
            print(f"Falling back to truncation of concatenated summaries")
        
        # Fall back to truncation of the concatenated summaries
        return concatenated_summaries[:max_context_size - 3] + "..."


# Example usage
if __name__ == "__main__":
    # Example with 5 articles of different lengths
    articles = [
        "A" * 1000,  # 1000-character article
        "B" * 1000,  # 1000-character article
        "C" * 2000,  # 2000-character article
        "D" * 3000,  # 3000-character article
        "E" * 2000   # 2000-character article
    ]
    
    # Total length: 9000 characters, max context: 2500
    result = reduce_text(articles, max_context_size=2500)
    print(f"Result length: {len(result)}")
    print(f"Result is under max context: {len(result) <= 2500}")
    print(f"First 100 characters: {result[:100]}...")


# Example usage
if __name__ == "__main__":
    # Example with 5 articles of different lengths
    articles = [
        "A" * 1000,  # 1000-character article
        "B" * 1000,  # 1000-character article
        "C" * 2000,  # 2000-character article
        "D" * 3000,  # 3000-character article
        "E" * 2000   # 2000-character article
    ]
    
    # Total length: 9000 characters, max context: 2500
    result = reduce_text(articles, max_context_size=2500)
    print(f"Result length: {len(result)}")
    print(f"Result is under max context: {len(result) <= 2500}")
    print(f"First 100 characters: {result[:100]}...")


# Example usage
if __name__ == "__main__":
    # Example with 5 articles of different lengths
    articles = [
        "A" * 1000,  # 1000-character article
        "B" * 1000,  # 1000-character article
        "C" * 2000,  # 2000-character article
        "D" * 3000,  # 3000-character article
        "E" * 2000   # 2000-character article
    ]
    
    # Total length: 9000 characters, max context: 2500
    result = reduce_text(articles, max_context_size=2500)
    print(f"Result length: {len(result)}")
    print(f"Result is under max context: {len(result) <= 2500}")
    print(f"First 100 characters: {result[:100]}...")
    print(result)