import ollama
import sys
import time
from tqdm import tqdm

def reduce_text(text_list, max_context_size=10000, model='deepseek-r1:32b', show_progress=True):
    """
    Takes a list of texts and reduces them to a single text under the specified maximum length,
    preserving important facts and key values.
    
    Args:
        text_list (list): List of text strings to be reduced
        max_context_size (int): Maximum desired length of the output text
        model (str): Ollama model to use for summarization
        show_progress (bool): Whether to display progress bars
    
    Returns:
        str: A single text containing the most important information, under max_context_size
    """
    # Join texts with clear section breaks
    concatenated = "\n\n---\n\n".join(text_list)
    total_input_len = len(concatenated)
    
    # Display initial information
    if show_progress:
        print(f"Input: {len(text_list)} articles, total {total_input_len} characters")
        print(f"Target: Single text under {max_context_size} characters")
        print(f"Compression ratio: {max_context_size/total_input_len:.2%}")
    
    # Check if summarization is needed
    if total_input_len <= max_context_size:
        if show_progress:
            print("No summarization needed - input is already under max context size")
        return concatenated
    
    # Calculate compression ratio needed
    compression_ratio = max_context_size / total_input_len
    target_length = int(max_context_size * 0.95)  # 5% buffer to ensure we stay under limit
    
    # Create progress bar for preparation phase
    if show_progress:
        print("\nPreparing summarization prompt...")
        prep_progress = tqdm(total=3, desc="Preparation", unit="step")
        prep_progress.update(1)
    
    # For extreme compression ratios, we need to be more aggressive
    if compression_ratio < 0.2:  # If we need to compress to less than 20% of original
        prompt = f"""
        I need you to create an extremely concise summary of these articles in exactly {target_length} characters.
        Include ONLY the most critical facts, statistics, and conclusions.
        Focus on:
        1. Key numerical values and statistics
        2. Major conclusions
        3. Essential context
        
        For each article, provide only 1-2 sentences of the most important information.
        
        Original text:
        {concatenated}
        
        Provide ONLY the summarized text without any meta commentary.
        """
        if show_progress:
            print("\nUsing aggressive summarization (compression ratio < 20%)")
    else:
        prompt = f"""
        I need you to summarize these articles in approximately {target_length} characters.
        Focus on preserving:
        1. Important facts and statistics
        2. Key values and figures
        3. Critical context
        4. Essential points and conclusions
        
        Original text:
        {concatenated}
        
        Provide ONLY the summarized text without any meta commentary.
        """
    
    if show_progress:
        prep_progress.update(1)
        prep_progress.update(1)
        print("\nSending to LLM for summarization...")
        # Create a progress bar that pulses to show the model is thinking
        thinking_progress = tqdm(desc="Model thinking", bar_format='{desc}: |{bar}|')
    
    # Animate the thinking progress bar
    stop_animation = False
    def animate_progress():
        i = 0
        while not stop_animation:
            if show_progress:
                thinking_progress.n = i % 100
                thinking_progress.refresh()
            time.sleep(0.1)
            i += 1
    
    # Start animation in separate thread if in interactive environment
    import threading
    if show_progress:
        animation_thread = threading.Thread(target=animate_progress)
        animation_thread.daemon = True
        animation_thread.start()
    
    # Call the Ollama model
    try:
        start_time = time.time()
        
        # First summarization round
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        first_round_time = time.time() - start_time
        summary = response['message']['content'].strip()
        
        if show_progress:
            stop_animation = True
            time.sleep(0.2)  # Give animation thread time to stop
            print(f"\nFirst summarization complete in {first_round_time:.2f}s")
            print(f"First summary length: {len(summary)} characters")
        
        # If we still exceed the max context size, try another round with stricter constraints
        if len(summary) > max_context_size:
            if show_progress:
                print(f"\nSummary still exceeds max context, running second summarization round...")
                stop_animation = False
                if 'thinking_progress' in locals():
                    thinking_progress.reset()
                thinking_progress = tqdm(desc="Second round", bar_format='{desc}: |{bar}|')
                animation_thread = threading.Thread(target=animate_progress)
                animation_thread.daemon = True
                animation_thread.start()
            
            second_start = time.time()
            final_prompt = f"""
            The previous summary is still too long. Create an extremely concise summary 
            in EXACTLY {max_context_size - 100} characters, focusing only on the most 
            critical facts and figures from these texts:
            
            {summary}
            
            Provide ONLY the summarized text without any meta commentary.
            """
            
            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': final_prompt}]
            )
            
            second_round_time = time.time() - second_start
            summary = response['message']['content'].strip()
            
            if show_progress:
                stop_animation = True
                time.sleep(0.2)  # Give animation thread time to stop
                print(f"\nSecond summarization complete in {second_round_time:.2f}s")
                print(f"Second summary length: {len(summary)} characters")
            
            # If still too long, truncate as last resort
            if len(summary) > max_context_size:
                if show_progress:
                    print("\nSummary still too long, truncating to fit max context size")
                summary = summary[:max_context_size - 3] + "..."
        
        total_time = time.time() - start_time
        if show_progress:
            compression_achieved = len(summary) / total_input_len
            print(f"\nFinal summary: {len(summary)} characters ({compression_achieved:.2%} of original)")
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Characters per second: {total_input_len / total_time:.2f}")
        
        return summary
        
    except Exception as e:
        stop_animation = True
        print(f"\nError summarizing texts: {e}")
        # Fall back to simple truncation on error
        truncated = concatenated[:max_context_size - 3] + "..."
        if show_progress:
            print(f"Falling back to simple truncation: {len(truncated)} characters")
        return truncated


# Example usage
if __name__ == "__main__":
    # Example with 5 articles of different lengths
    articles = [
        "A" * 4000,  # 1000-character article
        "B" * 1000,  # 1000-character article
        "C" * 2000,  # 2000-character article
        "D" * 3000,  # 3000-character article
        "E" * 2000   # 2000-character article
    ]
    
    max_context = 10000
    result = reduce_text(articles, max_context_size=max_context)
    print(f"Result length: {len(result)}")
    print(f"Result is under max context: {len(result) <= max_context}")
    print(f"Compression ratio: {len(result) / sum(len(a) for a in articles):.2%}")
    print(result)