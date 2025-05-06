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
    # Allow some buffer for dividers (100 chars per article)
    divider_space = min(100 * len(text_list), max_context_size * 0.1)
    available_space = max_context_size - divider_space
    
    proportions = [len(text) / total_input_len for text in text_list]
    individual_targets = [max(200, int(p * available_space)) for p in proportions]
    
    # If sum of targets is too large, scale them down
    if sum(individual_targets) > available_space:
        scale_factor = available_space / sum(individual_targets)
        individual_targets = [max(100, int(target * scale_factor)) for target in individual_targets]
    
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
        # Make sure each summary preserves the original content characteristics
        prompt = f"""
        I need you to summarize the following text to approximately {target_length} characters.
        Focus on preserving:
        1. Important facts and statistics
        2. Key values and figures
        3. Critical context
        4. Essential points and conclusions
        
        The text appears to be from file #{i+1} of {len(text_list)}.
        
        Original text:
        {text[:30000]}  # Limit input size to prevent token overflow
        
        Provide ONLY the summarized text without any meta commentary.
        """
        
        # Create spinner animation
        if show_progress:
            print(f"Summarizing file {i+1}...", end='', flush=True)
            spinner_chars = "|/-\\"
            start_time = time.time()
        
        # Call the Ollama model
        try:
            # Show spinner
            if show_progress:
                sys.stdout.write("\rSummarizing file {0} {1} ".format(
                    i+1, spinner_chars[0]))
                sys.stdout.flush()
            
            # Make the actual API call
            start_time = time.time()
            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            summary = response['message']['content'].strip()
            
            # Add file identifier to help debugging
            summary = f"[Summary of file {i+1}]\n{summary}"
            
            # Ensure summary isn't too long
            if len(summary) > target_length * 1.2:  # Allow some flexibility
                summary = summary[:target_length] + "..."
            
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
            individual_summaries.append(f"[Summary of file {i+1} (truncated due to error)]\n{text[:target_length]}...")
    
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
    These are already condensed summaries, so focus on preserving the most important points from EACH file.
    It's critical that the final summary contains information from EVERY article.
    
    The summaries are separated by "---" dividers.
    
    Summaries to condense:
    {concatenated_summaries[:25000]}  # Limit input size
    
    Provide ONLY the summarized text without any meta commentary.
    Make sure to preserve information from ALL {len(individual_summaries)} files.
    """
    
    # Create spinner for final summarization
    if show_progress:
        print("Final summarization in progress...", end='', flush=True)
        spinner_chars = "|/-\\"
        start_time = time.time()
    
    try:
        # Simple spinner animation
        if show_progress:
            sys.stdout.write("\rFinal summarization in progress... {0} ".format(spinner_chars[0]))
            sys.stdout.flush()
        
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
            print(f"Falling back to concatenation and truncation")
        
        # Try a simpler approach: just concatenate the first parts of each summary
        fallback_summary = ""
        chars_per_summary = max(50, int(max_context_size / len(individual_summaries)) - 10)
        
        for i, summary in enumerate(individual_summaries):
            # Extract the first part of each summary
            excerpt = summary[:chars_per_summary].strip()
            fallback_summary += f"[File {i+1}] {excerpt}\n\n"
        
        # Ensure we're under the limit
        return fallback_summary[:max_context_size]


# Example usage
if __name__ == "__main__":
    # Example with 5 articles of different lengths and different content
    articles = [
        "E" * 1000,  # 1000-character article of E's
        "A" * 1000,  # 1000-character article of A's
        "X" * 2000,  # 2000-character article of X's
        "Y" * 3000,  # 3000-character article of Y's
        "Z" * 2000   # 2000-character article of Z's
    ]
    
    # Total length: 9000 characters, max context: 2500
    result = reduce_text(articles, max_context_size=2500)
    print(f"Result length: {len(result)}")
    print(f"Result is under max context: {len(result) <= 2500}")
    print(f"First 100 characters: {result}...")
    
    # Test with real text to demonstrate it works with actual content
    real_articles = [
        "Climate change is a pressing global issue. Rising temperatures are causing melting ice caps, rising sea levels, and extreme weather events. Countries around the world are struggling to implement effective policies to reduce carbon emissions. The Paris Agreement aims to limit global warming to well below 2 degrees Celsius.",
        "Artificial intelligence has made significant advances in recent years. Machine learning models can now perform tasks that were once thought to require human intelligence. These include image recognition, natural language processing, and even creative tasks like art and music generation.",
        "Space exploration continues to push the boundaries of human knowledge. Recent missions to Mars have provided valuable data about the red planet. Private companies are now playing a larger role in space missions alongside traditional government agencies."
    ]
    
    # Process real articles
    real_result = reduce_text(real_articles, max_context_size=500)
    print("\n\nReal articles example:")
    print(f"Result length: {len(real_result)}")
    print(f"Result is under max context: {len(real_result) <= 500}")
    print(f"Full result:\n{real_result}")