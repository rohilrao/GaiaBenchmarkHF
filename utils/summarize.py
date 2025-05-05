import ollama

def reduce_text(text_list, max_context_size=2500, model='deepseek-r1:32b'):
    """
    Takes a list of texts and reduces them to a single text under the specified maximum length,
    preserving important facts and key values.
    
    Args:
        text_list (list): List of text strings to be reduced
        max_context_size (int): Maximum desired length of the output text
        model (str): Ollama model to use for summarization
    
    Returns:
        str: A single text containing the most important information, under max_context_size
    """
    # Join texts with clear section breaks
    concatenated = "\n\n---\n\n".join(text_list)
    
    # Check if summarization is needed
    if len(concatenated) <= max_context_size:
        return concatenated
    
    # Calculate compression ratio needed
    compression_ratio = max_context_size / len(concatenated)
    target_length = int(max_context_size * 0.95)  # 5% buffer to ensure we stay under limit
    
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
    
    # Call the Ollama model
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        summary = response['message']['content'].strip()
        
        # If we still exceed the max context size, try another round with stricter constraints
        if len(summary) > max_context_size:
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
            
            summary = response['message']['content'].strip()
            
            # If still too long, truncate as last resort
            if len(summary) > max_context_size:
                summary = summary[:max_context_size - 3] + "..."
        
        return summary
        
    except Exception as e:
        print(f"Error summarizing texts: {e}")
        # Fall back to simple truncation on error
        return concatenated[:max_context_size - 3] + "..."


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