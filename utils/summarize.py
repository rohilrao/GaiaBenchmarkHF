import ollama
from typing import List, Optional
import math

def summarize_text(
    text: str, 
    target_len: int = 150, 
    chunk_size: int = 8000,
    truncate: bool = False,
    model: str = 'qwen2.5:72b', 
    temperature: float = 0.3,
    show_progress: bool = True
) -> str:
    """
    Summarize text string using a language model with optional chunking.
    
    Args:
        text: The text to summarize (as a string)
        target_len: Target length (in characters) for the summary
        chunk_size: Maximum character count per chunk for long texts
        truncate: If True, truncate final summary to target_len
        model: The language model to use
        temperature: Controls randomness (lower = more deterministic)
        show_progress: Whether to print progress updates during summarization
    
    Returns:
        A summary of the input text
    """
    # For short texts, summarize directly
    if len(text) <= chunk_size:
        if show_progress:
            print("Text is short enough to summarize directly (no chunking needed)")
        summary = _summarize_chunk(text, target_len, model, temperature)
    else:
        # For longer texts, use chunking strategy
        summary = _recursive_summarize(text, target_len, chunk_size, model, temperature, show_progress)
    
    # Apply truncation if requested
    if truncate and len(summary) > target_len:
        if show_progress:
            print(f"Truncating summary from {len(summary)} to {target_len} characters")
        summary = summary[:target_len]
        # Try to avoid cutting off in the middle of a word
        last_space = summary.rfind(' ')
        if last_space > 0.9 * target_len:  # Only if we're not losing too much
            summary = summary[:last_space]
    
    if show_progress:
        print(f"Summarization complete. Final summary length: {len(summary)} characters")
    
    return summary

def _summarize_chunk(
    text: str, 
    target_len: int,
    model: str, 
    temperature: float
) -> str:
    """
    Summarize a single chunk of text using a language model.
    
    Args:
        text: The text to summarize
        target_len: Target length for the summary
        model: The language model to use
        temperature: Controls randomness
    
    Returns:
        A summary of the input text
    """
    # Estimate words based on average English word length of ~5 chars + space
    approx_words = math.ceil(target_len / 6)
    
    prompt = (
        "You are an expert abstractor.\n\n"
        f"Please summarize the following text in about {approx_words} words. "
        "Focus on including key facts, figures, and main points:\n\n{text}"
    )
    
    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'user', 'content': prompt.format(text=text)}
        ],
        options={'temperature': temperature}
    )
    
    return response['message']['content']

def _chunk_text(text: str, chunk_size: int) -> List[str]:
    """
    Split text into chunks of approximately equal size.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum number of characters per chunk
    
    Returns:
        List of text chunks
    """
    # Split by paragraphs first to preserve paragraph integrity
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph exceeds chunk size, start a new chunk
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    # Handle case where text is shorter than chunk_size but has no paragraphs
    if not chunks and text:
        words = text.split()
        current_chunk = ""
        for word in words:
            if len(current_chunk) + len(word) + 1 > chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = word
            else:
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word
        if current_chunk:
            chunks.append(current_chunk)
            
    return chunks

def _recursive_summarize(
    text: str, 
    target_len: int, 
    chunk_size: int,
    model: str, 
    temperature: float,
    show_progress: bool,
    depth: int = 0
) -> str:
    """
    Summarize large texts by chunking and recursively summarizing.
    
    Args:
        text: The text to summarize
        target_len: Target length for the final summary
        chunk_size: Maximum character count per chunk
        model: The language model to use
        temperature: Controls randomness
        show_progress: Whether to print progress updates
        depth: Current recursion depth (for progress reporting)
    
    Returns:
        A summary of the entire text
    """
    # Split text into chunks - fixing the chunking issue
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph exceeds chunk size, start a new chunk
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    # If no paragraphs were found, force chunking by character count
    if len(chunks) <= 1 and len(text) > chunk_size:
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
    
    if show_progress:
        if depth == 0:
            print(f"Text split into {len(chunks)} chunks for processing")
        else:
            print(f"Recursive level {depth}: Split into {len(chunks)} chunks")
    
    # Calculate proportional length for intermediate summaries
    # We use a larger size for intermediate summaries to preserve information
    intermediate_len = min(target_len * 2, 1800)  # ~300 words
    
    # Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        if show_progress:
            print(f"Processing chunk {i+1}/{len(chunks)} (length: {len(chunk)} characters)")
        
        summary = _summarize_chunk(chunk, intermediate_len, model, temperature)
        chunk_summaries.append(summary)
        
        if show_progress:
            print(f"Chunk {i+1} summarized to {len(summary)} characters")
    
    # Combine intermediate summaries
    combined_summary = "\n\n".join(chunk_summaries)
    
    return combined_summary
    '''
    if show_progress:
        print(f"Combined {len(chunks)} chunk summaries into {len(combined_summary)} characters")
    
    # Create final summary if combined summary is still too large
    if len(combined_summary) > chunk_size:
        if show_progress:
            print(f"Combined summary still too large ({len(combined_summary)} > {chunk_size}). Recursively summarizing...")
        return _recursive_summarize(combined_summary, target_len, chunk_size, model, temperature, show_progress, depth + 1)
    else:
        if show_progress:
            print("Creating final summary...")
        return _summarize_chunk(combined_summary, target_len, model, temperature)
    '''
if __name__ == "__main__":
    # Example long text
    long_text = "A"*4000 + "B"*4000 + "C"*4000 + "D"*4000 + "E"*4000
    
    # Advanced usage with custom parameters
    custom_summary = summarize_text(
        text=long_text,
        target_len=600,         # Target length in characters
        chunk_size=3000,        # Smaller chunks
        truncate=False,          # Enforce exact length limit
        model="llama3:8b",      # Use a different model
        temperature=0.5         # Slightly more creative
    )
    print("Custom summary:", custom_summary)