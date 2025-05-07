import ollama
from typing import List, Optional
import math

def summarize_text(
    text: str, 
    target_len: int = 150, 
    chunk_size: int = 8000,
    truncate: bool = False,
    model: str = 'qwen2.5:72b', 
    temperature: float = 0.3
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
    
    Returns:
        A summary of the input text
    """
    # For short texts, summarize directly
    if len(text) <= chunk_size:
        summary = _summarize_chunk(text, target_len, model, temperature)
    else:
        # For longer texts, use chunking strategy
        summary = _recursive_summarize(text, target_len, chunk_size, model, temperature)
    
    # Apply truncation if requested
    if truncate and len(summary) > target_len:
        summary = summary[:target_len]
        # Try to avoid cutting off in the middle of a word
        last_space = summary.rfind(' ')
        if last_space > 0.9 * target_len:  # Only if we're not losing too much
            summary = summary[:last_space]
    
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
        
    return chunks

def _recursive_summarize(
    text: str, 
    target_len: int, 
    chunk_size: int,
    model: str, 
    temperature: float
) -> str:
    """
    Summarize large texts by chunking and recursively summarizing.
    
    Args:
        text: The text to summarize
        target_len: Target length for the final summary
        chunk_size: Maximum character count per chunk
        model: The language model to use
        temperature: Controls randomness
    
    Returns:
        A summary of the entire text
    """
    # Split text into chunks
    chunks = _chunk_text(text, chunk_size)
    
    # Calculate proportional length for intermediate summaries
    # We use a larger size for intermediate summaries to preserve information
    intermediate_len = min(target_len * 2, 1800)  # ~300 words
    
    # Summarize each chunk
    chunk_summaries = []
    for chunk in chunks:
        summary = _summarize_chunk(chunk, intermediate_len, model, temperature)
        chunk_summaries.append(summary)
    
    # Combine intermediate summaries
    combined_summary = "\n\n".join(chunk_summaries)
    
    # Create final summary if combined summary is still too large
    if len(combined_summary) > chunk_size:
        return _recursive_summarize(combined_summary, target_len, chunk_size, model, temperature)
    else:
        return _summarize_chunk(combined_summary, target_len, model, temperature)

# Example usage
if __name__ == "__main__":
    text = "A"*1500 + "B"*1500 + "C"*4000 + "D"*4000
    summary = summarize_text(
        text=text,
        target_len=900, 
        chunk_size=8000,
        truncate=True
    )
    print(summary)