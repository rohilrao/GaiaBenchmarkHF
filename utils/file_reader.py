import os
from PyPDF2 import PdfReader
import sys
from bs4 import BeautifulSoup
import docx
import mammoth
import pandas as pd

def read_file(filename):
    """
    Read content from a file, supporting multiple formats including PDF, DOCX, DOC, HTML, XLSX, and text.
    
    Args:
        filename (str): Path to the file to read
        
    Returns:
        str: Content of the file
    """
    # Check if file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    # Extract file extension
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    
    # Handle different file types
    if ext == '.pdf':
        try:
            pdf_reader = PdfReader(filename)
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text()
            return content
        except Exception as e:
            raise Exception(f"Error reading PDF file: {e}")
    
    elif ext == '.docx':
        try:
            doc = docx.Document(filename)
            content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return content
        except Exception as e:
            raise Exception(f"Error reading DOCX file: {e}")
    
    elif ext == '.doc':
        try:
            # Use mammoth for older Word formats
            with open(filename, 'rb') as file:
                result = mammoth.extract_raw_text(file)
                return result.value
        except Exception as e:
            raise Exception(f"Error reading DOC file: {e}")
    
    elif ext in ['.xlsx', '.xls', '.xlsm', '.xlsb']:
        try:
            # Read Excel files and others with pandas
            df = pd.read_excel(filename, engine='openpyxl' if ext != '.xls' else 'xlrd')
            # Convert DataFrame to string representation
            return df.to_string()
        except Exception as e:
            raise Exception(f"Error reading Excel file: {e}")
    
    elif ext == '.html' or ext == '.htm':
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                return soup.get_text()
        except UnicodeDecodeError:
            with open(filename, 'r', encoding='latin-1') as file:
                soup = BeautifulSoup(file, 'html.parser')
                return soup.get_text()
        except Exception as e:
            raise Exception(f"Error reading HTML file: {e}")
    
    # Handle text-based files
    else:
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with a different encoding if utf-8 fails
            with open(filename, 'r', encoding='latin-1') as file:
                return file.read()


if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        try:
            content = read_file(file_path)
            print(f"Successfully read file: {file_path}")
            print(f"First 100 characters: {content[:100]}...")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python file_reader.py <path_to_file>")

