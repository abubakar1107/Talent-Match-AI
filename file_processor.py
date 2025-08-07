import streamlit as st
import PyPDF2
import io
from typing import Union

class FileProcessor:
    """
    Handles processing of different file formats for resume extraction.
    """
    
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, file) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file: Streamlit uploaded file object
            
        Returns:
            str: Extracted text content
        """
        try:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            
            # Extract text from all pages
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return text.strip()
            
        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")
    
    def extract_text_from_txt(self, file) -> str:
        """
        Extract text from a TXT file.
        
        Args:
            file: Streamlit uploaded file object
            
        Returns:
            str: File content as string
        """
        try:
            # Read the text file
            text = file.read().decode('utf-8')
            return text.strip()
            
        except UnicodeDecodeError:
            
            try:
                file.seek(0)  # Reset file pointer
                text = file.read().decode('latin-1')
                return text.strip()
            except Exception as e:
                raise Exception(f"Error reading text file with encoding: {str(e)}")
                
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")
    
    def extract_text(self, file) -> str:
        """
        Extract text from a file based on its type.
        
        Args:
            file: Streamlit uploaded file object
            
        Returns:
            str: Extracted text content
        """
        if file is None:
            raise ValueError("No file provided")
        
        # Get file extension
        file_type = file.type.lower() if file.type else ""
        file_name = file.name.lower() if file.name else ""
        
        # Reset file pointer to beginning
        file.seek(0)
        
        # Process based on file type
        if file_type == "application/pdf" or file_name.endswith('.pdf'):
            return self.extract_text_from_pdf(file)
        
        elif file_type == "text/plain" or file_name.endswith('.txt'):
            return self.extract_text_from_txt(file)
        
        else:
            # Try to read as text file by default
            try:
                return self.extract_text_from_txt(file)
            except:
                raise Exception(f"Unsupported file type: {file_type}. Please upload PDF or TXT files.")
    
    def validate_content(self, content: str, min_length: int = 50) -> bool:
        """
        Validate that extracted content is meaningful.
        
        Args:
            content (str): Extracted text content
            min_length (int): Minimum length for valid content
            
        Returns:
            bool: True if content is valid
        """
        if not content or not content.strip():
            return False
        
        # Check minimum length
        if len(content.strip()) < min_length:
            return False
        
        # Check if content has some alphabetic characters
        alpha_chars = sum(1 for c in content if c.isalpha())
        if alpha_chars < min_length // 2:
            return False
        
        return True
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                cleaned_lines.append(line)
        
        # Join lines with single newlines
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove multiple consecutive newlines
        while '\n\n\n' in cleaned_text:
            cleaned_text = cleaned_text.replace('\n\n\n', '\n\n')
        
        return cleaned_text.strip()
    
    def extract_and_validate(self, file) -> str:
        """
        Extract text from file and validate it.
        
        Args:
            file: Streamlit uploaded file object
            
        Returns:
            str: Cleaned and validated text content
            
        Raises:
            Exception: If file processing fails or content is invalid
        """
        # Extract text
        raw_text = self.extract_text(file)
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        
        # Validate content
        if not self.validate_content(cleaned_text):
            raise Exception(f"File '{file.name}' does not contain sufficient readable text content")
        
        return cleaned_text
