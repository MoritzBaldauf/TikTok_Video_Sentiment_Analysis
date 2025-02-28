# Shared utility functions for the TikTok bias detection pipeline

import os
import pandas as pd
import numpy as np
import torch
import re
from pathlib import Path

def is_cuda_available():
    """Check if CUDA is available for GPU acceleration"""
    return torch.cuda.is_available()

def get_device():
    """Get the appropriate device for computation"""
    return 'cuda' if is_cuda_available() else 'cpu'

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)
    return directory

def get_video_path(video_id, base_dir):
    """Build the path to a video file based on ID"""
    return os.path.join(base_dir, f"{video_id}.mp4")

def convert_score_string(score_str):
    """Convert the string representation of scores to a dictionary with float values"""
    try:
        # Remove np.float64() wrapper and convert to regular float
        score_str = score_str.replace('np.float64(', '').replace(')', '')
        # Convert to dictionary
        score_dict = eval(score_str)
        return {
            'political_bias': float(score_dict['political_bias']),
            'religious_bias': float(score_dict['religious_bias']),
            'combined_bias': float(score_dict['combined_bias'])
        }
    except:
        return None

def preprocess_text(text):
    """Preprocess text for analysis"""
    if pd.isna(text) or text == '':
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Handle hashtags - preserve the text but remove the # symbol
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Basic cleaning
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def scale_value(value, old_min=0, old_max=1, new_min=1, new_max=100):
    """Scale a value from one range to another"""
    # Ensure value is within the old range
    value = max(old_min, min(old_max, value))
    
    # Calculate the scaled value
    return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)

def clean_transcript(text, corrections=None):
    """
    Clean a transcript to remove gibberish and format it for analysis.
    """
    if not text or len(text.strip()) < 5:
        return ""
        
    # Convert to uppercase for consistency with most ASR output
    text = text.upper()
    
    # Apply specific corrections if provided
    if corrections:
        for error, correction in corrections.items():
            text = re.sub(r'\b' + error + r'\b', correction, text, flags=re.IGNORECASE)
    
    # Remove strings of random characters (likely noise/gibberish)
    # This regex looks for words with 3+ characters that have no vowels
    text = re.sub(r'\b[^AEIOU\s]{3,}\b', '', text)
    
    # Remove words that are likely gibberish (random consonant patterns)
    text = re.sub(r'\b[BCDFGHJKLMNPQRSTVWXYZ]{4,}\b', '', text)
    
    # Remove short nonsense words (adjust as needed)
    words = text.split()
    common_short_words = ['I', 'A', 'AN', 'THE', 'TO', 'OF', 'IN', 'ON', 'BY', 'AND', 'BUT', 'OR', 'SO', 'WE', 'US', 'AM', 'IS', 'ARE']
    cleaned_words = []
    
    for word in words:
        # Keep meaningful short words, words with vowels, or longer words
        if word in common_short_words or any(vowel in word for vowel in 'AEIOU') or len(word) > 3:
            cleaned_words.append(word)
    
    # Join the words back together
    cleaned_text = ' '.join(cleaned_words)
    
    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def get_bias_category(score):
    """Get the bias category based on the score"""
    if score < 20:
        return "Minimal Bias"
    elif score < 40:
        return "Low Bias"
    elif score < 60:
        return "Moderate Bias"
    elif score < 80:
        return "High Bias"
    else:
        return "Extreme Bias"

class Logger:
    """Simple logger for the bias detection pipeline"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def info(self, message):
        """Log an informational message"""
        if self.verbose:
            print(f"[INFO] {message}")
    
    def warning(self, message):
        """Log a warning message"""
        if self.verbose:
            print(f"[WARNING] {message}")
    
    def error(self, message):
        """Log an error message"""
        print(f"[ERROR] {message}")
    
    def section(self, title):
        """Log a section header"""
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"{title}")
            print(f"{'='*50}\n")