# Configuration settings for the TikTok bias detection pipeline

import os
import json
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "data")

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# File paths
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config", "default_config.json")
DEFAULT_INPUT_CSV = os.path.join(DATA_DIR, "data_table.csv")
DEFAULT_OUTPUT_CSV = os.path.join(DATA_DIR, "final_analysis.csv")

# Intermediate files
VIDEO_ANALYSIS_FILE = os.path.join(DATA_DIR, "video_analysis_results.csv")
TEXT_ANALYSIS_FILE = os.path.join(DATA_DIR, "text_analysis_results.csv")
AUDIO_ANALYSIS_FILE = os.path.join(DATA_DIR, "audio_analysis_results.csv")
MERGED_RESULTS_FILE = os.path.join(DATA_DIR, "merged_analysis_results.csv")
EXTRACTED_AUDIO_DIR = os.path.join(DATA_DIR, "extracted_audio")

# Model settings
BERT_MODEL_NAME = 'bert-base-uncased'
VIDEO_MODEL_NAME = 'MCG-NJU/videomae-base-finetuned-kinetics'
SPEECH_MODEL = 'facebook/wav2vec2-base-960h'
SENTIMENT_MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'

# Default weights
DEFAULT_WEIGHTS = {
    "text_weight": 0.5,
    "video_weight": 0.3,
    "audio_weight": 0.2,
    "political_weight": 0.5,
    "religious_weight": 0.5
}

def load_config(config_path=None):
    """
    Load configuration from a JSON file or use defaults.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration settings
    """
    config = DEFAULT_WEIGHTS.copy()
    
    # Create config directory if it doesn't exist
    os.makedirs(os.path.dirname(DEFAULT_CONFIG_PATH), exist_ok=True)
    
    # Create default config file if it doesn't exist
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        with open(DEFAULT_CONFIG_PATH, 'w') as f:
            json.dump(DEFAULT_WEIGHTS, f, indent=4)
    
    # Load custom config if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                config.update(custom_config)
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {str(e)}")
            print("Using default configuration.")
    
    return config

def save_config(config, config_path=DEFAULT_CONFIG_PATH):
    """
    Save configuration to a JSON file.
    
    Args:
        config (dict): Configuration settings
        config_path (str): Path to save the configuration file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving configuration to {config_path}: {str(e)}")