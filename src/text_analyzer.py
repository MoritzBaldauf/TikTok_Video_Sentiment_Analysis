# text_analyzer.py - Text-based bias analysis for TikTok videos

import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from utils import preprocess_text, get_device, Logger

class TextBiasAnalyzer:
    """
    Analyzes political and religious bias in text descriptions.
    """
    
    def __init__(self, model_name='bert-base-uncased', device=None, logger=None):
        """
        Initialize the text bias analyzer.
        
        Args:
            model_name (str): Name of the pre-trained BERT model to use
            device (str): Device to run the model on ('cuda' or 'cpu')
            logger (Logger): Logger instance for output
        """
        self.model_name = model_name
        self.device = device if device else get_device()
        self.logger = logger if logger else Logger()
        
        self.logger.info(f"Loading BERT model: {model_name} on {self.device}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # For feature extraction approach
        self.bert_model = BertModel.from_pretrained(model_name)
        self.bert_model.to(self.device)
        self.bert_model.eval()
        
        # Initialize political and religious bias word lists
        self.initialize_bias_indicators()
        
        # Initialize scalers for the scores
        self.political_scaler = MinMaxScaler(feature_range=(1, 100))
        self.religious_scaler = MinMaxScaler(feature_range=(1, 100))
        
        # Fit scalers with expected range
        self.political_scaler.fit(np.array([[0], [1]]))
        self.religious_scaler.fit(np.array([[0], [1]]))
    
    def initialize_bias_indicators(self):
        """Initialize word lists that indicate potential political or religious bias."""
        # Political bias indicators (simplified list - would be more comprehensive in production)
        self.political_terms = {
            'left_leaning': [
                'progressive', 'liberal', 'democrat', 'socialism', 'green new deal',
                'universal healthcare', 'defund', 'climate crisis', 'equity'
            ],
            'right_leaning': [
                'conservative', 'republican', 'trump', 'maga', 'border wall',
                'pro-life', 'freedom', 'patriot', 'anti-woke', 'second amendment'
            ],
            'politically_charged': [
                'radical', 'extremist', 'corrupt', 'conspiracy', 'fake news',
                'indoctrination', 'marxist', 'fascist', 'communist', 'destroy'
            ]
        }
        
        # Religious bias indicators (simplified list - would be more comprehensive in production)
        self.religious_terms = {
            'christian': [
                'jesus', 'bible', 'christ', 'gospel', 'sin', 'salvation', 'prayer',
                'church', 'faith', 'blessed', 'holy', 'god'
            ],
            'islamic': [
                'allah', 'quran', 'muhammad', 'islam', 'muslim', 'mosque',
                'ramadan', 'jihad', 'halal', 'haram'
            ],
            'other_religious': [
                'buddhist', 'hindu', 'jewish', 'torah', 'karma', 'religious',
                'atheist', 'secular', 'spiritual', 'divine'
            ],
            'religiously_charged': [
                'blasphemy', 'heresy', 'infidel', 'godless', 'sacred', 'unholy',
                'righteous', 'sinner', 'heathen', 'convert'
            ]
        }
    
    def extract_bert_embeddings(self, text):
        """
        Extract BERT embeddings for the text.
        
        Args:
            text (str): The text to extract embeddings for
            
        Returns:
            numpy.ndarray: The extracted embeddings
        """
        # Preprocess the text
        text = preprocess_text(text)
        
        # Return empty embedding for empty text
        if not text:
            return np.zeros(768)  # BERT base hidden size
        
        # Tokenize the text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get the embeddings
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use the [CLS] token embedding as the text representation
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings[0]
    
    def calculate_keyword_bias_score(self, text, term_lists, weights=None):
        """
        Calculate bias score based on presence of terms from term lists.
        
        Args:
            text (str): The text to analyze
            term_lists (dict): Dictionary of term categories and their terms
            weights (dict): Optional weights for each category
            
        Returns:
            float: Raw bias score between 0 and 1
        """
        if weights is None:
            # Default equal weights
            weights = {category: 1.0 for category in term_lists.keys()}
            
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate score for each category
        text = text.lower()
        scores = {}
        
        for category, terms in term_lists.items():
            category_hits = sum(1 for term in terms if term in text)
            category_score = min(1.0, category_hits / 3.0)  # Cap at 1.0, require 3 hits for max score
            scores[category] = category_score
        
        # Calculate weighted score
        weighted_score = sum(scores.get(category, 0) * normalized_weights.get(category, 0) 
                             for category in term_lists.keys())
        
        return weighted_score
    
    def analyze_political_bias(self, text):
        """
        Analyze the political bias in the text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            float: Political bias score between 1 and 100
        """
        # Get BERT embeddings
        embeddings = self.extract_bert_embeddings(text)
        
        # Calculate keyword-based bias score
        keyword_score = self.calculate_keyword_bias_score(
            text, 
            self.political_terms,
            weights={
                'left_leaning': 1.0,
                'right_leaning': 1.0,
                'politically_charged': 1.5  # Give more weight to charged language
            }
        )
        
        # Scale to 1-100 range
        scaled_score = self.political_scaler.transform([[keyword_score]])[0][0]
        
        return scaled_score
    
    def analyze_religious_bias(self, text):
        """
        Analyze the religious bias in the text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            float: Religious bias score between 1 and 100
        """
        # Get BERT embeddings
        embeddings = self.extract_bert_embeddings(text)
        
        # Calculate keyword-based bias score
        keyword_score = self.calculate_keyword_bias_score(
            text, 
            self.religious_terms,
            weights={
                'christian': 1.0,
                'islamic': 1.0,
                'other_religious': 1.0,
                'religiously_charged': 1.5  # Give more weight to charged language
            }
        )
        
        # Scale to 1-100 range
        scaled_score = self.religious_scaler.transform([[keyword_score]])[0][0]
        
        return scaled_score
    
    def get_combined_bias_score(self, text, political_weight=0.5, religious_weight=0.5):
        """
        Get a combined bias score for the text.
        
        Args:
            text (str): The text to analyze
            political_weight (float): Weight for political bias score
            religious_weight (float): Weight for religious bias score
            
        Returns:
            dict: Dictionary with individual and combined bias scores
        """
        political_score = self.analyze_political_bias(text)
        religious_score = self.analyze_religious_bias(text)
        
        # Calculate weighted average for combined score
        combined_score = (political_score * political_weight + 
                          religious_score * religious_weight)
        
        return {
            'political_bias': political_score,
            'religious_bias': religious_score,
            'combined_bias': combined_score
        }
    
    def analyze_batch(self, df, description_col='Description', output_col='text_bias_scores', political_weight=0.5, religious_weight=0.5):
        """
        Analyze a batch of text descriptions from a DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame containing text to analyze
            description_col (str): Column name containing the text descriptions
            output_col (str): Column name for the output scores
            political_weight (float): Weight for political bias
            religious_weight (float): Weight for religious bias
            
        Returns:
            pandas.DataFrame: DataFrame with added bias scores
        """
        # Create a copy of the input DataFrame
        result_df = df.copy()
        
        # Ensure the Description column has string values
        result_df[description_col] = result_df[description_col].fillna("")
        
        # Initialize columns for individual scores
        result_df['text_political_bias'] = None
        result_df['text_religious_bias'] = None
        result_df['text_combined_bias'] = None
        
        # Initialize column for storing scores as strings
        if output_col not in result_df.columns:
            result_df[output_col] = None
        
        # Get the descriptions as a list
        descriptions = result_df[description_col].tolist()
        
        self.logger.info(f"Analyzing {len(descriptions)} text descriptions")
        
        # Process each description
        for idx, text in enumerate(descriptions):
            self.logger.info(f"Analyzing text {idx+1}/{len(descriptions)}")
            
            # Get bias scores
            scores = self.get_combined_bias_score(text, political_weight, religious_weight)
            
            # Store individual scores
            result_df.loc[idx, 'text_political_bias'] = scores['political_bias']
            result_df.loc[idx, 'text_religious_bias'] = scores['religious_bias'] 
            result_df.loc[idx, 'text_combined_bias'] = scores['combined_bias']
            
            # Store the scores as a string representation
            result_df.at[idx, output_col] = str(scores)
        
        return result_df