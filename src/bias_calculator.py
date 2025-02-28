# bias_calculator.py - Combined bias score calculation for TikTok videos

import pandas as pd
import numpy as np
import json
import os

from utils import get_bias_category, Logger

class BiasScoringSystem:
    """
    Combines text, video, and audio bias scores to calculate an overall bias rating.
    """
    
    def __init__(self, text_weight=0.5, video_weight=0.3, audio_weight=0.2, 
                 political_weight=0.5, religious_weight=0.5, logger=None):
        """
        Initialize the bias scoring system.
        
        Args:
            text_weight (float): Weight for text bias scores
            video_weight (float): Weight for video bias scores
            audio_weight (float): Weight for audio bias scores
            political_weight (float): Weight for political bias
            religious_weight (float): Weight for religious bias
            logger (Logger): Logger instance for output
        """
        self.text_weight = text_weight
        self.video_weight = video_weight
        self.audio_weight = audio_weight
        self.political_weight = political_weight
        self.religious_weight = religious_weight
        self.logger = logger if logger else Logger()
        
        # Normalize weights
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalize weights to ensure they sum to 1.0"""
        # Normalize source weights
        source_sum = self.text_weight + self.video_weight + self.audio_weight
        if source_sum != 1.0:
            self.text_weight /= source_sum
            self.video_weight /= source_sum
            self.audio_weight /= source_sum
        
        # Normalize bias type weights
        bias_sum = self.political_weight + self.religious_weight
        if bias_sum != 1.0:
            self.political_weight /= bias_sum
            self.religious_weight /= bias_sum
    
    def update_weights(self, **kwargs):
        """
        Update weights based on provided keyword arguments.
        
        Args:
            **kwargs: Weights to update (text_weight, video_weight, audio_weight,
                      political_weight, religious_weight)
        """
        # Update weights if provided
        if 'text_weight' in kwargs:
            self.text_weight = kwargs['text_weight']
        if 'video_weight' in kwargs:
            self.video_weight = kwargs['video_weight']
        if 'audio_weight' in kwargs:
            self.audio_weight = kwargs['audio_weight']
        if 'political_weight' in kwargs:
            self.political_weight = kwargs['political_weight']
        if 'religious_weight' in kwargs:
            self.religious_weight = kwargs['religious_weight']
        
        # Normalize the updated weights
        self._normalize_weights()
        
        self.logger.info(f"Updated weights: text={self.text_weight:.2f}, video={self.video_weight:.2f}, "
                         f"audio={self.audio_weight:.2f}, political={self.political_weight:.2f}, "
                         f"religious={self.religious_weight:.2f}")
    
    def calculate_combined_score(self, text_scores, video_scores, audio_scores=None):
        """
        Calculate a combined bias score from all analysis scores.
        
        Args:
            text_scores (dict): Dictionary with text bias scores
            video_scores (dict): Dictionary with video bias scores
            audio_scores (dict): Dictionary with audio bias scores (optional)
            
        Returns:
            dict: Dictionary with combined bias scores
        """
        # Set default audio scores if not provided
        if audio_scores is None:
            audio_scores = {
                'political_bias': 1.0,
                'religious_bias': 1.0,
                'combined_bias': 1.0
            }
        
        # Calculate political bias
        political_bias = (
            text_scores['political_bias'] * self.text_weight +
            video_scores['political_bias'] * self.video_weight +
            audio_scores['political_bias'] * self.audio_weight
        )
        
        # Calculate religious bias
        religious_bias = (
            text_scores['religious_bias'] * self.text_weight +
            video_scores['religious_bias'] * self.video_weight +
            audio_scores['religious_bias'] * self.audio_weight
        )
        
        # Calculate overall bias
        overall_bias = (
            political_bias * self.political_weight +
            religious_bias * self.religious_weight
        )
        
        return {
            'political_bias': political_bias,
            'religious_bias': religious_bias,
            'overall_bias': overall_bias,
            'text_contribution': self.text_weight * 100,
            'video_contribution': self.video_weight * 100,
            'audio_contribution': self.audio_weight * 100
        }
    
    def create_bias_report(self, text_scores, video_scores, audio_scores=None):
        """
        Create a comprehensive bias report.
        
        Args:
            text_scores (dict): Dictionary with text bias scores
            video_scores (dict): Dictionary with video bias scores
            audio_scores (dict): Dictionary with audio bias scores (optional)
            
        Returns:
            dict: Detailed bias report
        """
        combined_scores = self.calculate_combined_score(text_scores, video_scores, audio_scores)
        
        report = {
            'scores': {
                'text': text_scores,
                'video': video_scores,
                'audio': audio_scores,
                'combined': combined_scores
            },
            'categories': {
                'political_bias': get_bias_category(combined_scores['political_bias']),
                'religious_bias': get_bias_category(combined_scores['religious_bias']),
                'overall_bias': get_bias_category(combined_scores['overall_bias'])
            },
            'weights': {
                'text_weight': self.text_weight,
                'video_weight': self.video_weight,
                'audio_weight': self.audio_weight,
                'political_weight': self.political_weight,
                'religious_weight': self.religious_weight
            }
        }
        
        return report
    
    def process_dataframe(self, df):
        """
        Process a DataFrame containing bias scores to calculate combined scores.
        
        Args:
            df (pandas.DataFrame): DataFrame with bias scores
            
        Returns:
            pandas.DataFrame: DataFrame with added combined bias scores
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Initialize the output columns
        result_df['political_bias'] = None
        result_df['religious_bias'] = None
        result_df['overall_bias_score'] = None
        result_df['political_bias_category'] = None
        result_df['religious_bias_category'] = None
        result_df['overall_bias_category'] = None
        
        # Initialize bias report column
        result_df['bias_report'] = None
        
        # Process each row
        for idx, row in result_df.iterrows():
            try:
                # Extract text bias scores
                text_scores = {
                    'political_bias': row.get('text_political_bias', 1.0),
                    'religious_bias': row.get('text_religious_bias', 1.0),
                    'combined_bias': row.get('text_combined_bias', 1.0)
                }
                
                # Extract video bias scores
                video_scores = {
                    'political_bias': row.get('video_political_bias', 1.0),
                    'religious_bias': row.get('video_religious_bias', 1.0),
                    'combined_bias': row.get('video_combined_bias', 1.0)
                }
                
                # Extract audio bias scores if available
                if 'audio_political_bias' in row and 'audio_religious_bias' in row and 'audio_combined_bias' in row:
                    audio_scores = {
                        'political_bias': row.get('audio_political_bias', 1.0),
                        'religious_bias': row.get('audio_religious_bias', 1.0),
                        'combined_bias': row.get('audio_combined_bias', 1.0)
                    }
                else:
                    audio_scores = None
                
                # Calculate combined scores
                combined_scores = self.calculate_combined_score(text_scores, video_scores, audio_scores)
                
                # Update the row with combined scores
                result_df.loc[idx, 'political_bias'] = combined_scores['political_bias']
                result_df.loc[idx, 'religious_bias'] = combined_scores['religious_bias']
                result_df.loc[idx, 'overall_bias_score'] = combined_scores['overall_bias']
                
                # Add category labels
                result_df.loc[idx, 'political_bias_category'] = get_bias_category(combined_scores['political_bias'])
                result_df.loc[idx, 'religious_bias_category'] = get_bias_category(combined_scores['religious_bias'])
                result_df.loc[idx, 'overall_bias_category'] = get_bias_category(combined_scores['overall_bias'])
                
                # Add a full report column if requested
                if 'bias_report' not in result_df.columns:
                    result_df['bias_report'] = None
                    
                result_df.at[idx, 'bias_report'] = str(self.create_bias_report(text_scores, video_scores, audio_scores))
                
            except Exception as e:
                self.logger.error(f"Error processing row {idx}: {e}")
        
        return result_df
    
    def generate_summary_statistics(self, df):
        """
        Generate summary statistics from the bias analysis results.
        
        Args:
            df (pandas.DataFrame): DataFrame with bias scores
            
        Returns:
            dict: Summary statistics
        """
        # Check if the required columns exist
        required_cols = ['political_bias', 'religious_bias', 'overall_bias_score']
        if not all(col in df.columns for col in required_cols):
            self.logger.error("Required columns not found in DataFrame")
            return None
        
        # Extract bias scores
        political_scores = df['political_bias'].dropna().tolist()
        religious_scores = df['religious_bias'].dropna().tolist()
        overall_scores = df['overall_bias_score'].dropna().tolist()
        
        # Count categories
        categories = {category: 0 for category in ['Minimal Bias', 'Low Bias', 'Moderate Bias', 'High Bias', 'Extreme Bias']}
        
        if 'overall_bias_category' in df.columns:
            for category in df['overall_bias_category'].dropna():
                if category in categories:
                    categories[category] += 1
        
        # Calculate statistics
        summary = {
            'count': len(overall_scores),
            'political_bias': {
                'mean': np.mean(political_scores) if political_scores else None,
                'median': np.median(political_scores) if political_scores else None,
                'min': min(political_scores) if political_scores else None,
                'max': max(political_scores) if political_scores else None
            },
            'religious_bias': {
                'mean': np.mean(religious_scores) if religious_scores else None,
                'median': np.median(religious_scores) if religious_scores else None,
                'min': min(religious_scores) if religious_scores else None,
                'max': max(religious_scores) if religious_scores else None
            },
            'overall_bias': {
                'mean': np.mean(overall_scores) if overall_scores else None,
                'median': np.median(overall_scores) if overall_scores else None,
                'min': min(overall_scores) if overall_scores else None,
                'max': max(overall_scores) if overall_scores else None
            },
            'categories': categories
        }
        
        return summary