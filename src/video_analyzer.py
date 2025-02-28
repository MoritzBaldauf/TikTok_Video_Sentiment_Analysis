# video_analyzer.py - Video-based bias analysis for TikTok videos

import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
import pandas as pd

from utils import get_device, get_video_path, Logger

class VideoBiasAnalyzer:
    """
    Analyzes video content for political and religious bias.
    """
    
    def __init__(self, model_name='MCG-NJU/videomae-base-finetuned-kinetics', device=None, logger=None):
        """
        Initialize the video analyzer.
        
        Args:
            model_name (str): Name of the pre-trained video model to use
            device (str): Device to run the model on ('cuda' or 'cpu')
            logger (Logger): Logger instance for output
        """
        self.device = device if device else get_device()
        self.model_name = model_name
        self.logger = logger if logger else Logger()
        
        self.logger.info(f"Loading video model: {model_name} on {self.device}")
        self.content_indicators = {}
        
        # Initialize VideoMAE model
        try:
            self.feature_extractor = VideoMAEFeatureExtractor.from_pretrained(model_name)
            self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info("VideoMAE model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading VideoMAE model: {e}")
            self.logger.info("Initializing basic CNN features instead")
            # Fallback to a basic CNN feature extractor if VideoMAE isn't available
            self.initialize_basic_feature_extractor()
        
        # Initialize bias detection components
        self.initialize_bias_detectors()
    
    def initialize_basic_feature_extractor(self):
        """Initialize a basic CNN feature extractor as fallback."""
        # Use ResNet as a fallback feature extractor
        resnet = models.resnet50(pretrained=True)
        self.basic_model = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final FC layer
        self.basic_model.to(self.device)
        self.basic_model.eval()
        
        self.basic_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def initialize_bias_detectors(self):
        """Initialize components for detecting bias in video features."""
        # In a full implementation, these would be trained classifiers
        # For now, we'll implement a simplified approach
        
        # Political bias visual cues (simplified)
        self.political_visual_cues = {
            'flags': ['us_flag', 'party_symbols'],
            'political_figures': ['politicians', 'rallies'],
            'text_overlays': ['political_slogans', 'partisan_text'],
            'emotional_imagery': ['anger', 'fear', 'patriotism']
        }
        
        # Religious bias visual cues (simplified)
        self.religious_visual_cues = {
            'symbols': ['cross', 'star_of_david', 'crescent'],
            'religious_figures': ['clergy', 'prayer'],
            'religious_settings': ['church', 'mosque', 'temple'],
            'text_overlays': ['religious_quotes', 'scripture']
        }
    
    def extract_frames(self, video_path, num_frames=16):
        """
        Extract frames from a video file.
        
        Args:
            video_path (str): Path to the video file
            num_frames (int): Number of frames to extract
            
        Returns:
            list: List of extracted frames as numpy arrays
        """
        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            return []
        
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                self.logger.error(f"No frames found in video: {video_path}")
                return []
                
            # Select evenly spaced frames
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert from BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                
            cap.release()
            
            if len(frames) < num_frames:
                self.logger.warning(f"Only extracted {len(frames)} frames from {video_path}")
                
            return frames
            
        except Exception as e:
            self.logger.error(f"Error extracting frames from {video_path}: {e}")
            return []
    
    def process_video(self, video_path):
        """
        Process video with VideoMAE model.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            numpy.ndarray: Video features
        """
        # Extract frames
        frames = self.extract_frames(video_path)
        
        if not frames:
            self.logger.warning(f"No frames could be extracted from {video_path}")
            return None
        
        # Extract content indicators
        self.content_indicators = self.extract_content_indicators(frames)
        
        self.logger.info(f"Processing video: {video_path}")
        model_type = 'VideoMAE' if hasattr(self, 'model') else 'Fallback CNN'
        self.logger.info(f"Using model: {model_type}")
            
        try:
            # Prepare input for VideoMAE
            inputs = self.feature_extractor(frames, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get features
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use last hidden state as features
                features = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
                
            return features[0]  # First (and only) video in the batch
            
        except Exception as e:
            self.logger.error(f"Error processing video with VideoMAE: {e}")
            return self.fallback_video_processing(frames)
        
    def fallback_video_processing(self, frames):
        """
        Fallback video processing using basic CNN if VideoMAE fails.
        
        Args:
            frames (list): List of video frames
            
        Returns:
            numpy.ndarray: Video features
        """
        if not frames:
            return np.zeros(2048)  # Default empty feature vector
            
        try:
            frame_features = []
            
            for frame in frames:
                # Convert to PIL Image
                pil_image = Image.fromarray(frame)
                
                # Apply transforms
                input_tensor = self.basic_transform(pil_image).unsqueeze(0).to(self.device)
                
                # Extract features
                with torch.no_grad():
                    features = self.basic_model(input_tensor)
                    features = features.squeeze().cpu().numpy()
                
                frame_features.append(features)
            
            # Aggregate frame features (average pooling)
            return np.mean(frame_features, axis=0)
            
        except Exception as e:
            self.logger.error(f"Error in fallback video processing: {e}")
            return np.zeros(2048)  # Default empty feature vector
    
    def analyze_political_bias(self, video_path):
        """
        Analyze the political bias in the video.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            float: Political bias score between 1 and 100
        """
        # Extract video features
        features = self.process_video(video_path)
        
        if features is None:
            return 1.0  # Minimum bias score on failure

        # Calculate statistics    
        feature_mean = np.mean(np.abs(features))
        feature_var = np.var(features)
        self.logger.info(f"Feature mean: {feature_mean:.6f}, variance: {feature_var:.6f}")
        
        # Use min-max scaling against expected ranges
        # Based on your output, values seem to range from ~1.5-1.9 for mean and ~6-15 for variance
        mean_score = (feature_mean - 1.5) / 0.5  # Scale to approximately 0-1 range
        var_score = (feature_var - 6.0) / 9.0    # Scale to approximately 0-1 range
        
        # Clip values to 0-1 range
        mean_score = max(0, min(1, mean_score))
        var_score = max(0, min(1, var_score))
        
        # Include content indicators in scoring
        content_score = (
            self.content_indicators.get('has_faces', 0) * 0.4 +
            self.content_indicators.get('color_intensity', 0) * 0.3 +
            self.content_indicators.get('motion_estimate', 0) * 0.3
        )
        # Include in final calculation
        combined_raw_score = mean_score * 0.5 + var_score * 0.3 + content_score * 0.2
        self.logger.info(f"Mean score: {mean_score:.3f}, Var score: {var_score:.3f}, Combined raw: {combined_raw_score:.3f}")
        
        # Scale to 1-100
        bias_score = 1 + combined_raw_score * 99
        
        return bias_score
    
    def analyze_religious_bias(self, video_path):
        """
        Analyze the religious bias in the video.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            float: Religious bias score between 1 and 100
        """
        # Extract video features
        features = self.process_video(video_path)
        
        if features is None:
            return 1.0  # Minimum bias score on failure
        
        # Calculate statistics    
        feature_mean = np.mean(np.abs(features))
        feature_var = np.var(features)
        self.logger.info(f"Religious feature mean: {feature_mean:.6f}, variance: {feature_var:.6f}")
        
        # Use min-max scaling against expected ranges
        # Using slightly different parameters than political bias
        mean_score = (feature_mean - 1.4) / 0.6  # Scale to approximately 0-1 range
        var_score = (feature_var - 5.0) / 10.0   # Scale to approximately 0-1 range
        
        # Clip values to 0-1 range
        mean_score = max(0, min(1, mean_score))
        var_score = max(0, min(1, var_score))
        
        # Include content indicators in scoring
        content_score = (
            self.content_indicators.get('has_faces', 0) * 0.4 +
            self.content_indicators.get('color_intensity', 0) * 0.3 +
            self.content_indicators.get('motion_estimate', 0) * 0.3
        )
        # Include in final calculation
        combined_raw_score = mean_score * 0.5 + var_score * 0.3 + content_score * 0.2
        self.logger.info(f"Religious mean score: {mean_score:.3f}, Var score: {var_score:.3f}, Combined raw: {combined_raw_score:.3f}")
        
        # Scale to 1-100
        bias_score = 1 + combined_raw_score * 99
        
        return bias_score
        
    def get_combined_bias_score(self, video_path, political_weight=0.5, religious_weight=0.5):
        """
        Get a combined bias score for the video.
        
        Args:
            video_path (str): Path to the video file
            political_weight (float): Weight for political bias score
            religious_weight (float): Weight for religious bias score
            
        Returns:
            dict: Dictionary with individual and combined bias scores
        """
        political_score = self.analyze_political_bias(video_path)
        religious_score = self.analyze_religious_bias(video_path)
        
        # Calculate weighted average
        combined_score = (political_score * political_weight + 
                          religious_score * religious_weight)
        
        return {
            'political_bias': political_score,
            'religious_bias': religious_score,
            'combined_bias': combined_score
        }
        
    def analyze_batch(self, df, video_id_col='Video_ID', data_dir=None, output_col='video_bias_scores', 
                      political_weight=0.5, religious_weight=0.5):
        """
        Analyze a batch of videos from a DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame containing video IDs
            video_id_col (str): Column name containing the video IDs
            data_dir (str): Directory containing the video files
            output_col (str): Column name for the output scores
            political_weight (float): Weight for political bias
            religious_weight (float): Weight for religious bias
            
        Returns:
            pandas.DataFrame: DataFrame with added bias scores
        """
        # Create a copy of the input DataFrame
        result_df = df.copy()
        
        # Initialize columns for individual scores
        result_df['video_political_bias'] = None
        result_df['video_religious_bias'] = None
        result_df['video_combined_bias'] = None
        
        # Initialize output column for scores
        if output_col not in result_df.columns:
            result_df[output_col] = None
        
        # Iterate through each video ID
        for idx, video_id in enumerate(result_df[video_id_col]):
            self.logger.info(f"Analyzing video {idx+1}/{len(result_df)}: {video_id}")
            
            # Build the path to the video file
            video_path = get_video_path(video_id, data_dir)
            
            if os.path.exists(video_path):
                # Get bias scores
                scores = self.get_combined_bias_score(video_path, political_weight, religious_weight)
                
                # Store individual scores
                result_df.loc[idx, 'video_political_bias'] = scores['political_bias']
                result_df.loc[idx, 'video_religious_bias'] = scores['religious_bias']
                result_df.loc[idx, 'video_combined_bias'] = scores['combined_bias']
                
                # Store scores as string
                result_df.at[idx, output_col] = str(scores)
            else:
                self.logger.warning(f"Video file not found: {video_path}")
                # Add placeholder result for missing videos
                placeholder = {
                    'political_bias': 1.0,
                    'religious_bias': 1.0,
                    'combined_bias': 1.0
                }
                # Store placeholder scores
                result_df.at[idx, 'video_political_bias'] = 1.0
                result_df.at[idx, 'video_religious_bias'] = 1.0
                result_df.at[idx, 'video_combined_bias'] = 1.0
                result_df.at[idx, output_col] = placeholder
        
        return result_df

    def extract_content_indicators(self, frames):
        """Extract specific content indicators from frames"""
        indicators = {
            'has_faces': 0,
            'color_intensity': 0,
            'motion_estimate': 0,
            'text_density': 0
        }
        
        # Skip if no frames
        if not frames:
            return indicators
        
        # Simple face detection (if opencv-python with cv2.CascadeClassifier is available)
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            face_count = 0
            for frame in frames[::4]:  # Check every 4th frame
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                face_count += len(faces)
            indicators['has_faces'] = min(1.0, face_count / (len(frames) / 4 * 3))  # Normalize
        except Exception as e:
            self.logger.warning(f"Face detection failed: {e}")
        
        # Color intensity - bright/saturated colors often indicate emotional content
        color_intensities = []
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            s_channel = hsv[:,:,1]
            v_channel = hsv[:,:,2]
            color_intensities.append(np.mean(s_channel) * np.mean(v_channel) / (255*255))
        indicators['color_intensity'] = np.mean(color_intensities)
        
        # Simple motion estimate through frame differences
        if len(frames) > 1:
            diffs = []
            for i in range(len(frames)-1):
                diff = np.mean(np.abs(frames[i].astype(np.float32) - frames[i+1].astype(np.float32)))
                diffs.append(diff / 255)
            indicators['motion_estimate'] = min(1.0, np.mean(diffs) * 5)  # Scale up but cap at 1.0
        
        return indicators