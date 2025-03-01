import os
import pandas as pd
import time
from pathlib import Path

from text_analyzer import TextBiasAnalyzer
from video_analyzer import VideoBiasAnalyzer
from audio_analyzer import AudioBiasAnalyzer
from bias_calculator import BiasScoringSystem
from utils import Logger, ensure_dir
import config

class TikTokBiasDetectionPipeline:
    """
    Main pipeline for TikTok bias detection that coordinates all components.
    """
    
    def __init__(self, config_path=None, verbose=True):
        """
        Initialize the TikTok bias detection pipeline.
        
        Args:
            config_path (str): Path to a custom configuration file
            verbose (bool): Whether to print verbose output
        """
        self.logger = Logger(verbose=verbose)
        self.logger.info("Initializing TikTok Bias Detection Pipeline")
        
        # Load configuration
        self.config = config.load_config(config_path)
        self.logger.info(f"Loaded configuration with weights: {self.config}")
        
        # Initialize component weights
        self.text_weight = self.config['text_weight']
        self.video_weight = self.config['video_weight']
        self.audio_weight = self.config['audio_weight']
        self.political_weight = self.config['political_weight']
        self.religious_weight = self.config['religious_weight']
        
        # Create component instances
        self.logger.info("Initializing analysis components...")
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all analysis components"""
        # Create the bias calculator
        self.bias_calculator = BiasScoringSystem(
            text_weight=self.text_weight,
            video_weight=self.video_weight,
            audio_weight=self.audio_weight,
            political_weight=self.political_weight,
            religious_weight=self.religious_weight,
            logger=self.logger
        )
        
        # Text analyzer is lightweight, so always initialize it
        self.text_analyzer = TextBiasAnalyzer(
            model_name=config.BERT_MODEL_NAME,
            logger=self.logger
        )
        
        # Video and audio analyzers are heavier, so initialize them on demand
        self.video_analyzer = None
        self.audio_analyzer = None
    
    def _ensure_video_analyzer(self):
        """Initialize the video analyzer if not already initialized"""
        if self.video_analyzer is None:
            self.logger.info("Initializing video analyzer...")
            self.video_analyzer = VideoBiasAnalyzer(
                model_name=config.VIDEO_MODEL_NAME,
                logger=self.logger
            )
    
    def _ensure_audio_analyzer(self):
        """Initialize the audio analyzer if not already initialized"""
        if self.audio_analyzer is None:
            self.logger.info("Initializing audio analyzer...")
            self.audio_analyzer = AudioBiasAnalyzer(
                speech_model=config.SPEECH_MODEL,
                sentiment_model=config.SENTIMENT_MODEL,
                logger=self.logger
            )
    
    def update_weights(self, **kwargs):
        """
        Update weights for the bias detection components.
        
        Args:
            **kwargs: Weights to update (text_weight, video_weight, audio_weight,
                      political_weight, religious_weight)
        """
        # Update configuration
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
        
        # Update instance variables
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
        
        # Update bias calculator weights
        self.bias_calculator.update_weights(**kwargs)
        
        # Save updated configuration
        config.save_config(self.config)
        
        self.logger.info(f"Updated weights: {self.config}")
    
    def analyze_single_video(self, video_path, description="", include_audio=True):
        """
        Analyze a single video for bias.
        
        Args:
            video_path (str): Path to the video file
            description (str): Text description of the video
            include_audio (bool): Whether to include audio analysis
            
        Returns:
            dict: Analysis results
        """
        self.logger.section(f"Analyzing video: {os.path.basename(video_path)}")
        
        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            return None
        
        start_time = time.time()
        
        # Analyze text description
        self.logger.info("Analyzing text description...")
        text_result = self.text_analyzer.get_combined_bias_score(
            description,
            political_weight=self.political_weight,
            religious_weight=self.religious_weight
        )
        
        # Analyze video content
        self.logger.info("Analyzing video content...")
        self._ensure_video_analyzer()
        video_result = self.video_analyzer.get_combined_bias_score(
            video_path,
            political_weight=self.political_weight,
            religious_weight=self.religious_weight
        )
        
        # Analyze audio if requested
        audio_result = None
        if include_audio:
            self.logger.info("Analyzing audio content...")
            self._ensure_audio_analyzer()
            
            # Extract audio directory if needed
            audio_dir = os.path.join(os.path.dirname(video_path), "extracted_audio")
            ensure_dir(audio_dir)
            
            # Get full audio analysis
            audio_analysis = self.audio_analyzer.get_complete_audio_analysis(
                video_path,
                output_dir=audio_dir,
                political_weight=self.political_weight,
                religious_weight=self.religious_weight
            )
            
            # Extract just the bias scores
            audio_result = audio_analysis['bias_scores']
        
        # Calculate combined bias
        self.logger.info("Calculating combined bias scores...")
        bias_report = self.bias_calculator.create_bias_report(
            text_result, video_result, audio_result
        )
        
        # Prepare final result
        final_result = {
            'video_path': video_path,
            'description': description,
            'text_bias': text_result,
            'video_bias': video_result,
            'audio_bias': audio_result,
            'bias_report': bias_report,
            'analysis_time': time.time() - start_time
        }
        
        self.logger.info(f"Analysis completed in {final_result['analysis_time']:.2f} seconds")
        self.logger.info(f"Overall bias score: {bias_report['scores']['combined']['overall_bias']:.2f}")
        self.logger.info(f"Overall bias category: {bias_report['categories']['overall_bias']}")
        
        return final_result
    
    def process_batch(self, input_csv=None, data_dir=None, output_csv=None, include_audio=True):
        """
        Process a batch of videos from a CSV file.
        
        Args:
            input_csv (str): Path to input CSV file
            data_dir (str): Directory containing video files
            output_csv (str): Path to output CSV file
            include_audio (bool): Whether to include audio analysis
            
        Returns:
            pandas.DataFrame: DataFrame with analysis results
        """
        # Use default paths if not provided
        input_csv = input_csv or config.DEFAULT_INPUT_CSV
        output_csv = output_csv or config.DEFAULT_OUTPUT_CSV
        data_dir = data_dir or config.DATA_DIR
        
        self.logger.section(f"Processing batch from {input_csv}")
        
        if not os.path.exists(input_csv):
            self.logger.error(f"Input CSV file not found: {input_csv}")
            return None
        
        # Try different possible locations for the input CSV
        potential_paths = [
            input_csv,
            os.path.join("data", os.path.basename(input_csv)),
            os.path.join("c:/Hackathon_Backend/data", os.path.basename(input_csv)),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", os.path.basename(input_csv))
        ]
        
        # Try each path
        df = None
        used_path = None
        
        for path in potential_paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    used_path = path
                    self.logger.info(f"Loaded {len(df)} videos from {path}")
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to load {path}: {e}")
        
        if df is None:
            self.logger.error(f"Could not find or load input CSV. Tried: {potential_paths}")
            return None
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            ensure_dir(output_dir)
        
        # Create audio extraction directory if needed
        if include_audio:
            ensure_dir(config.EXTRACTED_AUDIO_DIR)
        
        start_time = time.time()
        
        # Step 1: Analyze text descriptions
        self.logger.info("Analyzing text descriptions...")
        df = self.text_analyzer.analyze_batch(
            df, 
            description_col='Description',
            political_weight=self.political_weight,
            religious_weight=self.religious_weight
        )
        
        # Step 2: Analyze video content
        self.logger.info("Analyzing video content...")
        self._ensure_video_analyzer()
        df = self.video_analyzer.analyze_batch(
            df,
            video_id_col='Video_ID',
            data_dir=data_dir,
            political_weight=self.political_weight,
            religious_weight=self.religious_weight
        )
        
        # Step 3: Analyze audio if requested
        if include_audio:
            self.logger.info("Analyzing audio content...")
            self._ensure_audio_analyzer()
            df = self.audio_analyzer.analyze_batch(
                df,
                video_id_col='Video_ID',
                data_dir=data_dir,
                output_dir=config.EXTRACTED_AUDIO_DIR,
                political_weight=self.political_weight,
                religious_weight=self.religious_weight
            )
        
        # Step 4: Calculate combined bias scores
        self.logger.info("Calculating combined bias scores...")
        df = self.bias_calculator.process_dataframe(df)
        
        # Step 5: Save results
        df.to_csv(output_csv, index=False)
        self.logger.info(f"Results saved to {output_csv}")
        
        # Calculate summary statistics
        summary = self.bias_calculator.generate_summary_statistics(df)
        
        # Print summary
        self.logger.section("Summary")
        self.logger.info(f"Total videos analyzed: {summary['count']}")
        self.logger.info(f"Average overall bias score: {summary['overall_bias']['mean']:.2f}")
        self.logger.info(f"Bias categories distribution: {summary['categories']}")
        self.logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
        
        return df