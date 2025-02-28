# audio_analyzer.py - Audio-based bias analysis for TikTok videos

import os
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
import re
import traceback

from utils import get_device, ensure_dir, clean_transcript, Logger, get_video_path

class AudioBiasAnalyzer:
    """
    Extract and analyze audio features from TikTok videos.
    """
    
    def __init__(self, speech_model='facebook/wav2vec2-base-960h', 
                 sentiment_model='distilbert-base-uncased-finetuned-sst-2-english',
                 device=None, logger=None):
        """
        Initialize the audio analyzer.
        
        Args:
            speech_model (str): Name of the pre-trained speech model
            sentiment_model (str): Name of the pre-trained sentiment model
            device (str): Device to run the model on ('cuda' or 'cpu')
            logger (Logger): Logger instance for output
        """
        self.device = device if device else get_device()
        self.logger = logger if logger else Logger()
        
        self.logger.info(f"Initializing Audio Feature Extractor on {self.device}")
        
        # Speech recognition model
        try:
            self.logger.info(f"Loading speech model: {speech_model}")
            self.processor = Wav2Vec2Processor.from_pretrained(speech_model)
            self.speech_model = Wav2Vec2ForCTC.from_pretrained(speech_model).to(self.device)
            self.logger.info("Speech model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading speech model: {e}")
            self.processor = None
            self.speech_model = None
        
        # Sentiment analysis pipeline
        try:
            self.logger.info(f"Loading sentiment model: {sentiment_model}")
            self.sentiment_pipeline = pipeline("sentiment-analysis", 
                                             model=sentiment_model, 
                                             device=0 if self.device == 'cuda' else -1)
            self.logger.info("Sentiment model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading sentiment model: {e}")
            self.sentiment_pipeline = None
            
        # Define common speech-to-text errors and corrections
        self.speech_corrections = {
            'ebasi': 'basic',
            'teraffs': 'tariffs',
            'teriff': 'tariff',
            'terr': 'tariff',
            'bricxs': 'brics',
            'duller': 'dollar',
            'yetas': 'years',
            'cleer': 'clear',
            'shienbam': 'sheinbaum',
            'ilness': 'illness',
            'doltrum': 'trump',
            'terate': 'trade',
            'temper': 'ten per',
            'intrudau': 'trudeau',
            'uessthat': 'us that',
            'tere': 'there',
            'shienbam': 'sheinbaum',
            'shijingfing': 'xi jinping',
            'klaudya': 'claudia',
            'exraport': 'export',
            'asin': 'using',
            'navanada': 'in canada',
            'doesnt em': "doesn't",
            'teraphs': 'tariffs',
            'wati': 'wait',
            'giu': 'you',
            'toyo': 'to you'
        }
    
    def extract_audio_from_video(self, video_path, output_dir=None):
        """
        Extract audio from video file using MoviePy.
        
        Args:
            video_path (str): Path to the video file
            output_dir (str): Directory to save the extracted audio
            
        Returns:
            str: Path to the extracted audio file, or None if extraction failed
        """
        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            return None
        
        try:
            # Create output directory if it doesn't exist
            if output_dir:
                ensure_dir(output_dir)
            
            # Generate output path
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            audio_path = os.path.join(output_dir, f"{video_name}.wav") if output_dir else f"{video_name}.wav"
            
            # Skip extraction if audio file already exists
            if os.path.exists(audio_path):
                self.logger.info(f"Audio file already exists: {audio_path}")
                return audio_path
            
            # Extract audio using MoviePy
            with VideoFileClip(video_path) as video:
                if video.audio is not None:
                    self.logger.info(f"Extracting audio from {video_path}")
                    # No verbose or logger parameters
                    video.audio.write_audiofile(audio_path)
                    self.logger.info(f"Audio extracted and saved to {audio_path}")
                    return audio_path
                else:
                    self.logger.warning(f"No audio track found in {video_path}")
                    return None
            
        except Exception as e:
            self.logger.error(f"Error extracting audio from {video_path}: {e}")
            return None
        
    def extract_acoustic_features(self, audio_path):
        """
        Extract acoustic features (tempo, pitch, energy) from audio.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            dict: Dictionary of acoustic features
        """
        if not os.path.exists(audio_path):
            self.logger.error(f"Audio file not found: {audio_path}")
            return None
        
        try:
            # Load audio with librosa
            y, sr = librosa.load(audio_path, sr=None)
            
            # Extract acoustic features
            features = {}
            
            # Tempo/rhythm features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            
            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitches = pitches[magnitudes > np.median(magnitudes)]
            features['pitch_mean'] = float(np.mean(pitches) if len(pitches) > 0 else 0)
            features['pitch_std'] = float(np.std(pitches) if len(pitches) > 0 else 0)
            
            # Energy/intensity features
            features['energy_mean'] = float(np.mean(librosa.feature.rms(y=y)[0]))
            features['energy_std'] = float(np.std(librosa.feature.rms(y=y)[0]))
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
            
            # MFCC features (summarized)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = float(np.mean(mfccs))
            features['mfcc_std'] = float(np.std(mfccs))
            
            # Calculate emotional indicators
            features['emotional_intensity'] = min(1.0, features['energy_std'] / features['energy_mean'] if features['energy_mean'] > 0 else 0)
            features['voice_variation'] = min(1.0, features['pitch_std'] / 300) # Normalized pitch variation
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting acoustic features from {audio_path}: {e}")
            return None
    
    def transcribe_speech(self, audio_path):
        """
        Transcribe speech from audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        if not self.speech_model or not self.processor:
            self.logger.warning("Speech model not available. Skipping transcription.")
            return ""
        
        if not os.path.exists(audio_path):
            self.logger.error(f"Audio file not found: {audio_path}")
            return ""
        
        try:
            # Load audio with librosa (resampling to 16kHz for wav2vec2)
            speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
            
            # Check if audio is too quiet
            if np.abs(speech_array).mean() < 0.005:
                self.logger.info("Audio is too quiet, amplifying...")
                speech_array = speech_array * (0.05 / np.abs(speech_array).mean())
            
            # Process audio in smaller chunks for better results
            chunk_size = 16000 * 15  # 15 seconds per chunk
            transcriptions = []
            
            for i in range(0, len(speech_array), chunk_size):
                chunk = speech_array[i:i+chunk_size]
                if len(chunk) < 3000:  # Skip chunks that are too short (less than 0.2 seconds)
                    continue
                    
                # Tokenize
                input_values = self.processor(chunk, 
                                            sampling_rate=16000, 
                                            return_tensors="pt").input_values.to(self.device)
                
                # Retrieve logits
                with torch.no_grad():
                    logits = self.speech_model(input_values).logits
                
                # Take argmax and decode
                predicted_ids = torch.argmax(logits, dim=-1)
                chunk_text = self.processor.batch_decode(predicted_ids)[0]
                
                # Only add non-empty chunks
                if chunk_text.strip():
                    transcriptions.append(chunk_text)
            
            # Join all transcribed chunks
            transcription = " ".join(transcriptions)
            
            # Basic post-processing to clean up transcription
            # Replace multiple spaces with a single space
            transcription = " ".join(transcription.split())
            
            if transcription:
                self.logger.info(f"Transcription (first 100 chars): '{transcription[:100]}...'")
            else:
                self.logger.info("No transcription generated.")
                
            return transcription
                
        except Exception as e:
            self.logger.error(f"Error transcribing speech from {audio_path}: {e}")
            return ""
    
    @staticmethod
    def detect_political_content(text):
        """
        Detect political content in transcript.
        Returns score between 0-1 indicating likelihood of political content.
        """
        if not text:
            return 0.0
            
        # Political keywords (simplified list)
        political_terms = [
            'president', 'government', 'minister', 'administration', 'election',
            'vote', 'democracy', 'democratic', 'republican', 'congress', 
            'senate', 'policy', 'politician', 'political', 'politics',
            'tariff', 'trade', 'international', 'nation', 'economy',
            'tax', 'bill', 'law', 'legislation', 'regulation',
            'conservative', 'liberal', 'left', 'right', 'party',
            'campaign', 'ballot', 'candidate', 'official', 'leader',
            'war', 'military', 'security', 'defense', 'foreign',
            'china', 'russia', 'united states', 'u.s.', 'america',
            'europe', 'european', 'nato', 'treaty', 'agreement',
            'sanction', 'diplomat', 'embassy', 'immigration', 'border',
            'protest', 'activist', 'rights', 'freedom', 'justice'
        ]
        
        text_lower = text.lower()
        count = sum(1 for term in political_terms if term in text_lower)
        
        # Normalize score (10 or more hits = definitely political)
        political_score = min(1.0, count / 10.0)
        
        return political_score
    
    @staticmethod
    def detect_religious_content(text):
        """
        Detect religious content in transcript.
        Returns score between 0-1 indicating likelihood of religious content.
        """
        if not text:
            return 0.0
            
        # Religious keywords (simplified list)
        religious_terms = [
            'god', 'jesus', 'christ', 'christian', 'christianity',
            'bible', 'church', 'pray', 'prayer', 'faith',
            'religion', 'religious', 'spiritual', 'spirit', 'holy',
            'sacred', 'divine', 'blessing', 'soul', 'salvation',
            'sin', 'heaven', 'hell', 'prophet', 'worship',
            'allah', 'muslim', 'islam', 'islamic', 'quran',
            'mosque', 'muhammad', 'rabbi', 'jewish', 'judaism',
            'torah', 'synagogue', 'hindu', 'hinduism', 'buddhist',
            'buddhism', 'temple', 'monk', 'scripture', 'ritual',
            'atheist', 'atheism', 'belief', 'miracle', 'meditation'
        ]
        
        text_lower = text.lower()
        count = sum(1 for term in religious_terms if term in text_lower)
        
        # Normalize score (5 or more hits = definitely religious)
        religious_score = min(1.0, count / 5.0)
        
        return religious_score
    
    def detect_music(self, audio_path):
        """
        Detect if audio contains music vs. speech with improved algorithm.
        """
        if not os.path.exists(audio_path):
            self.logger.error(f"Audio file not found: {audio_path}")
            return 0.0
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Extract more reliable features for music detection
            # Spectral flatness (higher for noise/music, lower for speech)
            spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
            
            # Spectral contrast (higher for music with harmonics)
            spectral_contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
            
            # Spectral bandwidth (higher for music with wider frequency range)
            spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
            
            # Zero crossing rate (higher for percussive sounds and noise)
            zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
            
            # Rhythmic features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo)
            
            # Onset strength (higher for music with strong rhythmic patterns)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onset_strength = float(np.mean(onset_env))
            
            # Harmonic-percussive source separation
            harmonic, percussive = librosa.effects.hpss(y)
            harmonic_energy = float(np.mean(np.abs(harmonic)))
            percussive_energy = float(np.mean(np.abs(percussive)))
            
            # Calculate music score using a more balanced formula
            music_indicators = [
                spectral_flatness * 10,  # Scale up as it's typically < 0.1
                spectral_contrast / 100,  # Scale down as it can be large
                spectral_bandwidth / 5000,  # Normalize
                zero_crossing_rate * 5,    # Scale up
                tempo / 200,               # Normalize
                onset_strength * 5,        # Scale up
                harmonic_energy * 20,      # Harmonic content (music)
                percussive_energy * 15     # Percussive content (beats)
            ]
            
            # Use a weighted average
            weights = [0.15, 0.15, 0.10, 0.10, 0.15, 0.15, 0.10, 0.10]
            music_score = sum(x * w for x, w in zip(music_indicators, weights))
            
            # Normalize to 0-1 range with a more balanced curve
            music_score_normalized = min(1.0, max(0.0, music_score / 1.5))
            
            self.logger.info(f"Music detection score: {music_score_normalized:.4f}")
            return music_score_normalized
                
        except Exception as e:
            self.logger.error(f"Error detecting music: {e}")
            return 0.0
        
    def get_audio_bias_score(self, transcription, cleaned_transcript=None, 
                              acoustic_features=None, sentiment_result=None,
                              political_weight=0.5, religious_weight=0.5):
        """
        Calculate bias scores from audio analysis.
        
        Args:
            transcription (str): Raw transcription
            cleaned_transcript (str): Cleaned transcription
            acoustic_features (dict): Acoustic features
            sentiment_result (dict): Sentiment analysis results
            political_weight (float): Weight for political bias
            religious_weight (float): Weight for religious bias
            
        Returns:
            dict: Audio bias scores
        """
        # Default values if components are missing
        if not cleaned_transcript:
            cleaned_transcript = clean_transcript(transcription, self.speech_corrections)
        
        if not sentiment_result:
            sentiment_result = {'label': 'NEUTRAL', 'score': 0.5}
        
        # Detect political and religious content
        political_score = self.detect_political_content(cleaned_transcript) * 100
        religious_score = self.detect_religious_content(cleaned_transcript) * 100
        
        # Apply sentiment influence
        # If sentiment is strong (positive or negative), increase the bias scores
        sentiment_intensity = abs(sentiment_result['score'] - 0.5) * 2  # 0-1 range
        
        # Calculate acoustic influence if we have features
        acoustic_influence = 0
        if acoustic_features:
            # Use emotional intensity and voice variation as indicators
            emotional_intensity = acoustic_features.get('emotional_intensity', 0)
            voice_variation = acoustic_features.get('voice_variation', 0)
            
            # Higher emotional intensity and voice variation can indicate stronger bias
            acoustic_influence = (emotional_intensity + voice_variation) / 2
        
        # Apply modifiers
        political_bias = political_score * (1 + sentiment_intensity * 0.3 + acoustic_influence * 0.2)
        religious_bias = religious_score * (1 + sentiment_intensity * 0.3 + acoustic_influence * 0.2)
        
        # Scale to 1-100 range and ensure minimum value
        political_bias = max(1.0, min(100.0, political_bias))
        religious_bias = max(1.0, min(100.0, religious_bias))
        
        # Calculate combined bias
        combined_bias = (political_bias * political_weight + 
                          religious_bias * religious_weight)
        
        return {
            'political_bias': political_bias,
            'religious_bias': religious_bias,
            'combined_bias': combined_bias
        }
    
    def get_complete_audio_analysis(self, video_path, output_dir=None, political_weight=0.5, religious_weight=0.5):
        """
        Perform complete audio analysis on a video.
        
        Args:
            video_path (str): Path to the video file
            output_dir (str): Directory to save extracted audio
            political_weight (float): Weight for political bias
            religious_weight (float): Weight for religious bias
            
        Returns:
            dict: Complete audio analysis results
        """
        # Extract audio
        audio_path = self.extract_audio_from_video(video_path, output_dir)
        if not audio_path:
            return {
                'speech_sentiment_score': 0.5,
                'speech_sentiment_label': 'NEUTRAL',
                'music_score': 0.0,
                'acoustic_features': {},
                'transcription': '',
                'audio_present': False,
                'bias_scores': {
                    'political_bias': 1.0,
                    'religious_bias': 1.0,
                    'combined_bias': 1.0
                }
            }
        
        # Transcribe speech
        transcription = self.transcribe_speech(audio_path)
        
        # Clean transcript
        cleaned_transcript = clean_transcript(transcription, self.speech_corrections)
        
        # Analyze speech sentiment
        sentiment = self.analyze_speech_sentiment(cleaned_transcript)
        
        # Extract acoustic features
        acoustic_features = self.extract_acoustic_features(audio_path)
        
        # Detect music
        music_score = self.detect_music(audio_path)
        
        # Calculate bias scores
        bias_scores = self.get_audio_bias_score(
            transcription, 
            cleaned_transcript,
            acoustic_features,
            sentiment,
            political_weight,
            religious_weight
        )
        
        # Clean up temporary audio file if we created it
        if output_dir is None and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass
        
        # Combine all results
        return {
            'speech_sentiment_score': sentiment['score'],
            'speech_sentiment_label': sentiment['label'],
            'music_score': music_score,
            'acoustic_features': acoustic_features if acoustic_features else {},
            'transcription': transcription,
            'cleaned_transcription': cleaned_transcript,
            'audio_present': True,
            'bias_scores': bias_scores
        }
    
    def analyze_batch(self, df, video_id_col='Video_ID', data_dir=None, output_dir=None,
                      output_col='audio_bias_scores', political_weight=0.5, religious_weight=0.5):
        """
        Analyze a batch of videos from a DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame containing video IDs
            video_id_col (str): Column name containing the video IDs
            data_dir (str): Directory containing the video files
            output_dir (str): Directory to save extracted audio files
            output_col (str): Column name for the output scores
            political_weight (float): Weight for political bias
            religious_weight (float): Weight for religious bias
            
        Returns:
            pandas.DataFrame: DataFrame with added audio analysis results
        """
        # Create a copy of the input DataFrame
        result_df = df.copy()
        
        # Initialize columns for individual scores
        result_df['audio_political_bias'] = None
        result_df['audio_religious_bias'] = None
        result_df['audio_combined_bias'] = None
        result_df['transcription'] = None
        
        # Initialize output column for scores
        if output_col not in result_df.columns:
            result_df[output_col] = None
        
        # Ensure output directory exists
        if output_dir:
            ensure_dir(output_dir)
        
        # Potential data directories to check
        if data_dir is None:
            data_dirs = [
                "data",
                "c:/Hackathon_Backend/data",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
            ]
        else:
            data_dirs = [data_dir]
        
        # Iterate through each video ID
        for idx, video_id in enumerate(result_df[video_id_col]):
            self.logger.info(f"Analyzing audio for video {idx+1}/{len(result_df)}: {video_id}")
            
            # Try to find the video file in potential directories
            video_path = None
            for directory in data_dirs:
                path = os.path.join(directory, f"{video_id}.mp4")
                if os.path.exists(path):
                    video_path = path
                    break
            
            if video_path and os.path.exists(video_path):
                # Get complete audio analysis
                analysis = self.get_complete_audio_analysis(
                    video_path, 
                    output_dir=output_dir,
                    political_weight=political_weight,
                    religious_weight=religious_weight
                )
                
                # Store bias scores
                bias_scores = analysis['bias_scores']
                result_df.loc[idx, 'audio_political_bias'] = bias_scores['political_bias']
                result_df.loc[idx, 'audio_religious_bias'] = bias_scores['religious_bias']
                result_df.loc[idx, 'audio_combined_bias'] = bias_scores['combined_bias']
                
                # Store transcription
                result_df.loc[idx, 'transcription'] = analysis['transcription']
                
                # Store the full analysis as string
                result_df.at[idx, output_col] = str(analysis)
            else:
                self.logger.warning(f"Video file not found for ID: {video_id}")
                # Add placeholder result for missing videos
                placeholder_bias = {
                    'political_bias': 1.0,
                    'religious_bias': 1.0,
                    'combined_bias': 1.0
                }
                
                # Store placeholder scores
                result_df.loc[idx, 'audio_political_bias'] = 1.0
                result_df.loc[idx, 'audio_religious_bias'] = 1.0
                result_df.loc[idx, 'audio_combined_bias'] = 1.0
                result_df.loc[idx, 'transcription'] = ""
                
                # Store placeholder analysis as string
                placeholder_analysis = {
                    'speech_sentiment_score': 0.5,
                    'speech_sentiment_label': 'NEUTRAL',
                    'music_score': 0.0,
                    'acoustic_features': {},
                    'transcription': '',
                    'cleaned_transcription': '',
                    'audio_present': False,
                    'bias_scores': placeholder_bias
                }
                result_df.at[idx, output_col] = str(placeholder_analysis)
        
        return result_df
        
    def analyze_speech_sentiment(self, text):
        """
        Analyze sentiment of transcribed speech.
        
        Args:
            text (str): Transcribed text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        if not self.sentiment_pipeline:
            self.logger.warning("Sentiment model not available. Skipping sentiment analysis.")
            return {'label': 'NEUTRAL', 'score': 0.5}
        
        # Check for empty or too short text
        if not text or len(text.strip()) < 10:
            self.logger.info("Text too short for meaningful sentiment analysis")
            return {'label': 'NEUTRAL', 'score': 0.5}
        
        try:
            # Clean text for better analysis
            cleaned_text = ' '.join([word for word in text.split() 
                                if len(word) > 1 and word.isalpha()])
            
            # If after cleaning, text is too short, return neutral
            if len(cleaned_text) < 10:
                self.logger.info("Cleaned text too short for sentiment analysis")
                return {'label': 'NEUTRAL', 'score': 0.5}
                
            self.logger.info(f"Analyzing sentiment for: '{cleaned_text[:100]}...'")
            
            # Split text into chunks if it's too long for the model
            max_length = 512
            chunks = [cleaned_text[i:i+max_length] for i in range(0, len(cleaned_text), max_length)]
            
            # Analyze sentiment for each chunk
            results = []
            for chunk in chunks:
                if len(chunk.strip()) > 10:  # Ensure chunk has meaningful content
                    sentiment = self.sentiment_pipeline(chunk)[0]
                    self.logger.info(f"Chunk sentiment: {sentiment['label']} ({sentiment['score']:.4f})")
                    results.append(sentiment)
                
            # If no valid results, return neutral
            if not results:
                return {'label': 'NEUTRAL', 'score': 0.5}
            
            # Calculate average sentiment scores
            positive_scores = [r['score'] for r in results if r['label'] == 'POSITIVE']
            negative_scores = [r['score'] for r in results if r['label'] == 'NEGATIVE']
            
            if positive_scores and negative_scores:
                # If we have both positive and negative scores, calculate net sentiment
                avg_positive = sum(positive_scores) / len(positive_scores)
                avg_negative = sum(negative_scores) / len(negative_scores)
                
                if avg_positive > avg_negative:
                    final_label = 'POSITIVE'
                    final_score = avg_positive
                else:
                    final_label = 'NEGATIVE'
                    final_score = avg_negative
            elif positive_scores:
                final_label = 'POSITIVE'
                final_score = sum(positive_scores) / len(positive_scores)
            elif negative_scores:
                final_label = 'NEGATIVE'
                final_score = sum(negative_scores) / len(negative_scores)
            else:
                final_label = 'NEUTRAL'
                final_score = 0.5
                
            self.logger.info(f"Final sentiment: {final_label} ({final_score:.4f})")
            return {'label': final_label, 'score': final_score}
                
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return {'label': 'NEUTRAL', 'score': 0.5}