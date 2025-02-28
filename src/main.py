# main.py - Entry point for the TikTok bias detection pipeline

import os
import argparse
import json
import time

from pipeline import TikTokBiasDetectionPipeline
import config

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TikTok Bias Detection Pipeline')
    
    # Input/output options
    parser.add_argument('--input', type=str, help='Path to input CSV file')
    parser.add_argument('--output', type=str, help='Path to output CSV file')
    parser.add_argument('--data-dir', type=str, help='Directory containing video files')
    parser.add_argument('--hackathon-path', type=str, 
                      default='c:/Hackathon_Backend/data',
                      help='Path to the Hackathon_Backend/data directory')
    
    # Single video mode
    parser.add_argument('--video', type=str, help='Path to a single video file for analysis')
    parser.add_argument('--description', type=str, default='', help='Text description for single video analysis')
    
    # Configuration options
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--save-config', type=str, help='Save current configuration to file')
    
    # Weight options
    parser.add_argument('--text-weight', type=float, help='Weight for text analysis (0-1)')
    parser.add_argument('--video-weight', type=float, help='Weight for video analysis (0-1)')
    parser.add_argument('--audio-weight', type=float, help='Weight for audio analysis (0-1)')
    parser.add_argument('--political-weight', type=float, help='Weight for political bias (0-1)')
    parser.add_argument('--religious-weight', type=float, help='Weight for religious bias (0-1)')
    
    # Analysis options
    parser.add_argument('--skip-audio', action='store_true', help='Skip audio analysis')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    return parser.parse_args()

def check_data_files():
    """Check if we can access the necessary data files"""
    # Possible data paths
    data_paths = [
        "data",
        "c:/Hackathon_Backend/data",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    ]
    
    # Check for data_table.csv
    data_table_path = None
    for base_path in data_paths:
        path = os.path.join(base_path, "data_table.csv")
        if os.path.exists(path):
            data_table_path = path
            print(f"Found data_table.csv at: {path}")
            break
    
    if not data_table_path:
        print(f"WARNING: Could not find data_table.csv")
    
    # Check for a sample video file
    sample_video = None
    for base_path in data_paths:
        path = os.path.join(base_path, "1.mp4")
        if os.path.exists(path):
            sample_video = path
            print(f"Found sample video at: {path}")
            break
    
    if not sample_video:
        print(f"WARNING: Could not find sample video files")
    
    # Return the found data directory
    return os.path.dirname(data_table_path) if data_table_path else None

def main():
    """Main function for the TikTok bias detection pipeline"""
    args = parse_arguments()
    
    # Override the data directory if specified
    if args.hackathon_path and os.path.exists(args.hackathon_path):
        config.DATA_DIR = args.hackathon_path
        config.DEFAULT_INPUT_CSV = os.path.join(config.DATA_DIR, "data_table.csv")
        config.DEFAULT_OUTPUT_CSV = os.path.join(config.DATA_DIR, "final_analysis.csv")
        print(f"Using hackathon data path: {config.DATA_DIR}")
    else:
        # Try to automatically find data files
        data_dir = check_data_files()
        if data_dir:
            config.DATA_DIR = data_dir
            config.DEFAULT_INPUT_CSV = os.path.join(config.DATA_DIR, "data_table.csv")
            config.DEFAULT_OUTPUT_CSV = os.path.join(config.DATA_DIR, "final_analysis.csv")
            print(f"Automatically detected data directory: {config.DATA_DIR}")
    
    # Track execution time
    start_time = time.time()
    
    # Initialize the pipeline
    pipeline = TikTokBiasDetectionPipeline(
        config_path=args.config,
        verbose=not args.quiet
    )
    
    # Update weights if provided
    weights = {}
    if args.text_weight is not None:
        weights['text_weight'] = args.text_weight
    if args.video_weight is not None:
        weights['video_weight'] = args.video_weight
    if args.audio_weight is not None:
        weights['audio_weight'] = args.audio_weight
    if args.political_weight is not None:
        weights['political_weight'] = args.political_weight
    if args.religious_weight is not None:
        weights['religious_weight'] = args.religious_weight
    
    if weights:
        pipeline.update_weights(**weights)
    
    # Save configuration if requested
    if args.save_config:
        config.save_config(pipeline.config, args.save_config)
        print(f"Configuration saved to {args.save_config}")
    
    # Process based on mode
    if args.video:
        # Single video mode
        result = pipeline.analyze_single_video(
            args.video,
            description=args.description,
            include_audio=not args.skip_audio
        )
        
        if result:
            # Print result summary
            print("\nAnalysis Results:")
            print(f"Overall Bias Score: {result['bias_report']['scores']['combined']['overall_bias']:.2f}")
            print(f"Category: {result['bias_report']['categories']['overall_bias']}")
            print(f"Political Bias: {result['bias_report']['scores']['combined']['political_bias']:.2f}")
            print(f"Religious Bias: {result['bias_report']['scores']['combined']['religious_bias']:.2f}")
            
            # Save detailed result to JSON
            output_file = os.path.splitext(args.video)[0] + "_analysis.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nDetailed results saved to {output_file}")
    else:
        # Batch mode
        df = pipeline.process_batch(
            input_csv=args.input,
            data_dir=args.data_dir,
            output_csv=args.output,
            include_audio=not args.skip_audio
        )
    
    # Report total execution time
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()