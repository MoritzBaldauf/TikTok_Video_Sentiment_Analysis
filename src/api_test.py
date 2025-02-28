# api_test.py - Example of using the TikTok Bias Detection API

from pipeline import TikTokBiasDetectionPipeline

# Initialize the pipeline
pipeline = TikTokBiasDetectionPipeline()

# Update weights if needed
pipeline.update_weights(
    text_weight=0.4,
    video_weight=0.4,
    audio_weight=0.2,
    political_weight=0.3,
    religious_weight=0.3
)

# Process a batch of videos
df = pipeline.process_batch(
    input_csv="C:/Hackathon_Backend/data/data_table.csv", 
    data_dir="C:/Hackathon_Backend/data",  # Explicitly set the data directory (previously lot of issues with this)
    output_csv="C:/Hackathon_Backend/data/results_API.csv"
)

# Print a summary of the results
if df is not None:
    print("\nAnalysis Results Summary:")
    print(f"Total videos analyzed: {len(df)}")
    
    # Get average bias scores
    political_avg = df['political_bias'].mean()
    religious_avg = df['religious_bias'].mean()
    overall_avg = df['overall_bias_score'].mean()
    
    print(f"Average political bias: {political_avg:.2f}")
    print(f"Average religious bias: {religious_avg:.2f}")
    print(f"Average overall bias: {overall_avg:.2f}")
    
    # Count bias categories
    if 'overall_bias_category' in df.columns:
        categories = df['overall_bias_category'].value_counts()
        print("\nBias category distribution:")
        for category, count in categories.items():
            print(f"  {category}: {count}")
    
    print("\nResults have been saved to: C:/Hackathon_Backend/data/results_API.csv")
else:
    print("Analysis failed. Check the error messages above.")