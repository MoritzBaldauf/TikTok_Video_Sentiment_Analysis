#  - Test script for the BERT sequence classifier implementation

import pandas as pd
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from text_analyzer import TextBiasAnalyzer
from utils import Logger

def main():
    """Test the BERT sequence classifier implementation"""
    logger = Logger(verbose=True)
    logger.section("Testing BERT Sequence Classifier Implementation")
    
    # Initialize the analyzer with pre-trained models
    analyzer = TextBiasAnalyzer(
        model_name='bert-base-uncased',
        political_model_name='textattack/bert-base-uncased-yelp-polarity',  
        religious_model_name='distilbert-base-uncased-finetuned-sst-2-english',
        logger=logger
    )
    
    # Sample political and religious texts
    test_texts = [
        {
            'description': "A simple non-political message about cats and dogs.",
            'expected': "low bias"
        },
        {
            'description': "The government is corrupt and untrustworthy. The president is destroying our freedom.",
            'expected': "high political bias"
        },
        {
            'description': "The holy bible teaches us that Jesus Christ died for our sins and we must repent to find salvation.",
            'expected': "high religious bias"
        },
        {
            'description': "Liberal socialists want to destroy our great country with their radical green new deal agenda.",
            'expected': "high political bias"
        },
        {
            'description': "The mosque is a sacred place of worship where Muslims pray to Allah and follow the teachings of the Quran.",
            'expected': "high religious bias"
        },
        {
            'description': "Today's weather is nice and sunny. I might go for a walk later.",
            'expected': "low bias"
        }
    ]
    
    # Analyze each test text
    logger.info("Analyzing sample texts:")
    for i, test in enumerate(test_texts):
        logger.info(f"\nTest {i+1}: {test['description']}")
        logger.info(f"Expected: {test['expected']}")
        
        scores = analyzer.get_combined_bias_score(test['description'])
        
        logger.info(f"Political Bias: {scores['political_bias']:.2f}")
        logger.info(f"Religious Bias: {scores['religious_bias']:.2f}")
        logger.info(f"Combined Bias: {scores['combined_bias']:.2f}")
    
    # Test with a small batch of data
    logger.section("Testing with batch data")
    
    # Create a small test DataFrame
    data = {
        'Description': [text['description'] for text in test_texts]
    }
    df = pd.DataFrame(data)
    
    # Analyze the batch
    result_df = analyzer.analyze_batch(df)
    
    # Display results
    logger.info("Batch analysis results:")
    for idx, row in result_df.iterrows():
        logger.info(f"\nDescription: {row['Description'][:50]}...")
        logger.info(f"Political Bias: {row['text_political_bias']:.2f}")
        logger.info(f"Religious Bias: {row['text_religious_bias']:.2f}")
        logger.info(f"Combined Bias: {row['text_combined_bias']:.2f}")
    
    logger.info("\nTest completed successfully!")

if __name__ == "__main__":
    main()