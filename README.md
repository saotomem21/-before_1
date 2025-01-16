# Sentiment and Stance Analysis

This repository contains Python scripts for analyzing sentiment and stance in Japanese text using a pre-trained BERT model.

## Features
- Sentiment analysis using koheiduck/bert-japanese-finetuned-sentiment model
- Text preprocessing including URL removal and whitespace normalization
- Batch processing for efficient analysis of large datasets
- Detailed logging for monitoring and debugging

## Requirements
- Python 3.8+
- See requirements.txt for dependencies

## Usage
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables:
   ```bash
   export HUGGINGFACE_TOKEN=your_token_here
   ```
4. Run the analysis:
   ```bash
   python sentiment_stance_analysis.py
   ```

## Output
- Analysis results saved to analysis_output.csv
- Logs saved to sentiment_analysis.log

## License
MIT
