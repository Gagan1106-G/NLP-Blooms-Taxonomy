"""
Download all required NLTK data
Run this once before using the MCQ generator
"""

import nltk
import ssl

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_resources():
    """Download all required NLTK resources"""
    resources = [
        'punkt',
        'punkt_tab',  # New tokenizer format
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng',
        'maxent_ne_chunker',
        'maxent_ne_chunker_tab',
        'words',
        'wordnet',
        'omw-1.4',  # Open Multilingual WordNet
        'brown',
        'stopwords'
    ]
    
    print("📥 Downloading NLTK resources...")
    for resource in resources:
        try:
            nltk.download(resource, quiet=False)
            print(f"✅ {resource}")
        except Exception as e:
            print(f"⚠️ {resource} - {str(e)}")
    
    print("\n✅ NLTK resources downloaded successfully!")

if __name__ == "__main__":
    download_nltk_resources()