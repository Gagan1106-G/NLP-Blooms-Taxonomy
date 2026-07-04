"""
True/False Question Generator
Generates true/false questions from input text.
"""

import logging
import random
import re
from typing import List, Dict, Any

import nltk

# Download required NLTK data with better error handling
try:
    from nltk.tokenize import sent_tokenize
    # Test if punkt_tab works
    sent_tokenize("Test sentence.")
except (LookupError, OSError):
    # Download if not available
    try:
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)
    except:
        pass

from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)


class TrueFalseGenerator:
    """Generates True/False questions from text passages."""
    
    def __init__(self):
        """Initialize the True/False generator."""
        logger.info("TrueFalseGenerator initialized")
    
    def generate_true_false(
        self, 
        text: str, 
        num_questions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate true/false questions from the input text.
        
        Args:
            text: Input text to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            List of question dictionaries
        """
        logger.info(f"Generating {num_questions} True/False questions...")
        
        try:
            # Tokenize into sentences
            sentences = sent_tokenize(text)
            
            # Filter out short sentences
            sentences = [s for s in sentences if len(s.split()) > 5]
            
            if not sentences:
                logger.warning("No suitable sentences found for T/F generation")
                return []
            
            # Generate mix of true and false questions
            questions = []
            num_true = num_questions // 2
            num_false = num_questions - num_true
            
            # Generate TRUE questions (use original sentences)
            available_sentences = sentences.copy()
            random.shuffle(available_sentences)
            
            for i in range(min(num_true, len(available_sentences))):
                sentence = available_sentences[i]
                questions.append({
                    'question': sentence,
                    'answer': True,
                    'type': 'True/False',
                    'explanation': 'This statement is directly from the text.',
                    'user_answer': None
                })
            
            # Generate FALSE questions (modify sentences)
            for i in range(min(num_false, len(available_sentences) - num_true)):
                sentence = available_sentences[num_true + i]
                false_sentence, explanation = self._create_false_statement(sentence)
                
                questions.append({
                    'question': false_sentence,
                    'answer': False,
                    'type': 'True/False',
                    'explanation': explanation,
                    'user_answer': None
                })
            
            # Shuffle the order
            random.shuffle(questions)
            
            logger.info(f"Successfully generated {len(questions)} T/F questions")
            return questions
            
        except Exception as exc:
            logger.error(f"Error generating T/F questions: {exc}")
            return []
    
    def _create_false_statement(self, sentence: str) -> tuple:
        """
        Create a false statement by modifying the sentence.
        
        Returns:
            Tuple of (false_statement, explanation)
        """
        # Simple negation strategies
        strategies = [
            self._negate_verb,
            self._swap_numbers,
            self._replace_keywords,
        ]
        
        # Try each strategy
        random.shuffle(strategies)
        for strategy in strategies:
            result = strategy(sentence)
            if result:
                return result
        
        # Fallback: simple negation
        return (
            sentence.replace("is", "is not").replace("are", "are not"),
            "The statement has been negated."
        )
    
    def _negate_verb(self, sentence: str) -> tuple:
        """Add 'not' after helping verbs."""
        patterns = [
            (r'\bis\b', 'is not'),
            (r'\bare\b', 'are not'),
            (r'\bwas\b', 'was not'),
            (r'\bwere\b', 'were not'),
            (r'\bhas\b', 'has not'),
            (r'\bhave\b', 'have not'),
            (r'\bcan\b', 'cannot'),
            (r'\bwill\b', 'will not'),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                modified = re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)
                return modified, "The verb has been negated."
        
        return None
    
    def _swap_numbers(self, sentence: str) -> tuple:
        """Swap numbers in the sentence."""
        numbers = re.findall(r'\b\d+\b', sentence)
        if numbers:
            original = numbers[0]
            fake = str(int(original) + random.choice([1, 2, -1, -2, 10]))
            modified = sentence.replace(original, fake, 1)
            return modified, f"The number has been changed from {original} to {fake}."
        return None
    
    def _replace_keywords(self, sentence: str) -> tuple:
        """Replace keywords with opposite concepts."""
        replacements = {
            'increase': 'decrease',
            'decrease': 'increase',
            'more': 'less',
            'less': 'more',
            'always': 'never',
            'never': 'always',
            'all': 'none',
            'none': 'all',
            'before': 'after',
            'after': 'before',
        }
        
        for original, replacement in replacements.items():
            pattern = r'\b' + original + r'\b'
            if re.search(pattern, sentence, re.IGNORECASE):
                modified = re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)
                return modified, f"'{original}' has been replaced with '{replacement}'."
        
        return None