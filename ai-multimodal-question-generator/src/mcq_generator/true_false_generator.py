"""
True/False Question Generator
Generates true/false questions from input text strictly based on source content.
"""

import logging
import random
import re
from typing import List, Dict, Any

import nltk

try:
    from nltk.tokenize import sent_tokenize
    sent_tokenize("Test sentence.")
except (LookupError, OSError):
    try:
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)
    except:
        pass

from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)


class TrueFalseGenerator:
    """Generates True/False questions from text passages with difficulty support."""
    
    def __init__(self):
        """Initialize the True/False generator."""
        logger.info("TrueFalseGenerator initialized")
    
    def generate_true_false(
        self, 
        text: str, 
        num_questions: int = 5,
        target_difficulty: str = "Any"
    ) -> List[Dict[str, Any]]:
        """
        Generate true/false questions from the input text supporting up to 50 items.
        """
        logger.info(f"Generating {num_questions} True/False questions with difficulty: {target_difficulty}...")
        
        try:
            sentences = sent_tokenize(text)
            sentences = [s for s in sentences if len(s.split()) > 4]
            
            if not sentences:
                logger.warning("No suitable sentences found for T/F generation")
                return []
            
            questions = []
            num_true = num_questions // 2
            num_false = num_questions - num_true
            
            available_sentences = sentences.copy()
            
            # Ensure we have enough sentences even if requested count exceeds sentence count
            while len(available_sentences) < num_questions:
                available_sentences.extend(sentences)
                
            random.shuffle(available_sentences)
            
            # Generate TRUE questions
            for i in range(min(num_true, len(available_sentences))):
                sentence = available_sentences[i]
                questions.append({
                    'question': sentence,
                    'answer': True,
                    'type': 'True/False',
                    'difficulty': target_difficulty if target_difficulty != "Any" else "Easy",
                    'explanation': 'This statement is directly and accurately stated in the text.',
                    'user_answer': None
                })
            
            # Generate FALSE questions based on difficulty
            for i in range(min(num_false, len(available_sentences) - num_true)):
                sentence = available_sentences[num_true + i]
                false_sentence, explanation = self._create_false_statement(sentence, target_difficulty)
                
                questions.append({
                    'question': false_sentence,
                    'answer': False,
                    'type': 'True/False',
                    'difficulty': target_difficulty if target_difficulty != "Any" else "Medium",
                    'explanation': explanation,
                    'user_answer': None
                })
            
            random.shuffle(questions)
            logger.info(f"Successfully generated {len(questions)} T/F questions")
            return questions[:num_questions]
            
        except Exception as exc:
            logger.error(f"Error generating T/F questions: {exc}")
            return []
    
    def _create_false_statement(self, sentence: str, difficulty: str) -> tuple:
        """Create a false statement tailored to the requested difficulty level."""
        # Hard mode prioritizes subtle keyword or number swaps over simple 'not' negations
        if difficulty == "Hard":
            strategies = [self._swap_numbers, self._replace_keywords, self._negate_verb]
        elif difficulty == "Easy":
            strategies = [self._negate_verb, self._replace_keywords]
        else:
            strategies = [self._negate_verb, self._swap_numbers, self._replace_keywords]
            
        random.shuffle(strategies)
        for strategy in strategies:
            result = strategy(sentence)
            if result:
                return result
        
        return (
            sentence.replace("is", "is not").replace("are", "are not"),
            "The statement has been altered to contradict the source text."
        )
    
    def _negate_verb(self, sentence: str) -> tuple:
        """Add negation to helping verbs."""
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
                return modified, "The statement reverses the verb meaning found in the text."
        
        return None
    
    def _swap_numbers(self, sentence: str) -> tuple:
        """Swap numbers to create a subtle factual error."""
        numbers = re.findall(r'\b\d+\b', sentence)
        if numbers:
            original = numbers[0]
            fake = str(int(original) + random.choice([1, 2, 5, -1, -2]))
            if fake == original:
                fake = str(int(original) + 3)
            modified = sentence.replace(original, fake, 1)
            return modified, f"The numeric value was altered from {original} to {fake}."
        return None
    
    def _replace_keywords(self, sentence: str) -> tuple:
        """Replace key concepts with antonyms."""
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
            'primary': 'secondary',
            'major': 'minor'
        }
        
        for original, replacement in replacements.items():
            pattern = r'\b' + original + r'\b'
            if re.search(pattern, sentence, re.IGNORECASE):
                modified = re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)
                return modified, f"The keyword '{original}' was replaced with its opposite '{replacement}'."
        
        return None
