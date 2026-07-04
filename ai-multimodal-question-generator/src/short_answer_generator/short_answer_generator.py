"""
Short Answer Question Generator
Generates descriptive open-ended questions from input text.
"""

import logging
import random
import re
from typing import List, Dict, Any
from difflib import SequenceMatcher

import nltk

# Download required NLTK data with better error handling
try:
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk import pos_tag
    from nltk.corpus import stopwords
    # Test if it works
    sent_tokenize("Test sentence.")
except (LookupError, OSError):
    try:
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass

from nltk.tokenize import sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)


class ShortAnswerGenerator:
    """Generates descriptive short answer questions (2-4 sentence answers)."""
    
    def __init__(self):
        """Initialize the Short Answer generator."""
        logger.info("ShortAnswerGenerator initialized")
        
        # Question templates
        self.templates = [
            "What is {topic}?",
            "Explain what {topic} is.",
            "Describe {topic}.",
            "What does {topic} mean?",
            "How does {topic} work?",
            "Explain how {topic} works.",
            "Why is {topic} important?",
            "What is the purpose of {topic}?",
            "Define {topic}.",
            "What are the main features of {topic}?",
        ]
    
    def generate_short_answer(
        self, 
        text: str, 
        num_questions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate descriptive short answer questions.
        
        Args:
            text: Input text to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            List of question dictionaries
        """
        logger.info(f"Generating {num_questions} Short Answer questions...")
        
        try:
            # Tokenize into sentences
            sentences = sent_tokenize(text)
            
            # Group sentences into paragraphs (every 2-3 sentences)
            paragraphs = []
            for i in range(0, len(sentences), 2):
                para = " ".join(sentences[i:i+3])
                if len(para.split()) > 15:  # At least 15 words
                    paragraphs.append(para)
            
            if not paragraphs:
                logger.warning("No suitable paragraphs found for SA generation")
                return []
            
            questions = []
            random.shuffle(paragraphs)
            
            for para in paragraphs[:num_questions * 2]:  # Try more to get enough
                result = self._create_descriptive_question(para)
                if result:
                    questions.append(result)
                    if len(questions) >= num_questions:
                        break
            
            logger.info(f"Successfully generated {len(questions)} SA questions")
            return questions
            
        except Exception as exc:
            logger.error(f"Error generating SA questions: {exc}")
            return []
    
    def _create_descriptive_question(self, paragraph: str) -> Dict[str, Any]:
        """
        Create a descriptive question from a paragraph.
        
        Returns:
            Question dictionary or None if failed
        """
        try:
            # Extract key topic/concept from the paragraph
            topic = self._extract_topic(paragraph)
            
            if not topic:
                return None
            
            # Choose a random question template
            template = random.choice(self.templates)
            question_text = template.format(topic=topic)
            
            # Extract keywords for answer checking
            keywords = self._extract_keywords(paragraph)
            
            # Use the paragraph as the expected answer
            answer_text = paragraph.strip()
            
            return {
                'question': question_text,
                'answer': answer_text,
                'type': 'Short Answer',
                'keywords': keywords,
                'user_answer': None,
                'explanation': f"Expected answer should mention: {', '.join(keywords[:5])}"
            }
            
        except Exception as exc:
            logger.debug(f"Could not create question from paragraph: {exc}")
            return None
    
    def _extract_topic(self, text: str) -> str:
        """Extract the main topic/subject from text."""
        # Look for capitalized terms (likely proper nouns/concepts)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        if capitalized:
            # Return the first significant capitalized term
            for term in capitalized:
                if term not in ['The', 'A', 'An', 'This', 'That', 'These', 'Those']:
                    return term
        
        # Fallback: extract first noun phrase (simplified)
        words = text.split()
        for i, word in enumerate(words[:10]):
            if len(word) > 4 and word[0].isupper():
                # Get 1-3 words as topic
                topic_words = []
                for j in range(i, min(i+3, len(words))):
                    if words[j][0].isupper() or j == i:
                        topic_words.append(words[j])
                    else:
                        break
                if topic_words:
                    return " ".join(topic_words)
        
        return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        try:
            # Tokenize
            words = word_tokenize(text.lower())
            
            # Remove stopwords and short words
            try:
                from nltk.corpus import stopwords
                stop_words = set(stopwords.words('english'))
            except:
                stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but'}
            
            keywords = [w for w in words 
                       if w.isalnum() 
                       and len(w) > 3 
                       and w not in stop_words]
            
            # Get unique keywords
            unique_keywords = []
            seen = set()
            for kw in keywords:
                if kw not in seen:
                    unique_keywords.append(kw)
                    seen.add(kw)
            
            return unique_keywords[:15]  # Return top 15
            
        except Exception as exc:
            logger.debug(f"Keyword extraction failed: {exc}")
            # Fallback: just split and filter
            words = text.lower().split()
            return [w for w in words if len(w) > 4][:15]
    
    def check_answer(
        self, 
        user_answer: str, 
        correct_answer: str, 
        keywords: List[str]
    ) -> Dict[str, Any]:
        """
        Check if user's answer is adequate using keyword matching and length.
        
        Args:
            user_answer: User's submitted answer
            correct_answer: The expected answer
            keywords: List of important keywords
            
        Returns:
            Dictionary with is_correct, similarity, and feedback
        """
        if not user_answer or len(user_answer.strip()) < 10:
            return {
                'is_correct': False,
                'similarity': 0.0,
                'feedback': '❌ Answer too short. Please provide at least 2-3 sentences.'
            }
        
        user_lower = user_answer.lower()
        user_words = set(word_tokenize(user_lower))
        
        # Count keyword matches
        keyword_matches = sum(1 for kw in keywords if kw in user_lower)
        keyword_percentage = keyword_matches / len(keywords) if keywords else 0
        
        # Check answer length (words)
        word_count = len(user_answer.split())
        
        # Overall similarity with expected answer
        similarity = SequenceMatcher(None, user_lower, correct_answer.lower()).ratio()
        
        # Scoring logic
        if keyword_percentage >= 0.6 and word_count >= 20:
            return {
                'is_correct': True,
                'similarity': keyword_percentage,
                'feedback': f'✅ Excellent! Your answer covers the key points. ({keyword_matches}/{len(keywords)} keywords)'
            }
        elif keyword_percentage >= 0.4 and word_count >= 15:
            return {
                'is_correct': True,
                'similarity': keyword_percentage,
                'feedback': f'✅ Good answer! Covers most important points. ({keyword_matches}/{len(keywords)} keywords)'
            }
        elif keyword_percentage >= 0.3 or word_count >= 20:
            return {
                'is_correct': False,
                'similarity': keyword_percentage,
                'feedback': f'⚠️ Partially correct. Missing some key concepts. ({keyword_matches}/{len(keywords)} keywords)'
            }
        else:
            return {
                'is_correct': False,
                'similarity': keyword_percentage,
                'feedback': f'❌ Incomplete answer. Please include more details about the key concepts.'
            }