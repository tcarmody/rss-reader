"""
Image Prompt Generator Service

Generates AI image prompts from article content using Claude API.
Supports multiple artistic styles optimized for different image generation tools.
"""

import logging
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from summarization.article_summarizer import ArticleSummarizer
from cache.tiered_cache import TieredCache

logger = logging.getLogger(__name__)


class ImagePromptGenerator:
    """Generate editorial illustration prompts from article content using Claude API."""

    EDITORIAL_PROMPT_TEMPLATE = """FIND A SINGLE STRONG VISUAL DETAIL from this article:

Title: "{title}"
Original Content: {content}

TASK: Create an editorial illustration based on ONE specific visual detail or narrative moment from the article content.

Step 1 - SCAN FOR CONCRETE VISUAL DETAILS:
Read through the original content and look for:
- A specific person doing something (working, speaking, gesturing, interacting)
- A particular object, tool, or piece of technology mentioned
- A described physical setting (office, factory, street, building interior)
- An action or interaction between people
- A specific scene that illustrates the story's impact
- Objects that represent the central issue (documents, machines, products)

Step 2 - SELECT THE STRONGEST VISUAL:
Choose ONE detail that:
- Is specifically described in the content (not just implied by the headline)
- Has clear visual potential for illustration
- Represents or symbolizes the story's main point
- Could work as a standalone editorial illustration

Step 3 - CREATE EDITORIAL ILLUSTRATION PROMPT:
Write a detailed prompt for an editorial illustration showing this single visual detail:
- Describe the specific scene, object, or interaction you selected
- Include artistic style: "editorial illustration in a clean, modern style"
- Specify composition details (close-up, wide view, perspective)
- Add appropriate color palette and mood
- Keep it focused on this ONE visual element

CRITICAL RULES:
- Use ONLY details explicitly mentioned in the original content
- Focus on ONE strong visual, not multiple elements
- Ignore the headline - work from the article content only
- Make it specific and concrete, not abstract or metaphorical"""
    
    def __init__(self, summarizer: Optional[ArticleSummarizer] = None):
        """
        Initialize the image prompt generator.
        
        Args:
            summarizer: Optional ArticleSummarizer instance. If None, creates a new one.
        """
        self.summarizer = summarizer or ArticleSummarizer()
        self.cache = TieredCache(
            memory_size=100,
            disk_path="./cache/image_prompts",
            ttl_days=1  # Cache for 1 day
        )
        
        # Cache settings
        self.cache_ttl = timedelta(hours=24)  # Cache prompts for 24 hours
        
    def _get_cache_key(self, title: str, content: str, style: str) -> str:
        """Generate cache key for prompt."""
        content_hash = hashlib.md5(f"{title}{content}{style}".encode()).hexdigest()
        return f"image_prompt:{style}:{content_hash}"
    
    def _extract_key_content(self, title: str, content: str, max_words: int = 400) -> str:
        """
        Extract key content for prompt generation prioritizing concrete visual details.
        Focuses on specific people, places, actions, objects, and scenes.
        """
        # Increase word limit further for richer visual analysis
        full_text = f"{title}\n\n{content}"
        words = full_text.split()
        
        if len(words) <= max_words:
            return full_text
        
        # If content is too long, prioritize sentences with concrete visual elements
        sentences = content.split('. ')
        
        # Enhanced keywords for concrete visual content
        visual_keywords = [
            # Actions & Movement
            'walk', 'run', 'stand', 'sit', 'drive', 'fly', 'climb', 'fall', 'jump', 'dance',
            'work', 'build', 'destroy', 'create', 'break', 'fix', 'hold', 'carry', 'point',
            
            # People & Descriptions
            'people', 'person', 'man', 'woman', 'child', 'worker', 'employee', 'customer',
            'crowd', 'group', 'team', 'family', 'audience', 'protesters', 'officials',
            'wearing', 'dressed', 'uniform', 'suit', 'shirt', 'hat', 'glasses',
            
            # Places & Settings
            'building', 'office', 'factory', 'store', 'restaurant', 'hospital', 'school',
            'street', 'road', 'bridge', 'park', 'room', 'hall', 'stage', 'field',
            'inside', 'outside', 'downtown', 'upstairs', 'basement', 'rooftop',
            
            # Objects & Tools
            'machine', 'computer', 'phone', 'car', 'truck', 'equipment', 'device',
            'sign', 'banner', 'screen', 'table', 'chair', 'box', 'bag', 'tool',
            
            # Visual Qualities
            'see', 'look', 'show', 'appear', 'visible', 'bright', 'dark', 'light',
            'color', 'red', 'blue', 'green', 'yellow', 'black', 'white', 'gray',
            'large', 'small', 'huge', 'tiny', 'tall', 'short', 'wide', 'narrow',
            
            # Time & Weather
            'morning', 'afternoon', 'evening', 'night', 'sunny', 'cloudy', 'rain',
            'yesterday', 'today', 'during', 'while', 'when', 'as'
        ]
        
        # Score sentences based on visual content richness
        scored_sentences = []
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Count visual keywords (higher weight for specific terms)
            for keyword in visual_keywords:
                count = sentence_lower.count(keyword)
                if keyword in ['people', 'building', 'machine', 'work', 'see', 'show']:
                    score += count * 2  # Higher weight for key visual terms
                else:
                    score += count
            
            # Heavy bonus for numbers and measurements
            if any(char.isdigit() for char in sentence):
                score += 3
            
            # Bonus for proper nouns (likely specific people/places)
            words_in_sentence = sentence.split()
            for word in words_in_sentence:
                if word and word[0].isupper() and len(word) > 2:
                    score += 1
            
            # Bonus for quotes (often contain specific details)
            if '"' in sentence or "'" in sentence:
                score += 2
                
            # Bonus for concrete descriptive language
            descriptive_terms = ['new', 'old', 'first', 'last', 'main', 'central', 'local', 
                                'public', 'private', 'empty', 'full', 'open', 'closed']
            for term in descriptive_terms:
                if term in sentence_lower:
                    score += 1
                    
            scored_sentences.append((sentence, score))
        
        # Sort by score and keep highest scoring sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Build result with title and best sentences
        result_parts = [title]
        current_words = len(title.split())
        
        # Always include the first sentence (often contains key context)
        if sentences:
            first_sentence = sentences[0]
            first_words = len(first_sentence.split())
            if current_words + first_words <= max_words - 20:
                if first_sentence not in [s[0] for s in scored_sentences[:1]]:
                    result_parts.append(first_sentence)
                    current_words += first_words
        
        # Add highest scoring sentences
        for sentence, score in scored_sentences:
            sentence_words = len(sentence.split())
            if current_words + sentence_words <= max_words - 10 and sentence not in result_parts:
                result_parts.append(sentence)
                current_words += sentence_words
            elif current_words >= max_words * 0.9:  # Stop when near limit
                break
        
        return '\n\n'.join(result_parts)
    
    async def generate_prompt(
        self,
        title: str,
        content: str,
        style: str = "editorial"  # Only editorial style now
    ) -> Dict[str, Any]:
        """
        Generate an editorial illustration prompt from article content.

        Args:
            title: Article title
            content: Original article content (not summary)
            style: Ignored - always generates editorial illustrations

        Returns:
            Dict containing the generated prompt and metadata
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(title, content, "editorial")
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info("Using cached editorial illustration prompt")
                return cached_result

            # Extract and limit content length for analysis
            key_content = self._extract_key_content(title, content)

            # Format the prompt template
            formatted_prompt = self.EDITORIAL_PROMPT_TEMPLATE.format(
                title=title,
                content=key_content
            )

            logger.info(f"Generating editorial illustration prompt for: {title[:50]}...")

            # Generate prompt using Claude
            response = await self.summarizer.client.messages.create(
                model=self.summarizer.model_name,
                max_tokens=300,  # Focused, concise prompts
                temperature=0.7,  # Balanced creativity
                messages=[{
                    "role": "user",
                    "content": formatted_prompt
                }]
            )

            generated_prompt = response.content[0].text.strip()

            # Prepare result
            result = {
                "prompt": generated_prompt,
                "style": "editorial",
                "style_name": "Editorial Illustration",
                "style_description": "Editorial illustration focusing on a single strong visual detail",
                "title": title,
                "generated_at": datetime.now().isoformat(),
                "word_count": len(generated_prompt.split())
            }

            # Cache the result
            ttl_seconds = int(self.cache_ttl.total_seconds())
            self.cache.set(cache_key, result, ttl=ttl_seconds)

            logger.info(f"Generated editorial illustration prompt ({result['word_count']} words)")
            return result

        except Exception as e:
            logger.error(f"Error generating image prompt: {str(e)}")

            # Return fallback prompt
            fallback_prompt = self._generate_fallback_prompt(title, content)
            return {
                "prompt": fallback_prompt,
                "style": "editorial",
                "style_name": "Editorial Illustration",
                "style_description": "Fallback prompt due to generation error",
                "title": title,
                "generated_at": datetime.now().isoformat(),
                "word_count": len(fallback_prompt.split()),
                "error": str(e)
            }
    
    def _generate_fallback_prompt(self, title: str, content: str) -> str:
        """Generate a simple fallback prompt when AI generation fails."""
        # Extract key words from title and content
        key_words = []
        for text in [title, content]:
            words = text.lower().split()
            # Simple keyword extraction - avoid common stop words
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            meaningful_words = [w for w in words if len(w) > 3 and w not in stop_words]
            key_words.extend(meaningful_words[:5])  # Limit per text

        key_phrase = " ".join(key_words[:6])  # Use top keywords

        return f"Editorial illustration in a clean, modern style showing {key_phrase}, focused composition, professional design"

    def get_available_styles(self) -> Dict[str, Dict[str, str]]:
        """Get available editorial illustration style."""
        return {
            "editorial": {
                "name": "Editorial Illustration",
                "description": "Editorial illustration focusing on a single strong visual detail from the original content"
            }
        }


# Singleton instance for use across the application
_image_prompt_generator = None

def get_image_prompt_generator() -> ImagePromptGenerator:
    """Get or create the global image prompt generator instance."""
    global _image_prompt_generator
    if _image_prompt_generator is None:
        _image_prompt_generator = ImagePromptGenerator()
    return _image_prompt_generator