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
    """Generate image prompts from article content using Claude API."""
    
    STYLE_TEMPLATES = {
        "photojournalistic": {
            "name": "Photojournalistic",
            "description": "Realistic news photography style",
            "prompt_template": """EXTRACT SPECIFIC VISUAL ELEMENTS from this story:

Title: "{title}"
Content: {content}

Step 1 - FIND CONCRETE DETAILS:
Read through the content and identify:
- WHO: Specific people mentioned (names, roles, ages, descriptions)
- WHERE: Exact locations, buildings, rooms, outdoor settings
- WHAT: Specific objects, vehicles, equipment, products mentioned
- WHEN: Time of day, weather, season, lighting conditions described
- HOW: Actions, movements, gestures, interactions happening

Step 2 - IDENTIFY THE MOST VISUAL MOMENT:
From the content, find the single most dramatic, emotional, or visually striking moment. Look for:
- A specific scene where people are doing something
- A moment of tension, conflict, celebration, or change
- Physical interactions between people or with objects
- Environmental details that set the scene

Step 3 - CREATE PHOTOJOURNALISTIC PROMPT:
Write a detailed image generation prompt that shows this specific moment as a news photograph. Include:
- The exact people involved and what they're doing
- Their precise location and surroundings
- The specific time/lighting/weather from the story
- Objects, tools, or environmental details mentioned
- Camera angle that captures the human drama
- Professional photography specifications (lighting, composition, lens)

CRITICAL: Base everything on specific details found in the content, not generic interpretations of the headline."""
        },
        
        "illustration": {
            "name": "Editorial Illustration", 
            "description": "Artistic illustration style",
            "prompt_template": """CREATE SYMBOLIC EDITORIAL ILLUSTRATION from story elements:

Title: "{title}"
Content: {content}

Step 1 - EXTRACT STORY ELEMENTS:
From the content, identify:
- CENTRAL CONFLICT: What two forces are opposing each other?
- KEY PLAYERS: Who are the main people/organizations involved?
- STAKES: What is being gained, lost, or changed?
- SETTING: Where does this take place (industry, location, environment)?
- OBJECTS/TOOLS: What specific items, technologies, or symbols are mentioned?

Step 2 - FIND VISUAL METAPHORS:
Based on the extracted elements, identify:
- What physical objects could represent the main concepts?
- What spatial relationships show the conflict or change?
- What natural or architectural elements reflect the story's scale?
- How can size, position, or interaction show power dynamics?

Step 3 - CREATE ILLUSTRATION PROMPT:
Design an editorial illustration that uses the specific elements from the story:
- Transform key players into visual symbols or representations
- Use actual objects/settings mentioned in the content
- Show the conflict through spatial arrangement and visual metaphors
- Include specific colors, textures, or styles that match the story's tone
- Specify artistic technique (digital painting, vector art, mixed media)
- Create composition that guides viewer to understand the story's meaning

CRITICAL: Use specific details and elements found in the actual content, not abstract concepts from the headline."""
        },
        
        "abstract": {
            "name": "Abstract Conceptual",
            "description": "Abstract artistic representation", 
            "prompt_template": """TRANSLATE STORY DYNAMICS into abstract visual elements:

Title: "{title}"
Content: {content}

Step 1 - IDENTIFY STORY DYNAMICS:
From the content, find:
- MOVEMENT: What is growing, shrinking, accelerating, slowing, starting, ending?
- PRESSURE: What forces are building, releasing, colliding, or balancing?
- RELATIONSHIPS: How are different elements connected, separated, or interacting?
- RHYTHM: What patterns, cycles, or progressions are described?
- SCALE: What contrasts in size, importance, or impact are mentioned?

Step 2 - EXTRACT EMOTIONAL QUALITIES:
Based on the story content, identify:
- What emotional energy is present (tension, excitement, calm, chaos)?
- What is the dominant feeling of change (sudden vs gradual, violent vs peaceful)?
- Where is the focal point of intensity or importance in the story?
- What is the overall trajectory (rising, falling, cyclical, explosive)?

Step 3 - CREATE ABSTRACT PROMPT:
Design abstract art that translates these specific story elements:
- Convert the identified movements into visual flows and directions
- Transform relationships into spatial arrangements and connections
- Use color and form to represent the specific emotional qualities found
- Create composition that mirrors the story's rhythm and progression
- Specify abstract art techniques that match the story's energy
- Include scale and proportion that reflects the story's dynamics

CRITICAL: Base all abstract elements on specific dynamics and relationships found in the story content, not general impressions."""
        },
        
        "infographic": {
            "name": "Infographic Style",
            "description": "Data visualization and infographic",
            "prompt_template": """EXTRACT QUANTIFIABLE INFORMATION to create data visualization:

Title: "{title}"
Content: {content}

Step 1 - FIND SPECIFIC DATA:
Read through the content and identify:
- NUMBERS: Exact figures, percentages, amounts, quantities mentioned
- COMPARISONS: Before/after, bigger/smaller, more/less relationships
- TIMELINES: Dates, durations, sequences, or progressions described
- CATEGORIES: Different types, groups, or classifications mentioned
- PROCESSES: Step-by-step procedures or workflows described

Step 2 - IDENTIFY RELATIONSHIPS:
From the extracted data, find:
- What increases or decreases in relation to what else?
- What are the cause-and-effect relationships?
- How do different categories compare in size or importance?
- What patterns or trends can be shown over time?
- What hierarchies or structures exist?

Step 3 - CREATE INFOGRAPHIC PROMPT:
Design a data visualization using the specific information found:
- Choose appropriate chart types for each piece of data found
- Use the exact numbers and categories mentioned in the content
- Show the specific comparisons and relationships identified
- Create visual hierarchy based on importance mentioned in the story
- Include relevant icons or symbols that relate to the story's industry/context
- Use color coding that makes sense for the specific data categories
- Design layout that tells the story through the progression of information

CRITICAL: Only visualize data and relationships explicitly mentioned in the story content, not implied or general industry information."""
        }
    }
    
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
        style: str = "photojournalistic"
    ) -> Dict[str, Any]:
        """
        Generate an image prompt for the given article content.
        
        Args:
            title: Article title
            content: Article content or summary
            style: Image style (photojournalistic, illustration, abstract, infographic)
            
        Returns:
            Dict containing the generated prompt, style info, and metadata
        """
        try:
            # Validate style
            if style not in self.STYLE_TEMPLATES:
                logger.warning(f"Unknown style '{style}', using 'photojournalistic'")
                style = "photojournalistic"
            
            # Check cache first
            cache_key = self._get_cache_key(title, content, style)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"Using cached image prompt for style '{style}'")
                return cached_result
            
            # Extract and limit content length
            key_content = self._extract_key_content(title, content)
            
            # Get style template
            style_config = self.STYLE_TEMPLATES[style]
            prompt_template = style_config["prompt_template"]
            
            # Format the prompt template
            formatted_prompt = prompt_template.format(
                title=title,
                content=key_content
            )
            
            logger.info(f"Generating {style} image prompt for article: {title[:50]}...")
            
            # Generate prompt using Claude with more tokens for detailed analysis
            response = await self.summarizer.client.messages.create(
                model=self.summarizer.model_name,
                max_tokens=500,  # Increased for more detailed prompts
                temperature=0.8,  # More creative for visual generation
                messages=[{
                    "role": "user", 
                    "content": formatted_prompt
                }]
            )
            
            generated_prompt = response.content[0].text.strip()
            
            # Prepare result
            result = {
                "prompt": generated_prompt,
                "style": style,
                "style_name": style_config["name"],
                "style_description": style_config["description"],
                "title": title,
                "generated_at": datetime.now().isoformat(),
                "word_count": len(generated_prompt.split())
            }
            
            # Cache the result (convert timedelta to seconds)
            ttl_seconds = int(self.cache_ttl.total_seconds())
            self.cache.set(cache_key, result, ttl=ttl_seconds)
            
            logger.info(f"Generated {style} image prompt ({result['word_count']} words)")
            return result
            
        except Exception as e:
            logger.error(f"Error generating image prompt: {str(e)}")
            
            # Return fallback prompt
            fallback_prompt = self._generate_fallback_prompt(title, content, style)
            return {
                "prompt": fallback_prompt,
                "style": style,
                "style_name": self.STYLE_TEMPLATES.get(style, {}).get("name", style.title()),
                "style_description": "Fallback prompt due to generation error",
                "title": title,
                "generated_at": datetime.now().isoformat(),
                "word_count": len(fallback_prompt.split()),
                "error": str(e)
            }
    
    def _generate_fallback_prompt(self, title: str, content: str, style: str) -> str:
        """Generate a simple fallback prompt when AI generation fails."""
        # Extract key words from title and content
        key_words = []
        for text in [title, content]:
            words = text.lower().split()
            # Simple keyword extraction - avoid common stop words
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            meaningful_words = [w for w in words if len(w) > 3 and w not in stop_words]
            key_words.extend(meaningful_words[:5])  # Limit per text
        
        # Create basic prompt based on style
        style_descriptors = {
            "photojournalistic": "realistic photograph, journalistic style, professional lighting",
            "illustration": "editorial illustration, artistic style, symbolic elements",
            "abstract": "abstract art, conceptual design, emotional colors",
            "infographic": "clean infographic, data visualization, modern design"
        }
        
        descriptor = style_descriptors.get(style, "professional image")
        key_phrase = " ".join(key_words[:6])  # Use top keywords
        
        return f"{descriptor} showing {key_phrase}, high quality, detailed composition"
    
    def get_available_styles(self) -> Dict[str, Dict[str, str]]:
        """Get list of available image styles with descriptions."""
        return {
            style_key: {
                "name": style_config["name"],
                "description": style_config["description"]
            }
            for style_key, style_config in self.STYLE_TEMPLATES.items()
        }


# Singleton instance for use across the application
_image_prompt_generator = None

def get_image_prompt_generator() -> ImagePromptGenerator:
    """Get or create the global image prompt generator instance."""
    global _image_prompt_generator
    if _image_prompt_generator is None:
        _image_prompt_generator = ImagePromptGenerator()
    return _image_prompt_generator