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
            "prompt_template": """Create a photojournalistic image prompt for: "{title}"

Article summary: {content}

Generate a detailed prompt for creating a realistic, news-style photograph that captures the essence of this story. Include:
- Specific visual elements and composition
- Lighting and atmosphere that matches the story's tone
- Key subjects or objects that should be featured
- Realistic setting and context
- Professional photography techniques

The prompt should be optimized for AI image generators like DALL-E, Midjourney, or Stable Diffusion.
Keep it under 200 words and focus on visual, concrete details."""
        },
        
        "illustration": {
            "name": "Editorial Illustration", 
            "description": "Artistic illustration style",
            "prompt_template": """Create an editorial illustration prompt for: "{title}"

Article summary: {content}

Generate a detailed prompt for creating an artistic illustration that represents the key concepts of this story. Include:
- Visual metaphors and symbolic elements
- Color palette that reflects the story's mood
- Composition style (minimalist, detailed, abstract elements)
- Artistic technique (digital art, watercolor, vector style)
- Key concepts to visualize symbolically

The illustration should be suitable for editorial use in magazines or newspapers.
Keep it under 200 words and emphasize artistic and symbolic visual elements."""
        },
        
        "abstract": {
            "name": "Abstract Conceptual",
            "description": "Abstract artistic representation", 
            "prompt_template": """Create an abstract conceptual art prompt for: "{title}"

Article summary: {content}

Generate a detailed prompt for creating abstract art that represents the underlying themes and emotions of this story. Include:
- Abstract shapes, forms, and patterns
- Color psychology and emotional resonance
- Conceptual visual metaphors
- Artistic movement inspiration (modernist, expressionist, etc.)
- Composition and visual flow

The artwork should convey the story's essence through abstract visual language.
Keep it under 200 words and focus on emotional and conceptual visual elements."""
        },
        
        "infographic": {
            "name": "Infographic Style",
            "description": "Data visualization and infographic",
            "prompt_template": """Create an infographic-style image prompt for: "{title}"

Article summary: {content}

Generate a detailed prompt for creating an informational graphic that visualizes the key data, facts, or processes from this story. Include:
- Clean, modern design elements
- Data visualization components (charts, graphs, icons)
- Hierarchical information layout
- Professional color scheme
- Clear typography and visual hierarchy
- Key statistics or facts to highlight

The infographic should be informative and visually engaging for digital or print media.
Keep it under 200 words and emphasize clarity and information design."""
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
    
    def _extract_key_content(self, title: str, content: str, max_words: int = 150) -> str:
        """Extract key content for prompt generation, limiting length."""
        # Combine title and content, but limit total length
        full_text = f"{title}\n\n{content}"
        words = full_text.split()
        
        if len(words) > max_words:
            # Keep the title and truncate content
            title_words = title.split()
            remaining_words = max_words - len(title_words)
            content_words = content.split()[:remaining_words]
            return f"{title}\n\n{' '.join(content_words)}..."
        
        return full_text
    
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
            
            # Generate prompt using Claude
            response = await self.summarizer.client.messages.create(
                model=self.summarizer.model_name,
                max_tokens=300,  # Limit response length
                temperature=0.7,  # Slightly creative
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