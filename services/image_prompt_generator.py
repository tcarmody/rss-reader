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
            "prompt_template": """Analyze this news story and create a striking photojournalistic image prompt:

Title: "{title}"
Content: {content}

TASK: Identify the most visually compelling moment or scene from this story. Don't just describe the headline - dig into the content to find dramatic, human, or emotionally resonant visual elements.

Create a detailed photography prompt that includes:

SCENE & COMPOSITION:
- What is the most powerful visual moment in this story?
- What human emotion or dramatic tension can be captured?
- What specific setting/location would be most impactful?
- What foreground, middle ground, background elements tell the story?

TECHNICAL DETAILS:
- Camera angle and lens choice for maximum impact
- Lighting conditions (natural, artificial, dramatic shadows)
- Depth of field and focus points
- Color palette and mood

HUMAN ELEMENTS:
- What expressions, gestures, or body language convey the story?
- How are people positioned relative to each other?
- What objects or symbols add meaning to the scene?

Generate a concrete, detailed prompt that an AI image generator could use to create a compelling news photograph that goes beyond the obvious and captures the deeper story."""
        },
        
        "illustration": {
            "name": "Editorial Illustration", 
            "description": "Artistic illustration style",
            "prompt_template": """Create a conceptual editorial illustration that captures the deeper meaning of this story:

Title: "{title}"
Content: {content}

ANALYSIS TASK: Look beyond the literal events. What are the underlying themes, conflicts, or transformations in this story? What metaphors or symbols could powerfully represent these concepts?

Design an editorial illustration with:

CONCEPTUAL ELEMENTS:
- What central metaphor or visual analogy best represents this story?
- What symbols, objects, or imagery could represent the key players or forces?
- What visual storytelling technique would make the abstract concrete?
- How can you show change, conflict, or resolution visually?

ARTISTIC APPROACH:
- What illustration style best serves the story (realistic, stylized, minimalist)?
- What art medium/technique enhances the message (digital painting, line art, collage)?
- Should it be literal, symbolic, or surreal?

COMPOSITION & MOOD:
- What color psychology supports the story's emotional tone?
- How should elements be arranged to guide the viewer's eye?
- What lighting or shading creates the right atmosphere?
- What perspective or viewpoint is most impactful?

Create a detailed prompt for an AI image generator that results in a thought-provoking editorial illustration worthy of a major publication cover."""
        },
        
        "abstract": {
            "name": "Abstract Conceptual",
            "description": "Abstract artistic representation", 
            "prompt_template": """Transform this story into an abstract visual language that captures its emotional and conceptual essence:

Title: "{title}"
Content: {content}

DEEP ANALYSIS: What are the invisible forces, emotions, tensions, or transformations at the heart of this story? How can abstract visual elements embody these intangible concepts?

Create an abstract artwork prompt with:

CONCEPTUAL TRANSLATION:
- What emotions, energies, or psychological states does this story evoke?
- How can abstract forms represent the key relationships or conflicts?
- What visual metaphors could translate complex ideas into shapes, colors, movement?
- What is the story's rhythm, pace, or emotional arc, and how can this be visualized?

ABSTRACT ELEMENTS:
- What shapes, forms, or patterns best embody the story's core concepts?
- How should color, texture, and composition reflect the emotional journey?
- What kind of movement or flow should the eye experience?
- Should the abstraction suggest chaos/order, tension/harmony, growth/decay?

ARTISTIC EXECUTION:
- What abstract art style or movement best serves this story (expressionism, constructivism, fluid dynamics)?
- How should space, scale, and proportion create psychological impact?
- What interplay of light, shadow, positive/negative space enhances meaning?

Generate a detailed prompt for creating abstract art that makes viewers feel the story's deeper truths without literal representation."""
        },
        
        "infographic": {
            "name": "Infographic Style",
            "description": "Data visualization and infographic",
            "prompt_template": """Analyze this story and create a compelling data visualization that reveals insights beyond the obvious:

Title: "{title}"
Content: {content}

INFORMATION MINING: Look for quantitative elements, processes, relationships, comparisons, or trends within this story. What data points, statistics, timelines, or structural relationships can be visualized?

Design an infographic that includes:

DATA IDENTIFICATION:
- What are the key numbers, percentages, or measurements in this story?
- What processes, workflows, or cause-and-effect relationships exist?
- What comparisons, contrasts, or progressions can be shown visually?
- What hidden patterns or connections could be illuminated through design?

VISUAL STORYTELLING:
- How can the data be arranged to tell a compelling visual narrative?
- What's the most surprising or insightful angle on this information?
- How should the viewer's eye move through the information hierarchy?
- What visual metaphors could make complex data more accessible?

DESIGN EXECUTION:
- What chart types, icons, and visual elements best serve each data point?
- How should color coding, sizing, and spacing enhance comprehension?
- What balance of text, numbers, and visuals creates optimal impact?
- How can the overall design be both informative and visually striking?

Create a detailed prompt for an AI image generator to produce a sophisticated infographic that makes complex information both beautiful and immediately understandable."""
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
    
    def _extract_key_content(self, title: str, content: str, max_words: int = 300) -> str:
        """
        Extract key content for prompt generation with enhanced visual focus.
        Prioritizes content with visual, emotional, or descriptive elements.
        """
        # Increase word limit for better visual analysis
        full_text = f"{title}\n\n{content}"
        words = full_text.split()
        
        if len(words) <= max_words:
            return full_text
        
        # If content is too long, try to preserve the most visually rich parts
        sentences = content.split('. ')
        
        # Priority keywords for visual content
        visual_keywords = [
            'see', 'show', 'appear', 'look', 'visual', 'image', 'photo', 'video',
            'color', 'light', 'dark', 'bright', 'scene', 'view', 'display',
            'dramatic', 'striking', 'beautiful', 'shocking', 'massive', 'tiny',
            'crowd', 'empty', 'full', 'destroyed', 'built', 'created',
            'people', 'person', 'man', 'woman', 'child', 'group',
            'building', 'street', 'home', 'office', 'factory', 'store',
            'outside', 'inside', 'above', 'below', 'behind', 'front'
        ]
        
        # Score sentences based on visual content
        scored_sentences = []
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Count visual keywords
            for keyword in visual_keywords:
                score += sentence_lower.count(keyword)
            
            # Bonus for numbers and specific details
            if any(char.isdigit() for char in sentence):
                score += 2
            
            # Bonus for descriptive adjectives
            descriptive_words = ['new', 'old', 'big', 'small', 'red', 'blue', 'green', 'black', 'white']
            for word in descriptive_words:
                if word in sentence_lower:
                    score += 1
                    
            scored_sentences.append((sentence, score))
        
        # Sort by score and keep highest scoring sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Build result with title and best sentences
        result_parts = [title]
        current_words = len(title.split())
        
        for sentence, score in scored_sentences:
            sentence_words = len(sentence.split())
            if current_words + sentence_words <= max_words - 10:  # Leave buffer
                result_parts.append(sentence)
                current_words += sentence_words
            else:
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