"""
Compatibility layer for backward compatibility with old imports.
"""

import sys
import importlib
import logging
import warnings

# Logger setup
logger = logging.getLogger(__name__)

# Import redirections
IMPORT_REDIRECTS = {
    'reader': 'reader.base_reader',
    'summarizer': 'summarization.article_summarizer',
    'fast_summarizer': 'summarization.fast_summarizer',
    'batch_processing': 'common.batch_processing',
    'clustering': 'clustering.base',
    'enhanced_clustering': 'clustering.enhanced',
    'lm_cluster_analyzer': 'models.lm_analyzer',
    'utils.config': 'common.config',
    'utils.http': 'common.http',
    'utils.archive': 'common.archive',
    'utils.performance': 'common.performance',
    'utils.source_extractor': 'common.source_extractor'
}

# Class name redirections
CLASS_REDIRECTS = {
    'ArticleSummarizer': 'summarization.article_summarizer.ArticleSummarizer',
    'FastArticleSummarizer': 'summarization.fast_summarizer.FastSummarizer',
    'BatchProcessor': 'common.batch_processing.BatchProcessor',
    'RSSReader': 'reader.base_reader.RSSReader',
    'EnhancedRSSReader': 'reader.enhanced_reader.EnhancedRSSReader',
    'ArticleClusterer': 'clustering.base.ArticleClusterer'
}

# Function redirections
FUNCTION_REDIRECTS = {
    'create_fast_summarizer': 'summarization.fast_summarizer.create_fast_summarizer',
    'apply_fix_to_fast_summarizer': 'common.batch_processing.apply_fix_to_fast_summarizer',
    'create_enhanced_clusterer': 'clustering.enhanced.create_enhanced_clusterer'
}

def install_import_hook():
    """Install the import hook to redirect imports."""
    class RedirectImport:
        def __init__(self):
            self.original_import = __builtins__['__import__']
        
        def __call__(self, name, globals=None, locals=None, fromlist=(), level=0):
            # Check if this import needs redirection
            if name in IMPORT_REDIRECTS:
                new_name = IMPORT_REDIRECTS[name]
                logger.debug(f"Redirecting import: {name} -> {new_name}")
                
                # Issue deprecation warning
                warnings.warn(
                    f"Importing from '{name}' is deprecated. Use '{new_name}' instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
                
                # Try to import from new location
                try:
                    return self.original_import(new_name, globals, locals, fromlist, level)
                except ImportError:
                    logger.warning(f"Failed to import from new location {new_name}, falling back to {name}")
            
            # Use the original import for other imports
            return self.original_import(name, globals, locals, fromlist, level)
    
    # Install the hook
    __builtins__['__import__'] = RedirectImport()
    logger.info("Installed import redirection hook")

def apply_compatibility_layer():
    """Apply the full compatibility layer for backward compatibility."""
    # Install import hook
    install_import_hook()
    
    # Create module-level redirections
    for old_module, new_module in IMPORT_REDIRECTS.items():
        if old_module in sys.modules:
            # Module already imported, replace it
            try:
                new_mod = importlib.import_module(new_module)
                sys.modules[old_module] = new_mod
                logger.debug(f"Replaced existing module {old_module} with {new_module}")
            except ImportError:
                logger.warning(f"Failed to import {new_module} for replacing {old_module}")
    
    logger.info("Applied compatibility layer for backward compatibility")