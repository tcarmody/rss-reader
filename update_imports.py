#!/usr/bin/env python3
"""
Script to automatically update import statements for a refactored Python project.
This script scans Python files and updates import statements to match the new package structure.
"""

import os
import re
import sys
import argparse
from pathlib import Path

# Mapping of old module names to new module paths
IMPORT_MAP = {
    # Module imports
    'summarizer': 'summarization.article_summarizer',
    'fast_summarizer': 'summarization.fast_summarizer',
    'reader': 'reader.base_reader',
    'clustering': 'clustering.base',
    'enhanced_clustering': 'clustering.enhanced',
    'batch_processing': 'common.batch_processing',
    'async_batch': 'common.batch_processing',
    'parallel_batch_processor': 'common.batch_processing',
    'enhanced_batch_processor': 'common.batch_processing',
    'lm_cluster_analyzer': 'models.lm_analyzer',
    'model_selection': 'models.selection',
    'tiered_cache': 'cache.tiered_cache',
    'memory_cache': 'cache.memory_cache',
    'rate_limiter': 'api.rate_limiter',
    
    # Utils to common remapping
    'utils.config': 'common.config',
    'utils.http': 'common.http',
    'utils.archive': 'common.archive',
    'utils.performance': 'common.performance',
    'utils.source_extractor': 'common.source_extractor',
    'utils.logging': 'common.logging',
}

# Class name to module mappings
CLASS_MAP = {
    'ArticleSummarizer': 'summarization.article_summarizer.ArticleSummarizer',
    'FastArticleSummarizer': 'summarization.fast_summarizer.FastSummarizer',
    'BatchProcessor': 'common.batch_processing.BatchProcessor',
    'RSSReader': 'reader.base_reader.RSSReader',
    'EnhancedRSSReader': 'reader.enhanced_reader.EnhancedRSSReader',
    'ArticleClusterer': 'clustering.base.ArticleClusterer',
    'EnhancedArticleClusterer': 'clustering.enhanced.EnhancedArticleClusterer',
    'MemoryCache': 'cache.memory_cache.MemoryCache',
    'TieredCache': 'cache.tiered_cache.TieredCache',
    'RateLimiter': 'api.rate_limiter.RateLimiter',
    'LMClusterAnalyzer': 'models.lm_analyzer.LMClusterAnalyzer',
}

# Function name to module mappings
FUNCTION_MAP = {
    'create_fast_summarizer': 'summarization.fast_summarizer.create_fast_summarizer',
    'create_enhanced_clusterer': 'clustering.enhanced.create_enhanced_clusterer',
    'create_cluster_analyzer': 'models.lm_analyzer.create_cluster_analyzer',
    'apply_fix_to_fast_summarizer': 'common.batch_processing.apply_fix_to_fast_summarizer',
    'adaptive_retry': 'api.rate_limiter.adaptive_retry',
    'get_model_identifier': 'models.selection.get_model_identifier',
    'estimate_complexity': 'models.selection.estimate_complexity',
    'select_model_by_complexity': 'models.selection.select_model_by_complexity',
}

def update_imports(file_path, dry_run=False):
    """
    Update import statements in a Python file.
    
    Args:
        file_path: Path to the Python file
        dry_run: If True, only print changes without modifying the file
        
    Returns:
        int: Number of import statements changed
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Track original content to check if changes were made
    original_content = content
    changes = 0
    
    # Pattern 1: from module import X
    pattern = r'from\s+({0})\s+import\s+([^;\n]+)'.format('|'.join(map(re.escape, IMPORT_MAP.keys())))
    matches = re.finditer(pattern, content)
    
    for match in matches:
        old_import = match.group(0)
        old_module = match.group(1)
        imported_names = match.group(2).strip()
        
        # Get the new module path
        new_module = IMPORT_MAP[old_module]
        
        # Create the new import statement
        new_import = f"from {new_module} import {imported_names}"
        
        # Replace the import statement
        content = content.replace(old_import, new_import)
        if old_import != new_import:
            changes += 1
            if dry_run:
                print(f"In {file_path}:")
                print(f"  - {old_import}")
                print(f"  + {new_import}")
    
    # Pattern 2: import module
    pattern = r'import\s+({0})(?:\s+as\s+([^;\n]+))?'.format('|'.join(map(re.escape, IMPORT_MAP.keys())))
    matches = re.finditer(pattern, content)
    
    for match in matches:
        old_import = match.group(0)
        old_module = match.group(1)
        alias = match.group(2) if match.group(2) else None
        
        # Get the new module path
        new_module = IMPORT_MAP[old_module]
        
        # Create the new import statement
        if alias:
            new_import = f"import {new_module} as {alias}"
        else:
            new_import = f"import {new_module}"
        
        # Replace the import statement
        content = content.replace(old_import, new_import)
        if old_import != new_import:
            changes += 1
            if dry_run:
                print(f"In {file_path}:")
                print(f"  - {old_import}")
                print(f"  + {new_import}")
    
    # Only write the file if changes were made and not in dry run mode
    if content != original_content and not dry_run:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
            
        print(f"Updated {changes} import(s) in {file_path}")
    
    return changes

def find_python_files(directory, exclude_dirs=None):
    """
    Find all Python files in a directory and its subdirectories.
    
    Args:
        directory: Directory to search
        exclude_dirs: List of directory names to exclude
        
    Returns:
        List of Python file paths
    """
    if exclude_dirs is None:
        exclude_dirs = ['.git', '.venv', 'venv', '__pycache__', 'env']
    
    python_files = []
    
    for root, dirs, files in os.walk(directory):
        # Remove excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def main():
    parser = argparse.ArgumentParser(description='Update import statements in Python files')
    parser.add_argument('directory', help='Directory to search for Python files')
    parser.add_argument('--dry-run', action='store_true', help='Print changes without modifying files')
    parser.add_argument('--exclude', nargs='+', help='Directories to exclude from search')
    
    args = parser.parse_args()
    
    # Validate the directory
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a directory")
        return 1
    
    # Find all Python files
    exclude_dirs = args.exclude if args.exclude else None
    python_files = find_python_files(args.directory, exclude_dirs)
    
    print(f"Found {len(python_files)} Python files to check")
    
    # Update imports in each file
    total_changes = 0
    for file_path in python_files:
        changes = update_imports(file_path, args.dry_run)
        total_changes += changes
    
    # Print summary
    if args.dry_run:
        print(f"\nDry run completed. {total_changes} import statements would be changed.")
    else:
        print(f"\nUpdated {total_changes} import statements in {len(python_files)} files.")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
