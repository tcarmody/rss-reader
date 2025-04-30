"""
Test script to verify that imports are working correctly.
Used to diagnose Python path and module import issues.
"""

import os
import sys

# Add the parent directory to sys.path to ensure modules can be imported
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

print("Python Path:", sys.path)
print("Current Directory:", os.getcwd())
print("Parent Directory:", parent_dir)

try:
    import flask
    print("Flask version:", flask.__version__)
except ImportError as e:
    print("Flask import error:", e)

try:
    from reader import RSSReader
    print("RSSReader imported successfully")
except ImportError as e:
    print("RSSReader import error:", e)

try:
    from summarizer import ArticleSummarizer
    print("ArticleSummarizer imported successfully")
except ImportError as e:
    print("ArticleSummarizer import error:", e)

try:
    from clustering import ArticleClusterer
    print("ArticleClusterer imported successfully")
except ImportError as e:
    print("ArticleClusterer import error:", e)

try:
    from enhanced_clustering import EnhancedArticleClusterer
    print("EnhancedArticleClusterer imported successfully")
except ImportError as e:
    print("EnhancedArticleClusterer import error:", e)

try:
    from tests.lm_cluster_analyzer import LMClusterAnalyzer
    print("LMClusterAnalyzer imported successfully")
except ImportError as e:
    print("LMClusterAnalyzer import error:", e)

try:
    from utils.config import get_env_var
    print("utils.config imported successfully")
except ImportError as e:
    print("utils.config import error:", e)

if __name__ == "__main__":
    print("\nImport test completed.")
    print("If any errors occurred, check Python path and module locations.")