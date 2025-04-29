# test_imports.py
import sys
print("Python Path:", sys.path)

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
