#!/usr/bin/env python3
"""
Runner script to apply the batch processor fix.

Include this at the start of your main.py:
```
# Apply batch processor fixes
import apply_batch_fix
apply_batch_fix.apply()
```
"""

import sys
import logging
import importlib.util
import traceback

def apply():
    """Apply the batch processor fix."""
    try:
        # Try to import the fix module
        spec = importlib.util.spec_from_file_location("fix_rss_batch_processor", "fix_rss_batch_processor.py")
        if not spec or not spec.loader:
            print("Error: Could not find fix_rss_batch_processor.py")
            return False
            
        fix_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fix_module)
        
        # Apply the fixes
        if hasattr(fix_module, 'apply_all_fixes') and callable(fix_module.apply_all_fixes):
            result = fix_module.apply_all_fixes()
            if result:
                print("Successfully applied batch processor fix")
            else:
                print("Failed to apply batch processor fix")
            return result
        else:
            print("Error: apply_all_fixes function not found in fix module")
            return False
    except Exception as e:
        print(f"Error applying batch processor fix: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    apply()