#!/usr/bin/env python3
"""
Script to fix issues with the RSS Reader application.
This script addresses:
1. Anthropic client initialization errors with 'proxies' parameter
2. Import errors with 'latest_data' from main.py
"""

import os
import re
import sys
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_script")

def backup_file(file_path):
    """Create a backup of a file before modifying it."""
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup: {backup_path}")
    return backup_path

def fix_anthropic_client_initialization(file_path):
    """Fix Anthropic client initialization to handle different library versions."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    try:
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Look for Anthropic client initialization
        client_init_pattern = r'self\.client\s*=\s*anthropic\.Anthropic\(api_key=([^)]+)\)'
        match = re.search(client_init_pattern, content)
        
        if match:
            # Create the fixed initialization code
            fixed_code = (
                "try:\n"
                f"            # Try the current API format\n"
                f"            self.client = anthropic.Anthropic(api_key={match.group(1)})\n"
                f"        except TypeError as e:\n"
                f"            if 'proxies' in str(e):\n"
                f"                # Fall back to older API format if needed\n"
                f"                import inspect\n"
                f"                if 'proxies' in inspect.signature(anthropic.Anthropic.__init__).parameters:\n"
                f"                    self.client = anthropic.Anthropic(api_key={match.group(1)}, proxies=None)\n"
                f"                else:\n"
                f"                    raise"
            )
            
            # Replace the initialization
            new_content = re.sub(client_init_pattern, fixed_code, content)
            
            # Write the updated file
            with open(file_path, 'w') as f:
                f.write(new_content)
                
            logger.info(f"Fixed Anthropic client initialization in {file_path}")
            return True
        else:
            logger.warning(f"No Anthropic client initialization found in {file_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error fixing Anthropic client in {file_path}: {e}")
        return False

def fix_server_latest_data_import(file_path):
    """Fix the import of 'latest_data' from main.py in server.py."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    try:
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Look for the problematic code in two parts
        import_pattern = r'from main import latest_data as main_latest_data'
        clusters_pattern = r'clusters\s*=\s*main_latest_data\.get\([\'"]clusters[\'"]\s*,\s*\[\]\)'
        
        if re.search(import_pattern, content) and re.search(clusters_pattern, content):
            # Create the fixed code
            fixed_import = "# Import directly from EnhancedRSSReader\nfrom main import EnhancedRSSReader"
            fixed_clusters = "# Get clusters directly from a temporary reader instance\ntemp_reader = EnhancedRSSReader()\noutput_file = asyncio.run(temp_reader.process_feeds())\nclusters = temp_reader.last_processed_clusters if hasattr(temp_reader, 'last_processed_clusters') else []"
            
            # Replace the problematic parts
            new_content = re.sub(import_pattern, fixed_import, content)
            new_content = re.sub(clusters_pattern, fixed_clusters, new_content)
            
            # Write the updated file
            with open(file_path, 'w') as f:
                f.write(new_content)
                
            logger.info(f"Fixed latest_data import in {file_path}")
            return True
        else:
            logger.warning(f"No problematic imports found in {file_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error fixing imports in {file_path}: {e}")
        return False

def add_latest_data_to_main(file_path):
    """Add 'latest_data' variable to main.py as an alternative fix."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    try:
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if latest_data already exists
        if 'latest_data' in content:
            logger.info(f"latest_data already exists in {file_path}")
            return True
        
        # Find a good spot to add the variable - after imports
        import_section_end = max(
            content.rfind('\n\n', 0, content.find('def ')) if content.find('def ') > 0 else len(content) - 1,
            content.rfind('\n\n', 0, content.find('class ')) if content.find('class ') > 0 else len(content) - 1
        )
        
        if import_section_end < 0:
            import_section_end = 0
        
        # Create the latest_data variable
        latest_data_code = (
            "\n\n# Store the latest processed data for server.py access\n"
            "latest_data = {\n"
            "    'clusters': [],\n"
            "    'timestamp': None,\n"
            "    'output_file': None\n"
            "}\n"
        )
        
        # Insert the code
        new_content = content[:import_section_end] + latest_data_code + content[import_section_end:]
        
        # Write the updated file
        with open(file_path, 'w') as f:
            f.write(new_content)
            
        logger.info(f"Added latest_data variable to {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error adding latest_data to {file_path}: {e}")
        return False

def fix_optimized_integration(file_path):
    """Fix the optimized_integration.py file to handle client creation correctly."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    try:
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Look for imports
        if 'import anthropic' not in content:
            # Add import at the top of the file after other imports
            import_section_end = content.find('\n\n', content.find('import'))
            if import_section_end < 0:
                import_section_end = content.find('\n', content.find('import'))
            
            import_code = "\nimport anthropic"
            new_content = content[:import_section_end] + import_code + content[import_section_end:]
            
            # Write the updated file
            with open(file_path, 'w') as f:
                f.write(new_content)
            
            # Re-read the updated content
            with open(file_path, 'r') as f:
                content = f.read()
        
        # Now add a helper function to create Anthropic client
        if 'def create_anthropic_client(' not in content:
            # Find a good spot - before any other function definition
            first_function = content.find('def ')
            if first_function < 0:
                first_function = len(content)
            
            helper_function = (
                "\ndef create_anthropic_client(api_key):\n"
                "    \"\"\"Create an Anthropic client handling API version differences.\"\"\"\n"
                "    try:\n"
                "        # Try current API format\n"
                "        return anthropic.Anthropic(api_key=api_key)\n"
                "    except TypeError as e:\n"
                "        if 'proxies' in str(e):\n"
                "            # Try older API format\n"
                "            import inspect\n"
                "            if 'proxies' in inspect.signature(anthropic.Anthropic.__init__).parameters:\n"
                "                return anthropic.Anthropic(api_key=api_key, proxies=None)\n"
                "        # Re-raise if we can't handle the error\n"
                "        raise\n\n"
            )
            
            new_content = content[:first_function] + helper_function + content[first_function:]
            
            # Write the updated file
            with open(file_path, 'w') as f:
                f.write(new_content)
                
            logger.info(f"Added Anthropic client helper function to {file_path}")
            return True
        else:
            logger.info(f"Helper function already exists in {file_path}")
            return True
    
    except Exception as e:
        logger.error(f"Error fixing optimized_integration.py: {e}")
        return False

def main():
    """Main function to run all fixes."""
    print("Starting RSS Reader Fix Script...")
    
    # Get the script directory to find files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths relative to the script directory
    summarizer_path = os.path.join(script_dir, 'summarizer.py')
    server_path = os.path.join(script_dir, 'server.py')
    main_path = os.path.join(script_dir, 'main.py')
    optimized_integration_path = os.path.join(script_dir, 'optimized_integration.py')
    
    # Create backups
    print("Creating backups of files...")
    backup_file(summarizer_path)
    backup_file(server_path)
    backup_file(main_path)
    if os.path.exists(optimized_integration_path):
        backup_file(optimized_integration_path)
    
    # Apply fixes
    print("\nApplying fixes...")
    
    # Fix 1: Anthropic client initialization in summarizer.py
    if fix_anthropic_client_initialization(summarizer_path):
        print("✓ Fixed Anthropic client initialization in summarizer.py")
    else:
        print("✗ Failed to fix Anthropic client initialization in summarizer.py")
    
    # Fix 2: Option A - Fix server.py to not use latest_data
    if fix_server_latest_data_import(server_path):
        print("✓ Fixed latest_data import in server.py")
    else:
        print("✗ Failed to fix latest_data import in server.py")
    
    # Fix 2: Option B - Add latest_data to main.py
    if add_latest_data_to_main(main_path):
        print("✓ Added latest_data variable to main.py")
    else:
        print("✗ Failed to add latest_data to main.py")
    
    # Fix 3: Fix optimized_integration.py if it exists
    if os.path.exists(optimized_integration_path):
        if fix_optimized_integration(optimized_integration_path):
            print("✓ Fixed optimized_integration.py")
        else:
            print("✗ Failed to fix optimized_integration.py")
    
    print("\nFix script completed! Try running server.py again.")
    print("If issues persist, you can restore from the .bak backup files.")

if __name__ == "__main__":
    main()
