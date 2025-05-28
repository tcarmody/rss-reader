#!/usr/bin/env python3
"""
Implementation script to automatically add WSJ bypass enhancements
Run this script to modify your common/archive.py file
"""

import os
import re

def backup_file(filepath):
    """Create a backup of the original file"""
    backup_path = f"{filepath}.backup"
    if os.path.exists(filepath):
        with open(filepath, 'r') as original:
            with open(backup_path, 'w') as backup:
                backup.write(original.read())
        print(f"✅ Created backup: {backup_path}")
    return backup_path

def add_wsj_imports(content):
    """Add required imports for WSJ functionality"""
    import_section = '''import re
import time
import random
import json
import hashlib
from urllib.parse import urlparse, quote
import requests
from bs4 import BeautifulSoup'''
    
    # Find the existing imports section and add our imports if not present
    if 'import random' not in content:
        # Add after the existing imports
        lines = content.split('\n')
        insert_index = 0
        for i, line in enumerate(lines):
            if line.startswith('from bs4') or line.startswith('import requests'):
                insert_index = i + 1
        
        # Insert our additional imports
        additional_imports = [
            'import random',
            'import json',
            'import re'
        ]
        
        for imp in reversed(additional_imports):
            if imp not in content:
                lines.insert(insert_index, imp)
        
        content = '\n'.join(lines)
    
    return content

def add_wsj_constants(content):
    """Add WSJ-specific constants"""
    wsj_constants = '''
# WSJ-specific user agents for enhanced bypass
WSJ_USER_AGENTS = [
    'facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)',
    'Twitterbot/1.0',
    'LinkedInBot/1.0 (compatible; Mozilla/5.0; Apache-HttpClient +http://www.linkedin.com)',
    'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/W.X.Y.Z Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    'Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/600.2.5 (KHTML, like Gecko) Version/8.0.2 Safari/600.2.5 (Applebot/0.1; +http://www.apple.com/go/applebot)',
]
'''
    
    # Find where to insert constants (after existing PAYWALL_DOMAINS)
    if 'WSJ_USER_AGENTS' not in content:
        # Insert after COMPLEX_PAYWALL_DOMAINS
        insertion_point = content.find('COMPLEX_PAYWALL_DOMAINS = [')
        if insertion_point != -1:
            # Find the end of this list
            end_point = content.find(']', insertion_point) + 1
            # Find the next newline
            next_newline = content.find('\n', end_point)
            content = content[:next_newline] + wsj_constants + content[next_newline:]
    
    return content

def add_wsj_functions(content):
    """Add WSJ-specific functions"""
    
    wsj_functions = '''
def get_wsj_specific_session(url):
    """Create a session specifically optimized for WSJ with advanced evasion."""
    session = requests.Session()
    
    user_agent = random.choice(WSJ_USER_AGENTS)
    
    headers = {
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
    }
    
    referrer_strategies = [
        f'https://www.google.com/search?q={quote(url)}',
        'https://www.facebook.com/',
        'https://twitter.com/',
        'https://www.linkedin.com/',
        'https://news.google.com/',
        'https://t.co/',
        '',
    ]
    
    headers['Referer'] = random.choice(referrer_strategies)
    
    session.cookies.update({
        'wsjregion': 'na,us',
        'gdprApplies': 'false',
        'ccpaApplies': 'false',
        'usr_bkt': 'C0H2lPLMlD',
    })
    
    session.headers.update(headers)
    return session

def try_wsj_amp_version(url):
    """Try to access the AMP version of WSJ articles."""
    if 'wsj.com/articles/' in url:
        match = re.search(r'/articles/([^/?]+)', url)
        if match:
            article_id = match.group(1)
            return f'https://www.wsj.com/amp/articles/{article_id}'
    return None

def try_wsj_mobile_version(url):
    """Try mobile version which sometimes has less strict paywalls."""
    if 'wsj.com' in url and 'm.wsj.com' not in url:
        return url.replace('www.wsj.com', 'm.wsj.com')
    return None

def try_wsj_print_version(url):
    """Try print version which sometimes bypasses paywalls."""
    if '?' in url:
        return f"{url}&mod=article_inline"
    else:
        return f"{url}?mod=article_inline"

def is_wsj_actual_content(soup):
    """Check if the soup contains actual WSJ article content vs paywall/cleaning page."""
    paywall_indicators = [
        'subscribe to continue reading',
        'sign in to continue reading',
        'this content is reserved for subscribers',
        'cleaning webpage',
        'advertisement',
        'paywall',
        'subscribe now',
        'digital subscription'
    ]
    
    page_text = soup.get_text().lower()
    
    for indicator in paywall_indicators:
        if indicator in page_text:
            logging.warning(f"Detected paywall indicator: {indicator}")
            return False
    
    content_indicators = [
        soup.find('div', {'class': re.compile(r'article.*body|story.*body|content.*body')}),
        soup.find('div', {'data-module': 'ArticleBody'}),
        soup.find('section', {'class': re.compile(r'article|story|content')}),
        soup.find('div', {'class': 'wsj-snippet-body'}),
    ]
    
    for element in content_indicators:
        if element and len(element.get_text().strip()) > 200:
            return True
    
    return False

def extract_wsj_article_content(soup):
    """Extract the actual article content from WSJ page."""
    content_selectors = [
        'div[data-module="ArticleBody"]',
        '.article-content',
        '.wsj-snippet-body',
        '.article-wrap',
        '[data-module="BodyText"]',
        '.StoryBody',
        '.snippet-promotion',
    ]
    
    for selector in content_selectors:
        elements = soup.select(selector)
        if elements:
            content_div = elements[0]
            
            for unwanted in content_div.select('.advertisement, .ad, .promo, .related'):
                unwanted.decompose()
            
            paragraphs = content_div.find_all('p')
            if paragraphs:
                content = '\\n\\n'.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20)
                if len(content) > 200:
                    return content
    
    return None

def fetch_wsj_article_content_enhanced(url, session=None):
    """Enhanced WSJ content fetching with multiple strategies."""
    logging.info(f"Starting enhanced WSJ bypass for: {url}")
    
    if not session:
        session = get_wsj_specific_session(url)
    
    strategies = [
        ('Original URL', url),
        ('AMP Version', try_wsj_amp_version(url)),
        ('Mobile Version', try_wsj_mobile_version(url)),
        ('Print Version', try_wsj_print_version(url)),
    ]
    
    for strategy_name, test_url in strategies:
        if not test_url:
            continue
            
        logging.info(f"Trying WSJ strategy: {strategy_name} - {test_url}")
        
        try:
            time.sleep(random.uniform(1, 3))
            
            response = session.get(test_url, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                if is_wsj_actual_content(soup):
                    content = extract_wsj_article_content(soup)
                    if content and len(content) > 500:
                        logging.info(f"Successfully extracted content using {strategy_name}")
                        return content
                else:
                    logging.warning(f"{strategy_name} returned paywall/cleaning page")
                    
        except Exception as e:
            logging.warning(f"Error with {strategy_name}: {str(e)}")
            continue
    
    # Try archive services
    try:
        twelve_ft_url = f"https://12ft.io/{url}"
        twelve_ft_session = get_wsj_specific_session(url)
        twelve_ft_session.headers.update({
            'Referer': 'https://12ft.io/',
            'Origin': 'https://12ft.io'
        })
        
        response = twelve_ft_session.get(twelve_ft_url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            if is_wsj_actual_content(soup):
                content = extract_wsj_article_content(soup)
                if content:
                    logging.info("Successfully extracted content using 12ft.io")
                    return content
    except Exception as e:
        logging.warning(f"12ft.io failed for WSJ: {str(e)}")
    
    try:
        outline_url = f"https://outline.com/{url}"
        response = session.get(outline_url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            content_div = soup.find('div', {'class': 'outline-content'})
            if content_div:
                paragraphs = content_div.find_all('p')
                content = '\\n\\n'.join(p.get_text().strip() for p in paragraphs)
                if len(content) > 200:
                    logging.info("Successfully extracted content using Outline.com")
                    return content
    except Exception as e:
        logging.warning(f"Outline.com failed for WSJ: {str(e)}")
    
    logging.warning(f"All WSJ bypass strategies failed for: {url}")
    return None
'''
    
    # Insert the functions before the main fetch_article_content function
    if 'get_wsj_specific_session' not in content:
        # Find a good insertion point
        insertion_points = [
            'def get_domain_specific_bypass(',
            'def fetch_article_content(',
            'def get_archive_url('
        ]
        
        for point in insertion_points:
            index = content.find(point)
            if index != -1:
                content = content[:index] + wsj_functions + '\n\n' + content[index:]
                break
    
    return content

def modify_domain_specific_bypass(content):
    """Modify the get_domain_specific_bypass function to use WSJ session"""
    
    # Find the WSJ section and replace it
    wsj_pattern = r"(# WSJ-specific.*?return session)"
    replacement = '''# WSJ-specific
    elif 'wsj.com' in domain:
        # Use the enhanced WSJ session
        return get_wsj_specific_session(url)'''
    
    content = re.sub(wsj_pattern, replacement, content, flags=re.DOTALL)
    
    return content

def modify_fetch_article_content(content):
    """Add WSJ-specific handling to fetch_article_content"""
    
    # Find the function and add WSJ check
    pattern = r"(domain = urlparse\(original_url\)\.netloc\.lower\(\))"
    
    wsj_check = '''domain = urlparse(original_url).netloc.lower()
    
    # Enhanced WSJ handling
    if 'wsj.com' in domain:
        logging.info(f"Detected WSJ URL, using enhanced bypass: {original_url}")
        wsj_content = fetch_wsj_article_content_enhanced(original_url, session)
        if wsj_content and is_valid_content(wsj_content):
            logging.info(f"Successfully bypassed WSJ paywall: {original_url}")
            cache_content(url, wsj_content)
            return wsj_content
        else:
            logging.warning(f"Enhanced WSJ bypass failed, falling back to standard methods")'''
    
    content = re.sub(pattern, wsj_check, content)
    
    return content

def implement_wsj_enhancements():
    """Main implementation function"""
    archive_file = 'common/archive.py'
    
    if not os.path.exists(archive_file):
        print(f"❌ File not found: {archive_file}")
        print("Make sure you're running this from the project root directory")
        return False
    
    print(f"🔧 Implementing WSJ enhancements in {archive_file}")
    
    # Create backup
    backup_path = backup_file(archive_file)
    
    try:
        # Read the current content
        with open(archive_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("📝 Adding imports...")
        content = add_wsj_imports(content)
        
        print("📝 Adding constants...")
        content = add_wsj_constants(content)
        
        print("📝 Adding WSJ functions...")
        content = add_wsj_functions(content)
        
        print("📝 Modifying domain-specific bypass...")
        content = modify_domain_specific_bypass(content)
        
        print("📝 Modifying fetch_article_content...")
        content = modify_fetch_article_content(content)
        
        # Write the modified content
        with open(archive_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Successfully implemented WSJ enhancements!")
        print(f"📁 Backup saved as: {backup_path}")
        print("\n🧪 To test the implementation:")
        print("1. Run your RSS reader on a WSJ article")
        print("2. Check the logs for 'Detected WSJ URL, using enhanced bypass'")
        print("3. If it fails, you can restore from the backup file")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during implementation: {str(e)}")
        print(f"🔄 You can restore from backup: {backup_path}")
        return False

if __name__ == "__main__":
    print("🚀 WSJ Bypass Enhancement Implementation")
    print("=" * 50)
    implement_wsj_enhancements()
