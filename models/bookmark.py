"""
Bookmark model for storing saved articles and summaries.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import json
import csv
import io

Base = declarative_base()

class Bookmark(Base):
    """
    Bookmark model for storing saved articles and summaries.
    
    Attributes:
        id: Unique identifier
        title: Article title
        url: Article URL
        summary: Article summary text
        content: Full article content (optional)
        date_added: When the bookmark was created
        tags: Comma-separated tags for categorization
        read_status: Whether the article has been read
    """
    __tablename__ = 'bookmarks'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    url = Column(String(1024), nullable=False)
    summary = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    date_added = Column(DateTime, default=datetime.now)
    tags = Column(String(255), nullable=True)  # Comma-separated tags
    read_status = Column(Boolean, default=False)
    
    def to_dict(self):
        """Convert bookmark to dictionary format."""
        return {
            'id': self.id,
            'title': self.title,
            'url': self.url,
            'summary': self.summary,
            'content': self.content,
            'date_added': self.date_added.isoformat(),
            'tags': self.tags.split(',') if self.tags else [],
            'read_status': self.read_status
        }
    
    @classmethod
    def export_to_json(cls, bookmarks):
        """
        Export a list of bookmarks to JSON format.
        
        Args:
            bookmarks: List of Bookmark objects
            
        Returns:
            str: JSON string representation of bookmarks
        """
        bookmark_dicts = [b.to_dict() for b in bookmarks]
        return json.dumps(bookmark_dicts, indent=2)
    
    @classmethod
    def export_to_csv(cls, bookmarks):
        """
        Export a list of bookmarks to CSV format.
        
        Args:
            bookmarks: List of Bookmark objects
            
        Returns:
            str: CSV string representation of bookmarks
        """
        output = io.StringIO()
        fieldnames = ['id', 'title', 'url', 'summary', 'tags', 'date_added', 'read_status']
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for bookmark in bookmarks:
            b_dict = bookmark.to_dict()
            # Convert tags list to comma-separated string for CSV
            b_dict['tags'] = ','.join(b_dict['tags'])
            # Remove content field as it can be very large
            if 'content' in b_dict:
                del b_dict['content']
            writer.writerow(b_dict)
        
        return output.getvalue()
