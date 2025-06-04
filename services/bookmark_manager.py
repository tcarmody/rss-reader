"""
Bookmark manager service for handling bookmark CRUD operations.
"""

import os
import logging
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
from models.bookmark import Base, Bookmark
from datetime import datetime
import json
import csv
from typing import List, Optional, Dict, Any, Union
import re

logger = logging.getLogger(__name__)

def escape_sql_like(value: str) -> str:
    """
    Escape special characters in SQL LIKE patterns to prevent injection.
    
    Args:
        value: The value to escape
        
    Returns:
        Escaped value safe for SQL LIKE operations
    """
    if not value:
        return value
    
    # Escape SQL LIKE special characters: % _ \ 
    escaped = value.replace('\\', '\\\\')  # Must be first
    escaped = escaped.replace('%', '\\%')
    escaped = escaped.replace('_', '\\_')
    
    # Remove any potential SQL injection patterns
    escaped = re.sub(r'[^\w\s\-.]', '', escaped)
    
    return escaped

class BookmarkManager:
    """
    Service for managing bookmarks in the database.
    
    Provides methods for creating, reading, updating, and deleting bookmarks,
    as well as exporting bookmarks to different formats.
    """
    
    def __init__(self, db_path=None):
        """
        Initialize the bookmark manager with a database connection.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default path.
        """
        if db_path is None:
            # Create data directory if it doesn't exist
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = f'sqlite:///{os.path.join(data_dir, "bookmarks.db")}'
        
        self.engine = create_engine(db_path)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"Initialized BookmarkManager with database at {db_path}")
    
    def add_bookmark(self, title: str, url: str, summary: Optional[str] = None, 
                    content: Optional[str] = None, tags: Optional[List[str]] = None) -> int:
        """
        Add a new bookmark to the database.
        
        Args:
            title: Article title
            url: Article URL
            summary: Article summary text
            content: Full article content (optional)
            tags: List of tags for categorization
            
        Returns:
            int: ID of the newly created bookmark
        """
        session = self.Session()
        try:
            # Check if bookmark with same URL already exists
            existing = session.query(Bookmark).filter(Bookmark.url == url).first()
            if existing:
                logger.info(f"Bookmark already exists for URL: {url}")
                return existing.id
            
            bookmark = Bookmark(
                title=title,
                url=url,
                summary=summary,
                content=content,
                tags=','.join(tags) if tags else None
            )
            session.add(bookmark)
            session.commit()
            bookmark_id = bookmark.id
            logger.info(f"Added bookmark: {title} (ID: {bookmark_id})")
            return bookmark_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding bookmark: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_bookmark(self, bookmark_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a single bookmark by ID.
        
        Args:
            bookmark_id: ID of the bookmark to retrieve
            
        Returns:
            dict: Bookmark data or None if not found
        """
        session = self.Session()
        try:
            bookmark = session.query(Bookmark).get(bookmark_id)
            if bookmark:
                return bookmark.to_dict()
            return None
        finally:
            session.close()
    
    def get_bookmarks(self, filter_read: Optional[bool] = None, 
                     tags: Optional[List[str]] = None, 
                     limit: int = 100, 
                     offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get bookmarks with optional filtering.
        
        Args:
            filter_read: Filter by read status (True/False) or None for all
            tags: Filter by tags (bookmark must have at least one of these tags)
            limit: Maximum number of bookmarks to return
            offset: Number of bookmarks to skip (for pagination)
            
        Returns:
            list: List of bookmark dictionaries
        """
        session = self.Session()
        try:
            query = session.query(Bookmark).order_by(desc(Bookmark.date_added))
            
            if filter_read is not None:
                query = query.filter(Bookmark.read_status == filter_read)
                
            if tags and len(tags) > 0:
                # Filter bookmarks that contain any of the specified tags
                tag_filters = []
                for tag in tags:
                    # Safely escape the tag to prevent SQL injection
                    escaped_tag = escape_sql_like(str(tag))
                    tag_filters.append(Bookmark.tags.like(f'%{escaped_tag}%'))
                
                from sqlalchemy import or_
                query = query.filter(or_(*tag_filters))
            
            query = query.limit(limit).offset(offset)
            bookmarks = [b.to_dict() for b in query.all()]
            return bookmarks
        finally:
            session.close()
    
    def update_bookmark(self, bookmark_id: int, **kwargs) -> bool:
        """
        Update bookmark properties.
        
        Args:
            bookmark_id: ID of the bookmark to update
            **kwargs: Properties to update (title, url, summary, content, tags, read_status)
            
        Returns:
            bool: True if successful, False if bookmark not found
        """
        session = self.Session()
        try:
            bookmark = session.query(Bookmark).get(bookmark_id)
            if not bookmark:
                return False
            
            # Update provided fields
            for key, value in kwargs.items():
                if key == 'tags' and isinstance(value, list):
                    value = ','.join(value)
                if hasattr(bookmark, key):
                    setattr(bookmark, key, value)
            
            session.commit()
            logger.info(f"Updated bookmark ID {bookmark_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating bookmark {bookmark_id}: {str(e)}")
            return False
        finally:
            session.close()
    
    def mark_as_read(self, bookmark_id: int, read_status: bool = True) -> bool:
        """
        Mark a bookmark as read or unread.
        
        Args:
            bookmark_id: ID of the bookmark to update
            read_status: True for read, False for unread
            
        Returns:
            bool: True if successful, False if bookmark not found
        """
        return self.update_bookmark(bookmark_id, read_status=read_status)
    
    def delete_bookmark(self, bookmark_id: int) -> bool:
        """
        Delete a bookmark.
        
        Args:
            bookmark_id: ID of the bookmark to delete
            
        Returns:
            bool: True if successful, False if bookmark not found
        """
        session = self.Session()
        try:
            bookmark = session.query(Bookmark).get(bookmark_id)
            if not bookmark:
                return False
            
            session.delete(bookmark)
            session.commit()
            logger.info(f"Deleted bookmark ID {bookmark_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting bookmark {bookmark_id}: {str(e)}")
            return False
        finally:
            session.close()
    
    def export_bookmarks(self, format_type: str = 'json', 
                        filter_read: Optional[bool] = None,
                        tags: Optional[List[str]] = None) -> str:
        """
        Export bookmarks to JSON or CSV format.
        
        Args:
            format_type: 'json' or 'csv'
            filter_read: Filter by read status
            tags: Filter by tags
            
        Returns:
            str: Exported data as string
        """
        session = self.Session()
        try:
            query = session.query(Bookmark).order_by(desc(Bookmark.date_added))
            
            if filter_read is not None:
                query = query.filter(Bookmark.read_status == filter_read)
                
            if tags and len(tags) > 0:
                # Filter bookmarks that contain any of the specified tags
                tag_filters = []
                for tag in tags:
                    # Safely escape the tag to prevent SQL injection
                    escaped_tag = escape_sql_like(str(tag))
                    tag_filters.append(Bookmark.tags.like(f'%{escaped_tag}%'))
                
                from sqlalchemy import or_
                query = query.filter(or_(*tag_filters))
            
            bookmarks = query.all()
            
            if format_type.lower() == 'json':
                return Bookmark.export_to_json(bookmarks)
            elif format_type.lower() == 'csv':
                return Bookmark.export_to_csv(bookmarks)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
        finally:
            session.close()
    
    def import_from_json(self, json_data: str) -> int:
        """
        Import bookmarks from JSON data.
        
        Args:
            json_data: JSON string with bookmark data
            
        Returns:
            int: Number of bookmarks imported
        """
        try:
            bookmarks_data = json.loads(json_data)
            if not isinstance(bookmarks_data, list):
                raise ValueError("JSON data must be a list of bookmarks")
            
            count = 0
            for bookmark_data in bookmarks_data:
                # Extract required fields
                title = bookmark_data.get('title')
                url = bookmark_data.get('url')
                
                if not title or not url:
                    logger.warning(f"Skipping import of bookmark missing title or URL: {bookmark_data}")
                    continue
                
                # Extract optional fields
                summary = bookmark_data.get('summary')
                content = bookmark_data.get('content')
                tags = bookmark_data.get('tags', [])
                read_status = bookmark_data.get('read_status', False)
                
                # Add bookmark
                bookmark_id = self.add_bookmark(title, url, summary, content, tags)
                
                # Update read status if needed
                if read_status:
                    self.mark_as_read(bookmark_id, read_status)
                
                count += 1
            
            logger.info(f"Imported {count} bookmarks from JSON")
            return count
        except Exception as e:
            logger.error(f"Error importing bookmarks from JSON: {str(e)}")
            raise
