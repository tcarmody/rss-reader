"""
User data manager for per-user database isolation.

Each user gets their own SQLite database containing:
- Bookmarks
- RSS feed subscriptions
- User settings
"""

import os
import shutil
import logging
from typing import Optional, List

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from models.user import Feed, UserSettings
from models.bookmark import Base as BookmarkBase, Bookmark

logger = logging.getLogger(__name__)


class UserDataManager:
    """
    Manages per-user data storage with complete database isolation.

    Each user's data is stored in: data/users/{user_id}/user_data.db
    """

    def __init__(self, user_id: int):
        """
        Initialize the UserDataManager for a specific user.

        Args:
            user_id: The user's ID
        """
        self.user_id = user_id
        self.user_dir = self._get_user_directory()
        self.db_path = os.path.join(self.user_dir, "user_data.db")
        self.engine = None
        self.SessionLocal = None

        self._initialize_database()

    def _get_user_directory(self) -> str:
        """Get the user's data directory, creating if needed."""
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'users', str(self.user_id)
        )
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def _initialize_database(self) -> None:
        """Initialize the user's database with all required tables."""
        db_url = f'sqlite:///{self.db_path}'
        self.engine = create_engine(db_url)

        # Create all tables from both base classes
        BookmarkBase.metadata.create_all(self.engine)

        # Create Feed and UserSettings tables
        from models.user import Base as UserBase
        UserBase.metadata.create_all(self.engine)

        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.debug(f"Initialized database for user {self.user_id}")

    def _get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    # -------------------------------------------------------------------------
    # Feed Management
    # -------------------------------------------------------------------------

    def get_feeds(self, active_only: bool = True) -> List[Feed]:
        """
        Get all feeds for this user.

        Args:
            active_only: If True, only return active feeds

        Returns:
            List of Feed objects
        """
        session = self._get_session()
        try:
            query = session.query(Feed)
            if active_only:
                query = query.filter(Feed.is_active == True)
            return query.order_by(Feed.added_at.desc()).all()
        finally:
            session.close()

    def get_feed_urls(self, active_only: bool = True) -> List[str]:
        """
        Get all feed URLs for this user.

        Args:
            active_only: If True, only return active feeds

        Returns:
            List of feed URL strings
        """
        feeds = self.get_feeds(active_only=active_only)
        return [f.url for f in feeds]

    def add_feed(self, url: str, name: Optional[str] = None,
                 category: Optional[str] = None) -> Optional[Feed]:
        """
        Add a new feed subscription.

        Args:
            url: Feed URL
            name: Optional display name
            category: Optional category

        Returns:
            Created Feed object or None if already exists
        """
        session = self._get_session()
        try:
            # Check if already exists
            existing = session.query(Feed).filter(Feed.url == url).first()
            if existing:
                # Reactivate if inactive
                if not existing.is_active:
                    existing.is_active = True
                    session.commit()
                    return existing
                return None

            feed = Feed(url=url, name=name, category=category)
            session.add(feed)
            session.commit()
            session.refresh(feed)
            logger.info(f"Added feed for user {self.user_id}: {url}")
            return feed
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding feed: {e}")
            return None
        finally:
            session.close()

    def remove_feed(self, url: str, hard_delete: bool = False) -> bool:
        """
        Remove a feed subscription.

        Args:
            url: Feed URL to remove
            hard_delete: If True, delete from DB. If False, just deactivate.

        Returns:
            True if successful
        """
        session = self._get_session()
        try:
            feed = session.query(Feed).filter(Feed.url == url).first()
            if not feed:
                return False

            if hard_delete:
                session.delete(feed)
            else:
                feed.is_active = False

            session.commit()
            logger.info(f"Removed feed for user {self.user_id}: {url}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error removing feed: {e}")
            return False
        finally:
            session.close()

    def import_feeds_from_list(self, urls: List[str]) -> int:
        """
        Import multiple feeds from a list of URLs.

        Args:
            urls: List of feed URLs

        Returns:
            Number of feeds successfully imported
        """
        count = 0
        for url in urls:
            url = url.strip()
            if url and not url.startswith('#'):
                # Remove inline comments
                url = url.split('#')[0].strip()
                if url and self.add_feed(url):
                    count += 1
        return count

    def import_default_feeds(self) -> int:
        """
        Import the default feeds from rss_feeds.txt.

        Returns:
            Number of feeds imported
        """
        feeds_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'rss_feeds.txt'
        )

        if not os.path.exists(feeds_file):
            logger.warning(f"Default feeds file not found: {feeds_file}")
            return 0

        with open(feeds_file, 'r') as f:
            urls = f.readlines()

        return self.import_feeds_from_list(urls)

    # -------------------------------------------------------------------------
    # Bookmark Management (delegated)
    # -------------------------------------------------------------------------

    def get_bookmark_manager(self):
        """
        Get a BookmarkManager instance for this user's database.

        Returns:
            BookmarkManager instance
        """
        from services.bookmark_manager import BookmarkManager
        return BookmarkManager(db_path=f'sqlite:///{self.db_path}')

    # -------------------------------------------------------------------------
    # Settings Management
    # -------------------------------------------------------------------------

    def get_setting(self, key: str, default=None):
        """
        Get a user setting by key.

        Args:
            key: Setting key
            default: Default value if not found

        Returns:
            Setting value or default
        """
        session = self._get_session()
        try:
            setting = session.query(UserSettings).filter(
                UserSettings.key == key
            ).first()
            return setting.value if setting else default
        finally:
            session.close()

    def set_setting(self, key: str, value) -> bool:
        """
        Set a user setting.

        Args:
            key: Setting key
            value: Setting value (must be JSON-serializable)

        Returns:
            True if successful
        """
        session = self._get_session()
        try:
            setting = session.query(UserSettings).filter(
                UserSettings.key == key
            ).first()

            if setting:
                setting.value = value
            else:
                setting = UserSettings(key=key, value=value)
                session.add(setting)

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error setting {key}: {e}")
            return False
        finally:
            session.close()

    def get_all_settings(self) -> dict:
        """
        Get all user settings.

        Returns:
            Dictionary of all settings
        """
        session = self._get_session()
        try:
            settings = session.query(UserSettings).all()
            return {s.key: s.value for s in settings}
        finally:
            session.close()

    def set_all_settings(self, settings: dict) -> bool:
        """
        Set multiple settings at once.

        Args:
            settings: Dictionary of settings

        Returns:
            True if all successful
        """
        for key, value in settings.items():
            if not self.set_setting(key, value):
                return False
        return True

    # -------------------------------------------------------------------------
    # Data Export/Import
    # -------------------------------------------------------------------------

    def export_data(self) -> dict:
        """
        Export all user data for backup.

        Returns:
            Dictionary containing all user data
        """
        bookmark_manager = self.get_bookmark_manager()
        bookmarks = bookmark_manager.get_all_bookmarks()
        feeds = self.get_feeds(active_only=False)
        settings = self.get_all_settings()

        return {
            'bookmarks': [b.to_dict() for b in bookmarks],
            'feeds': [f.to_dict() for f in feeds],
            'settings': settings
        }

    def delete_all_data(self) -> bool:
        """
        Delete all user data (dangerous!).

        Returns:
            True if successful
        """
        try:
            if os.path.exists(self.user_dir):
                shutil.rmtree(self.user_dir)
                logger.info(f"Deleted all data for user {self.user_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting user data: {e}")
            return False


# Cache of UserDataManager instances
_user_data_managers: dict = {}


def get_user_data_manager(user_id: int) -> UserDataManager:
    """
    Get or create a UserDataManager for a specific user.

    Args:
        user_id: User's ID

    Returns:
        UserDataManager instance
    """
    if user_id not in _user_data_managers:
        _user_data_managers[user_id] = UserDataManager(user_id)
    return _user_data_managers[user_id]


def clear_user_data_manager_cache():
    """Clear the UserDataManager cache (useful for testing)."""
    global _user_data_managers
    _user_data_managers = {}
