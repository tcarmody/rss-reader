"""
User model for authentication and user management.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import bcrypt
import secrets

Base = declarative_base()


class User(Base):
    """
    User model for authentication.

    Attributes:
        id: Unique identifier
        username: Unique username for login
        email: User's email address
        password_hash: bcrypt hashed password
        is_admin: Admin privileges flag
        created_at: Account creation timestamp
        last_login: Last successful login timestamp
        settings: JSON blob for user preferences
    """
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    settings = Column(JSON, nullable=True)

    def set_password(self, password: str) -> None:
        """Hash and set the user's password."""
        salt = bcrypt.gensalt(rounds=12)
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def verify_password(self, password: str) -> bool:
        """Verify a password against the stored hash."""
        return bcrypt.checkpw(
            password.encode('utf-8'),
            self.password_hash.encode('utf-8')
        )

    def to_dict(self) -> dict:
        """Convert user to dictionary (excluding sensitive data)."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'is_admin': self.is_admin,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'settings': self.settings or {}
        }

    def update_last_login(self) -> None:
        """Update the last login timestamp."""
        self.last_login = datetime.utcnow()


class UserSession(Base):
    """
    Persistent session tokens for "remember me" functionality.

    Attributes:
        id: Unique identifier
        user_id: Foreign key to users table
        session_token: Unique session token
        created_at: Session creation timestamp
        expires_at: Session expiration timestamp
    """
    __tablename__ = 'user_sessions'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)

    @classmethod
    def generate_token(cls) -> str:
        """Generate a cryptographically secure session token."""
        return secrets.token_urlsafe(32)

    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return datetime.utcnow() > self.expires_at


class Feed(Base):
    """
    RSS feed subscription model (per-user).

    Attributes:
        id: Unique identifier
        url: Feed URL
        name: Display name for the feed
        category: Optional category for grouping
        added_at: When the feed was added
        last_fetched: Last successful fetch timestamp
        is_active: Whether the feed is active
    """
    __tablename__ = 'feeds'

    id = Column(Integer, primary_key=True)
    url = Column(String(1024), unique=True, nullable=False)
    name = Column(String(255), nullable=True)
    category = Column(String(100), nullable=True)
    added_at = Column(DateTime, default=datetime.utcnow)
    last_fetched = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

    def to_dict(self) -> dict:
        """Convert feed to dictionary."""
        return {
            'id': self.id,
            'url': self.url,
            'name': self.name,
            'category': self.category,
            'added_at': self.added_at.isoformat() if self.added_at else None,
            'last_fetched': self.last_fetched.isoformat() if self.last_fetched else None,
            'is_active': self.is_active
        }


class UserSettings(Base):
    """
    Key-value settings storage for user preferences.

    Attributes:
        key: Setting key/name
        value: JSON value for the setting
        updated_at: Last update timestamp
    """
    __tablename__ = 'user_settings'

    key = Column(String(100), primary_key=True)
    value = Column(JSON, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
