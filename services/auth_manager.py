"""
Authentication manager for user registration, login, and session management.
"""

import os
import re
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from models.user import Base, User, UserSession

logger = logging.getLogger(__name__)


class AuthManager:
    """
    Manages user authentication, registration, and session handling.

    Uses a shared auth.db for all user accounts, while each user's
    data (bookmarks, feeds) is stored in their own database.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the AuthManager.

        Args:
            db_path: Path to the auth database. Defaults to data/auth.db
        """
        if db_path is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data'
            )
            os.makedirs(data_dir, exist_ok=True)
            db_path = f'sqlite:///{os.path.join(data_dir, "auth.db")}'

        self.engine = create_engine(db_path)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info("AuthManager initialized with database")

    def _get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def _validate_username(self, username: str) -> Tuple[bool, str]:
        """
        Validate username format.

        Args:
            username: Username to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not username:
            return False, "Username is required"
        if len(username) < 3:
            return False, "Username must be at least 3 characters"
        if len(username) > 50:
            return False, "Username must be less than 50 characters"
        if not re.match(r'^[a-zA-Z0-9_-]+$', username):
            return False, "Username can only contain letters, numbers, underscores, and hyphens"
        return True, ""

    def _validate_email(self, email: str) -> Tuple[bool, str]:
        """
        Validate email format.

        Args:
            email: Email to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not email:
            return False, "Email is required"
        if len(email) > 255:
            return False, "Email must be less than 255 characters"
        # Basic email regex
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            return False, "Invalid email format"
        return True, ""

    def _validate_password(self, password: str) -> Tuple[bool, str]:
        """
        Validate password strength.

        Args:
            password: Password to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not password:
            return False, "Password is required"
        if len(password) < 8:
            return False, "Password must be at least 8 characters"
        if len(password) > 128:
            return False, "Password must be less than 128 characters"
        return True, ""

    def register_user(
        self,
        username: str,
        email: str,
        password: str
    ) -> Tuple[Optional[User], str]:
        """
        Register a new user.

        Args:
            username: Desired username
            email: User's email address
            password: Plain text password (will be hashed)

        Returns:
            Tuple of (User object or None, error message or empty string)
        """
        # Validate inputs
        valid, error = self._validate_username(username)
        if not valid:
            return None, error

        valid, error = self._validate_email(email)
        if not valid:
            return None, error

        valid, error = self._validate_password(password)
        if not valid:
            return None, error

        session = self._get_session()
        try:
            # Check for existing username
            existing = session.query(User).filter(
                User.username == username.lower()
            ).first()
            if existing:
                return None, "Username already taken"

            # Check for existing email
            existing = session.query(User).filter(
                User.email == email.lower()
            ).first()
            if existing:
                return None, "Email already registered"

            # Check if this is the first user (make them admin)
            user_count = session.query(User).count()
            is_first_user = user_count == 0

            # Create new user
            user = User(
                username=username.lower(),
                email=email.lower(),
                is_admin=is_first_user
            )
            user.set_password(password)

            session.add(user)
            session.commit()
            session.refresh(user)

            logger.info(f"User registered: {username} (admin={is_first_user})")
            return user, ""

        except Exception as e:
            session.rollback()
            logger.error(f"Error registering user: {e}")
            return None, "Registration failed. Please try again."
        finally:
            session.close()

    def authenticate_user(
        self,
        username_or_email: str,
        password: str
    ) -> Tuple[Optional[User], str]:
        """
        Authenticate a user by username/email and password.

        Args:
            username_or_email: Username or email address
            password: Plain text password

        Returns:
            Tuple of (User object or None, error message or empty string)
        """
        if not username_or_email or not password:
            return None, "Username and password are required"

        session = self._get_session()
        try:
            # Find user by username or email
            identifier = username_or_email.lower()
            user = session.query(User).filter(
                (User.username == identifier) | (User.email == identifier)
            ).first()

            if not user:
                return None, "Invalid username or password"

            if not user.verify_password(password):
                return None, "Invalid username or password"

            # Update last login
            user.update_last_login()
            session.commit()

            logger.info(f"User authenticated: {user.username}")
            return user, ""

        except Exception as e:
            session.rollback()
            logger.error(f"Error authenticating user: {e}")
            return None, "Authentication failed. Please try again."
        finally:
            session.close()

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """
        Get a user by their ID.

        Args:
            user_id: User's ID

        Returns:
            User object or None
        """
        session = self._get_session()
        try:
            return session.query(User).filter(User.id == user_id).first()
        finally:
            session.close()

    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get a user by their username.

        Args:
            username: Username to look up

        Returns:
            User object or None
        """
        session = self._get_session()
        try:
            return session.query(User).filter(
                User.username == username.lower()
            ).first()
        finally:
            session.close()

    def create_session_token(
        self,
        user_id: int,
        expires_days: int = 30
    ) -> Optional[str]:
        """
        Create a persistent session token for "remember me" functionality.

        Args:
            user_id: User's ID
            expires_days: Number of days until token expires

        Returns:
            Session token string or None
        """
        session = self._get_session()
        try:
            token = UserSession.generate_token()
            expires_at = datetime.utcnow() + timedelta(days=expires_days)

            user_session = UserSession(
                user_id=user_id,
                session_token=token,
                expires_at=expires_at
            )

            session.add(user_session)
            session.commit()

            logger.info(f"Session token created for user {user_id}")
            return token

        except Exception as e:
            session.rollback()
            logger.error(f"Error creating session token: {e}")
            return None
        finally:
            session.close()

    def validate_session_token(self, token: str) -> Optional[User]:
        """
        Validate a session token and return the associated user.

        Args:
            token: Session token to validate

        Returns:
            User object or None if invalid/expired
        """
        if not token:
            return None

        session = self._get_session()
        try:
            user_session = session.query(UserSession).filter(
                UserSession.session_token == token
            ).first()

            if not user_session:
                return None

            if user_session.is_expired():
                # Clean up expired session
                session.delete(user_session)
                session.commit()
                return None

            # Get the user
            user = session.query(User).filter(
                User.id == user_session.user_id
            ).first()

            return user

        except Exception as e:
            logger.error(f"Error validating session token: {e}")
            return None
        finally:
            session.close()

    def invalidate_session_token(self, token: str) -> bool:
        """
        Invalidate (delete) a session token.

        Args:
            token: Session token to invalidate

        Returns:
            True if successful, False otherwise
        """
        session = self._get_session()
        try:
            result = session.query(UserSession).filter(
                UserSession.session_token == token
            ).delete()
            session.commit()
            return result > 0
        except Exception as e:
            session.rollback()
            logger.error(f"Error invalidating session token: {e}")
            return False
        finally:
            session.close()

    def invalidate_all_user_sessions(self, user_id: int) -> int:
        """
        Invalidate all session tokens for a user.

        Args:
            user_id: User's ID

        Returns:
            Number of sessions invalidated
        """
        session = self._get_session()
        try:
            result = session.query(UserSession).filter(
                UserSession.user_id == user_id
            ).delete()
            session.commit()
            logger.info(f"Invalidated {result} sessions for user {user_id}")
            return result
        except Exception as e:
            session.rollback()
            logger.error(f"Error invalidating user sessions: {e}")
            return 0
        finally:
            session.close()

    def cleanup_expired_sessions(self) -> int:
        """
        Remove all expired session tokens from the database.

        Returns:
            Number of sessions cleaned up
        """
        session = self._get_session()
        try:
            result = session.query(UserSession).filter(
                UserSession.expires_at < datetime.utcnow()
            ).delete()
            session.commit()
            if result > 0:
                logger.info(f"Cleaned up {result} expired sessions")
            return result
        except Exception as e:
            session.rollback()
            logger.error(f"Error cleaning up sessions: {e}")
            return 0
        finally:
            session.close()

    def update_user_settings(self, user_id: int, settings: dict) -> bool:
        """
        Update a user's settings.

        Args:
            user_id: User's ID
            settings: Settings dictionary to merge

        Returns:
            True if successful, False otherwise
        """
        session = self._get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return False

            # Merge with existing settings
            current = user.settings or {}
            current.update(settings)
            user.settings = current

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating user settings: {e}")
            return False
        finally:
            session.close()

    def get_all_users(self) -> list:
        """
        Get all users (admin function).

        Returns:
            List of User dictionaries
        """
        session = self._get_session()
        try:
            users = session.query(User).all()
            return [u.to_dict() for u in users]
        finally:
            session.close()

    def delete_user(self, user_id: int) -> bool:
        """
        Delete a user account.

        Args:
            user_id: User's ID

        Returns:
            True if successful, False otherwise
        """
        session = self._get_session()
        try:
            # First invalidate all sessions
            self.invalidate_all_user_sessions(user_id)

            # Delete the user
            result = session.query(User).filter(User.id == user_id).delete()
            session.commit()
            return result > 0
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting user: {e}")
            return False
        finally:
            session.close()


# Global instance (lazy initialization)
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get or create the global AuthManager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager
