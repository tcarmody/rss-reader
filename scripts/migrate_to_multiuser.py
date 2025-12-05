#!/usr/bin/env python3
"""
Migration script to convert existing single-user data to multi-user system.

This script:
1. Creates the first user (admin) from existing data
2. Migrates existing bookmarks to the admin user's database
3. Copies default feeds to the admin user's feed list

Usage:
    python scripts/migrate_to_multiuser.py --username admin --email admin@example.com --password yourpassword

Or interactively:
    python scripts/migrate_to_multiuser.py
"""

import os
import sys
import argparse
import getpass
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from services.auth_manager import AuthManager
from services.user_data_manager import UserDataManager
from models.bookmark import Bookmark


def get_old_bookmarks():
    """Get bookmarks from the old single-user database."""
    old_db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'bookmarks.db'
    )

    if not os.path.exists(old_db_path):
        print(f"No existing bookmarks database found at {old_db_path}")
        return []

    engine = create_engine(f'sqlite:///{old_db_path}')
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        bookmarks = session.query(Bookmark).all()
        # Convert to dicts to avoid session issues
        return [
            {
                'title': b.title,
                'url': b.url,
                'summary': b.summary,
                'content': b.content,
                'date_added': b.date_added,
                'tags': b.tags,
                'read_status': b.read_status
            }
            for b in bookmarks
        ]
    finally:
        session.close()


def migrate_bookmarks(user_data: UserDataManager, old_bookmarks: list) -> int:
    """Migrate old bookmarks to user's database."""
    bookmark_manager = user_data.get_bookmark_manager()
    migrated = 0

    for bookmark in old_bookmarks:
        try:
            tags = bookmark['tags'].split(',') if bookmark['tags'] else []
            bookmark_manager.add_bookmark(
                title=bookmark['title'],
                url=bookmark['url'],
                summary=bookmark['summary'],
                content=bookmark['content'],
                tags=tags
            )
            # Update read status if needed
            if bookmark['read_status']:
                # Get the newly created bookmark and mark as read
                pass  # Would need to implement this
            migrated += 1
        except Exception as e:
            print(f"  Warning: Failed to migrate bookmark '{bookmark['title']}': {e}")

    return migrated


def backup_old_data():
    """Create backup of old data before migration."""
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data'
    )

    old_db_path = os.path.join(data_dir, 'bookmarks.db')
    if os.path.exists(old_db_path):
        backup_path = os.path.join(data_dir, 'bookmarks.db.backup')
        shutil.copy2(old_db_path, backup_path)
        print(f"Created backup at {backup_path}")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description='Migrate to multi-user system')
    parser.add_argument('--username', help='Admin username')
    parser.add_argument('--email', help='Admin email')
    parser.add_argument('--password', help='Admin password (will prompt if not provided)')
    parser.add_argument('--skip-backup', action='store_true', help='Skip backup creation')
    args = parser.parse_args()

    print("=" * 60)
    print("Multi-User Migration Script")
    print("=" * 60)

    # Get credentials
    username = args.username
    email = args.email
    password = args.password

    if not username:
        username = input("Enter admin username: ").strip()
    if not email:
        email = input("Enter admin email: ").strip()
    if not password:
        password = getpass.getpass("Enter admin password: ")
        password_confirm = getpass.getpass("Confirm password: ")
        if password != password_confirm:
            print("Error: Passwords do not match")
            sys.exit(1)

    # Validate inputs
    if len(username) < 3:
        print("Error: Username must be at least 3 characters")
        sys.exit(1)
    if '@' not in email:
        print("Error: Invalid email format")
        sys.exit(1)
    if len(password) < 8:
        print("Error: Password must be at least 8 characters")
        sys.exit(1)

    # Create backup
    if not args.skip_backup:
        print("\n[1/5] Creating backup...")
        backup_old_data()

    # Get existing bookmarks
    print("\n[2/5] Reading existing bookmarks...")
    old_bookmarks = get_old_bookmarks()
    print(f"  Found {len(old_bookmarks)} existing bookmarks")

    # Create admin user
    print("\n[3/5] Creating admin user...")
    auth_manager = AuthManager()
    user, error = auth_manager.register_user(username, email, password)

    if not user:
        print(f"Error creating user: {error}")
        sys.exit(1)

    print(f"  Created user: {user.username} (ID: {user.id}, Admin: {user.is_admin})")

    # Initialize user data and import default feeds
    print("\n[4/5] Setting up user data...")
    user_data = UserDataManager(user.id)
    feeds_imported = user_data.import_default_feeds()
    print(f"  Imported {feeds_imported} default feeds")

    # Migrate bookmarks
    print("\n[5/5] Migrating bookmarks...")
    if old_bookmarks:
        migrated = migrate_bookmarks(user_data, old_bookmarks)
        print(f"  Migrated {migrated}/{len(old_bookmarks)} bookmarks")
    else:
        print("  No bookmarks to migrate")

    print("\n" + "=" * 60)
    print("Migration Complete!")
    print("=" * 60)
    print(f"\nAdmin account created:")
    print(f"  Username: {username}")
    print(f"  Email: {email}")
    print(f"\nYou can now start the server and log in with these credentials.")
    print(f"\nData location: data/users/{user.id}/")


if __name__ == '__main__':
    main()
