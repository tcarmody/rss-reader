"""
Authentication middleware and decorators.
"""

import logging
from functools import wraps
from typing import Optional, Callable

from starlette.requests import Request
from starlette.responses import RedirectResponse
from fastapi import HTTPException

from services.auth_manager import get_auth_manager
from services.user_data_manager import get_user_data_manager

logger = logging.getLogger(__name__)

# Session keys
SESSION_USER_ID = 'auth_user_id'
SESSION_USERNAME = 'auth_username'
SESSION_IS_ADMIN = 'auth_is_admin'
COOKIE_REMEMBER_TOKEN = 'remember_token'


def get_current_user(request: Request) -> Optional[dict]:
    """
    Get the current authenticated user from the request.

    Checks:
    1. Session for user_id
    2. "Remember me" cookie token

    Args:
        request: The incoming request

    Returns:
        User dict with id, username, is_admin or None if not authenticated
    """
    # Check session first
    user_id = request.session.get(SESSION_USER_ID)
    if user_id:
        return {
            'id': user_id,
            'username': request.session.get(SESSION_USERNAME, ''),
            'is_admin': request.session.get(SESSION_IS_ADMIN, False)
        }

    # Check for remember token cookie
    remember_token = request.cookies.get(COOKIE_REMEMBER_TOKEN)
    if remember_token:
        auth_manager = get_auth_manager()
        user = auth_manager.validate_session_token(remember_token)
        if user:
            # Restore session from token
            request.session[SESSION_USER_ID] = user.id
            request.session[SESSION_USERNAME] = user.username
            request.session[SESSION_IS_ADMIN] = user.is_admin
            return {
                'id': user.id,
                'username': user.username,
                'is_admin': user.is_admin
            }

    return None


def set_user_session(request: Request, user) -> None:
    """
    Set the user session after successful authentication.

    Args:
        request: The incoming request
        user: User object from authentication
    """
    request.session[SESSION_USER_ID] = user.id
    request.session[SESSION_USERNAME] = user.username
    request.session[SESSION_IS_ADMIN] = user.is_admin
    logger.info(f"Session set for user: {user.username}")


def clear_user_session(request: Request) -> None:
    """
    Clear the user session on logout.

    Args:
        request: The incoming request
    """
    request.session.pop(SESSION_USER_ID, None)
    request.session.pop(SESSION_USERNAME, None)
    request.session.pop(SESSION_IS_ADMIN, None)


def require_login(redirect_to_login: bool = True):
    """
    Decorator to require authentication for a route.

    For HTML routes, redirects to login page.
    For API routes, returns 401 Unauthorized.

    Args:
        redirect_to_login: If True, redirect to login. If False, return 401.

    Usage:
        @app.get("/protected")
        @require_login()
        async def protected_route(request: Request):
            user = get_current_user(request)
            ...

        @app.get("/api/protected")
        @require_login(redirect_to_login=False)
        async def api_route(request: Request):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            user = get_current_user(request)

            if not user:
                if redirect_to_login:
                    # Store the original URL for redirect after login
                    request.session['next_url'] = str(request.url)
                    return RedirectResponse(url='/login', status_code=303)
                else:
                    raise HTTPException(
                        status_code=401,
                        detail="Authentication required"
                    )

            # Add user to request state for easy access
            request.state.user = user

            return await func(request, *args, **kwargs)

        return wrapper
    return decorator


def require_admin(redirect_to_login: bool = True):
    """
    Decorator to require admin privileges for a route.

    Args:
        redirect_to_login: If True, redirect to login. If False, return 403.

    Usage:
        @app.get("/admin")
        @require_admin()
        async def admin_route(request: Request):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            user = get_current_user(request)

            if not user:
                if redirect_to_login:
                    request.session['next_url'] = str(request.url)
                    return RedirectResponse(url='/login', status_code=303)
                else:
                    raise HTTPException(
                        status_code=401,
                        detail="Authentication required"
                    )

            if not user.get('is_admin'):
                raise HTTPException(
                    status_code=403,
                    detail="Admin privileges required"
                )

            request.state.user = user
            return await func(request, *args, **kwargs)

        return wrapper
    return decorator


def get_user_data(request: Request):
    """
    Get the UserDataManager for the current authenticated user.

    Args:
        request: The incoming request

    Returns:
        UserDataManager instance or None if not authenticated
    """
    user = get_current_user(request)
    if not user:
        return None
    return get_user_data_manager(user['id'])


class AuthMiddleware:
    """
    ASGI middleware to attach user info to all requests.

    This makes user info available via request.state.user
    without requiring the @require_login decorator.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # User info will be populated by get_current_user when needed
            pass
        await self.app(scope, receive, send)
