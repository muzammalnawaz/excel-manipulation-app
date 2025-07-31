from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .services.auth_service import AuthService
from .services.file_service import FileService
from .models.user import UserInfo
from .core.exceptions import AuthenticationError, InsufficientPermissionsError

# Global instances - these will be initialized in main.py
auth_service: AuthService = None
file_service: FileService = None

security = HTTPBasic()

def get_auth_service() -> AuthService:
    """Dependency to get auth service instance"""
    if auth_service is None:
        raise HTTPException(status_code=500, detail="Auth service not initialized")
    return auth_service

def get_file_service() -> FileService:
    """Dependency to get file service instance"""
    if file_service is None:
        raise HTTPException(status_code=500, detail="File service not initialized")
    return file_service

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> UserInfo:
    """Verify user credentials and return user info"""
    try:
        auth = get_auth_service()
        return auth.verify_credentials(credentials.username, credentials.password)
    except AuthenticationError:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

def admin_required(user_info: UserInfo = Depends(verify_credentials)) -> UserInfo:
    """Require admin role for access"""
    try:
        auth = get_auth_service()
        auth.require_admin(user_info)
        return user_info
    except InsufficientPermissionsError:
        raise HTTPException(status_code=403, detail="Admin access required")

def initialize_services():
    """Initialize global service instances"""
    global auth_service, file_service
    auth_service = AuthService()
    file_service = FileService()
    return auth_service, file_service