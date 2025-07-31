from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..dependencies import admin_required, get_auth_service, get_file_service
from ..services.auth_service import AuthService
from ..services.file_service import FileService
from ..models.user import UserInfo

router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request, 
    user_info: UserInfo = Depends(admin_required),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Admin dashboard page"""
    users = auth_service.get_all_users()
    return templates.TemplateResponse("admin.html", {
        "request": request, 
        "users": users,
        "user_info": user_info.dict()
    })

@router.post("/create_user")
async def create_user(
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    user_info: UserInfo = Depends(admin_required),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Create a new user"""
    try:
        auth_service.create_user(username, password, role)
        return {"message": "User created successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/update_user/{username}")
async def update_user(
    username: str,
    password: str = Form(...),
    role: str = Form(...),
    user_info: UserInfo = Depends(admin_required),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Update an existing user"""
    try:
        auth_service.update_user(username, password, role, user_info)
        return {"message": "User updated successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/delete_user/{username}")
async def delete_user(
    username: str,
    user_info: UserInfo = Depends(admin_required),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Delete a user"""
    try:
        auth_service.delete_user(username, user_info)
        return {"message": "User deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/system_status")
async def get_system_status(
    user_info: UserInfo = Depends(admin_required),
    file_service: FileService = Depends(get_file_service)
):
    """Get system status including file cleanup info"""
    try:
        return file_service.get_system_status()
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": "unknown"
        }