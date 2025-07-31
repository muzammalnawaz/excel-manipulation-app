from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..dependencies import verify_credentials
from ..models.user import UserInfo
from ..core.config import TOOLS_CONFIG

router = APIRouter(tags=["auth"])
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Root endpoint - shows login page"""
    return templates.TemplateResponse("login.html", {"request": request})

@router.get("/dashboard", response_class=HTMLResponse)
async def user_dashboard(request: Request, user_info: UserInfo = Depends(verify_credentials)):
    """User dashboard page"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "tools": TOOLS_CONFIG,
        "user_info": user_info.dict()
    })

@router.get("/user_role")
async def get_user_role(user_info: UserInfo = Depends(verify_credentials)):
    """Get current user role information"""
    return {"role": user_info.role, "username": user_info.username}