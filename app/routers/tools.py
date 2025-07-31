from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..dependencies import verify_credentials
from ..models.user import UserInfo
from ..core.config import TOOLS_CONFIG

router = APIRouter(prefix="/tool", tags=["tools"])
templates = Jinja2Templates(directory="templates")

@router.get("/{tool_id}", response_class=HTMLResponse)
async def tool_page(request: Request, tool_id: str, user_info: UserInfo = Depends(verify_credentials)):
    """Display tool-specific page"""
    if tool_id not in TOOLS_CONFIG:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    tool_config = TOOLS_CONFIG[tool_id]
    
    # Select appropriate template based on tool
    if tool_id == "data_summarizer":
        template_name = "summarizer.html"
    elif tool_id == "stock_distribution":
        template_name = "stock_distribution.html"
    else:
        template_name = "tool.html"
    
    return templates.TemplateResponse(template_name, {
        "request": request,
        "tool_id": tool_id,
        "tool_config": tool_config,
        "user_info": user_info.dict()
    })