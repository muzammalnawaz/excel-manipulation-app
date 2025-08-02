from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel

class FileSession(BaseModel):
    file_id: str
    tool_id: str
    user: str
    uploaded_at: datetime
    expires_at: datetime
    processed: bool = False
    
    # File paths
    original_path: Optional[str] = None
    processed_path: Optional[str] = None
    warehouse_path: Optional[str] = None  # For stock distribution
    sales_path: Optional[str] = None      # For stock distribution
    
    # File metadata
    original_filename: Optional[str] = None
    warehouse_filename: Optional[str] = None
    sales_filename: Optional[str] = None
    columns: Optional[List[str]] = None
    column_info: Optional[Dict[str, str]] = None
    
    # Enhanced: Summary data for dynamic display
    summary_data: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "file_id": "abc123",
                "tool_id": "excel_splitter",
                "user": "user1",
                "uploaded_at": "2024-01-01T12:00:00",
                "expires_at": "2024-01-01T12:05:00",
                "processed": False
            }
        }

class FileStatus(BaseModel):
    file_id: str
    processed: bool
    time_left_seconds: int
    expires_at: str
    summary_data: Optional[Dict[str, Any]] = None  # Enhanced: Include summary if available