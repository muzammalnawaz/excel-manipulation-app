# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
import hashlib
from pathlib import Path
import secrets
# Get port from environment variable for deployment
PORT = int(os.environ.get("PORT", 8000))

app = FastAPI(title="Excel Manipulation Tool")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBasic()

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("downloads", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# File to store users persistently
USERS_FILE = "users_data.json"

def load_users():
    """Load users from file, create default if doesn't exist"""
    if Path(USERS_FILE).exists():
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If file is corrupted, recreate with defaults
            pass
    
    # Default users if file doesn't exist
    default_users = {
        "admin": {"password": "admin123", "role": "admin"},
        "user1": {"password": "user123", "role": "user"}
    }
    save_users(default_users)
    return default_users

def save_users(users_data):
    """Save users to file"""
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users_data, f, indent=2)
    except IOError as e:
        print(f"Error saving users: {e}")


users_db = load_users()

# File storage with expiration
file_storage: Dict[str, Dict[str, Any]] = {}

# Tools configuration
TOOLS_CONFIG = {
    "excel_splitter": {
        "name": "Excel Sheet Splitter",
        "description": "Split Excel data into multiple sheets based on column values",
        "processor": "split_excel_by_column"
    }
    # Future tools can be added here
}

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    
    if username not in users_db:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    stored_password = users_db[username]["password"]
    if password != stored_password:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return {"username": username, "role": users_db[username]["role"]}

def admin_required(user_info: dict = Depends(verify_credentials)):
    if user_info["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user_info

async def cleanup_expired_files():
    """Background task to cleanup expired files"""
    while True:
        current_time = datetime.now()
        expired_files = []
        
        for file_id, file_info in file_storage.items():
            if current_time > file_info["expires_at"]:
                expired_files.append(file_id)
        
        for file_id in expired_files:
            file_info = file_storage[file_id]
            # Remove physical files
            for file_path in [file_info.get("original_path"), file_info.get("processed_path")]:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error removing file {file_path}: {e}")
            # Remove from storage
            del file_storage[file_id]
        
        await asyncio.sleep(60)  # Check every minute

@app.on_event("startup")
async def startup_event():
    # Start cleanup task
    asyncio.create_task(cleanup_expired_files())

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, user_info: dict = Depends(admin_required)):
    return templates.TemplateResponse("admin.html", {
        "request": request, 
        "users": users_db,
        "user_info": user_info
    })

@app.post("/admin/create_user")
async def create_user(
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    user_info: dict = Depends(admin_required)
):
    if username in users_db:
        raise HTTPException(status_code=400, detail="User already exists")
    
    users_db[username] = {"password": password, "role": role}
    save_users(users_db)  # ← Save to file
    return {"message": "User created successfully"}

@app.put("/admin/update_user/{username}")
async def update_user(
    username: str,
    password: str = Form(...),
    role: str = Form(...),
    user_info: dict = Depends(admin_required)
):
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Prevent admin from changing their own role or deleting themselves
    if username == "admin" and user_info["username"] == "admin":
        if role != "admin":
            raise HTTPException(status_code=400, detail="Cannot change admin role")
    
    users_db[username]["password"] = password
    users_db[username]["role"] = role
    save_users(users_db)  # ← Save to file
    return {"message": "User updated successfully"}

@app.delete("/admin/delete_user/{username}")
async def delete_user(
    username: str,
    user_info: dict = Depends(admin_required)
):
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Prevent deletion of admin user
    if username == "admin":
        raise HTTPException(status_code=400, detail="Cannot delete admin user")
    
    # Prevent admin from deleting themselves if they're the only admin
    if username == user_info["username"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    
    del users_db[username]
    save_users(users_db)  # ← Save to file
    return {"message": "User deleted successfully"}

@app.get("/dashboard", response_class=HTMLResponse)
async def user_dashboard(request: Request, user_info: dict = Depends(verify_credentials)):
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "tools": TOOLS_CONFIG,
        "user_info": user_info
    })

@app.get("/user_role")
async def get_user_role(user_info: dict = Depends(verify_credentials)):
    return {"role": user_info["role"], "username": user_info["username"]}

@app.get("/tool/{tool_id}", response_class=HTMLResponse)
async def tool_page(request: Request, tool_id: str, user_info: dict = Depends(verify_credentials)):
    if tool_id not in TOOLS_CONFIG:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    tool_config = TOOLS_CONFIG[tool_id]
    return templates.TemplateResponse("tool.html", {
        "request": request,
        "tool_id": tool_id,
        "tool_config": tool_config,
        "user_info": user_info
    })

@app.post("/upload/{tool_id}")
async def upload_file(
    tool_id: str,
    file: UploadFile = File(...),
    user_info: dict = Depends(verify_credentials)
):
    if tool_id not in TOOLS_CONFIG:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    # Validate file type
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are allowed")
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_extension = os.path.splitext(file.filename)[1]
    file_path = f"uploads/{file_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    # Read Excel file to get columns
    try:
        df = pd.read_excel(file_path, sheet_name=0)  # Read first sheet
        columns = df.columns.tolist()
        
        # Check if dataframe is empty
        if df.empty:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Excel file is empty")
            
    except Exception as e:
        # Clean up file if error occurs
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Error reading Excel file: {str(e)}")
    
    # Store file info
    file_storage[file_id] = {
        "original_path": file_path,
        "original_filename": file.filename,
        "columns": columns,
        "tool_id": tool_id,
        "user": user_info["username"],
        "uploaded_at": datetime.now(),
        "expires_at": datetime.now() + timedelta(minutes=5),
        "processed": False
    }
    
    return {
        "file_id": file_id,
        "columns": columns,
        "filename": file.filename
    }

@app.post("/process/{file_id}")
async def process_file(
    file_id: str,
    column_name: str = Form(...),
    user_info: dict = Depends(verify_credentials)
):
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found or expired")
    
    file_info = file_storage[file_id]
    
    # Check if file belongs to user (optional security check)
    if file_info["user"] != user_info["username"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Check if file expired
    if datetime.now() > file_info["expires_at"]:
        raise HTTPException(status_code=410, detail="File has expired")
    
    # Check if column exists
    if column_name not in file_info["columns"]:
        raise HTTPException(status_code=400, detail="Column not found")
    
    try:
        # Process the file based on tool
        tool_id = file_info["tool_id"]
        processor_name = TOOLS_CONFIG[tool_id]["processor"]
        
        if processor_name == "split_excel_by_column":
            processed_path = split_excel_by_column(file_info["original_path"], column_name, file_id)
        else:
            raise HTTPException(status_code=400, detail="Unknown processor")
        
        # Update file info
        file_storage[file_id]["processed"] = True
        file_storage[file_id]["processed_path"] = processed_path
        file_storage[file_id]["column_used"] = column_name
        
        return {"message": "File processed successfully", "download_ready": True}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

def split_excel_by_column(file_path: str, column_name: str, file_id: str) -> str:
    """Split Excel file by column values into multiple sheets"""
    try:
        # Read the Excel file
        df = pd.read_excel(file_path, sheet_name=0)
        
        # Check if column exists and has data
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the file")
        
        if df[column_name].isna().all():
            raise ValueError(f"Column '{column_name}' contains no data")
        
        # Group by the specified column
        grouped_data = df.groupby(column_name)
        
        # Create output file path
        output_path = f"downloads/{file_id}_processed.xlsx"
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for group_name, group_data in grouped_data:
                # Clean sheet name (Excel sheet names have restrictions)
                sheet_name = str(group_name)[:31]  # Max 31 characters
                
                # Remove invalid characters for Excel sheet names
                invalid_chars = ['/', '\\', '*', '?', '[', ']', ':']
                for char in invalid_chars:
                    sheet_name = sheet_name.replace(char, '_')
                
                # Ensure sheet name is not empty
                if not sheet_name.strip():
                    sheet_name = f"Sheet_{len(writer.sheets) + 1}"
                
                # Write to sheet
                group_data.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return output_path
        
    except Exception as e:
        raise Exception(f"Error splitting Excel file: {str(e)}")

@app.get("/download/{file_id}")
async def download_file(file_id: str, user_info: dict = Depends(verify_credentials)):
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found or expired")
    
    file_info = file_storage[file_id]
    
    # Check if file belongs to user
    if file_info["user"] != user_info["username"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Check if file expired
    if datetime.now() > file_info["expires_at"]:
        raise HTTPException(status_code=410, detail="File has expired")
    
    if not file_info["processed"]:
        raise HTTPException(status_code=400, detail="File not processed yet")
    
    processed_path = file_info["processed_path"]
    if not os.path.exists(processed_path):
        raise HTTPException(status_code=404, detail="Processed file not found")
    
    # Generate download filename
    original_name = file_info["original_filename"]
    name_without_ext = os.path.splitext(original_name)[0]
    download_filename = f"{name_without_ext}_processed.xlsx"
    
    return FileResponse(
        processed_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=download_filename
    )

@app.get("/file_status/{file_id}")
async def file_status(file_id: str, user_info: dict = Depends(verify_credentials)):
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found or expired")
    
    file_info = file_storage[file_id]
    
    # Check if file belongs to user
    if file_info["user"] != user_info["username"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    time_left = file_info["expires_at"] - datetime.now()
    
    return {
        "file_id": file_id,
        "processed": file_info["processed"],
        "time_left_seconds": max(0, int(time_left.total_seconds())),
        "expires_at": file_info["expires_at"].isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)