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
import glob
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import json
import hashlib
from pathlib import Path
import secrets
import numpy as np

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
os.makedirs("data", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# File to store users persistently - moved to data directory
USERS_FILE = "data/users_data.json"

def load_users():
    """Load users from file, create default if doesn't exist"""
    if Path(USERS_FILE).exists():
        try:
            with open(USERS_FILE, 'r') as f:
                users_data = json.load(f)
                print(f"Loaded {len(users_data)} users from {USERS_FILE}")
                return users_data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading users file: {e}")
            backup_file = f"{USERS_FILE}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                if Path(USERS_FILE).exists():
                    Path(USERS_FILE).rename(backup_file)
                    print(f"Corrupted users file backed up to {backup_file}")
            except Exception as backup_error:
                print(f"Could not backup corrupted file: {backup_error}")
    
    # Default users if file doesn't exist
    default_users = {
        "admin": {"password": "admin123", "role": "admin"},
        "user1": {"password": "user123", "role": "user"}
    }
    save_users(default_users)
    print("Created default users")
    return default_users

def save_users(users_data):
    """Save users to file with backup"""
    try:
        # Create backup of existing file before saving
        if Path(USERS_FILE).exists():
            backup_file = f"{USERS_FILE}.backup"
            Path(USERS_FILE).rename(backup_file)
        
        # Save new data
        with open(USERS_FILE, 'w') as f:
            json.dump(users_data, f, indent=2)
        
        print(f"Saved {len(users_data)} users to {USERS_FILE}")
        
        # Remove old backup if save was successful
        backup_file = f"{USERS_FILE}.backup"
        if Path(backup_file).exists():
            Path(backup_file).unlink()
            
    except IOError as e:
        print(f"Error saving users: {e}")
        # Restore backup if save failed
        backup_file = f"{USERS_FILE}.backup"
        if Path(backup_file).exists():
            Path(backup_file).rename(USERS_FILE)
            print("Restored backup due to save failure")

users_db = load_users()

# File storage with expiration
file_storage: Dict[str, Dict[str, Any]] = {}

# Tools configuration
TOOLS_CONFIG = {
    "excel_splitter": {
        "name": "Excel Sheet Splitter",
        "description": "Split Excel data into multiple sheets based on column values",
        "processor": "split_excel_by_column"
    },
    "data_summarizer": {
        "name": "Summarize Data",
        "description": "Group data by selected columns and perform aggregation operations on multiple numerical columns",
        "processor": "summarize_data"
    },
    "stock_distribution": {
        "name": "Stock Distribution System",
        "description": "Analyze sales conversion rates and distribute warehouse stock to shops based on performance metrics",
        "processor": "process_stock_distribution"
    }
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

def get_file_age_minutes(file_path: str) -> float:
    """Get file age in minutes"""
    try:
        if not os.path.exists(file_path):
            return float('inf')  # File doesn't exist, consider it very old
        
        file_time = os.path.getmtime(file_path)
        current_time = datetime.now().timestamp()
        age_seconds = current_time - file_time
        return age_seconds / 60.0  # Convert to minutes
    except Exception as e:
        print(f"Error getting file age for {file_path}: {e}")
        return float('inf')

def cleanup_old_files_by_pattern(directory: str, pattern: str, max_age_minutes: int) -> int:
    """Clean up files matching pattern older than max_age_minutes"""
    cleaned_count = 0
    try:
        full_pattern = os.path.join(directory, pattern)
        files = glob.glob(full_pattern)
        
        for file_path in files:
            try:
                file_age = get_file_age_minutes(file_path)
                if file_age > max_age_minutes:
                    os.remove(file_path)
                    print(f"Cleaned up old file: {file_path} (age: {file_age:.1f} minutes)")
                    cleaned_count += 1
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")
                
    except Exception as e:
        print(f"Error during pattern cleanup in {directory} with pattern {pattern}: {e}")
    
    return cleaned_count

async def cleanup_expired_files():
    """Enhanced background task to cleanup expired files"""
    print("🧹 File cleanup system started")
    
    while True:
        try:
            current_time = datetime.now()
            expired_files = []
            total_cleaned = 0
            
            # 1. Clean up expired files from file_storage tracking
            for file_id, file_info in list(file_storage.items()):
                if current_time > file_info["expires_at"]:
                    expired_files.append(file_id)
            
            # Remove tracked expired files
            for file_id in expired_files:
                file_info = file_storage[file_id]
                
                # Remove all associated physical files
                files_to_remove = []
                
                # Add original files
                if "original_path" in file_info:
                    files_to_remove.append(file_info["original_path"])
                
                # Add stock distribution files (dual file upload)
                if "warehouse_path" in file_info:
                    files_to_remove.append(file_info["warehouse_path"])
                if "sales_path" in file_info:
                    files_to_remove.append(file_info["sales_path"])
                
                # Add processed files
                if "processed_path" in file_info:
                    files_to_remove.append(file_info["processed_path"])
                
                # Remove all associated files
                for file_path in files_to_remove:
                    if file_path and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            print(f"🗑️  Removed tracked expired file: {file_path}")
                            total_cleaned += 1
                        except Exception as e:
                            print(f"❌ Error removing tracked file {file_path}: {e}")
                
                # Remove from tracking
                del file_storage[file_id]
                print(f"📋 Removed file_id {file_id} from tracking")
            
            # 2. Clean up orphaned files (files not in tracking system)
            print("🔍 Scanning for orphaned files...")
            
            # Clean uploads directory - files older than 15 minutes
            uploads_cleaned = cleanup_old_files_by_pattern("uploads", "*", 15)
            total_cleaned += uploads_cleaned
            
            # Clean downloads directory - files older than 30 minutes  
            downloads_cleaned = cleanup_old_files_by_pattern("downloads", "*", 30)
            total_cleaned += downloads_cleaned
            
            # 3. Clean up temporary and backup files
            # Clean old backup files in data directory (older than 7 days)
            backup_cleaned = cleanup_old_files_by_pattern("data", "*.backup*", 7 * 24 * 60)
            total_cleaned += backup_cleaned
            
            # Clean any .tmp files older than 5 minutes
            tmp_cleaned = cleanup_old_files_by_pattern("uploads", "*.tmp", 5)
            tmp_cleaned += cleanup_old_files_by_pattern("downloads", "*.tmp", 5)
            total_cleaned += tmp_cleaned
            
            # 4. Log cleanup summary
            if total_cleaned > 0:
                print(f"🧹 Cleanup completed: {total_cleaned} files removed")
                print(f"   - Tracked expired: {len(expired_files)} file sets")
                print(f"   - Orphaned uploads: {uploads_cleaned}")
                print(f"   - Orphaned downloads: {downloads_cleaned}")
                print(f"   - Old backups: {backup_cleaned}")
                print(f"   - Temp files: {tmp_cleaned}")
            
            # 5. Log current storage status
            if len(file_storage) > 0:
                print(f"📊 Active file tracking: {len(file_storage)} file sets")
                
                # Show upcoming expirations
                upcoming_expirations = []
                for file_id, file_info in file_storage.items():
                    time_left = file_info["expires_at"] - current_time
                    if time_left.total_seconds() > 0:
                        upcoming_expirations.append({
                            'file_id': file_id[:8] + '...',
                            'minutes_left': int(time_left.total_seconds() / 60),
                            'user': file_info.get('user', 'unknown'),
                            'tool': file_info.get('tool_id', 'unknown')
                        })
                
                if upcoming_expirations:
                    upcoming_expirations.sort(key=lambda x: x['minutes_left'])
                    print("⏰ Upcoming expirations:")
                    for exp in upcoming_expirations[:3]:  # Show next 3
                        print(f"   - {exp['file_id']} ({exp['tool']}) by {exp['user']}: {exp['minutes_left']}min left")
            
        except Exception as e:
            print(f"❌ Error during cleanup cycle: {e}")
            import traceback
            traceback.print_exc()
        
        # Run cleanup every 2 minutes for more responsive cleanup
        await asyncio.sleep(120)

@app.on_event("startup")
async def startup_event():
    # Start cleanup task
    asyncio.create_task(cleanup_expired_files())
    print("✅ Application started successfully with enhanced file cleanup")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on application shutdown"""
    print("🛑 Application shutting down, performing final cleanup...")
    
    try:
        # Clean up all tracked files
        for file_id, file_info in file_storage.items():
            files_to_remove = []
            
            if "original_path" in file_info:
                files_to_remove.append(file_info["original_path"])
            if "warehouse_path" in file_info:
                files_to_remove.append(file_info["warehouse_path"])
            if "sales_path" in file_info:
                files_to_remove.append(file_info["sales_path"])
            if "processed_path" in file_info:
                files_to_remove.append(file_info["processed_path"])
            
            for file_path in files_to_remove:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"🗑️  Shutdown cleanup: {file_path}")
                    except Exception as e:
                        print(f"❌ Error during shutdown cleanup: {e}")
        
        print("✅ Shutdown cleanup completed")
        
    except Exception as e:
        print(f"❌ Error during shutdown cleanup: {e}")

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
    save_users(users_db)
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
    
    if username == "admin" and user_info["username"] == "admin":
        if role != "admin":
            raise HTTPException(status_code=400, detail="Cannot change admin role")
    
    users_db[username]["password"] = password
    users_db[username]["role"] = role
    save_users(users_db)
    return {"message": "User updated successfully"}

@app.delete("/admin/delete_user/{username}")
async def delete_user(
    username: str,
    user_info: dict = Depends(admin_required)
):
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    if username == "admin":
        raise HTTPException(status_code=400, detail="Cannot delete admin user")
    
    if username == user_info["username"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    
    del users_db[username]
    save_users(users_db)
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

@app.get("/admin/system_status")
async def get_system_status(user_info: dict = Depends(admin_required)):
    """Get system status including file cleanup info"""
    try:
        current_time = datetime.now()
        
        # Count files in directories
        uploads_count = len([f for f in os.listdir("uploads") if os.path.isfile(os.path.join("uploads", f))])
        downloads_count = len([f for f in os.listdir("downloads") if os.path.isfile(os.path.join("downloads", f))])
        
        # Active file tracking info
        active_files = len(file_storage)
        active_users = len(set(info.get('user', 'unknown') for info in file_storage.values()))
        
        # Calculate total disk usage
        def get_dir_size(directory):
            total = 0
            try:
                for dirpath, dirnames, filenames in os.walk(directory):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        total += os.path.getsize(filepath)
            except Exception:
                pass
            return total
        
        uploads_size = get_dir_size("uploads")
        downloads_size = get_dir_size("downloads")
        
        # Upcoming expirations
        upcoming = []
        for file_id, file_info in file_storage.items():
            time_left = file_info["expires_at"] - current_time
            if time_left.total_seconds() > 0:
                upcoming.append({
                    'file_id': file_id[:8] + '...',
                    'minutes_left': int(time_left.total_seconds() / 60),
                    'user': file_info.get('user', 'unknown'),
                    'tool': file_info.get('tool_id', 'unknown')
                })
        
        upcoming.sort(key=lambda x: x['minutes_left'])
        
        return {
            "status": "healthy",
            "timestamp": current_time.isoformat(),
            "file_system": {
                "uploads": {
                    "count": uploads_count,
                    "size_bytes": uploads_size,
                    "size_mb": round(uploads_size / 1024 / 1024, 2)
                },
                "downloads": {
                    "count": downloads_count,
                    "size_bytes": downloads_size,
                    "size_mb": round(downloads_size / 1024 / 1024, 2)
                }
            },
            "active_sessions": {
                "file_sets": active_files,
                "active_users": active_users,
                "upcoming_expirations": upcoming[:5]
            },
            "cleanup_system": {
                "enabled": True,
                "check_interval_seconds": 120,
                "upload_retention_minutes": 15,
                "download_retention_minutes": 30
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/tool/{tool_id}", response_class=HTMLResponse)
async def tool_page(request: Request, tool_id: str, user_info: dict = Depends(verify_credentials)):
    if tool_id not in TOOLS_CONFIG:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    tool_config = TOOLS_CONFIG[tool_id]
    
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
        "user_info": user_info
    })

# [Continue with the rest of the functions - process_file, split_excel_by_column, summarize_data, etc.]
# The remaining functions stay the same as in the previous version...

@app.post("/process/{file_id}")
async def process_file(
    file_id: str,
    request: Request,
    user_info: dict = Depends(verify_credentials)
):
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found or expired")
    
    file_info = file_storage[file_id]
    
    if file_info["user"] != user_info["username"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if datetime.now() > file_info["expires_at"]:
        raise HTTPException(status_code=410, detail="File has expired")
    
    try:
        tool_id = file_info["tool_id"]
        processor_name = TOOLS_CONFIG[tool_id]["processor"]
        
        if processor_name == "process_stock_distribution":
            processed_path = process_stock_distribution(
                file_info["warehouse_path"],
                file_info["sales_path"],
                file_id
            )
            
        elif processor_name == "split_excel_by_column":
            form = await request.form()
            column_name = form.get("column_name")
            
            if not column_name or column_name not in file_info["columns"]:
                raise HTTPException(status_code=400, detail="Valid column name required")
            processed_path = split_excel_by_column(file_info["original_path"], column_name, file_id)
            
        elif processor_name == "summarize_data":
            form = await request.form()
            groupby_columns = form.get("groupby_columns")
            numeric_columns = form.get("numeric_columns")
            operation = form.get("operation")
            
            if not groupby_columns or not numeric_columns or not operation:
                raise HTTPException(status_code=400, detail="All fields are required for summarization")
            
            try:
                groupby_cols = json.loads(groupby_columns)
                numeric_cols = json.loads(numeric_columns)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid columns format")
            
            processed_path = summarize_data(
                file_info["original_path"], 
                groupby_cols, 
                numeric_cols,
                operation, 
                file_id
            )
            
        else:
            raise HTTPException(status_code=400, detail="Unknown processor")
        
        file_storage[file_id]["processed"] = True
        file_storage[file_id]["processed_path"] = processed_path
        
        return {"message": "File processed successfully", "download_ready": True}
    
    except Exception as e:
        print(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

def split_excel_by_column(file_path: str, column_name: str, file_id: str) -> str:
    """Split Excel file by column values into multiple sheets"""
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the file")
        
        if df[column_name].isna().all():
            raise ValueError(f"Column '{column_name}' contains no data")
        
        grouped_data = df.groupby(column_name)
        output_path = f"downloads/{file_id}_processed.xlsx"
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for group_name, group_data in grouped_data:
                sheet_name = str(group_name)[:31]
                
                invalid_chars = ['/', '\\', '*', '?', '[', ']', ':']
                for char in invalid_chars:
                    sheet_name = sheet_name.replace(char, '_')
                
                if not sheet_name.strip():
                    sheet_name = f"Sheet_{len(writer.sheets) + 1}"
                
                group_data.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return output_path
        
    except Exception as e:
        raise Exception(f"Error splitting Excel file: {str(e)}")

def summarize_data(file_path: str, groupby_columns: List[str], numeric_columns: List[str], operation: str, file_id: str) -> str:
    """Summarize data by grouping and applying aggregation operations to multiple numeric columns"""
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        
        for col in groupby_columns:
            if col not in df.columns:
                raise ValueError(f"Groupby column '{col}' not found in the file")
        
        for col in numeric_columns:
            if col not in df.columns:
                raise ValueError(f"Numeric column '{col}' not found in the file")
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' is not numeric")
        
        valid_operations = ['sum', 'mean', 'median', 'max', 'min', 'count']
        if operation not in valid_operations:
            raise ValueError(f"Operation '{operation}' is not valid. Choose from: {', '.join(valid_operations)}")
        
        df_clean = df.dropna(subset=groupby_columns + numeric_columns)
        
        if df_clean.empty:
            raise ValueError("No valid data remaining after removing missing values")
        
        grouped = df_clean.groupby(groupby_columns)
        
        if operation == 'sum':
            result = grouped[numeric_columns].sum().reset_index()
        elif operation == 'mean':
            result = grouped[numeric_columns].mean().reset_index()
        elif operation == 'median':
            result = grouped[numeric_columns].median().reset_index()
        elif operation == 'max':
            result = grouped[numeric_columns].max().reset_index()
        elif operation == 'min':
            result = grouped[numeric_columns].min().reset_index()
        elif operation == 'count':
            result = grouped[numeric_columns].count().reset_index()
        
        rename_dict = {}
        for col in numeric_columns:
            rename_dict[col] = f"{col}_{operation}"
        result = result.rename(columns=rename_dict)
        
        if operation in ['sum', 'mean', 'count']:
            total_row = {}
            for col in groupby_columns:
                total_row[col] = 'TOTAL'
            
            for col in numeric_columns:
                result_col = f"{col}_{operation}"
                if operation == 'sum':
                    total_row[result_col] = result[result_col].sum()
                elif operation == 'mean':
                    total_row[result_col] = result[result_col].mean()
                elif operation == 'count':
                    total_row[result_col] = result[result_col].sum()
            
            result = pd.concat([result, pd.DataFrame([total_row])], ignore_index=True)
        
        output_path = f"downloads/{file_id}_summarized.xlsx"
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            result.to_excel(writer, sheet_name='Summary', index=False)
            
            workbook = writer.book
            worksheet = writer.sheets['Summary']
            
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        return output_path
        
    except Exception as e:
        raise Exception(f"Error summarizing data: {str(e)}")

@app.get("/download/{file_id}")
async def download_file(file_id: str, user_info: dict = Depends(verify_credentials)):
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found or expired")
    
    file_info = file_storage[file_id]
    
    if file_info["user"] != user_info["username"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if datetime.now() > file_info["expires_at"]:
        raise HTTPException(status_code=410, detail="File has expired")
    
    if not file_info["processed"]:
        raise HTTPException(status_code=400, detail="File not processed yet")
    
    processed_path = file_info["processed_path"]
    if not os.path.exists(processed_path):
        raise HTTPException(status_code=404, detail="Processed file not found")
    
    if file_info["tool_id"] == "stock_distribution":
        download_filename = "Stock_Distribution_Analysis.xlsx"
    elif file_info["tool_id"] == "data_summarizer":
        original_name = file_info["original_filename"]
        name_without_ext = os.path.splitext(original_name)[0]
        download_filename = f"{name_without_ext}_summarized.xlsx"
    else:
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
    
    if file_info["user"] != user_info["username"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    time_left = file_info["expires_at"] - datetime.now()
    
    return {
        "file_id": file_id,
        "processed": file_info["processed"],
        "time_left_seconds": max(0, int(time_left.total_seconds())),
        "expires_at": file_info["expires_at"].isoformat()
    }

def read_sap_bi_excel(file_path: str, file_type: str = "unknown"):
    """Read SAP BI Excel files by trying all sheets and finding the one with actual data"""
    try:
        print(f"Reading SAP BI Excel file: {file_path} (type: {file_type})")
        
        excel_data = pd.read_excel(file_path, sheet_name=None)
        print(f"Found sheets: {list(excel_data.keys())}")
        
        valid_dfs = []
        
        for sheet_name, sheet_df in excel_data.items():
            print(f"Checking sheet '{sheet_name}': shape {sheet_df.shape}")
            
            if sheet_name.startswith('_com.sap.ip.bi.xl.hiddensheet'):
                print(f"  Skipping hidden sheet: {sheet_name}")
                continue
                
            if sheet_name.startswith('__'):
                print(f"  Skipping system sheet: {sheet_name}")
                continue
            
            if not sheet_df.empty and sheet_df.shape[0] > 0 and sheet_df.shape[1] > 0:
                non_null_rows = sheet_df.dropna(how='all').shape[0]
                if non_null_rows > 1:
                    print(f"  Valid data sheet found: {sheet_name} with {non_null_rows} rows")
                    sheet_df['source_file'] = file_path
                    sheet_df['source_sheet'] = sheet_name
                    valid_dfs.append(sheet_df)
                else:
                    print(f"  Sheet {sheet_name} has only headers/metadata")
            else:
                print(f"  Sheet {sheet_name} is empty")
        
        if valid_dfs:
            combined_df = pd.concat(valid_dfs, ignore_index=True)
            print(f"Combined data shape: {combined_df.shape}")
            print(f"Combined data columns: {list(combined_df.columns)}")
            return combined_df
        else:
            print("No valid data sheets found")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error reading SAP BI Excel file {file_path}: {e}")
        return pd.DataFrame()

def process_stock_distribution(warehouse_file_path: str, sales_file_path: str, file_id: str) -> str:
    """Process stock distribution based on SAP BI warehouse and sales data"""
    try:
        import pandas as pd
        import numpy as np
        from collections import defaultdict
        
        print(f"Processing stock distribution for file ID: {file_id}")
        
        print("Reading warehouse data from SAP BI file...")
        warehouse_df = read_sap_bi_excel(warehouse_file_path, "warehouse")
        
        print("Reading sales data from SAP BI file...")
        sales_df = read_sap_bi_excel(sales_file_path, "sales")
        
        if warehouse_df.empty:
            raise Exception("No valid warehouse data found")
        
        if sales_df.empty:
            raise Exception("No valid sales data found")
        
        print(f"Data loaded - Warehouse: {warehouse_df.shape}, Sales: {sales_df.shape}")
        
        # Remove tracking columns if they exist
        for col in ['source_file', 'source_sheet']:
            if col in warehouse_df.columns:
                warehouse_df = warehouse_df.drop(columns=[col])
            if col in sales_df.columns:
                sales_df = sales_df.drop(columns=[col])
        
        print(f"Original warehouse columns: {list(warehouse_df.columns)}")
        print(f"Original sales columns: {list(sales_df.columns)}")
        
        # WAREHOUSE DATA COLUMN RENAMING
        warehouse_columns = list(warehouse_df.columns)
        new_warehouse_columns = {}
        
        if len(warehouse_columns) >= 12:
            new_warehouse_columns = {
                warehouse_columns[0]: 'Company Code',
                warehouse_columns[1]: 'Company Code Name',
                warehouse_columns[2]: 'Hierarchy Level 2 Code',
                warehouse_columns[3]: 'Department Name',
                warehouse_columns[4]: 'Hierarchy Level 1 Code', 
                warehouse_columns[5]: 'Hierarchy Level 1 Description',
                warehouse_columns[6]: 'Nesto MC Code',
                warehouse_columns[7]: 'Nesto MC Description',
                warehouse_columns[8]: 'Brand Code',
                warehouse_columns[9]: 'Brand Description',
                warehouse_columns[10]: 'Article Code',
                warehouse_columns[11]: 'Article Description'
            }
            
            for i in range(12, len(warehouse_columns)):
                new_warehouse_columns[warehouse_columns[i]] = warehouse_columns[i]
        
        warehouse_renamed = warehouse_df.rename(columns=new_warehouse_columns)
        print(f"Warehouse columns after renaming: {list(warehouse_renamed.columns)}")
        
        # SALES DATA COLUMN RENAMING
        sales_columns = list(sales_df.columns)
        new_sales_columns = {}
        
        if len(sales_columns) >= 12:
            new_sales_columns = {
                sales_columns[0]: 'Company Code',
                sales_columns[1]: 'Company Code Name',
                sales_columns[2]: 'Hierarchy Level 2 Code',
                sales_columns[3]: 'Department Name',
                sales_columns[4]: 'Hierarchy Level 1 Code',
                sales_columns[5]: 'Hierarchy Level 1 Description', 
                sales_columns[6]: 'Nesto MC Code',
                sales_columns[7]: 'Nesto MC Description',
                sales_columns[8]: 'Brand Code',
                sales_columns[9]: 'Brand Description',
                sales_columns[10]: 'Article Code',
                sales_columns[11]: 'Article Description'
            }
            
            for i in range(12, len(sales_columns)):
                new_sales_columns[sales_columns[i]] = sales_columns[i]
        
        sales_renamed = sales_df.rename(columns=new_sales_columns)
        print(f"Sales columns after renaming: {list(sales_renamed.columns)}")
        
        print("Cleaning data by removing rows with null Article Description...")
        
        warehouse_clean = warehouse_renamed.dropna(subset=['Article Description'])
        sales_clean = sales_renamed.dropna(subset=['Article Description'])
        
        print(f"After removing null Article Description - Warehouse: {warehouse_clean.shape}, Sales: {sales_clean.shape}")
        
        # Create working datasets
        warehouse_processed = pd.DataFrame()
        warehouse_processed['company_code'] = warehouse_clean['Company Code']
        warehouse_processed['department'] = warehouse_clean['Department Name']
        warehouse_processed['hierarchy_level1'] = warehouse_clean['Hierarchy Level 1 Description']
        warehouse_processed['nesto_mc'] = warehouse_clean['Nesto MC Description']
        warehouse_processed['brand'] = warehouse_clean['Brand Description']
        warehouse_processed['article_code'] = warehouse_clean['Article Code']
        warehouse_processed['article_description'] = warehouse_clean['Article Description']
        warehouse_processed['article_key'] = warehouse_clean['Article Code'].astype(str)
        
        # Find quantity columns
        for col in warehouse_clean.columns:
            col_str = str(col).lower()
            if 'total' in col_str and 'quantity' in col_str:
                warehouse_processed['total_quantity'] = pd.to_numeric(warehouse_clean[col], errors='coerce').fillna(0)
                print(f"Found warehouse quantity column: {col}")
                break
        
        if 'total_quantity' not in warehouse_processed.columns:
            raise Exception("Could not find total quantity column in warehouse data")
        
        # Sales working dataset  
        sales_processed = pd.DataFrame()
        sales_processed['company_code'] = sales_clean['Company Code']
        sales_processed['department'] = sales_clean['Department Name']
        sales_processed['category'] = sales_clean['Hierarchy Level 1 Description']
        sales_processed['nesto_mc'] = sales_clean['Nesto MC Description']
        sales_processed['brand'] = sales_clean['Brand Description']
        sales_processed['article_code'] = sales_clean['Article Code']
        sales_processed['article_description'] = sales_clean['Article Description']
        sales_processed['article_key'] = sales_clean['Article Code'].astype(str)
        
        # Find stock/sales quantity columns
        for col in sales_clean.columns:
            col_str = str(col).lower()
            if 'stock' in col_str and 'qty' in col_str and 'current' in col_str:
                sales_processed['current_stock'] = pd.to_numeric(sales_clean[col], errors='coerce').fillna(0)
                print(f"Found sales stock column: {col}")
            elif 'sales' in col_str and 'qty' in col_str:
                sales_processed['sales_qty'] = pd.to_numeric(sales_clean[col], errors='coerce').fillna(0)
                print(f"Found sales quantity column: {col}")
        
        if 'current_stock' not in sales_processed.columns:
            sales_processed['current_stock'] = 0
            print("Current stock column not found, defaulting to 0")
        
        if 'sales_qty' not in sales_processed.columns:
            sales_processed['sales_qty'] = 0  
            print("Sales quantity column not found, defaulting to 0")
        
        # Filter valid records
        warehouse_processed = warehouse_processed[
            (warehouse_processed['article_code'].notna()) & 
            (warehouse_processed['article_code'] != '') & 
            (warehouse_processed['total_quantity'] > 0)
        ]
        
        sales_processed = sales_processed[
            (sales_processed['article_code'].notna()) & 
            (sales_processed['article_code'] != '')
        ]
        
        print(f"After filtering - Warehouse: {warehouse_processed.shape}, Sales: {sales_processed.shape}")
        
        # Handle duplicate articles  
        duplicate_articles = warehouse_processed['article_key'].duplicated().sum()
        print(f"Found {duplicate_articles} duplicate articles in warehouse data")
        
        if duplicate_articles > 0:
            warehouse_processed = warehouse_processed.groupby('article_key').agg({
                'company_code': 'first',
                'department': 'first',
                'hierarchy_level1': 'first', 
                'nesto_mc': 'first',
                'brand': 'first',
                'article_code': 'first',
                'article_description': 'first',
                'total_quantity': 'sum'
            }).reset_index()
            print(f"After aggregating duplicates: {warehouse_processed.shape}")
        
        # Calculate sales conversion
        sales_processed['total_grn'] = sales_processed['sales_qty'] + sales_processed['current_stock']
        sales_processed['sales_conversion'] = np.where(
            sales_processed['total_grn'] > 0,
            (sales_processed['sales_qty'] / sales_processed['total_grn']) * 100,
            0
        )
        
        # Find high conversion items (>=80%)
        high_conversion_items = sales_processed[sales_processed['sales_conversion'] >= 80].copy()
        print(f"High conversion items found: {len(high_conversion_items)}")
        
        # Calculate MC-level conversions
        mc_conversions = {}
        for (company_code, nesto_mc), group in sales_processed.groupby(['company_code', 'nesto_mc']):
            total_sales = group['sales_qty'].sum()
            total_grn = group['total_grn'].sum()
            mc_conversion = (total_sales / total_grn * 100) if total_grn > 0 else 0
            mc_conversions[f"{company_code}_{nesto_mc}"] = mc_conversion
        
        # Create warehouse lookup
        warehouse_lookup = warehouse_processed.set_index('article_key').to_dict('index')
        print(f"Warehouse articles available: {len(warehouse_lookup)}")
        
        # Distribution logic
        distribution_results = []
        
        for _, item in high_conversion_items.iterrows():
            article_key = item['article_key']
            
            if article_key not in warehouse_lookup:
                continue
                
            warehouse_stock = warehouse_lookup[article_key]
            
            mc_key = f"{item['company_code']}_{item['nesto_mc']}"
            mc_conversion = mc_conversions.get(mc_key, 0)
            
            if mc_conversion > 60:
                max_distribution = min(12, warehouse_stock['total_quantity'])
                
                distribution_results.append({
                    'warehouse_code': warehouse_stock['company_code'],
                    'article_code': warehouse_stock['article_code'],
                    'article_description': warehouse_stock['article_description'],
                    'nesto_mc': item['nesto_mc'],
                    'brand': item['brand'],
                    'site_code': item['company_code'],
                    'article_sales_conversion': round(item['sales_conversion'], 2),
                    'mc_conversion': round(mc_conversion, 2),
                    'warehouse_total_qty': warehouse_stock['total_quantity'],
                    'distribution_qty': max_distribution,
                    'current_site_stock': item['current_stock']
                })
        
        print(f"Distribution results before conflict resolution: {len(distribution_results)}")
        
        # Handle conflicts
        distribution_by_article = defaultdict(list)
        for dist in distribution_results:
            distribution_by_article[dist['article_code']].append(dist)
        
        final_distribution = []
        for article_code, sites in distribution_by_article.items():
            if len(sites) == 1:
                final_distribution.append(sites[0])
            else:
                sites.sort(key=lambda x: x['article_sales_conversion'], reverse=True)
                
                warehouse_qty = sites[0]['warehouse_total_qty']
                remaining_qty = warehouse_qty
                
                for site in sites:
                    if remaining_qty <= 0:
                        break
                    
                    allocated_qty = min(site['distribution_qty'], remaining_qty)
                    if allocated_qty > 0:
                        site['distribution_qty'] = allocated_qty
                        final_distribution.append(site)
                        remaining_qty -= allocated_qty
        
        print(f"Final distribution count: {len(final_distribution)}")
        
        # Create pivot table structure
        pivot_data = []
        distribution_grouped = defaultdict(list)
        
        for item in final_distribution:
            key = f"{item['warehouse_code']}_{item['article_code']}"
            distribution_grouped[key].append(item)
        
        for key, distributions in distribution_grouped.items():
            warehouse_code, article_code = key.split('_', 1)
            
            pivot_row = {
                'Warehouse Code': warehouse_code,
                'Warehouse Total Quantity': distributions[0]['warehouse_total_qty'],
                'Article Code': distributions[0]['article_code'],
                'Article Description': distributions[0]['article_description'],
                'Nesto MC': distributions[0]['nesto_mc'],
                'Brand': distributions[0]['brand']
            }
            
            for dist in distributions:
                pivot_row[dist['site_code']] = dist['distribution_qty']
            
            pivot_data.append(pivot_row)
        
        # Create detailed data
        detailed_data = [{
            'Warehouse Code': item['warehouse_code'],
            'Article Code': item['article_code'],
            'Article Description': item['article_description'],
            'Nesto MC': item['nesto_mc'],
            'Brand': item['brand'],
            'Site Code': item['site_code'],
            'Warehouse Total Qty': item['warehouse_total_qty'],
            'Distribution Qty': item['distribution_qty'],
            'Article Sales Conversion %': item['article_sales_conversion'],
            'MC Conversion %': item['mc_conversion'],
            'Current Site Stock': item['current_site_stock']
        } for item in final_distribution]
        
        # Create summary statistics
        summary_stats = {
            'Total Distributions': len(final_distribution),
            'Unique Articles': len(set(item['article_code'] for item in final_distribution)),
            'Unique Sites': len(set(item['site_code'] for item in final_distribution)),
            'Total Qty Distributed': sum(item['distribution_qty'] for item in final_distribution),
            'Average Sales Conversion %': round(sum(item['article_sales_conversion'] for item in final_distribution) / len(final_distribution), 2) if final_distribution else 0,
            'Average MC Conversion %': round(sum(item['mc_conversion'] for item in final_distribution) / len(final_distribution), 2) if final_distribution else 0
        }
        
        print(f"Summary: {summary_stats}")
        
        output_path = f"downloads/{file_id}_stock_distribution.xlsx"
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            if pivot_data:
                pivot_df = pd.DataFrame(pivot_data)  
                pivot_df.to_excel(writer, sheet_name='Stock Distribution Pivot', index=False)
            
            if detailed_data:
                detailed_df = pd.DataFrame(detailed_data)
                detailed_df.to_excel(writer, sheet_name='Detailed Distribution', index=False)
            
            summary_df = pd.DataFrame([summary_stats])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"Excel file created successfully: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error in process_stock_distribution: {e}")
        import traceback
        traceback.print_exc()
        raise Exception(f"Error processing stock distribution: {str(e)}")

@app.post("/upload/{tool_id}")
async def upload_file(
    tool_id: str,
    request: Request,
    user_info: dict = Depends(verify_credentials)
):
    print(f"\n=== UPLOAD DEBUG START ===")
    print(f"Tool ID: {tool_id}")
    print(f"User: {user_info['username']}")
    
    if tool_id not in TOOLS_CONFIG:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    try:
        form = await request.form()
        print(f"Form keys: {list(form.keys())}")
        
        if tool_id == "stock_distribution":
            warehouse_file = form.get("warehouse_file")
            sales_file = form.get("sales_file")
            
            if not warehouse_file or not sales_file:
                raise HTTPException(status_code=400, detail="Both warehouse_file and sales_file are required")
            
            if not hasattr(warehouse_file, 'filename') or not hasattr(sales_file, 'filename'):
                raise HTTPException(status_code=400, detail="Invalid file uploads")
            
            print(f"Files received - Warehouse: {warehouse_file.filename}, Sales: {sales_file.filename}")
            
            if not warehouse_file.filename.endswith(('.xlsx', '.xls')):
                raise HTTPException(status_code=400, detail="Warehouse file must be Excel format")
            
            if not sales_file.filename.endswith(('.xlsx', '.xls')):
                raise HTTPException(status_code=400, detail="Sales file must be Excel format")
            
            file_id = str(uuid.uuid4())
            
            warehouse_path = f"uploads/{file_id}_warehouse_{warehouse_file.filename}"
            sales_path = f"uploads/{file_id}_sales_{sales_file.filename}"
            
            try:
                warehouse_content = await warehouse_file.read()
                with open(warehouse_path, "wb") as buffer:
                    buffer.write(warehouse_content)
                print(f"Warehouse file saved: {len(warehouse_content)} bytes")
                
                sales_content = await sales_file.read()
                with open(sales_path, "wb") as buffer:
                    buffer.write(sales_content)
                print(f"Sales file saved: {len(sales_content)} bytes")
                
            except Exception as e:
                for path in [warehouse_path, sales_path]:
                    if os.path.exists(path):
                        os.remove(path)
                raise HTTPException(status_code=500, detail=f"Error saving files: {str(e)}")
            
            try:
                print("=== READING SAP BI FILES ===")
                warehouse_df = read_sap_bi_excel(warehouse_path, "warehouse")
                sales_df = read_sap_bi_excel(sales_path, "sales")
                
                if warehouse_df.empty:
                    raise Exception("No valid data found in warehouse file")
                
                if sales_df.empty:
                    raise Exception("No valid data found in sales file")
                
                print(f"Final validation - Warehouse: {warehouse_df.shape}, Sales: {sales_df.shape}")
                
            except Exception as e:
                print(f"ERROR during file reading: {e}")
                for path in [warehouse_path, sales_path]:
                    if os.path.exists(path):
                        os.remove(path)
                raise HTTPException(status_code=400, detail=f"Error reading Excel files: {str(e)}")
            
            file_storage[file_id] = {
                "warehouse_path": warehouse_path,
                "sales_path": sales_path,
                "warehouse_filename": warehouse_file.filename,
                "sales_filename": sales_file.filename,
                "tool_id": tool_id,
                "user": user_info["username"],
                "uploaded_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(minutes=10),
                "processed": False
            }
            
            print(f"SUCCESS: Files uploaded with ID {file_id}")
            print(f"=== UPLOAD DEBUG END ===\n")
            
            return {
                "file_id": file_id,
                "warehouse_filename": warehouse_file.filename,
                "sales_filename": sales_file.filename,
                "message": "Files uploaded successfully. Ready for processing."
            }
        
        else:
            file = form.get("file")
            
            if not file:
                raise HTTPException(status_code=400, detail="File is required")
            
            if not hasattr(file, 'filename'):
                raise HTTPException(status_code=400, detail="Invalid file upload")
            
            if not file.filename.endswith(('.xlsx', '.xls')):
                raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are allowed")
            
            file_id = str(uuid.uuid4())
            file_path = f"uploads/{file_id}_{file.filename}"
            
            try:
                content = await file.read()
                with open(file_path, "wb") as buffer:
                    buffer.write(content)
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
            
            try:
                df = read_sap_bi_excel(file_path, "general")
                
                if df.empty:
                    os.remove(file_path)
                    raise HTTPException(status_code=400, detail="No valid data found in Excel file")
                
                columns = df.columns.tolist()
                
                column_info = {}
                if tool_id == "data_summarizer":
                    for col in columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            column_info[col] = "numeric"
                        else:
                            column_info[col] = "categorical"
                        
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise HTTPException(status_code=400, detail=f"Error reading Excel file: {str(e)}")
            
            file_storage[file_id] = {
                "original_path": file_path,
                "original_filename": file.filename,
                "columns": columns,
                "column_info": column_info if tool_id == "data_summarizer" else {},
                "tool_id": tool_id,
                "user": user_info["username"],
                "uploaded_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(minutes=5),
                "processed": False
            }
            
            response_data = {
                "file_id": file_id,
                "columns": columns,
                "filename": file.filename
            }
            
            if tool_id == "data_summarizer":
                response_data["column_info"] = column_info
            
            return response_data
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in upload: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)