import os
import uuid
import glob
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

from ..core.config import FILE_EXPIRATION, ALLOWED_EXTENSIONS
from ..core.exceptions import FileNotFoundError, FileExpiredError, InvalidFileTypeError
from ..models.file_session import FileSession, FileStatus

class FileService:
    def __init__(self):
        self.file_storage: Dict[str, FileSession] = {}
    
    def create_file_session(
        self,
        tool_id: str,
        user: str,
        expiration_minutes: Optional[int] = None
    ) -> str:
        """Create a new file session"""
        file_id = str(uuid.uuid4())
        
        if expiration_minutes is None:
            if tool_id == "stock_distribution":
                expiration_minutes = FILE_EXPIRATION["stock_distribution"]
            else:
                expiration_minutes = FILE_EXPIRATION["default"]
        
        expires_at = datetime.now() + timedelta(minutes=expiration_minutes)
        
        session = FileSession(
            file_id=file_id,
            tool_id=tool_id,
            user=user,
            uploaded_at=datetime.now(),
            expires_at=expires_at
        )
        
        self.file_storage[file_id] = session
        return file_id
    
    def get_file_session(self, file_id: str) -> FileSession:
        """Get file session by ID"""
        if file_id not in self.file_storage:
            raise FileNotFoundError("File not found or expired")
        
        session = self.file_storage[file_id]
        
        if datetime.now() > session.expires_at:
            raise FileExpiredError("File has expired")
        
        return session
    
    def update_file_session(self, file_id: str, **updates) -> None:
        """Update file session with new data"""
        if file_id not in self.file_storage:
            raise FileNotFoundError("File not found")
        
        session = self.file_storage[file_id]
        for key, value in updates.items():
            if hasattr(session, key):
                setattr(session, key, value)
    
    def validate_file_type(self, filename: str) -> None:
        """Validate that file has allowed extension"""
        if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
            raise InvalidFileTypeError(f"Only {', '.join(ALLOWED_EXTENSIONS)} files are allowed")
    
    def save_uploaded_file(self, file_content: bytes, filename: str, file_id: str, prefix: str = "") -> str:
        """Save uploaded file to disk"""
        self.validate_file_type(filename)
        
        if prefix:
            file_path = f"uploads/{file_id}_{prefix}_{filename}"
        else:
            file_path = f"uploads/{file_id}_{filename}"
        
        try:
            with open(file_path, "wb") as buffer:
                buffer.write(file_content)
            print(f"File saved: {len(file_content)} bytes to {file_path}")
            return file_path
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise Exception(f"Error saving file: {str(e)}")
    
    def get_file_status(self, file_id: str) -> FileStatus:
        """Get file status information"""
        session = self.get_file_session(file_id)  # This handles validation
        
        time_left = session.expires_at - datetime.now()
        time_left_seconds = max(0, int(time_left.total_seconds()))
        
        return FileStatus(
            file_id=file_id,
            processed=session.processed,
            time_left_seconds=time_left_seconds,
            expires_at=session.expires_at.isoformat()
        )
    
    def get_file_age_minutes(self, file_path: str) -> float:
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
    
    def cleanup_old_files_by_pattern(self, directory: str, pattern: str, max_age_minutes: int) -> int:
        """Clean up files matching pattern older than max_age_minutes"""
        cleaned_count = 0
        try:
            full_pattern = os.path.join(directory, pattern)
            files = glob.glob(full_pattern)
            
            for file_path in files:
                try:
                    file_age = self.get_file_age_minutes(file_path)
                    if file_age > max_age_minutes:
                        os.remove(file_path)
                        print(f"Cleaned up old file: {file_path} (age: {file_age:.1f} minutes)")
                        cleaned_count += 1
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
                    
        except Exception as e:
            print(f"Error during pattern cleanup in {directory} with pattern {pattern}: {e}")
        
        return cleaned_count
    
    def cleanup_expired_sessions(self) -> Dict[str, int]:
        """Clean up expired file sessions and return cleanup statistics"""
        current_time = datetime.now()
        expired_files = []
        total_cleaned = 0
        
        # 1. Clean up expired files from file_storage tracking
        for file_id, session in list(self.file_storage.items()):
            if current_time > session.expires_at:
                expired_files.append(file_id)
        
        # Remove tracked expired files
        for file_id in expired_files:
            session = self.file_storage[file_id]
            
            # Remove all associated physical files
            files_to_remove = []
            
            # Add original files
            if session.original_path:
                files_to_remove.append(session.original_path)
            
            # Add stock distribution files (dual file upload)
            if session.warehouse_path:
                files_to_remove.append(session.warehouse_path)
            if session.sales_path:
                files_to_remove.append(session.sales_path)
            
            # Add processed files
            if session.processed_path:
                files_to_remove.append(session.processed_path)
            
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
            del self.file_storage[file_id]
            print(f"📋 Removed file_id {file_id} from tracking")
        
        # 2. Clean up orphaned files (files not in tracking system)
        print("🔍 Scanning for orphaned files...")
        
        # Clean uploads directory - files older than 15 minutes
        uploads_cleaned = self.cleanup_old_files_by_pattern("uploads", "*", FILE_EXPIRATION["uploads"])
        total_cleaned += uploads_cleaned
        
        # Clean downloads directory - files older than 30 minutes  
        downloads_cleaned = self.cleanup_old_files_by_pattern("downloads", "*", FILE_EXPIRATION["downloads"])
        total_cleaned += downloads_cleaned
        
        # 3. Clean up temporary and backup files
        # Clean old backup files in data directory (older than 7 days)
        backup_cleaned = self.cleanup_old_files_by_pattern("data", "*.backup*", 7 * 24 * 60)
        total_cleaned += backup_cleaned
        
        # Clean any .tmp files older than 5 minutes
        tmp_cleaned = self.cleanup_old_files_by_pattern("uploads", "*.tmp", 5)
        tmp_cleaned += self.cleanup_old_files_by_pattern("downloads", "*.tmp", 5)
        total_cleaned += tmp_cleaned
        
        return {
            "total_cleaned": total_cleaned,
            "tracked_expired": len(expired_files),
            "orphaned_uploads": uploads_cleaned,
            "orphaned_downloads": downloads_cleaned,
            "old_backups": backup_cleaned,
            "temp_files": tmp_cleaned
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status including file information"""
        current_time = datetime.now()
        
        # Count files in directories
        uploads_count = len([f for f in os.listdir("uploads") if os.path.isfile(os.path.join("uploads", f))])
        downloads_count = len([f for f in os.listdir("downloads") if os.path.isfile(os.path.join("downloads", f))])
        
        # Active file tracking info
        active_files = len(self.file_storage)
        active_users = len(set(session.user for session in self.file_storage.values()))
        
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
        for file_id, session in self.file_storage.items():
            time_left = session.expires_at - current_time
            if time_left.total_seconds() > 0:
                upcoming.append({
                    'file_id': file_id[:8] + '...',
                    'minutes_left': int(time_left.total_seconds() / 60),
                    'user': session.user,
                    'tool': session.tool_id
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
                "upload_retention_minutes": FILE_EXPIRATION["uploads"],
                "download_retention_minutes": FILE_EXPIRATION["downloads"]
            }
        }