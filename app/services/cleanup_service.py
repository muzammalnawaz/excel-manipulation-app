import asyncio
import os
from datetime import datetime
from typing import Dict, Any

from .file_service import FileService
from ..core.config import CLEANUP_CONFIG

class CleanupService:
    """Service for handling background cleanup tasks"""
    
    def __init__(self, file_service: FileService):
        self.file_service = file_service
        self.cleanup_task = None
        self.is_running = False
    
    async def start_cleanup_task(self):
        """Start the background cleanup task"""
        if self.is_running:
            return
        
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        print("✅ Cleanup service started successfully")
    
    async def stop_cleanup_task(self):
        """Stop the background cleanup task"""
        if not self.is_running or not self.cleanup_task:
            return
        
        self.is_running = False
        self.cleanup_task.cancel()
        try:
            await self.cleanup_task
        except asyncio.CancelledError:
            pass
        print("🛑 Cleanup service stopped")
    
    async def _cleanup_loop(self):
        """Main cleanup loop"""
        print("🧹 File cleanup system started")
        
        while self.is_running:
            try:
                await self._perform_cleanup()
            except Exception as e:
                print(f"❌ Error during cleanup cycle: {e}")
                import traceback
                traceback.print_exc()
            
            # Wait for next cleanup cycle
            await asyncio.sleep(CLEANUP_CONFIG["check_interval_seconds"])
    
    async def _perform_cleanup(self):
        """Perform a single cleanup cycle"""
        current_time = datetime.now()
        
        try:
            # Perform cleanup using file service
            cleanup_stats = self.file_service.cleanup_expired_sessions()
            
            # Log cleanup summary
            if cleanup_stats["total_cleaned"] > 0:
                print(f"🧹 Cleanup completed: {cleanup_stats['total_cleaned']} files removed")
                print(f"   - Tracked expired: {cleanup_stats['tracked_expired']} file sets")
                print(f"   - Orphaned uploads: {cleanup_stats['orphaned_uploads']}")
                print(f"   - Orphaned downloads: {cleanup_stats['orphaned_downloads']}")
                print(f"   - Old backups: {cleanup_stats['old_backups']}")
                print(f"   - Temp files: {cleanup_stats['temp_files']}")
            
            # Log current storage status
            active_sessions = len(self.file_service.file_storage)
            if active_sessions > 0:
                print(f"📊 Active file tracking: {active_sessions} file sets")
                
                # Show upcoming expirations
                upcoming_expirations = []
                for file_id, session in self.file_service.file_storage.items():
                    time_left = session.expires_at - current_time
                    if time_left.total_seconds() > 0:
                        upcoming_expirations.append({
                            'file_id': file_id[:8] + '...',
                            'minutes_left': int(time_left.total_seconds() / 60),
                            'user': session.user,
                            'tool': session.tool_id
                        })
                
                if upcoming_expirations:
                    upcoming_expirations.sort(key=lambda x: x['minutes_left'])
                    print("⏰ Upcoming expirations:")
                    for exp in upcoming_expirations[:3]:  # Show next 3
                        print(f"   - {exp['file_id']} ({exp['tool']}) by {exp['user']}: {exp['minutes_left']}min left")
        
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
    
    async def perform_shutdown_cleanup(self):
        """Perform final cleanup on application shutdown"""
        print("🛑 Application shutting down, performing final cleanup...")
        
        try:
            # Clean up all tracked files
            for file_id, session in self.file_service.file_storage.items():
                files_to_remove = []
                
                if session.original_path:
                    files_to_remove.append(session.original_path)
                if session.warehouse_path:
                    files_to_remove.append(session.warehouse_path)
                if session.sales_path:
                    files_to_remove.append(session.sales_path)
                if session.processed_path:
                    files_to_remove.append(session.processed_path)
                
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