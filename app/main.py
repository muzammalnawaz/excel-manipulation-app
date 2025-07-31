import os
import asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from .core.config import PORT, DIRECTORIES
from .dependencies import initialize_services
from .services.cleanup_service import CleanupService
from .routers import auth, admin, files, tools

# Create FastAPI app
app = FastAPI(title="Excel Manipulation Tool")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
for directory in DIRECTORIES.values():
    os.makedirs(directory, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global services
cleanup_service: CleanupService = None

@app.on_event("startup")
async def startup_event():
    """Initialize services and start background tasks"""
    global cleanup_service
    
    # Initialize services
    auth_service, file_service = initialize_services()
    
    # Create and start cleanup service
    cleanup_service = CleanupService(file_service)
    await cleanup_service.start_cleanup_task()
    
    print("✅ Application started successfully with enhanced file cleanup")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on application shutdown"""
    global cleanup_service
    
    if cleanup_service:
        await cleanup_service.perform_shutdown_cleanup()
        await cleanup_service.stop_cleanup_task()

# Include routers
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(files.router)
app.include_router(tools.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=True)