import os
from typing import Dict, Any

# Get port from environment variable for deployment
PORT = int(os.environ.get("PORT", 8000))

# Directory configuration
DIRECTORIES = {
    "uploads": "uploads",
    "downloads": "downloads", 
    "templates": "templates",
    "static": "static",
    "data": "data"
}

# File configuration
USERS_FILE = "data/users_data.json"
ALLOWED_EXTENSIONS = {'.xlsx', '.xls'}

# File expiration times (in minutes)
FILE_EXPIRATION = {
    "default": 5,
    "stock_distribution": 10,
    "downloads": 30,
    "uploads": 15
}

# Cleanup configuration
CLEANUP_CONFIG = {
    "check_interval_seconds": 120,
    "backup_retention_days": 7,
    "temp_file_retention_minutes": 5
}

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

# Default users
DEFAULT_USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "user1": {"password": "user123", "role": "user"}
}
