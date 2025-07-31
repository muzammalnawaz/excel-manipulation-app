import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from ..core.config import USERS_FILE, DEFAULT_USERS
from ..core.exceptions import AuthenticationError, InsufficientPermissionsError
from ..models.user import User, UserInfo

class AuthService:
    def __init__(self):
        self.users_db = self.load_users()
    
    def load_users(self) -> Dict[str, Dict[str, str]]:
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
        
        # Use default users if file doesn't exist
        self.save_users(DEFAULT_USERS)
        print("Created default users")
        return DEFAULT_USERS.copy()
    
    def save_users(self, users_data: Dict[str, Dict[str, str]]) -> None:
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
    
    def verify_credentials(self, username: str, password: str) -> UserInfo:
        """Verify user credentials and return user info"""
        if username not in self.users_db:
            raise AuthenticationError("Invalid credentials")
        
        stored_password = self.users_db[username]["password"]
        if password != stored_password:
            raise AuthenticationError("Invalid credentials")
        
        return UserInfo(
            username=username,
            role=self.users_db[username]["role"]
        )
    
    def require_admin(self, user_info: UserInfo) -> None:
        """Check if user has admin role"""
        if user_info.role != "admin":
            raise InsufficientPermissionsError("Admin access required")
    
    def create_user(self, username: str, password: str, role: str) -> None:
        """Create a new user"""
        if username in self.users_db:
            raise ValueError("User already exists")
        
        self.users_db[username] = {"password": password, "role": role}
        self.save_users(self.users_db)
    
    def update_user(self, username: str, password: str, role: str, current_user: UserInfo) -> None:
        """Update an existing user"""
        if username not in self.users_db:
            raise ValueError("User not found")
        
        # Prevent admin from changing their own role
        if username == "admin" and current_user.username == "admin" and role != "admin":
            raise ValueError("Cannot change admin role")
        
        self.users_db[username]["password"] = password
        self.users_db[username]["role"] = role
        self.save_users(self.users_db)
    
    def delete_user(self, username: str, current_user: UserInfo) -> None:
        """Delete a user"""
        if username not in self.users_db:
            raise ValueError("User not found")
        
        if username == "admin":
            raise ValueError("Cannot delete admin user")
        
        if username == current_user.username:
            raise ValueError("Cannot delete yourself")
        
        del self.users_db[username]
        self.save_users(self.users_db)
    
    def get_all_users(self) -> Dict[str, Dict[str, str]]:
        """Get all users"""
        return self.users_db.copy()