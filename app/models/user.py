from typing import Dict, Any
from pydantic import BaseModel

class User(BaseModel):
    username: str
    password: str
    role: str
    
    class Config:
        schema_extra = {
            "example": {
                "username": "user1",
                "password": "password123", 
                "role": "user"
            }
        }

class UserInfo(BaseModel):
    username: str
    role: str
    
    class Config:
        schema_extra = {
            "example": {
                "username": "user1",
                "role": "user"
            }
        }
