class FileProcessingError(Exception):
    """Raised when file processing fails"""
    pass

class FileNotFoundError(Exception):
    """Raised when file is not found or expired"""
    pass

class InvalidFileTypeError(Exception):
    """Raised when file type is not supported"""
    pass

class AuthenticationError(Exception):
    """Raised when authentication fails"""
    pass

class InsufficientPermissionsError(Exception):
    """Raised when user lacks required permissions"""
    pass

class FileExpiredError(Exception):
    """Raised when file session has expired"""
    pass
