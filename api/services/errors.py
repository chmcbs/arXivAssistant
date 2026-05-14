"""
Domain errors raised by service layer
"""

class BadRequestError(ValueError):
    """Raised when client input violates API preconditions."""

class NotFoundError(ValueError):
    """Raised when a requested resource is absent."""
