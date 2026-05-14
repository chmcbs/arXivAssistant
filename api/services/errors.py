"""
Domain errors raised by service layer
"""


class BadRequestError(ValueError):
    """Raised when client input violates API preconditions."""


class NotFoundError(ValueError):
    """Raised when a requested resource is absent."""


class InternalServerError(ValueError):
    """Raised when an internal pipeline failure should be surfaced as HTTP 500."""
