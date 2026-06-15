"""DeepSeek Web Chat automation — exception classes."""


class DeepSeekError(Exception):
    """Base exception for DeepSeek automation errors."""


class LoginError(DeepSeekError):
    """Raised when login fails."""


class SessionExpiredError(DeepSeekError):
    """Raised when the saved session is no longer valid."""


class ChatInputError(DeepSeekError):
    """Raised when the chat input cannot be found or interacted with."""


class ResponseTimeoutError(DeepSeekError):
    """Raised when no AI response is received within the timeout."""


class ResponseExtractionError(DeepSeekError):
    """Raised when response text cannot be extracted."""


class PoWTimeoutError(DeepSeekError):
    """Raised when the Proof-of-Work challenge does not resolve."""
