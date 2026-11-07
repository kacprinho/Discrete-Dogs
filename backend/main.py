"""Convenience entrypoint so both `uvicorn app.main:app` and `uvicorn main:app` work."""

from app.main import app

__all__ = ["app"]
