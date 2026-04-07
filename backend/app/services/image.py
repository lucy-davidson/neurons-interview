"""Utilities for encoding, decoding, and validating images."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import structlog
from PIL import Image

logger = structlog.get_logger()


def load_image_b64(path: str | Path) -> str:
    """Read an image file from disk and return its base-64 encoded string."""
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode()


def b64_to_bytes(b64: str) -> bytes:
    """Decode a base-64 string into raw bytes."""
    return base64.b64decode(b64)


def bytes_to_b64(data: bytes) -> str:
    """Encode raw bytes into a base-64 string."""
    return base64.b64encode(data).decode()


def validate_image(data: bytes) -> bool:
    """Return ``True`` if *data* represents a decodable image.

    Uses PIL's ``verify()`` to catch truncated or corrupted files.
    """
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        return True
    except Exception as exc:
        logger.warning("image_validation_failed", error=str(exc))
        return False
