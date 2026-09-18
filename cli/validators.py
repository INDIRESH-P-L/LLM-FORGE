"""File, image, and document validators for LegalMindAI CLI."""

import mimetypes
import os
from pathlib import Path
from typing import Optional, Tuple
from cli.config import config


class ValidationError(Exception):
    """Raised when file or input validation fails."""
    pass


def validate_file_path(file_path: str) -> Path:
    """Validate that a file path exists, is a file, and is readable."""
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise ValidationError(f"[ERROR] File does not exist: {file_path}")
    if not path.is_file():
        raise ValidationError(f"[ERROR] Path is not a file: {file_path}")
    if not os.access(path, os.R_OK):
        raise ValidationError(f"[ERROR] Permission denied: {file_path}")
    
    file_size = path.stat().st_size
    if file_size == 0:
        raise ValidationError(f"[ERROR] File is empty: {file_path}")

    return path


def validate_image_file(file_path: str) -> Tuple[Path, str, Tuple[int, int]]:
    """Validate an image file: extension, size, dimensions, readability.
    
    Returns:
        (resolved_path, mime_type, (width, height))
    """
    path = validate_file_path(file_path)
    ext = path.suffix.lower()
    if ext not in config.supported_image_exts:
        raise ValidationError(
            f"[ERROR] Unsupported image type: {ext}. "
            f"Supported formats: {', '.join(config.supported_image_exts)}"
        )

    file_size = path.stat().st_size
    if file_size > config.max_image_size_bytes:
        max_mb = config.max_image_size_bytes // (1024 * 1024)
        raise ValidationError(
            f"[ERROR] Image file exceeds maximum allowed size ({max_mb} MB): {file_size / (1024 * 1024):.1f} MB"
        )

    # Validate image data using Pillow
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()  # Verify that it is, in fact, an image
        
        # Reopen to read dimensions and format
        with Image.open(path) as img:
            width, height = img.size
            if width <= 0 or height <= 0:
                raise ValidationError(f"[ERROR] Corrupted image with invalid dimensions ({width}x{height})")
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"[ERROR] Corrupted or unreadable image file: {e}")

    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        mime_type = f"image/{ext.lstrip('.')}"

    return path, mime_type, (width, height)


def validate_document_file(file_path: str) -> Tuple[Path, str, Optional[int]]:
    """Validate a document file (PDF, DOCX, TXT): extension, size, readability.
    
    Returns:
        (resolved_path, extension, page_count_or_none)
    """
    path = validate_file_path(file_path)
    ext = path.suffix.lower()
    
    # Allow document extensions as well as images for general analysis
    allowed_exts = config.supported_doc_exts + config.supported_image_exts
    if ext not in allowed_exts:
        raise ValidationError(
            f"[ERROR] Unsupported file type: {ext}. "
            f"Supported formats: {', '.join(allowed_exts)}"
        )

    file_size = path.stat().st_size
    if file_size > config.max_doc_size_bytes:
        max_mb = config.max_doc_size_bytes // (1024 * 1024)
        raise ValidationError(
            f"[ERROR] Document exceeds maximum allowed size ({max_mb} MB): {file_size / (1024 * 1024):.1f} MB"
        )

    page_count = None
    if ext == ".pdf":
        # Check PDF magic bytes
        try:
            with open(path, "rb") as f:
                header = f.read(5)
                if not header.startswith(b"%PDF-"):
                    raise ValidationError(f"[ERROR] Corrupted PDF file: invalid PDF header in {path.name}")
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"[ERROR] Unable to read PDF header: {e}")

        # Try to count pages
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            page_count = len(doc)
            doc.close()
        except Exception:
            try:
                import pypdf
                reader = pypdf.PdfReader(str(path))
                page_count = len(reader.pages)
            except Exception as e:
                raise ValidationError(f"[ERROR] Corrupted or password-protected PDF file: {e}")

    return path, ext, page_count
