"""
Multimodal Parser for NovaSearch.

Capable of extracting text from raw strings, PDFs, and Images (OCR).
"""
import io
import logging
from typing import Optional

try:
    from PIL import Image
except ImportError:
    pass

try:
    import pytesseract
except ImportError:
    pass

# For PDF parsing if needed (simplified fallback)
try:
    import pypdf
except ImportError:
    pass

logger = logging.getLogger(__name__)

class MultimodalParser:
    """
    Unified extraction class to parse text from various file formats.
    """
    
    def __init__(self):
        # Check if tesseract is installed on the system
        self.tesseract_available = False
        try:
            # A simple call to get tesseract version to verify installation
            if 'pytesseract' in globals():
                pytesseract.get_tesseract_version()
                self.tesseract_available = True
                logger.info("Tesseract OS binary verified. Image OCR enabled.")
        except Exception as e:
            logger.warning("Tesseract OCR not found on host OS. Image parsing will fallback to empty strings. (%s)", str(e))

    def parse(self, file_bytes: bytes, mime_type: str) -> str:
        """
        Routes the file bytes to the appropriate parsing engine based on mime type.
        """
        if not file_bytes:
            return ""
            
        mime_type = mime_type.lower()
        
        # 1. Image OCR (PNG, JPG, TIFF)
        if mime_type.startswith("image/"):
            return self._parse_image(file_bytes)
            
        # 2. PDF Parsing
        if mime_type == "application/pdf":
            return self._parse_pdf(file_bytes)
            
        # 3. Raw Text/Markdown
        if mime_type in ["text/plain", "text/markdown", "text/csv"]:
            return file_bytes.decode("utf-8", errors="ignore")
            
        # Fallback
        logger.warning(f"Unsupported MIME type: {mime_type}. Attempting default raw string decode.")
        return file_bytes.decode("utf-8", errors="ignore")

    def _parse_image(self, file_bytes: bytes) -> str:
        """Extracts text from an image using Tesseract OCR."""
        if not self.tesseract_available or 'Image' not in globals() or 'pytesseract' not in globals():
            logger.warning("Image parsing requested but OCR dependencies are missing.")
            return ""
            
        try:
            image = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.error(f"OCR Parsing failed: {e}")
            return ""
            
    def _parse_pdf(self, file_bytes: bytes) -> str:
        """Extracts text from PDF layers."""
        if 'pypdf' not in globals():
            logger.warning("PDF parsing requested but pypdf is not installed.")
            return ""
            
        try:
            pdf = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"PDF Parsing failed: {e}")
            return ""
