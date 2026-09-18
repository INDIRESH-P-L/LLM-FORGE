import sys

print("Python executable:", sys.executable)

packages_to_check = [
    "rich",
    "prompt_toolkit",
    "httpx",
    "requests",
    "PIL",
    "fitz",  # PyMuPDF
    "pypdf",
    "pdfplumber",
    "pytesseract",
    "easyocr",
    "whisper",
    "faster_whisper",
    "sounddevice",
    "pyaudio",
    "pyttsx3",
    "gtts",
    "fastapi",
    "uvicorn",
]

for pkg in packages_to_check:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        loc = getattr(mod, "__file__", "unknown")
        print(f"  [YES] {pkg:<16} version: {ver}")
    except ImportError:
        print(f"  [NO ] {pkg:<16} not installed")
    except Exception as e:
        print(f"  [ERR] {pkg:<16} error: {e}")
