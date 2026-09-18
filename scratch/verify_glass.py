from pathlib import Path

html = Path("app/static/index.html").read_text()
css = Path("app/static/style.css").read_text()
js = Path("app/static/app.js").read_text()

assert "hero-suggestions" in html and "suggestion-chip" in html, "Apple suggestion chips in HTML"
assert "backdrop-filter: blur(" in css, "Apple glass blur in CSS"
assert "input-area" in css and "overflow-y: hidden" in css, "Auto-sized clean input in CSS"
assert "extractFileClientSide" in js, "Client-side extraction fallback in JS"
assert "fillPrompt" in js, "Suggestion chip prompt filler in JS"

print("ALL APPLE DARK GLASS ASSETS VERIFIED SUCCESSFULLY ON DISK!")

