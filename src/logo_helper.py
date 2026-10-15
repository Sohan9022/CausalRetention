"""
CausalRetention Brand Identity & Vector Emblem Generator
Top-tier Google DeepMind / Meta AI Research Prototype Visual Standard
"""

import base64
import os

# Clean standalone SVG mark (minified, single-line, zero raw markdown parsing bugs)
RAW_SVG = (
    '<svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<defs>'
    '<linearGradient id="brandBg" x1="0%" y1="0%" x2="100%" y2="100%">'
    '<stop offset="0%" stop-color="#4F46E5"/>'
    '<stop offset="100%" stop-color="#06B6D4"/>'
    '</linearGradient>'
    '<linearGradient id="arrowGradient" x1="0%" y1="100%" x2="100%" y2="0%">'
    '<stop offset="0%" stop-color="#FFFFFF"/>'
    '<stop offset="100%" stop-color="#34D399"/>'
    '</linearGradient>'
    '</defs>'
    '<rect width="64" height="64" rx="16" fill="url(#brandBg)"/>'
    '<path d="M41 21 C36 16.5 29 15.5 23 18 C15 21.5 11 30 13 38.5 C15 47 23 52 32 50.5 C38 49.5 43 45.5 45.5 40" stroke="#FFFFFF" stroke-width="4.5" stroke-linecap="round"/>'
    '<path d="M25 39 L47 17 M47 17 H33 M47 17 V31" stroke="url(#arrowGradient)" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
)

# Pre-calculate base64 string so rendering is instantaneous and 100% immune to Markdown indentation bugs
LOGO_B64 = base64.b64encode(RAW_SVG.encode("utf-8")).decode("utf-8")


def get_logo_svg(size: int = 40, show_glow: bool = True) -> str:
    """
    Returns an HTML <img> tag with an embedded Base64 Data URI.
    This is 100% immune to Markdown indented-code-block bugs (which occur when raw SVG contains indentation).
    """
    radius = max(4, size // 4)
    return (
        f'<img src="data:image/svg+xml;base64,{LOGO_B64}" '
        f'width="{size}" height="{size}" alt="CausalRetention Logo" '
        f'style="display: inline-block; vertical-align: middle; border-radius: {radius}px; box-shadow: 0 2px 8px rgba(99, 102, 241, 0.25);" />'
    )
