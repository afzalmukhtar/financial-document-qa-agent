from __future__ import annotations

import re

import html2text
from bs4 import BeautifulSoup

TAGS_TO_REMOVE = [
    "script", "style", "noscript", "meta", "link", "header",
    "footer", "nav", "aside", "iframe", "object", "embed",
    "ix:header",
]

ATTRIBUTES_TO_STRIP = [
    "style", "class", "id", "onclick", "onload", "onmouseover",
    "align", "valign", "bgcolor", "color", "face", "size",
    "width", "height", "border", "cellpadding", "cellspacing",
]


def clean_and_convert_html_2_markdown(raw_html: str) -> str:
    """Clean SEC HTML of non-content elements and convert to markdown."""
    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove non-content tags
    for tag_name in TAGS_TO_REMOVE:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove hidden elements
    for tag in soup.find_all(attrs={"style": re.compile(r"display\s*:\s*none", re.IGNORECASE)}):
        tag.decompose()

    # Unwrap XBRL inline tags (keep their text content)
    for tag in soup.find_all(re.compile(r"^ix:")):
        tag.unwrap()

    # Strip styling/layout attributes from all remaining tags
    for tag in soup.find_all(True):
        for attr in ATTRIBUTES_TO_STRIP:
            tag.attrs.pop(attr, None)

    # Convert to markdown
    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    converter.body_width = 0
    converter.ignore_emphasis = False
    converter.unicode_snob = True
    return converter.handle(str(soup))
