#!/usr/bin/env python3
"""Render the bibliography from its JSON records."""
from pathlib import Path
import json
import re

root = Path(__file__).resolve().parents[1] / 'docs'
rows = json.loads((root / 'references.json').read_text())
sections = ['# References\n']
for row in rows:
    publication = row['publication']
    match = re.search(r'; doi:(\S+)', publication)
    if match:
        doi = match[1].rstrip('.')
        publication = publication[:match.start()] + f'. [DOI](https://doi.org/{doi})'
    sections.append(
        f"## [{row['id']}] {row['authors']} ({row['year']})\n\n"
        f"*[{row['title']}]({row['url']})*. {publication}.\n\n"
        f"{row['grounding']}\n"
    )
(root / 'REFERENCES.md').write_text('\n'.join(sections))
