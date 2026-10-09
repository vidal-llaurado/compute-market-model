#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v pandoc >/dev/null || { echo 'Install pandoc.' >&2; exit 1; }
command -v xelatex >/dev/null || { echo 'Install XeLaTeX.' >&2; exit 1; }
common=(--from=markdown+tex_math_dollars --pdf-engine=xelatex --resource-path=docs \
 --include-in-header=scripts/pdf_header.tex -V mainfont=LinLibertine_R.otf -V 'mainfontoptions=BoldFont=LinLibertine_RB.otf,ItalicFont=LinLibertine_RI.otf,BoldItalicFont=LinLibertine_RBI.otf' \
 -V monofont=lmmono10-regular.otf -V fontsize=11pt -V documentclass=article \
 -V papersize=a4 -V geometry:margin=0.82in -V linestretch=1.08)
tmp=$(mktemp "${TMPDIR:-/tmp}/compute-market-model.XXXXXX")
trap 'rm -f "$tmp"' EXIT
python3 - "$tmp" <<'PY'
from pathlib import Path
import re
import sys
model=Path('docs/MODEL.md').read_text()
refs=Path('docs/REFERENCES.md').read_text()
start=refs.index('## [R01]')
bibliography=re.sub(r'^## (.+)$', lambda m: r'\Needspace{5\baselineskip}'+'\n\n**'+m[1]+'**', refs[start:], flags=re.MULTILINE)
Path(sys.argv[1]).write_text(model+'\n\\newpage\n\n# References {.unnumbered}\n\n'+bibliography)
PY
pandoc "$tmp" "${common[@]}" --toc --toc-depth=1 --number-sections -o docs/MODEL.pdf
if [[ "${1:-}" != '--model-only' ]]; then
 pandoc docs/RESEARCH_REPORT.md "${common[@]}" --number-sections -o docs/RESEARCH_REPORT.pdf
fi
