"""Render timeline_track_key_correction_and_impact.md to PDF.

The PDF is an export of the markdown, never edited by hand: the 7 Oct export
in ~/Downloads went stale because it was made separately and the markdown was
corrected after it. Re-run this after every edit to the markdown.

Links in the markdown point at local absolute paths (/Users/.../file.md:74);
in the PDF they are rewritten to the GitHub copy of the same file and line,
so they work for a reader without the checkout.

Needs pandoc and Google Chrome (headless print). No Python packages.

    python3 scripts/build_timeline_pdf.py
"""

import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "timeline_track_key_correction_and_impact.md"
OUT = ROOT / "timeline_track_key_correction_and_impact.pdf"
REPO = "https://github.com/chenghongm/digital-mind-sprint/blob/main/"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4 landscape; margin: 14mm 12mm; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
       font-size: 9.2pt; line-height: 1.38; color: #1b1b1b; }
h1 { font-size: 17pt; margin: 0 0 4pt; }
h2 { font-size: 12pt; margin: 14pt 0 5pt; border-bottom: 1px solid #ccc;
     padding-bottom: 2pt; }
table { border-collapse: collapse; width: 100%; margin-bottom: 6pt; }
th { background: #dce6f2; text-align: left; }
th, td { border: 1px solid #c8c8c8; padding: 4pt 6pt; vertical-align: top; }
tr { page-break-inside: avoid; }
td:first-child { white-space: nowrap; font-weight: 600; }
code { font-family: Menlo, monospace; font-size: 8.4pt; color: #24613b; }
a { color: #1f5fa8; text-decoration: none; }
.foot { margin-top: 10pt; font-size: 7.6pt; color: #666; }
colgroup col { width: auto !important; }
#detailed-evidence { break-before: page; }
#at-a-glance-what-changed-and-when + table td:nth-child(1) { width: 9%; }
#at-a-glance-what-changed-and-when + table td:nth-child(2) { width: 46%; }
#detailed-evidence + table td:nth-child(1) { width: 9%; }
#detailed-evidence + table td:nth-child(2) { width: 11%; }
"""


def rewrite_links(md):
    def fix(m):
        text, target = m.group(1), m.group(2)
        p = re.match(r"/Users/[^)]*?/digital-mind/(.+)", target)
        if not p:
            return m.group(0)
        path, _, line = p.group(1).partition(":")
        url = REPO + path + (f"#L{line}" if line.isdigit() else "")
        return f"[{text}]({url})"
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", fix, md)


def main():
    md = rewrite_links(SRC.read_text())
    left = re.findall(r"\]\((/Users/[^)]+)\)", md)
    if left:
        raise SystemExit(f"unrewritten local links: {left[:3]}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.md").write_text(md)
        (tmp / "style.css").write_text(CSS)
        html = tmp / "out.html"
        subprocess.run(["pandoc", str(tmp / "in.md"), "-s", "-o", str(html),
                        "--css", "style.css", "--embed-resources",
                        "--metadata", "pagetitle=Digital Mind timeline"],
                       check=True, cwd=tmp)
        foot = ('<p class="foot">Generated from '
                '<code>timeline_track_key_correction_and_impact.md</code> by '
                '<code>scripts/build_timeline_pdf.py</code>. Links point to '
                'the files on GitHub (main).</p></body>')
        html.write_text(html.read_text().replace("</body>", foot))
        subprocess.run([CHROME, "--headless", "--disable-gpu",
                        "--no-pdf-header-footer", f"--print-to-pdf={OUT}",
                        html.as_uri()], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[pdf] {OUT}")


if __name__ == "__main__":
    main()
