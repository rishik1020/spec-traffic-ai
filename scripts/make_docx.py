"""
make_docx.py
============
Render the Markdown report as a .docx formatted to the Woxsen guidelines.

WHY THIS BUILDS OOXML BY HAND
pandoc, python-docx and LibreOffice are all absent from this machine, and the
report is due. A .docx is a zip holding a handful of XML parts, so it can be
written directly -- and writing it directly means the formatting the guidelines
specify is set explicitly rather than inherited from a converter's defaults and
then hunted down afterwards.

WHAT THE GUIDELINES REQUIRE, AND WHERE EACH IS SET
    Times New Roman 12pt .......... document default + every run
    1.5 line spacing .............. w:line="360" w:lineRule="auto"
    1 inch margins ................ w:pgMar 1440 twips on all four sides
    justified body ................ w:jc val="both"
    H1 bold 14pt, H2 13pt, H3 12pt  real Heading styles, so the TOC can see them
    page numbers FROM the intro ... two sections; only the second has a footer
    title page .................... logo, title, students, guide, department
    index (TOC) ................... a live TOC field, updated on open
    figure captions BELOW, 10pt ... emitted after the image
    table captions ABOVE, 10pt .... the Markdown already places them above

OOXML measures font size in HALF-POINTS, so 12pt is 24 and 10pt is 20. Line
spacing is in twentieths of a point, so 1.5 lines of 12pt is 360. Getting
either wrong produces a document that looks almost right, which is worse than
one that looks obviously wrong.

HEADING LEVELS
The Markdown '#' is the report title and belongs on the title page, so it is
not emitted into the body. '##' is therefore Heading 1, '###' Heading 2 and
'####' Heading 3, which is what makes the generated contents list match the
numbering printed in the text.

USAGE
    python scripts/make_docx.py
    python scripts/make_docx.py --in docs/FINAL_REPORT.md --out report.docx
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

W = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
     'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/'
     'relationships" '
     'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/'
     'wordprocessingDrawing" '
     'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
     'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"')

FONT = 'Times New Roman'
BODY_SZ = 24        # half-points -> 12pt
CAPTION_SZ = 20     # 10pt
LINE = 360          # 1.5 x 12pt, in twentieths of a point

EMU_PER_INCH = 914400
MAX_W_IN = 6.1      # A4 is 8.27in wide; 1in margins leave 6.27in of column
MAX_H_IN = 7.0      # keeps a figure and its caption together on one page

# --- title page --------------------------------------------------------
# Taken from the team's Review-II submission. CONFIRM the department and the
# academic year before submitting: neither appears on that document.
TITLE_PAGE = {
    'logo': 'figures/woxsen_logo.png',
    'course': 'CAPSTONE PROJECT – I',
    'title': ('SPEC Traffic AI: An Explainable Computer Vision Platform for '
              'Indian Traffic Violation Detection with Human-in-the-Loop '
              'E-Challan Verification and Advisory Signal Timing'),
    'students': [
        ('Rishik Reddy P', '23WU0102149'),
        ('Rohan Reddy', '23WU0102169'),
        ('Hari Ramaneti', '23WU0102135'),
    ],
    'guide': 'Prof. Meher Gayatri',
    'department': 'Department of Computer Science and Engineering',
    'institute': 'Woxsen University, School of Technology',
    'year': 'Academic Year 2026–2027  |  7th Semester',
}

HEADING_STYLES = {1: ('Heading1', 28), 2: ('Heading2', 26), 3: ('Heading3', 24)}


def esc(t: str) -> str:
    return t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def runs(text: str, sz: int = BODY_SZ, bold: bool = False,
         italic: bool = False) -> str:
    """Inline **bold**, *italic* and `code` become real runs."""
    out = []
    for part in re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)', text):
        if not part:
            continue
        b, i, t = bold, italic, part
        if part.startswith('**') and part.endswith('**'):
            b, t = True, part[2:-2]
        elif part.startswith('*') and part.endswith('*'):
            i, t = True, part[1:-1]
        elif part.startswith('`') and part.endswith('`'):
            t = part[1:-1]
        props = f'<w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}"/><w:sz w:val="{sz}"/>'
        if b:
            props += '<w:b/>'
        if i:
            props += '<w:i/>'
        out.append(f'<w:r><w:rPr>{props}</w:rPr>'
                   f'<w:t xml:space="preserve">{esc(t)}</w:t></w:r>')
    return ''.join(out)


def para(text: str, sz: int = BODY_SZ, bold: bool = False, align: str = 'both',
         before: int = 0, after: int = 120, italic: bool = False,
         style: str = '', page_break_before: bool = False) -> str:
    ppr = '<w:pPr>'
    if style:
        ppr += f'<w:pStyle w:val="{style}"/>'
    if page_break_before:
        ppr += '<w:pageBreakBefore/>'
    ppr += (f'<w:spacing w:before="{before}" w:after="{after}" '
            f'w:line="{LINE}" w:lineRule="auto"/><w:jc w:val="{align}"/></w:pPr>')
    return f'<w:p>{ppr}{runs(text, sz, bold, italic)}</w:p>'


def table(rows: list[list[str]]) -> str:
    """Markdown table -> w:tbl, header row bold, single borders."""
    borders = ('<w:tblBorders>' + ''.join(
        f'<w:{e} w:val="single" w:sz="4" w:color="000000"/>'
        for e in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'))
        + '</w:tblBorders>')
    out = [f'<w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/>{borders}'
           '</w:tblPr>']
    for ri, row in enumerate(rows):
        cells = []
        for c in row:
            body = para(c, sz=CAPTION_SZ, bold=(ri == 0), align='left', after=40)
            cells.append(f'<w:tc><w:tcPr><w:tcW w:w="0" w:type="auto"/>'
                         f'</w:tcPr>{body}</w:tc>')
        out.append(f'<w:tr>{"".join(cells)}</w:tr>')
    out.append('</w:tbl>')
    # a table must be followed by a paragraph or Word complains
    return ''.join(out) + para('', after=120)


def image_xml(rel_id: str, idx: int, path: Path, max_w: float = MAX_W_IN,
              max_h: float = MAX_H_IN) -> str:
    """A centred inline image scaled to fit the text column."""
    from PIL import Image
    with Image.open(path) as im:
        px_w, px_h = im.size
    w_in = max_w
    h_in = px_h * (w_in / px_w)
    if h_in > max_h:
        w_in *= max_h / h_in
        h_in = max_h
    cx, cy = int(w_in * EMU_PER_INCH), int(h_in * EMU_PER_INCH)
    return (
        f'<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        f'<wp:docPr id="{idx}" name="Figure {idx}"/>'
        f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/'
        f'drawingml/2006/picture"><pic:pic>'
        f'<pic:nvPicPr><pic:cNvPr id="{idx}" name="{path.name}"/>'
        f'<pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rel_id}"/>'
        f'<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/>'
        f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        f'</pic:pic></a:graphicData></a:graphic>'
        f'</wp:inline></w:drawing>')


def figure(rel_id: str, idx: int, path: Path, caption: str) -> str:
    img_p = (f'<w:p><w:pPr><w:spacing w:before="200" w:after="60" '
             f'w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>'
             f'<w:r>{image_xml(rel_id, idx, path)}</w:r></w:p>')
    return img_p + para(caption, sz=CAPTION_SZ, align='center', after=240)


def section_properties(footer: bool) -> str:
    """Only the body section carries page numbers, per the guidelines."""
    ref = '<w:footerReference w:type="default" r:id="rId3"/>' if footer else ''
    start = ('<w:pgNumType w:start="1"/>' if footer
             else '<w:pgNumType w:start="1" w:fmt="lowerRoman"/>')
    return (f'<w:sectPr>{ref}{start}'
            f'<w:pgSz w:w="11906" w:h="16838"/>'
            f'<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" '
            f'w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
            f'</w:sectPr>')


def title_page(media: list[tuple[str, Path]], docs_dir: Path) -> str:
    t = TITLE_PAGE
    out = []

    logo = (docs_dir / t['logo']).resolve()
    if logo.exists():
        rid = f'rIdImg{len(media) + 1}'
        media.append((rid, logo))
        out.append(
            f'<w:p><w:pPr><w:spacing w:before="480" w:after="360" '
            f'w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>'
            f'<w:r>{image_xml(rid, len(media), logo, max_w=2.4, max_h=1.4)}'
            f'</w:r></w:p>')

    out.append(para(t['course'], sz=28, bold=True, align='center', after=360))
    out.append(para(t['title'], sz=32, bold=True, align='center', after=480))
    out.append(para('Submitted by', sz=BODY_SZ, align='center', after=120))
    for name, roll in t['students']:
        out.append(para(f'{name}  ({roll})', sz=BODY_SZ, bold=True,
                        align='center', after=60))
    out.append(para('', after=240))
    out.append(para('Under the guidance of', sz=BODY_SZ, align='center',
                    after=120))
    out.append(para(t['guide'], sz=BODY_SZ, bold=True, align='center',
                    after=480))
    out.append(para(t['department'], sz=BODY_SZ, align='center', after=60))
    out.append(para(t['institute'], sz=BODY_SZ, bold=True, align='center',
                    after=240))
    out.append(para(t['year'], sz=BODY_SZ, align='center', after=0))
    return ''.join(out)


def toc() -> str:
    """A live TOC field. settings.xml asks Word to populate it on open."""
    rpr = (f'<w:rPr><w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}"/>'
           f'<w:sz w:val="{BODY_SZ}"/></w:rPr>')
    heading = para('TABLE OF CONTENTS', sz=28, bold=True, align='center',
                   before=240, after=240, page_break_before=True)
    field = (
        f'<w:p><w:pPr><w:spacing w:line="{LINE}" w:lineRule="auto"/></w:pPr>'
        f'<w:r>{rpr}<w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
        f'<w:r>{rpr}<w:instrText xml:space="preserve">'
        f' TOC \\o "1-3" \\h \\z \\u </w:instrText></w:r>'
        f'<w:r>{rpr}<w:fldChar w:fldCharType="separate"/></w:r>'
        f'<w:r>{rpr}<w:t>Right-click here and choose '
        f'&quot;Update Field&quot; to build the contents.</w:t></w:r>'
        f'<w:r>{rpr}<w:fldChar w:fldCharType="end"/></w:r></w:p>')
    return heading + field


def convert(md: str, docs_dir: Path) -> tuple[str, list[tuple[str, Path]]]:
    media: list[tuple[str, Path]] = []
    front: list[str] = [title_page(media, docs_dir), toc()]
    body: list[str] = []
    out = front                      # front matter until the Introduction
    split_done = False

    lines = md.split('\n')
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()

        if not ln.strip() or ln.strip() == '---':
            i += 1
            continue

        # ![caption](path) -- a figure, caption beneath it
        m_img = re.match(r'^!\[(.*?)\]\((.*?)\)\s*$', ln)
        if m_img:
            caption, rel = m_img.group(1), m_img.group(2)
            src = (docs_dir / rel).resolve()
            if src.exists():
                rid = f'rIdImg{len(media) + 1}'
                media.append((rid, src))
                out.append(figure(rid, len(media), src, caption))
            else:
                out.append(para(f'[missing figure: {rel}]', sz=CAPTION_SZ,
                                align='center'))
            i += 1
            continue

        if ln.lstrip().startswith('|'):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith('|'):
                cells = [c.strip()
                         for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-{2,}:?', c) for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                out.append(table(rows))
            continue

        m = re.match(r'^(#{1,4})\s+(.*)', ln)
        if m:
            level, text = len(m.group(1)), m.group(2)
            if level == 1:
                i += 1              # the report title lives on the title page
                continue
            # the body section starts at the Introduction, where the
            # guidelines require page numbering to begin
            if not split_done and re.match(r'^1\.\s', text):
                front.append(f'<w:p><w:pPr>{section_properties(False)}'
                             f'</w:pPr></w:p>')
                out = body
                split_done = True
            style, sz = HEADING_STYLES[level - 1]
            out.append(para(text, sz=sz, bold=True, align='left',
                            before=240, after=120, style=style))
            i += 1
            continue

        # "**Table 1: ...**" is a caption, which the guidelines put
        # above its table at 10pt rather than in the 12pt body size
        if re.match(r'^\*\*Table \d+[:.]', ln):
            out.append(para(ln, sz=CAPTION_SZ, align='center',
                            before=180, after=60))
            i += 1
            continue

        if re.match(r'^\s*[-*]\s+', ln):
            out.append(para('•  ' + re.sub(r'^\s*[-*]\s+', '', ln),
                            align='left', after=60))
            i += 1
            continue

        if re.match(r'^\s*\d+\.\s+', ln):
            out.append(para(ln.strip(), align='left', after=60))
            i += 1
            continue

        buf = [ln]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
                r'^(#{1,4}\s|\s*[-*]\s|\s*\d+\.\s|\||!\[)', lines[i]):
            buf.append(lines[i].rstrip())
            i += 1
        out.append(para(' '.join(buf)))

    doc = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           f'<w:document {W}><w:body>{"".join(front)}{"".join(body)}'
           f'{section_properties(True)}</w:body></w:document>')
    return doc, media


def heading_style(style_id: str, name: str, sz: int, outline: int) -> str:
    return (f'<w:style w:type="paragraph" w:styleId="{style_id}">'
            f'<w:name w:val="{name}"/><w:basedOn w:val="Normal"/>'
            f'<w:next w:val="Normal"/><w:qFormat/>'
            f'<w:pPr><w:keepNext/><w:outlineLvl w:val="{outline}"/>'
            f'<w:spacing w:before="240" w:after="120" w:line="{LINE}" '
            f'w:lineRule="auto"/><w:jc w:val="left"/></w:pPr>'
            f'<w:rPr><w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}"/><w:b/>'
            f'<w:sz w:val="{sz}"/></w:rPr></w:style>')


STYLES = (
    f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    f'<w:styles {W}><w:docDefaults><w:rPrDefault><w:rPr>'
    f'<w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:cs="{FONT}"/>'
    f'<w:sz w:val="{BODY_SZ}"/><w:szCs w:val="{BODY_SZ}"/>'
    f'</w:rPr></w:rPrDefault><w:pPrDefault><w:pPr>'
    f'<w:spacing w:line="{LINE}" w:lineRule="auto"/>'
    f'</w:pPr></w:pPrDefault></w:docDefaults>'
    f'<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
    f'<w:name w:val="Normal"/><w:qFormat/></w:style>'
    + heading_style('Heading1', 'heading 1', 28, 0)
    + heading_style('Heading2', 'heading 2', 26, 1)
    + heading_style('Heading3', 'heading 3', 24, 2)
    + '</w:styles>')

# asks Word to refresh the TOC when the document is opened
SETTINGS = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:settings {W}><w:updateFields w:val="true"/></w:settings>')

FOOTER = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          f'<w:ftr {W}><w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
          f'<w:fldSimple w:instr=" PAGE "><w:r>'
          f'<w:rPr><w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}"/>'
          f'<w:sz w:val="{CAPTION_SZ}"/></w:rPr><w:t>1</w:t></w:r>'
          f'</w:fldSimple></w:p></w:ftr>')

CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Default Extension="png" ContentType="image/png"/>'
    '<Default Extension="jpg" ContentType="image/jpeg"/>'
    '<Default Extension="jpeg" ContentType="image/jpeg"/>'
    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
    '<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>'
    '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
    '</Types>')

RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
    '</Relationships>')

DOC_RELS_HEAD = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>'
    '<Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>')

IMG_REL = ('<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
           'officeDocument/2006/relationships/image" Target="media/{name}"/>')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src',
                    default=str(ROOT / 'docs' / 'FINAL_REPORT.md'))
    ap.add_argument('--out', dest='dst',
                    default=str(ROOT / 'docs' /
                                'SPEC Traffic AI - Capstone Report.docx'))
    args = ap.parse_args()

    src = Path(args.src)
    doc, media = convert(src.read_text(encoding='utf-8'), src.parent)

    out = Path(args.dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', CONTENT_TYPES)
        z.writestr('_rels/.rels', RELS)
        z.writestr('word/document.xml', doc)
        z.writestr('word/styles.xml', STYLES)
        z.writestr('word/settings.xml', SETTINGS)
        z.writestr('word/footer1.xml', FOOTER)

        rels = [DOC_RELS_HEAD]
        for rid, path in media:
            name = f'{rid}{path.suffix.lower()}'
            z.write(path, f'word/media/{name}')
            rels.append(IMG_REL.format(rid=rid, name=name))
        rels.append('</Relationships>')
        z.writestr('word/_rels/document.xml.rels', ''.join(rels))

    print(f'wrote {out}  ({out.stat().st_size / 1024:.0f} KB)')
    print(f'  {FONT} {BODY_SZ // 2}pt, 1.5 spacing, 1in margins, justified')
    print('  title page + TOC field; page numbers restart at the Introduction')
    print(f'  {len(media) - 1} figure(s) embedded, captions below at '
          f'{CAPTION_SZ // 2}pt')
    print('  IN WORD: the contents list fills in on open; if prompted, allow '
          'it to update fields.')


if __name__ == '__main__':
    main()
