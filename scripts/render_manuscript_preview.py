from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "planning_trajectory_manuscript_draft.md"
OUT = ROOT / ".docx_review" / "manuscript_final" / "planning_trajectory_manuscript_preview.pdf"


def inline(text: str) -> str:
    value = html.escape(text)
    value = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", value)
    return value


def page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#595959"))
    if doc.page > 1:
        canvas.drawRightString(7.5 * inch, 10.55 * inch, "PLANNING TRAJECTORY SUPERVISION")
        canvas.drawCentredString(4.25 * inch, 0.45 * inch, str(doc.page - 1))
    canvas.restoreState()


def build() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        alignment=TA_LEFT,
        firstLineIndent=0.25 * inch,
        spaceAfter=1,
        allowWidows=0,
        allowOrphans=0,
    )
    title = ParagraphStyle(
        "Title",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=19,
        textColor=colors.HexColor("#122D4E"),
        firstLineIndent=0,
        spaceBefore=36,
        spaceAfter=18,
    )
    h1 = ParagraphStyle("H1", parent=body, fontName="Helvetica-Bold", fontSize=13, leading=15, textColor=colors.HexColor("#1F385C"), firstLineIndent=0, spaceBefore=12, spaceAfter=6, keepWithNext=True)
    h2 = ParagraphStyle("H2", parent=body, fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=colors.HexColor("#1F385C"), firstLineIndent=0, spaceBefore=9, spaceAfter=3, keepWithNext=True)
    caption = ParagraphStyle("Caption", parent=body, fontSize=8.5, leading=10.5, firstLineIndent=0, spaceAfter=7, keepWithNext=True)
    reference = ParagraphStyle("Reference", parent=body, fontSize=8.5, leading=10.5, firstLineIndent=-0.22 * inch, leftIndent=0.22 * inch, spaceAfter=3)
    placeholder = ParagraphStyle("Placeholder", parent=body, fontSize=9.5, leading=14, firstLineIndent=0, leftIndent=8, rightIndent=8, borderColor=colors.HexColor("#BF9000"), borderWidth=0.5, borderPadding=6, backColor=colors.HexColor("#FFF2CC"), spaceBefore=4, spaceAfter=6)

    story = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    in_refs = False
    first_title = True
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "\\pagebreak":
            story.append(PageBreak())
            i += 1
            continue
        if line.startswith("# "):
            text = line[2:].strip()
            story.append(Paragraph(inline(text), title if first_title else h1))
            first_title = False
            in_refs = text == "References"
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline(line[3:].strip()), h2))
            i += 1
            continue
        image_match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
        if image_match:
            path = (SOURCE.parent / image_match.group(2)).resolve()
            story.append(Image(str(path), width=6.2 * inch, height=6.2 * inch * 0.44))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells):
                    rows.append([Paragraph(inline(c), ParagraphStyle("Cell", parent=body, fontSize=7.5, leading=9, firstLineIndent=0)) for c in cells])
                i += 1
            cols = len(rows[0])
            widths = [6.5 * inch / cols] * cols
            if cols == 2:
                widths = [1.65 * inch, 4.85 * inch]
            elif cols == 3:
                widths = [1.3 * inch, 2.55 * inch, 2.65 * inch]
            elif cols == 4:
                widths = [1.4 * inch, 1.7 * inch, 1.7 * inch, 1.7 * inch]
            elif cols == 5:
                widths = [1.5 * inch, 1.15 * inch, 1.35 * inch, 1.6 * inch, 0.9 * inch]
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="CENTER")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E5F3")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#7F7F7F")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.extend([table, Spacer(1, 6)])
            continue
        if line.startswith("**Figure") or line.startswith("**Table"):
            story.append(Paragraph(inline(line), caption))
        elif line.startswith("[RESULTS PLACEHOLDER") or line.startswith("[CONCLUSION PLACEHOLDER"):
            story.append(Paragraph(inline(line), placeholder))
        elif in_refs and re.match(r"^\d+\.", line):
            story.append(Paragraph(inline(line), reference))
        else:
            story.append(Paragraph(inline(line), body))
        i += 1

    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=0.7 * inch, title="Planning trajectory supervision", author="David Thomas")
    doc.build(story, onFirstPage=page, onLaterPages=page)
    print(OUT)


if __name__ == "__main__":
    build()
