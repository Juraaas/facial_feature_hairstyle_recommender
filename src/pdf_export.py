from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable
)
import io

W, H = A4

def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", fontSize=20, fontName="Helvetica-Bold",
                                spaceAfter=6, textColor=colors.HexColor("#111111")),
        "heading": ParagraphStyle("heading", fontSize=13, fontName="Helvetica-Bold",
                                  spaceBefore=14, spaceAfter=4),
        "subhead": ParagraphStyle("subhead", fontSize=11, fontName="Helvetica-Bold",
                                  spaceBefore=8, spaceAfter=2),
        "body": ParagraphStyle("body", fontSize=10, fontName="Helvetica",
                               spaceAfter=3, leading=14),
        "small": ParagraphStyle("small", fontSize=9, fontName="Helvetica",
                                textColor=colors.grey),
        "positive": ParagraphStyle("positive", fontSize=10, fontName="Helvetica",
                                   textColor=colors.HexColor("#1a6e2e")),
        "negative": ParagraphStyle("negative", fontSize=10, fontName="Helvetica",
                                   textColor=colors.HexColor("#8b1a1a")),
    }

def _bar_table(label_left, label_right, value, min_val, max_val, avg_val):
    BAR_CELLS = 20
    pct = min(1.0, max(0.0, (value - min_val) / (max_val - min_val)))
    avg_pct = min(1.0, max(0.0, (avg_val - min_val) / (max_val - min_val)))
    filled = round(pct * BAR_CELLS)
    avg_cell = round(avg_pct * BAR_CELLS)

    cell_w = 0.38 * cm
    bar_colors = []
    for i in range(BAR_CELLS):
        if i < filled:
            bar_colors.append(colors.HexColor("#185FA5"))
        else:
            bar_colors.append(colors.HexColor("#e0e0e0"))
    
    cells = [[""] * BAR_CELLS]
    style_cmds = [
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white]),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 1),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 1),
        ("ROWHEIGHT",     (0, 0), (-1, -1), 10),
    ]
    for i, c in enumerate(bar_colors):
        style_cmds.append(("BACKGROUND", (i, 0), (i, 0), c))
    
    if 0 <= avg_cell < BAR_CELLS:
        style_cmds.append(("BOX", (avg_cell, 0), (avg_cell, 0),
                           1, colors.HexColor("#888888")))
    
    bar = Table(cells, colWidths=[cell_w] * BAR_CELLS)
    bar.setStyle(TableStyle(style_cmds))

    styles = _styles()
    wrapper = Table(
        [[Paragraph(label_left, styles["small"]), "",
          Paragraph(label_right, styles["small"])]],
        colWidths=[5 * cm, BAR_CELLS * cell_w - 10 * cm + 5 * cm, 5 * cm]
    )

    return bar, wrapper

def generate_pdf(features, traits, recs, norms_df) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm,  bottomMargin=2*cm)
    s = _styles()
    story = []

    story.append(Paragraph("Hairstyle AI Recommender", s["title"]))
    story.append(Paragraph("Facial Analysis Report", s["body"]))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#cccccc"), spaceAfter=10))
    
    story.append(Paragraph("Face Analysis", s["heading"]))
    for exp in recs["face_analysis"]:
        story.append(Paragraph(f"• {exp}", s["body"]))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Facial Proportions", s["heading"]))

    BARS = [
        ("face_ratio",       "Wide face",       "Long face"),
        ("jaw_ratio",        "Narrow jaw",      "Wide jaw"),
        ("eye_ratio",        "Close-set eyes",  "Wide-set eyes"),
        ("eye_height",       "Narrow eyes",     "Wide eyes"),
        ("lip_ratio",        "Narrow lips",     "Wide lips"),
        ("nose_position",    "High nose",       "Low nose"),
        ("lower_face_ratio", "Short lower face","Long lower face"),
        ("chin_prominence",  "Flat chin",       "Strong chin"),
        ("symmetry",         "Symmetrical",     "Asymmetrical"),
    ]

    bar_data = []
    for feat, lbl_left, lbl_right in BARS:
        if feat not in features:
            continue
        val = features[feat]
        min_val = float(norms_df.loc["p5",   feat])
        max_val = float(norms_df.loc["p95",  feat])
        avg_val = float(norms_df.loc["mean", feat])

        pct = (val - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else 0.5
        pct = min(1.0, max(0.0, pct))

        if pct < 0.33:
            position_label = lbl_left
            pos_color = colors.HexColor("#185FA5")
        elif pct > 0.67:
            position_label = lbl_right
            pos_color = colors.HexColor("#185FA5")
        else:
            position_label = "Average"
            pos_color = colors.HexColor("#444444")

        out_of_range = val < min_val or val > max_val

        row = [
            Paragraph(lbl_left, s["small"]),
            Paragraph(
                f'<font color="#185FA5"><b>{val:.3f}</b></font>'
                f'  →  <font color="{"#888888" if out_of_range else "#185FA5"}'
                f'"><b>{position_label}</b></font>'
                f'<br/><font color="#999999" size="8">'
                f'typical range: {min_val:.3f} – {max_val:.3f}'
                f'  |  avg: {avg_val:.3f}</font>',
                s["body"]
            ),
            Paragraph(lbl_right, s["small"]),
        ]
        bar_data.append(row)

    bar_table = Table(bar_data, colWidths=[4.5*cm, 8*cm, 4.5*cm])
    bar_table.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (0, 0), (0, -1),  "RIGHT"),
        ("ALIGN",         (2, 0), (2, -1),  "LEFT"),
        ("ALIGN",         (1, 0), (1, -1),  "LEFT"),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [colors.HexColor("#f9f9f9"),
                                            colors.white]),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(bar_table)
    story.append(Spacer(1, 0.4*cm))

    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#cccccc"), spaceAfter=6))
    story.append(Paragraph("Top Hairstyle Recommendations", s["heading"]))

    for i, style in enumerate(recs["top_styles"], 1):
        story.append(Paragraph(
            f"{i}. {style['name']} — {style['score']:.1f} points",
            s["subhead"]
        ))

        if style.get("contributions"):
            story.append(Paragraph("Why it works:", s["body"]))
            for c in style["contributions"][:3]:
                story.append(Paragraph(
                    f"  ✓  {c['desc']}  ({c['percent']*100:.0f}%)",
                    s["positive"]
                ))

        if style.get("negatives"):
            story.append(Paragraph("Potential drawbacks:", s["body"]))
            for c in style["negatives"][:2]:
                story.append(Paragraph(
                    f"  ✗  {c['desc']}",
                    s["negative"]
                ))

        story.append(Spacer(1, 0.2*cm))

    doc.build(story)
    return buf.getvalue()