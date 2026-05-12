import streamlit.components.v1 as components
import base64
from pathlib import Path

def trait_bar(title, value, min_val, max_val, avg_val,
              min_label, max_label, height=115):

    clamped = min(100, max(0, (value - min_val) / (max_val - min_val) * 100))
    avg_pct = min(100, max(0, (avg_val - min_val) / (max_val - min_val) * 100))

    out_of_range = "below" if value < min_val else ("above" if value > max_val else None)

    if out_of_range == "below":
      dot_pct, dot_label, dot_color = 2, f"&lt; {min_val:.2f}", "#888"
    elif out_of_range == "above":
      dot_pct, dot_label, dot_color = 98, f"&gt; {max_val:.2f}", "#888"
    else:
      dot_pct, dot_label, dot_color = clamped, f"{value:.2f}", "#185FA5"

    if out_of_range:
      interp = "out of range"
    elif clamped < 33:
      interp = min_label.lower()
    elif clamped > 67:
      interp = max_label.lower()
    else:
      interp = "average"

    html = f"""
    <style>
      .tb{{font-family:sans-serif;margin-bottom:4px}}
      .tb-hdr{{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px}}
      .tb-title{{font-size:13px;font-weight:500;color:var(--color-text-primary,#111)}}
      .tb-val{{font-size:12px;color:#666}}
      .tb-interp{{font-size:11px;color:#999;margin-left:4px}}
      .tb-track{{position:relative;height:6px;border-radius:3px;background:#eee}}
      .tb-fill{{position:absolute;left:0;top:0;height:100%;border-radius:3px;background:#378ADD}}
      .tb-avg{{position:absolute;top:-4px;width:1.5px;height:14px;background:#bbb;transform:translateX(-50%)}}
      .tb-dot{{position:absolute;top:-5px;width:16px;height:16px;border-radius:50%;border:2.5px solid #fff;transform:translateX(-50%)}}
      .tb-lbls{{display:flex;justify-content:space-between;margin-top:4px}}
      .tb-lbl{{font-size:11px;color:#aaa}}
    </style>
    <div class="tb">
      <div class="tb-hdr">
        <span class="tb-title">{title}</span>
        <span class="tb-val">{dot_label}<span class="tb-interp">→ {interp}</span></span>
      </div>
      <div class="tb-track">
        <div class="tb-fill" style="width:{clamped:.1f}%"></div>
        <div class="tb-avg" style="left:{avg_pct:.1f}%"></div>
        <div class="tb-dot" style="left:{dot_pct:.1f}%;background:{dot_color}"></div>
      </div>
      <div class="tb-lbls">
        <span class="tb-lbl">{min_label} ≥{min_val:.2f}</span>
        <span class="tb-lbl">{max_label} ≤{max_val:.2f}</span>
      </div>
    </div>"""
    components.html(html, height=height)

def load_image_b64(path: str) -> str | None:
  try:
    data = Path(path).read_bytes()
    ext = Path(path).suffix.lstrip(".")
    return f"data:image/{ext};base64,{base64.b64encode(data).decode()}"
  except Exception:
    return None


def style_card(style, rank=0):
  is_top = rank == 0
  border = "border:.5px solid #378ADD" if is_top else "border:.5px solid #e0e0e0"
  badge = (
      '<div style="position:absolute;top:8px;left:8px;background:#185FA5;'
      'color:#E6F1FB;font-size:10px;font-weight:500;padding:2px 8px;'
      'border-radius:20px">Best match</div>'
      if is_top else ""
  )

  score = round(style["score"], 1)

  img_src = load_image_b64(style.get("image", "")) if style.get("image") else None
  if img_src:
      img_html = (
          f'<img src="{img_src}" style="width:100%;height:300px;'
          f'object-fit:cover;display:block">'
      )
  else:
      img_html = (
          '<div style="width:100%;height:300px;background:#f4f4f4;'
          'display:flex;align-items:center;justify-content:center;'
          'font-size:28px;color:#ccc">✂</div>'
      )

  tags_html = "".join(
      f'<span style="font-size:10px;padding:2px 7px;border-radius:20px;'
      f'background:#f4f4f4;color:#666;border:.5px solid #e0e0e0;'
      f'margin-right:4px">{t}</span>'
      for t in style.get("tags", [])[:2]
  )

  contribs_html = ""
  for c in style["contributions"][:2]:
      pct = c["percent"] * 100
      contribs_html += f"""
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:5px">
        <span style="font-size:11px;color:#666;flex:0 0 auto;width:90px;
              overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
          {c['desc']}</span>
        <div style="flex:1;height:3px;border-radius:2px;background:#eee">
          <div style="width:{pct:.0f}%;height:100%;background:#378ADD;
                border-radius:2px"></div>
        </div>
        <span style="font-size:11px;font-weight:500;color:#111;
              width:28px;text-align:right">{pct:.0f}%</span>
      </div>"""

  neg_html = ""
  if style.get("negatives"):
      c = style["negatives"][0]
      neg_html = f"""
      <div style="font-size:11px;color:#888;padding:5px 8px;
            background:#f9f9f9;border-radius:6px;
            border-left:2px solid #ddd;margin-top:6px">
        ⚠ {c['reason']}
      </div>"""

  html = f"""
  <style>
    .sc{{font-family:sans-serif;background:#fff;border-radius:12px;
          overflow:hidden;{border};margin-bottom:12px}}
    .sc-img-wrap{{position:relative}}
    .sc-score{{position:absolute;top:8px;right:8px;width:30px;height:30px;
                border-radius:50%;background:#fff;border:.5px solid #e0e0e0;
                display:flex;align-items:center;justify-content:center;
                font-size:11px;font-weight:500;color:#111}}
    .sc-body{{padding:12px}}
    .sc-name{{font-size:14px;font-weight:500;color:#111;margin:0 0 6px}}
  </style>
  <div class="sc">
    <div class="sc-img-wrap">
      {img_html}
      {badge}
      <div class="sc-score">{score}</div>
    </div>
    <div class="sc-body">
      <p class="sc-name">{style['name']}</p>
      <div style="margin-bottom:8px">{tags_html}</div>
      {contribs_html}
      {neg_html}
    </div>
  </div>"""

  h = 450 if style.get("negatives") else 420
  components.html(html, height=h)