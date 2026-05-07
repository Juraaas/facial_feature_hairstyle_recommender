import streamlit.components.v1 as components

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

    min_sub = f"typical: ≥ {min_val:.2f}"
    max_sub = f"typical: ≤ {max_val:.2f}"

    html = f"""
    <style>
      .tbw{{font-family:sans-serif;padding:0.5rem 0 0}}
      .tb-title{{font-size:13px;font-weight:500;color:#111;margin-bottom:8px}}
      .tb-labels{{display:flex;justify-content:space-between;margin-bottom:4px}}
      .tb-label{{font-size:12px;color:#666;max-width:140px}}
      .tb-label.r{{text-align:right}}
      .tb-sub{{display:block;font-size:11px;color:#999;margin-top:1px}}
      .tb-track{{position:relative;height:10px;border-radius:5px;background:#eee;border:0.5px solid #ddd;margin-top:4px}}
      .tb-fill{{position:absolute;left:0;top:0;height:100%;border-radius:5px;background:linear-gradient(to right,#B5D4F4,#185FA5)}}
      .tb-avg{{position:absolute;top:-4px;width:2px;height:18px;background:#aaa;border-radius:1px;transform:translateX(-50%)}}
      .tb-avg-lbl{{position:absolute;top:17px;transform:translateX(-50%);font-size:10px;color:#999;white-space:nowrap}}
      .tb-dot{{position:absolute;top:-5px;width:20px;height:20px;border-radius:50%;border:2px solid #fff;transform:translateX(-50%)}}
      .tb-dot-lbl{{position:absolute;top:-24px;transform:translateX(-50%);font-size:11px;font-weight:500;white-space:nowrap}}
      .tb-spacer{{height:36px}}
    </style>
    <div class="tbw">
      <div class="tb-title">{title}</div>
      <div class="tb-labels">
        <div class="tb-label">{min_label}<span class="tb-sub">{min_sub}</span></div>
        <div class="tb-label r">{max_label}<span class="tb-sub">{max_sub}</span></div>
      </div>
      <div class="tb-track">
        <div class="tb-dot-lbl" style="left:{dot_pct:.1f}%;color:{dot_color}">{dot_label}</div>
        <div class="tb-fill" style="width:{clamped:.1f}%"></div>
        <div class="tb-avg" style="left:{avg_pct:.1f}%">
          <div class="tb-avg-lbl">avg {avg_val:.2f}</div>
        </div>
        <div class="tb-dot" style="left:{dot_pct:.1f}%;background:{dot_color}"></div>
      </div>
      <div class="tb-spacer"></div>
    </div>"""

    components.html(html, height=height)