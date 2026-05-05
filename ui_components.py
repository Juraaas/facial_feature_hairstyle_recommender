import streamlit.components.v1 as components

def trait_bar(title, value, min_val, max_val, avg_val,
              min_label, max_label, min_sub="", max_sub="", height=150):
    def pct(v):
        return min(100, max(0, (v - min_val) / (max_val - min_val) * 100))

    user_pct = pct(value)
    avg_pct = pct(avg_val)

    html = f"""
    <style>
      .tbw {{ font-family: sans-serif; padding: 0.5rem 0 0; }}
      .tb-title {{ font-size: 13px; font-weight: 500; color: #eaeaea; margin-bottom: 8px; }}
      .tb-labels {{ display: flex; justify-content: space-between; margin-bottom: 5px; }}
      .tb-label {{ font-size: 12px; color: #666; max-width: 130px; }}
      .tb-label.r {{ text-align: right; }}
      .tb-label span {{ display: block; font-size: 11px; color: #999; margin-top: 2px; }}
      .tb-track {{ position: relative; height: 10px; border-radius: 5px;
                   background: #eee; border: 0.5px solid #ddd; margin-top: 4px; }}
      .tb-fill {{ position: absolute; left: 0; top: 0; height: 100%; border-radius: 5px;
                  background: linear-gradient(to right, #B5D4F4, #185FA5); }}
      .tb-avg {{ position: absolute; top: -4px; width: 2px; height: 18px;
                 background: #888; border-radius: 1px; transform: translateX(-50%); }}
      .tb-avg-lbl {{ position: absolute; top: 17px; transform: translateX(-50%);
                     font-size: 10px; color: #777; white-space: nowrap; }}
      .tb-dot {{ position: absolute; top: -5px; width: 20px; height: 20px; border-radius: 50%;
                 background: #185FA5; border: 2px solid #fff; transform: translateX(-50%); }}
      .tb-dot-lbl {{ position: absolute; top: -24px; transform: translateX(-50%);
                     font-size: 11px; font-weight: 600; color: #185FA5; white-space: nowrap; }}
    </style>
    <div class="tbw">
      <div class="tb-title">{title}</div>
      <div class="tb-labels">
        <div class="tb-label">{min_label}<span>{min_sub}</span></div>
        <div class="tb-label r">{max_label}<span>{max_sub}</span></div>
      </div>
      <div class="tb-track">
        <div class="tb-dot-lbl" style="left:{user_pct:.1f}%">you: {value:.2f}</div>
        <div class="tb-fill" style="width:{user_pct:.1f}%"></div>
        <div class="tb-avg" style="left:{avg_pct:.1f}%">
          <div class="tb-avg-lbl">avg {avg_val:.2f}</div>
        </div>
        <div class="tb-dot" style="left:{user_pct:.1f}%"></div>
      </div>
    </div>
    """
    components.html(html, height=height)