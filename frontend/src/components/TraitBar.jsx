export function TraitBar({ title, value, minVal, maxVal, avgVal, minLabel, maxLabel }) {
    const clamp = (v) => Math.min(100, Math.max(0, (v - minVal) / (maxVal - minVal) * 100))

    const pct = clamp(value)
    const avgPct = clamp(avgVal)
    const outOfRange = value < minVal ? 'below' : value > maxVal ? 'above' : null

    let dotPct, dotLabel, dotColor
    if (outOfRange === 'below') {
        dotPct = 2; dotLabel = `<${minVal.toFixed(2)}`; dotColor = '#888'
    } else if (outOfRange === 'above') {
        dotPct = 98; dotLabel = `>${maxVal.toFixed(2)}`; dotColor = '#888'
    } else {
        dotPct = pct; dotLabel = value.toFixed(3); dotColor = '#185FA5'
    }

    const interp = outOfRange
    ? (outOfRange === 'below' ? `notably ${minLabel.toLowerCase()}` : `notably ${maxLabel.toLowerCase()}`)
    : pct < 33 ? `tends to ${minLabel.toLowerCase()}`
    : pct > 67 ? `tends to ${maxLabel.toLowerCase()}`
    : 'average'

    return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
        <span style={{ fontSize: 13, fontWeight: 500 }}>{title}</span>
        <span style={{ fontSize: 12, color: '#666' }}>
          {dotLabel}
          <span style={{ fontSize: 11, color: '#999', marginLeft: 4 }}>→ {interp}</span>
        </span>
      </div>
      <div style={{ position: 'relative', height: 6, borderRadius: 3, background: '#eee' }}>
        <div style={{
          position: 'absolute', left: 0, top: 0, height: '100%',
          width: `${pct}%`, borderRadius: 3, background: '#378ADD'
        }} />
        <div style={{
          position: 'absolute', top: -4, width: 1.5, height: 14,
          background: '#bbb', transform: 'translateX(-50%)',
          left: `${avgPct}%`
        }} />
        <div style={{
          position: 'absolute', top: -5, width: 16, height: 16,
          borderRadius: '50%', background: dotColor,
          border: '2.5px solid #fff', transform: 'translateX(-50%)',
          left: `${dotPct}%`
        }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
        <span style={{ fontSize: 11, color: '#aaa' }}>{minLabel} ≥{minVal.toFixed(2)}</span>
        <span style={{ fontSize: 11, color: '#aaa' }}>{maxLabel} ≤{maxVal.toFixed(2)}</span>
      </div>
    </div>
  )
}