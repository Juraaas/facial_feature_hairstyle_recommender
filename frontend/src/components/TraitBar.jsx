import { useTranslation } from 'react-i18next'

export function TraitBar({ title, value, minVal, maxVal, avgVal, minLabel, maxLabel }) {
    const { t } = useTranslation()
    const clamp = v => Math.min(100, Math.max(0, (v - minVal) / (maxVal - minVal) * 100))

    const pct = clamp(value)
    const avgPct = clamp(avgVal)
    const outOfRange = value < minVal ? 'below' : value > maxVal ? 'above' : null

    let dotPct, dotColor
    if (outOfRange === 'below') {
        dotPct = 2;  dotColor = 'var(--text-hint)'
    } else if (outOfRange === 'above') {
        dotPct = 98; dotColor = 'var(--text-hint)'
    } else {
        dotPct = pct; dotColor = 'var(--accent)'
    }

    const interp = outOfRange
        ? (outOfRange === 'below' ? t('trait_below') : t('trait_above'))
        : pct < 33 ? t('trait_low')
        : pct > 67 ? t('trait_high')
        : t('trait_average')

    return (
        <div style={{ marginBottom: 20 }}>
            {/* header row */}
            <div style={{
                display: 'flex', justifyContent: 'space-between',
                alignItems: 'baseline', marginBottom: 7,
            }}>
                <span style={{
                    fontSize: 12, fontWeight: 500,
                    color: 'var(--text)', letterSpacing: '.01em',
                }}>{title}</span>
                <span style={{
                    fontSize: 11, fontFamily: 'var(--font-mono)',
                    color: 'var(--text-muted)',
                }}>
                    <span style={{ color: 'var(--text)', marginRight: 6 }}>
                        {value.toFixed(3)}
                    </span>
                    <span style={{ color: 'var(--text-hint)' }}>→ {interp}</span>
                </span>
            </div>

            {/* track */}
            <div style={{
                position: 'relative', height: 4,
                borderRadius: 2, background: 'var(--surface-2)',
                border: '1px solid var(--border)',
            }}>
                {/* fill */}
                <div style={{
                    position: 'absolute', left: 0, top: 0, height: '100%',
                    width: `${pct}%`, borderRadius: 2,
                    background: outOfRange ? 'var(--border)' : 'var(--accent)',
                    opacity: 0.35,
                }} />

                {/* avg marker */}
                <div style={{
                    position: 'absolute', top: -4, width: 1,
                    height: 12, background: 'var(--border)',
                    transform: 'translateX(-50%)', left: `${avgPct}%`,
                }} />

                {/* value dot */}
                <div style={{
                    position: 'absolute', top: -5,
                    width: 14, height: 14, borderRadius: '50%',
                    background: dotColor,
                    border: '2px solid var(--bg)',
                    transform: 'translateX(-50%)',
                    left: `${dotPct}%`,
                    boxShadow: '0 1px 3px rgba(0,0,0,.15)',
                }} />
            </div>

            {/* min / max labels */}
            <div style={{
                display: 'flex', justifyContent: 'space-between',
                marginTop: 5,
            }}>
                <span style={{
                    fontSize: 10, color: 'var(--text-hint)',
                    fontFamily: 'var(--font-mono)',
                }}>{minLabel}</span>
                <span style={{
                    fontSize: 10, color: 'var(--text-hint)',
                    fontFamily: 'var(--font-mono)',
                }}>{maxLabel}</span>
            </div>
        </div>
    )
}