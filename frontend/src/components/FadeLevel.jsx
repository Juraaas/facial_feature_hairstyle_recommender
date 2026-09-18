import { useTranslation } from 'react-i18next'
import { RulerDimensionLine } from 'lucide-react'

const FADE_PL = {
  low: 'Niski',
  mid: 'Średni',
  high: 'Wysoki',
  skin: 'Skin',
}

const FADE_DESC_PL = {
  low: 'delikatne przejście przy karku',
  mid: 'klasyczne przejście na wysokości ucha',
  high: 'wyraźne przejście powyżej uszu',
  skin: 'przejście do gołej skóry',
}

const FADE_DESC_EN = {
  low: 'subtle taper at the neckline',
  mid: 'classic blend at ear level',
  high: 'defined fade above the ears',
  skin: 'tapers down to bare skin',
}

const LEVELS = ['low', 'mid', 'high', 'skin']

export function FadeLevel({ recommendation }) {
  const { i18n } = useTranslation()
  const pl = i18n.language === 'pl'
  const lvl = recommendation?.level ?? 'mid'
  const idx = LEVELS.indexOf(lvl)

  return (
    <section style={{ marginBottom: 32 }}>
      <h2 className="section-title" style={{
        display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16,
      }}>
        <RulerDimensionLine size={18} color="var(--accent)" strokeWidth={1.5} />
        {pl ? 'Rekomendowany fade' : 'Recommended fade'}
      </h2>

      <div style={{
        background: 'var(--surface)', borderRadius: 'var(--radius-lg)',
        border: '1px solid var(--border)', padding: '20px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 18 }}>
          <div style={{
            background: 'var(--accent-soft)', border: '1.5px solid var(--accent)',
            borderRadius: 'var(--radius-md)', padding: '8px 14px',
            flexShrink: 0, textAlign: 'center', minWidth: 72,
          }}>
            <p style={{
              fontFamily: 'var(--font-display)', fontSize: 15, fontWeight: 500,
              color: 'var(--accent)', lineHeight: 1.1,
            }}>
              {pl ? FADE_PL[lvl] : lvl.charAt(0).toUpperCase() + lvl.slice(1)}
            </p>
            <p style={{
              fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--accent)',
              letterSpacing: '.06em', opacity: 0.7, marginTop: 2, textTransform: 'uppercase',
            }}>
              fade
            </p>
          </div>

          <p style={{
            fontSize: 13, color: 'var(--text-muted)', fontWeight: 300,
            lineHeight: 1.6, flex: 1,
          }}>
            {pl
              ? recommendation?.reason_pl
              : recommendation?.reason_en}
          </p>
        </div>

        <div>
          <div style={{ display: 'flex', gap: 4, marginBottom: 6 }}>
            {LEVELS.map((l, i) => (
              <div key={l} style={{
                flex: 1, height: 6, borderRadius: 3,
                background: i <= idx ? 'var(--accent)' : 'var(--border)',
                opacity: i <= idx ? (0.4 + i * 0.2) : 1, transition: 'background .2s',
              }} />
            ))}
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            {LEVELS.map(l => (
              <span key={l} style={{
                fontSize: 9, fontFamily: 'var(--font-mono)', letterSpacing: '.04em',
                textTransform: 'uppercase',
                color: l === lvl ? 'var(--accent)' : 'var(--text-hint)',
                fontWeight: l === lvl ? 600 : 400,
              }}>
                {pl ? FADE_PL[l] : l}
              </span>
            ))}
          </div>

          <p style={{
            fontSize: 11, color: 'var(--text-hint)', fontWeight: 300,
            marginTop: 8, fontFamily: 'var(--font-mono)', textAlign: 'center',
          }}>
            {pl ? FADE_DESC_PL[lvl] : FADE_DESC_EN[lvl]}
          </p>
        </div>
      </div>
    </section>
  )
}