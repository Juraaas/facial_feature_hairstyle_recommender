import { useTranslation } from 'react-i18next'
import { useState } from 'react'
import { Palette, ChevronDown } from 'lucide-react'

export function HairColorSuggestions({ recommendation }) {
  const { i18n } = useTranslation()
  const pl = i18n.language === 'pl'
  const [open, setOpen] = useState(false)

  if (!recommendation) return null

  const undertoneLabel = {
    warm: { pl: 'ciepły',    en: 'warm'    },
    cool: { pl: 'chłodny',   en: 'cool'    },
    neutral: { pl: 'neutralny', en: 'neutral' },
  }[recommendation.undertone] ?? { pl: 'neutralny', en: 'neutral' }

  const lowConfidence = recommendation.confidence < 0.5

  return (
    <section style={{ marginBottom: 32 }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', display: 'flex', alignItems: 'center',
          justifyContent: 'space-between', background: 'none', border: 'none',
          cursor: 'pointer', padding: 0, marginBottom: open ? 12 : 0,
        }}
      >
        <h2 className='section-title' style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Palette size={18} color="var(--accent)" strokeWidth={1.5} />
          {pl ? 'Sugestie koloru włosów' : 'Hair color suggestions'}
        </h2>
        <ChevronDown size={16} color="var(--text-hint)" style={{
            transform:  open ? 'rotate(180deg)' : 'none', transition: 'transform .2s' }}/>
      </button>

      {open && (
        <div style={{ animation: 'fadeIn .2s ease' }}>
          {/* disclaimer */}
          <div style={{
            fontSize: 11, color: 'var(--text-hint)', fontWeight: 300,
            lineHeight: 1.6, marginBottom: 14, padding: '8px 12px',
            background: 'var(--surface-2)', borderRadius: 'var(--radius-sm)',
            borderLeft:   '2px solid var(--border)',
          }}>
            {pl
              ? 'Wynik zależy od oświetlenia. Dla najdokładniejszego rezultatu użyj zdjęcia w naturalnym świetle dziennym.'
              : 'Result depends on lighting. For best accuracy use a photo in natural daylight.'}
          </div>

          <div style={{
            background: 'var(--surface)', borderRadius: 'var(--radius-lg)',
            border: '1px solid var(--border)', padding: '16px 20px',
          }}>
            {/* undertone badge */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16,
            }}>
              <span style={{
                fontSize: 10, padding: '3px 10px', borderRadius: 20,
                background: 'var(--surface-2)', border: '1px solid var(--border)',
                color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', letterSpacing: '.04em',
              }}>
                {pl ? 'odcień skóry: ' : 'skin undertone: '}
                <strong style={{ color: 'var(--text)' }}>
                  {pl ? undertoneLabel.pl : undertoneLabel.en}
                </strong>
              </span>
              {lowConfidence && (
                <span style={{
                  fontSize: 10, color: 'var(--text-hint)', fontWeight: 300, fontStyle: 'italic',
                }}>
                  {pl ? '(niska pewność)' : '(low confidence)'}
                </span>
              )}
            </div>

            {/* color list */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {recommendation.colors.map((c, i) => (
                <div key={i} style={{
                  display: 'flex', alignItems: 'center', gap: 12, padding: '10px 12px',
                  borderRadius: 'var(--radius-md)', background: 'var(--surface-2)',
                  border: '1px solid var(--border)',
                }}>
                  <div style={{
                    width: 36, height: 36, borderRadius: 8, background: c.hex,
                    border: '1px solid rgba(0,0,0,.1)', flexShrink: 0,
                    boxShadow: '0 1px 4px rgba(0,0,0,.15)',
                  }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <p style={{
                      fontSize: 12, fontWeight: 500, color: 'var(--text)', marginBottom: 2,
                    }}>
                      {pl ? c.name_pl : c.name_en}
                    </p>
                    <p style={{
                      fontSize: 11, color: 'var(--text-muted)', fontWeight: 300, lineHeight: 1.5,
                    }}>
                      {pl ? c.desc_pl : c.desc_en}
                    </p>
                  </div>
                  {/* hex code */}
                  <span style={{
                    fontSize: 9, fontFamily: 'var(--font-mono)',
                    color: 'var(--text-hint)', letterSpacing: '.04em', flexShrink: 0,
                  }}>
                    {c.hex}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  )
}