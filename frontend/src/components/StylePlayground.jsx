import { useState } from 'react'
import { useTranslation } from 'react-i18next'

const BASE = import.meta.env.VITE_API_URL

const HAIR_COLORS = [
  { id: 'natural',  label_pl: 'Naturalny',     label_en: 'Natural',      prompt: 'keep the natural hair color' },
  { id: 'blonde',   label_pl: 'Blond',          label_en: 'Blonde',       prompt: 'platinum blonde hair color' },
  { id: 'dark',     label_pl: 'Ciemny brąz',    label_en: 'Dark brown',   prompt: 'dark brown hair color' },
  { id: 'black',    label_pl: 'Czarny',         label_en: 'Black',        prompt: 'jet black hair color' },
  { id: 'auburn',   label_pl: 'Rudy',           label_en: 'Auburn',       prompt: 'auburn red hair color' },
  { id: 'grey',     label_pl: 'Siwy',           label_en: 'Grey',         prompt: 'silver grey hair color' },
]

export function StylePlayground({ styles, originalFile, onClose, isPremium, onUpgrade }) {
  const { i18n } = useTranslation()
  const pl = i18n.language === 'pl'

  const [selectedStyle, setSelectedStyle] = useState(styles[0]?.name || '')
  const [selectedColor, setSelectedColor] = useState('natural')
  const [generating, setGenerating] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function handleGenerate() {
    if (!isPremium) { onUpgrade(); return }
    if (!originalFile) return

    setGenerating(true)
    setError(null)
    setResult(null)

    try {
      const form = new FormData()
      form.append('file', originalFile)
      form.append('style_name', selectedStyle)
      form.append('color_id', selectedColor)

      const res = await fetch(`${BASE}/style-preview`, {
        method: 'POST', body: form,
      })

      if (!res.ok) throw new Error('Generation failed')

      const blob = await res.blob()
      setResult(URL.createObjectURL(blob))
    } catch (e) {
      setError(pl ? 'Generowanie nie powiodło się. Spróbuj ponownie.' : 'Generation failed. Please try again.')
    } finally {
      setGenerating(false)
    }
  }

  return (
    <>
      {/* overlay */}
      <div onClick={onClose} style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,.4)', zIndex: 200,
      }} />

      {/* drawer */}
      <div style={{
        position: 'fixed', top: 0, right: 0,
        height: '100vh', width: '100%', maxWidth: 420,background: 'var(--surface)',
        borderLeft: '1px solid var(--border)', zIndex: 201, overflowY: 'auto',
        display: 'flex', flexDirection: 'column', animation: 'slideIn .25s ease',
      }}>
        {/* header */}
        <div style={{
          padding: '20px 20px 16px', borderBottom: '1px solid var(--border)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          position: 'sticky', top: 0, background: 'var(--surface)', zIndex: 1,
        }}>
          <div>
            <h2 style={{
              fontFamily: 'var(--font-display)', fontSize: 17, fontWeight: 500,
              color: 'var(--text)', marginBottom: 2,
            }}>
              ✨ {pl ? 'Przymierzalnia' : 'Style Playground'}
            </h2>
            <p style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 300 }}>
              {pl ? 'Podgląd fryzury na Twoim zdjęciu' : 'Preview hairstyles on your photo'}
            </p>
          </div>
          <button onClick={onClose} style={{
            background: 'none', border: 'none',
            cursor: 'pointer', fontSize: 18, color: 'var(--text-hint)',
          }}>✕</button>
        </div>

        {/* content */}
        <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: 20, flex: 1 }}>

          {/* style */}
          <div>
            <label style={{
              fontSize: 10, fontWeight: 600, letterSpacing: '.08em',
              textTransform: 'uppercase', color: 'var(--text-hint)',
              display: 'block', marginBottom: 10, fontFamily: 'var(--font-body)',
            }}>
              {pl ? 'Fryzura' : 'Hairstyle'}
            </label>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {styles.slice(0, 8).map(s => (
                <button
                  key={s.name}
                  onClick={() => setSelectedStyle(s.name)}
                  style={{
                    padding: '9px 12px', borderRadius: 'var(--radius-sm)',
                    border: `1px solid ${selectedStyle === s.name ? 'var(--accent)' : 'var(--border)'}`,
                    background: selectedStyle === s.name ? 'var(--accent-soft)' : 'var(--surface-2)',
                    color: selectedStyle === s.name ? 'var(--accent)' : 'var(--text)',
                    fontSize: 13, textAlign: 'left', cursor: 'pointer',
                    fontFamily: 'var(--font-body)', transition: 'all .15s',
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  }}
                >
                  <span>{s.name}</span>
                  {selectedStyle === s.name && <span style={{ fontSize: 10 }}>✓</span>}
                </button>
              ))}
            </div>
          </div>

          {/* color */}
          <div>
            <label style={{
              fontSize: 10, fontWeight: 600, letterSpacing: '.08em',
              textTransform: 'uppercase', color: 'var(--text-hint)',
              display: 'block', marginBottom: 10, fontFamily: 'var(--font-body)',
            }}>
              {pl ? 'Kolor włosów' : 'Hair color'}
            </label>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {HAIR_COLORS.map(c => (
                <button
                  key={c.id}
                  onClick={() => setSelectedColor(c.id)}
                  style={{
                    padding: '6px 12px', borderRadius: 20,
                    border: `1px solid ${selectedColor === c.id ? 'var(--accent)' : 'var(--border)'}`,
                    background: selectedColor === c.id ? 'var(--accent-soft)' : 'none',
                    color: selectedColor === c.id ? 'var(--accent)' : 'var(--text-muted)',
                    fontSize: 12, cursor: 'pointer', fontFamily: 'var(--font-body)',
                    transition: 'all .15s',
                  }}
                >
                  {pl ? c.label_pl : c.label_en}
                </button>
              ))}
            </div>
          </div>

          {/* generate button */}
          <button
            onClick={handleGenerate}
            disabled={generating}
            style={{
              background: isPremium ? 'var(--accent)' : 'var(--surface-2)',
              color: isPremium ? '#fff' : 'var(--text-muted)',
              border: isPremium ? 'none' : '1px solid var(--border)',
              borderRadius: 'var(--radius-md)', padding: '13px', fontSize: 14,
              fontWeight: 500, cursor: generating ? 'wait' : 'pointer',
              fontFamily: 'var(--font-body)', display: 'flex', alignItems: 'center',
              justifyContent: 'center', gap: 8, opacity: generating ? 0.7 : 1, transition: 'all .15s',
            }}
          >
            {generating ? (
              <>
                <div style={{
                  width: 14, height: 14, borderRadius: '50%',
                  border: '2px solid rgba(255,255,255,.3)',
                  borderTopColor: '#fff', animation: 'spin .7s linear infinite',
                }} />
                {pl ? 'Generowanie...' : 'Generating...'}
              </>
            ) : isPremium ? (
              `✨ ${pl ? 'Generuj podgląd' : 'Generate preview'}`
            ) : (
              `🔒 ${pl ? 'Funkcja premium' : 'Premium feature'}`
            )}
          </button>

          {error && (
            <p style={{
              fontSize: 12, color: '#c0392b',
              padding: '8px 12px', background: '#fef4f2',
              borderRadius: 'var(--radius-sm)', border: '1px solid #f5c6bc',
            }}>{error}</p>
          )}

          {/* result */}
          {result && (
            <div style={{ animation: 'fadeIn .3s ease' }}>
              <label style={{
                fontSize: 10, fontWeight: 600, letterSpacing: '.08em',
                textTransform: 'uppercase', color: 'var(--text-hint)',
                display: 'block', marginBottom: 10, fontFamily: 'var(--font-body)',
              }}>
                {pl ? 'Wynik' : 'Result'}
              </label>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8}}>
                <div>
                  <img src={URL.createObjectURL(originalFile)} alt="before"
                    style={{ width: '100%', borderRadius: 'var(--radius-md)', objectFit: 'cover', height: 200, objectPosition: 'top' }} />
                  <p style={{ fontSize: 10, color: 'var(--text-hint)', textAlign: 'center', marginTop: 4, fontFamily: 'var(--font-mono)' }}>
                    {pl ? 'Przed' : 'Before'}
                  </p>
                </div>
                <div>
                  <img src={result} alt="after"
                    style={{ width: '100%', borderRadius: 'var(--radius-md)', objectFit: 'cover',
                    height: 200, objectPosition: 'top' }} />
                  <p style={{ fontSize: 10, color: 'var(--text-hint)', textAlign: 'center',
                    marginTop: 4, fontFamily: 'var(--font-mono)' }}>
                    {pl ? 'Po' : 'After'}
                  </p>
                </div>
              </div>
              
              <a href={result}
                download={`stylizzer-${selectedStyle.toLowerCase().replace(/ /g, '-')}.jpg`}
                style={{
                  display: 'block', textAlign: 'center', marginTop: 10, fontSize: 12,
                  color: 'var(--accent)', fontFamily: 'var(--font-body)',
                }}
              >
                {pl ? '↓ Pobierz zdjęcie' : '↓ Download photo'}
              </a>
            </div>
          )}
        </div>
      </div>
    </>
  )
}