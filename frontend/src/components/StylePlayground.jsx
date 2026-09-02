import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { supabase } from '../lib/supabase'

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
      const { data } = await supabase.auth.getSession()
      const token = data.session?.access_token
      
      const form = new FormData()
      form.append('file', originalFile)
      form.append('style_name', selectedStyle)
      form.append('color_id', selectedColor)

      const res = await fetch(`${BASE}/style-preview`, {
        method: 'POST', 
        body: form,
        headers: token ? { Authorization: `Bearer ${token}` } : {},
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
      <div onClick={onClose} style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,.6)',
        backdropFilter: 'blur(4px)', zIndex: 200,
      }} />

      <div style={{
        position: 'fixed', top: '50%', left: '50%',transform: 'translate(-50%, -50%)',
        zIndex: 201, width: '100%', maxWidth: 760, maxHeight: '90vh',
        background: 'var(--surface)', borderRadius: 'var(--radius-lg)',
        border: '1px solid var(--border)', boxShadow: '0 32px 80px rgba(0,0,0,.3)',
        overflow: 'hidden', display:'flex',
        flexDirection: 'column', animation: 'modalIn .2s ease',
      }}>

        {/* header */}
        <div style={{
          padding: '18px 24px', borderBottom: '1px solid var(--border)', flexShrink: 0,
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          <div>
            <h2 style={{
              fontFamily: 'var(--font-display)',
              fontSize: 18, fontWeight: 500, color: 'var(--text)',
            }}>
              ✨ {pl ? 'Przymierzalnia' : 'Style Playground'}
            </h2>
            <p style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 300, marginTop: 2 }}>
              {pl ? 'Podgląd fryzury na Twoim zdjęciu' : 'Preview hairstyles on your photo'}
            </p>
          </div>
          <button onClick={onClose} style={{
            background: 'none', border: 'none',
            cursor: 'pointer', fontSize: 20, color: 'var(--text-hint)',
          }}>✕</button>
        </div>

        {/* body */}
        <div style={{
          display: 'grid', gridTemplateColumns: result ? '1fr 1fr' : '280px 1fr',
          flex: 1, overflow: 'hidden', minHeight: 0,
        }}>

          {/* left panel - controls */}
          <div style={{
            padding: '20px', borderRight: '1px solid var(--border)', overflowY: 'auto',
            display: 'flex', flexDirection: 'column', gap: 20,
          }}>

            {/* style */}
            <div>
              <label style={{
                fontSize: 10, fontWeight: 600, letterSpacing: '.08em',
                textTransform: 'uppercase', color: 'var(--text-hint)',
                display: 'block', marginBottom: 8, fontFamily: 'var(--font-body)',
              }}>
                {pl ? 'Fryzura' : 'Hairstyle'}
              </label>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                {styles.slice(0, 8).map(s => (
                  <button key={s.name} onClick={() => { setSelectedStyle(s.name); setResult(null) }}
                    style={{
                      padding: '8px 12px', borderRadius: 'var(--radius-sm)',
                      border: `1px solid ${selectedStyle === s.name ? 'var(--accent)' : 'var(--border)'}`,
                      background: selectedStyle === s.name ? 'var(--accent-soft)' : 'none',
                      color: selectedStyle === s.name ? 'var(--accent)' : 'var(--text)',
                      fontSize: 12, textAlign: 'left', cursor: 'pointer',
                      fontFamily: 'var(--font-body)', transition: 'all .15s',
                      display: 'flex', justifyContent: 'space-between',
                    }}
                  >
                    <span>{s.name}</span>
                    {selectedStyle === s.name && <span>✓</span>}
                  </button>
                ))}
              </div>
            </div>

            {/* hair color */}
            <div>
              <label style={{
                fontSize: 10, fontWeight: 600, letterSpacing: '.08em',
                textTransform: 'uppercase', color: 'var(--text-hint)',
                display: 'block', marginBottom: 8, fontFamily: 'var(--font-body)',
              }}>
                {pl ? 'Kolor włosów' : 'Hair color'}
              </label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                {HAIR_COLORS.map(c => (
                  <button key={c.id} onClick={() => { setSelectedColor(c.id); setResult(null) }}
                    style={{
                      padding: '5px 10px', borderRadius: 20,
                      border: `1px solid ${selectedColor === c.id ? 'var(--accent)' : 'var(--border)'}`,
                      background: selectedColor === c.id ? 'var(--accent-soft)' : 'none',
                      color: selectedColor === c.id ? 'var(--accent)' : 'var(--text-muted)',
                      fontSize: 11, cursor: 'pointer', fontFamily: 'var(--font-body)',
                    }}
                  >
                    {pl ? c.label_pl : c.label_en}
                  </button>
                ))}
              </div>
            </div>

            {/* generate */}
            <button onClick={handleGenerate} disabled={generating} style={{
              background: 'var(--accent)', color: '#fff',
              border: 'none', borderRadius: 'var(--radius-md)',
              padding: '12px', fontSize: 13, fontWeight: 500,
              cursor: generating ? 'wait' : 'pointer',
              fontFamily: 'var(--font-body)',
              display: 'flex', alignItems: 'center',
              justifyContent: 'center', gap: 8,
              opacity: generating ? 0.7 : 1,
              marginTop: 'auto',
            }}>
              {generating ? (
                <>
                  <div style={{
                    width: 14, height: 14, borderRadius: '50%',
                    border: '2px solid rgba(255,255,255,.3)',
                    borderTopColor: '#fff',
                    animation: 'spin .7s linear infinite',
                  }} />
                  {pl ? 'Generowanie' : 'Generating'}
                </>
              ) : (
                `✨ ${pl ? 'Generuj podgląd' : 'Generate preview'}`
              )}
            </button>

            {error && (
              <p style={{
                fontSize: 12, color: '#c0392b', padding: '8px 10px',
                background: '#fef4f2', borderRadius: 'var(--radius-sm)',
                border: '1px solid #f5c6bc',
              }}>{error}</p>
            )}
          </div>

          {/* right panel - before/after */}
          <div style={{
            padding: '20px', display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: 16,
            overflowY: 'auto', background: 'var(--surface-2)',
          }}>
            {!result && !generating && (
              <div style={{ textAlign: 'center', color: 'var(--text-hint)' }}>
                <div style={{ fontSize: 48, marginBottom: 12 }}>✂️</div>
                <p style={{ fontSize: 13, fontWeight: 300 }}>
                  {pl
                    ? 'Wybierz fryzurę i kolor, następnie kliknij Generuj'
                    : 'Select a style and color, then click Generate'}
                </p>
              </div>
            )}

            {generating && (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                <div style={{
                  width: 40, height: 40, borderRadius: '50%',
                  border: '3px solid var(--border)', borderTopColor: 'var(--accent)',
                  animation: 'spin .7s linear infinite', margin: '0 auto 16px',
                }} />
                <p style={{ fontSize: 13, fontWeight: 300 }}>
                  {pl ? 'Generowanie podglądu...' : 'Generating preview...'}
                </p>
                <p style={{ fontSize: 11, color: 'var(--text-hint)', marginTop: 4 }}>
                  {pl ? 'To zajmie około 10 sekund' : 'This takes about 10 seconds'}
                </p>
              </div>
            )}

            {result && (
              <div style={{ width: '100%', animation: 'fadeIn .3s ease' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
                  <div>
                    <img
                      src={URL.createObjectURL(originalFile)}
                      alt="before"
                      style={{
                        width: '100%', borderRadius: 'var(--radius-md)',
                        objectFit: 'cover', height: 280, objectPosition: 'top',
                      }}
                    />
                    <p style={{
                      fontSize: 10, color: 'var(--text-hint)', textAlign: 'center',
                      marginTop: 6, fontFamily: 'var(--font-mono)',
                      textTransform: 'uppercase', letterSpacing: '.06em',
                    }}>
                      {pl ? 'Przed' : 'Before'}
                    </p>
                  </div>
                  <div>
                    <img
                      src={result}
                      alt="after"
                      style={{
                        width: '100%', borderRadius: 'var(--radius-md)',
                        objectFit: 'cover', height: 280, objectPosition: 'top',
                      }}
                    />
                    <p style={{
                      fontSize: 10, color: 'var(--text-hint)', textAlign: 'center',
                      marginTop: 6, fontFamily: 'var(--font-mono)',
                      textTransform: 'uppercase', letterSpacing: '.06em',
                    }}>
                      {pl ? 'Po' : 'After'} — {selectedStyle}
                    </p>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: 8 }}>
                  <a href={result}
                    download={`stylizzer-${selectedStyle.toLowerCase().replace(/ /g, '-')}.jpg`}
                    style={{
                      flex: 1, textAlign: 'center', padding: '9px',
                      background: 'var(--accent)', color: '#fff',
                      borderRadius: 'var(--radius-sm)', fontSize: 12,
                      fontFamily: 'var(--font-body)', textDecoration: 'none',
                      fontWeight: 500,
                    }}
                  >
                    {pl ? '↓ Pobierz' : '↓ Download'}
                  </a>
                  <button onClick={() => setResult(null)} style={{
                    flex: 1, padding: '9px',
                    border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                    background: 'none', color: 'var(--text-muted)',
                    fontSize: 12, cursor: 'pointer', fontFamily: 'var(--font-body)',
                  }}>
                    {pl ? '↺ Generuj ponownie' : '↺ Regenerate'}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}