import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { X, Scissors } from 'lucide-react'
import { supabase } from '../lib/supabase'

const BASE = import.meta.env.VITE_API_URL

const HAIR_COLORS = [
  { id: 'natural',  label_pl: 'Naturalny',     label_en: 'Natural',      prompt: 'keep the natural hair color' },
  { id: 'blonde',   label_pl: 'Blond',          label_en: 'Blonde',       prompt: 'platinum blonde hair color' },
  { id: 'dark',     label_pl: 'Ciemny brąz',    label_en: 'Dark brown',   prompt: 'dark brown hair color' },
  { id: 'black',    label_pl: 'Czarny',         label_en: 'Black',        prompt: 'jet black hair color' },
  { id: 'auburn',   label_pl: 'Rudy',           label_en: 'Auburn',       prompt: 'auburn red hair color' },
]

export function StylePlayground({ styles, originalFile, onClose, isPremium, onUpgrade, gender }) {
  const { i18n } = useTranslation()
  const pl = i18n.language === 'pl'

  const [selectedStyle, setSelectedStyle] = useState(styles[0]?.name || '')
  const [selectedColor, setSelectedColor] = useState('natural')
  const [generating, setGenerating] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [transformation, setTransformation] = useState(null)

  async function handleGenerate() {
    if (!isPremium) { onUpgrade(); return }
    if (!originalFile) return

    setGenerating(true)
    setError(null)
    setResult(null)
    setTransformation(null)

    try {
      const { data } = await supabase.auth.getSession()
      const token = data.session?.access_token
      
      const form = new FormData()
      form.append('file', originalFile)
      form.append('style_name', selectedStyle)
      form.append('color_id', selectedColor)
      form.append('gender', gender)
      form.append('hair_type', result?.traits?.hair_type ?? '')
      form.append('hair_coverage', result?.quality?.hair_coverage ?? '0.05')

      const res = await fetch(`${BASE}/style-preview`, {
        method: 'POST', 
        body: form,
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      })

      if (!res.ok) throw new Error('Generation failed')
      const json = await res.json()
      const imgSrc = `data:image/jpeg;base64,${json.image_b64}`
      setResult(imgSrc)
      setTransformation(json.transformation ?? null)
      setResult(URL.createObjectURL(blob))
    } catch (e) {
      setError(pl ? 'Generowanie nie powiodło się. Spróbuj ponownie.' : 'Generation failed. Please try again.')
    } finally {
      setGenerating(false)
    }
  }


  return (
    <>
      <div style={{
        position: 'fixed', inset: 0, zIndex: 201,
        background: 'var(--surface)', overflow: 'hidden',
        display: 'flex', flexDirection: 'column', animation: 'fadeIn .2s ease',
      }}>
        {/* header */}
        <div style={{
          padding: '16px 24px', borderBottom: '1px solid var(--border)',
          display: 'flex', justifyContent: 'space-between',
          alignItems: 'center', flexShrink: 0, background: 'var(--surface)',
        }}>
          <h2 style={{
            fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 500, color: 'var(--text)',
          }}>
            ✨ {pl ? 'Przymierzalnia' : 'Style Playground'}
          </h2>
          <button onClick={onClose} style={{
            background: 'none', border: 'none', cursor: 'pointer',
            color: 'var(--text-hint)', display: 'flex',
          }}>
            <X size={20} strokeWidth={1.5} />
          </button>
        </div>

        {/* body two cols */}
        <div className="playground-grid" style={{
          display: 'grid', gridTemplateColumns: '300px 1fr', flex: 1, 
          overflow: 'hidden', minHeight: 0,
        }}>

          {/* left panel controls */}
          <div style={{
            borderRight: '1px solid var(--border)', overflowY: 'auto', padding: '20px',
            display: 'flex', flexDirection: 'column', gap: 20,
          }}>
            {/* style */}
            <div>
              <label style={{ fontSize: 10, fontWeight: 600, letterSpacing: '.08em',
                textTransform: 'uppercase', color: 'var(--text-hint)',
                display: 'block', marginBottom: 8, fontFamily: 'var(--font-body)'}}
                >{pl ? 'Fryzura' : 'Hairstyle'}</label>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                {styles.slice(0, 8).map(s => (
                  <button key={s.name}
                    onClick={() => { setSelectedStyle(s.name); setResult(null); setTransformation(null) }}
                    style={{
                      padding: '8px 12px', borderRadius: 'var(--radius-sm)', textAlign: 'left',
                      border: `1px solid ${selectedStyle === s.name ? 'var(--accent)' : 'var(--border)'}`,
                      background: selectedStyle === s.name ? 'var(--accent-soft)' : 'none',
                      color: selectedStyle === s.name ? 'var(--accent)' : 'var(--text)',
                      fontSize: 12, cursor: 'pointer', fontFamily: 'var(--font-body)',
                      display: 'flex', justifyContent: 'space-between',
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
              <label style={{ fontSize: 10, fontWeight: 600, letterSpacing: '.08em',
                textTransform: 'uppercase', color: 'var(--text-hint)',
                display: 'block', marginBottom: 8, fontFamily: 'var(--font-body)'}}
                >{pl ? 'Kolor włosów' : 'Hair color'}</label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                {HAIR_COLORS.map(c => (
                  <button key={c.id}
                    onClick={() => { setSelectedColor(c.id); setResult(null) }}
                    style={{
                      padding: '5px 10px', borderRadius: 20, fontSize: 11,
                      border: `1px solid ${selectedColor === c.id ? 'var(--accent)' : 'var(--border)'}`,
                      background: selectedColor === c.id ? 'var(--accent-soft)' : 'none',
                      color: selectedColor === c.id ? 'var(--accent)' : 'var(--text-muted)',
                      cursor: 'pointer', fontFamily: 'var(--font-body)',
                    }}
                  >
                    {pl ? c.label_pl : c.label_en}
                  </button>
                ))}
              </div>
            </div>

            {/* generate */}
            <button onClick={handleGenerate} disabled={generating} style={{
              background: 'var(--accent)', color: '#fff', border: 'none',
              borderRadius: 'var(--radius-md)', padding: '13px',
              fontSize: 14, fontWeight: 500, cursor: generating ? 'wait' : 'pointer',
              fontFamily: 'var(--font-body)',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              opacity: generating ? 0.7 : 1, marginTop: 'auto',
            }}>
              {generating ? (
                <>
                  <div style={{
                    width: 14, height: 14, borderRadius: '50%',
                    border: '2px solid rgba(255,255,255,.3)',
                    borderTopColor: '#fff', animation: 'spin .7s linear infinite',
                  }} />
                  {pl ? 'Generowanie (~15s)' : 'Generating (~15s)'}
                </>
              ) : `✨ ${pl ? 'Generuj podgląd' : 'Generate preview'}`}
            </button>

            {error && (
              <p style={{
                fontSize: 12, color: '#c0392b', padding: '8px 10px',
                background: '#fef4f2', borderRadius: 'var(--radius-sm)',
              }}>{error}</p>
            )}
          </div>

          {/* right panel before/after + transformation */}
          <div style={{
            overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', 
            alignItems: 'center', gap: 20, background: 'var(--surface-2)',
          }}>
            {!result && !generating && (
              <div style={{ textAlign: 'center', color: 'var(--text-hint)', marginTop: 60 }}>
                <Scissors size={48} color="var(--border)" strokeWidth={1} style={{ marginBottom: 16 }} />
                <p style={{ fontSize: 14, fontWeight: 300 }}>
                  {pl ? 'Wybierz fryzurę i kolor, kliknij Generuj'
                      : 'Select a style and color, then click Generate'}
                </p>
              </div>
            )}

            {generating && (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)', marginTop: 60 }}>
                <div style={{
                  width: 48, height: 48, borderRadius: '50%',
                  border: '3px solid var(--border)', borderTopColor: 'var(--accent)',
                  animation: 'spin .7s linear infinite', margin: '0 auto 20px',
                }} />
                <p style={{ fontSize: 14, fontWeight: 300 }}>
                  {pl ? 'Generowanie podglądu...' : 'Generating preview...'}
                </p>
              </div>
            )}

            {result && (
              <div style={{ width: '100%', maxWidth: 800, animation: 'fadeIn .3s ease' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
                  {[
                    { src: URL.createObjectURL(originalFile), label: pl ? 'Przed' : 'Before' },
                    { src: result, label: `${pl ? 'Po' : 'After'} — ${selectedStyle}` },
                  ].map(({ src, label }) => (
                    <div key={label}>
                      <img src={src} alt={label} style={{
                        width: '100%', borderRadius: 'var(--radius-lg)',
                        objectFit: 'cover', height: 400, objectPosition: 'top',
                        border: '1px solid var(--border)',
                      }} />
                      <p style={{
                        fontSize: 10, color: 'var(--text-hint)', textAlign: 'center',
                        marginTop: 6, fontFamily: 'var(--font-mono)',
                        textTransform: 'uppercase', letterSpacing: '.06em',
                      }}>
                        {label}
                      </p>
                    </div>
                  ))}
                </div>

                {/* action buttons */}
                <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
                  <a href={result}
                    download={`stylizzer-${selectedStyle.toLowerCase().replace(/ /g, '-')}.jpg`}
                    style={{
                      flex: 1, textAlign: 'center', padding: '10px',
                      background: 'var(--accent)', color: '#fff',
                      borderRadius: 'var(--radius-sm)', fontSize: 13,
                      fontFamily: 'var(--font-body)', textDecoration: 'none', fontWeight: 500,
                    }}>
                    {pl ? '↓ Pobierz' : '↓ Download'}
                  </a>
                  <button onClick={() => { setResult(null); setTransformation(null) }} style={{
                    flex: 1, padding: '10px',
                    border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                    background: 'none', color: 'var(--text-muted)',
                    fontSize: 13, cursor: 'pointer', fontFamily: 'var(--font-body)',
                  }}>
                    {pl ? '↺ Generuj ponownie' : '↺ Regenerate'}
                  </button>
                </div>

                {/* transformation estimate */}
                {transformation && (
                  <div style={{
                    padding: '16px', background: 'var(--surface)',
                    borderRadius: 'var(--radius-lg)', border: '1px solid var(--border)',
                  }}>
                    <p style={{
                      fontSize: 9, fontWeight: 600, letterSpacing: '.08em',
                      textTransform: 'uppercase', color: 'var(--text-hint)',
                      marginBottom: 12, fontFamily: 'var(--font-body)',
                    }}>
                      {pl ? 'Szacowana transformacja u fryzjera' : 'Estimated salon transformation'}
                    </p>

                    <div style={{ display: 'flex', gap: 24, marginBottom: 12 }}>
                      {[
                        { val: transformation.visits, label_pl: 'wizyta/y',   label_en: 'visits'  },
                        { val: transformation.months, label_pl: 'miesięce/y', label_en: 'months'  },
                      ].map(item => (
                        <div key={item.label_en} style={{ textAlign: 'center' }}>
                          <p style={{
                            fontFamily: 'var(--font-mono)', fontSize: 28,
                            fontWeight: 600, color: 'var(--accent)', lineHeight: 1,
                          }}>
                            {item.val}
                          </p>
                          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                            {pl ? item.label_pl : item.label_en}
                          </p>
                        </div>
                      ))}

                      <div style={{ textAlign: 'center' }}>
                        <p style={{
                          fontSize: 13, fontWeight: 600,
                          color: {
                            easy: '#2d8f4e',
                            moderate: '#C8975A',
                            challenging: '#c0392b',
                          }[transformation.difficulty],
                          textTransform: 'capitalize', lineHeight: 1,
                        }}>
                          {{ easy: pl ? 'Łatwa' : 'Easy',
                            moderate: pl ? 'Umiarkowana' : 'Moderate',
                            challenging: pl ? 'Wymagająca' : 'Challenging',
                          }[transformation.difficulty]}
                        </p>
                        <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                          {pl ? 'trudność' : 'difficulty'}
                        </p>
                      </div>
                    </div>

                    <p style={{
                      fontSize: 12, color: 'var(--text-muted)', fontWeight: 300,
                      lineHeight: 1.6, borderTop: '1px solid var(--border)', paddingTop: 10,
                    }}>
                      {pl ? transformation.note_pl : transformation.note_en}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* mobile — stack columns */}
        <style>{`
          @media (max-width: 640px) {
            .playground-grid { grid-template-columns: 1fr !important; }
          }
        `}</style>
      </div>
    </>
  )
}