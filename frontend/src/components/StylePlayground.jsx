import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { X, Scissors, WandSparkles } from 'lucide-react'
import { supabase } from '../lib/supabase'

const BASE = import.meta.env.VITE_API_URL

const HAIR_COLORS = [
  { id: 'natural', label_pl: 'Naturalny', label_en: 'Natural', prompt: 'keep the natural hair color' },
  { id: 'blonde', label_pl: 'Blond', label_en: 'Blonde', prompt: 'platinum blonde hair color' },
  { id: 'dark', label_pl: 'Ciemny brąz', label_en: 'Dark brown', prompt: 'dark brown hair color' },
  { id: 'black', label_pl: 'Czarny', label_en: 'Black', prompt: 'jet black hair color' },
  { id: 'auburn', label_pl: 'Rudy', label_en: 'Auburn', prompt: 'auburn red hair color' },
]

export function StylePlayground({ styles, originalFile, onClose, isPremium, onUpgrade, gender, analysisResult }) {
  const { i18n } = useTranslation()
  const pl = i18n.language === 'pl'

  const [selectedStyle, setSelectedStyle] = useState(styles[0]?.name || '')
  const [selectedColor, setSelectedColor] = useState('natural')
  const [generating, setGenerating] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [transformation, setTransformation] = useState(null)
  const [previewRating, setPreviewRating] = useState(null)
  const [origUrl, setOrigUrl] = useState(null)

  useEffect(() => {
    if (originalFile) setOrigUrl(URL.createObjectURL(originalFile))
    return () => { if (origUrl) URL.revokeObjectURL(origUrl) }
  }, [originalFile])

  async function handleGenerate() {
    if (!isPremium) { onUpgrade(); return }
    if (!originalFile) return

    setGenerating(true); setError(null); setResult(null);
    setTransformation(null); setPreviewRating(null)

    try {
      const { data } = await supabase.auth.getSession()
      const token = data.session?.access_token
      const form = new FormData()
      form.append('file', originalFile)
      form.append('style_name', selectedStyle)
      form.append('color_id', selectedColor)
      form.append('gender', gender)
      form.append('lang', i18n.language)
      form.append('hair_type', analysisResult?.traits?.hair_type ?? '')
      form.append('hair_coverage', String(analysisResult?.quality?.hair_coverage ?? 0.05))

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
    } catch (e) {
      setError(pl ? 'Generowanie nie powiodło się. Spróbuj ponownie.' : 'Generation failed. Please try again.')
    } finally { setGenerating(false) }
  }

  async function handlePreviewRating(v) {
    setPreviewRating(v)
    try {
      const { data } = await supabase.auth.getSession()
      const token    = data.session?.access_token
      await fetch(`${BASE}/preview-rating`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json',
                   ...(token ? { Authorization: `Bearer ${token}` } : {}) },
        body: JSON.stringify({ style_name: selectedStyle,
                               color_id: selectedColor, gender, rating: v }),
      })
    } catch {}
  }


  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 300,
      background: 'var(--bg)', display: 'flex', flexDirection: 'column',
    }}>
      {/* ── header ── */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 24px', height: 56, flexShrink: 0,
        borderBottom: '1px solid var(--border)', background: 'var(--surface)',
      }}>
        <h2 style={{
          fontFamily: 'var(--font-display)', fontSize: 17,
          fontWeight: 500, color: 'var(--text)',
          display: 'flex', alignItems: 'center', gap: 8,
        }}>
          <WandSparkles size={16} color="var(--accent)" strokeWidth={1.5} />
          {pl ? 'Przymierzalnia' : 'Style Playground'}
        </h2>
        <button onClick={onClose} style={{
          background: 'none', border: 'none', cursor: 'pointer',
          color: 'var(--text-hint)', display: 'flex', padding: 4,
        }}>
          <X size={20} strokeWidth={1.5} />
        </button>
      </div>

      {/* ── body ── */}
      <div className="pg-body" style={{flex: 1, display: 'grid', minHeight: 0, gridTemplateColumns: '280px 1fr'}}>
        {/* ── left panel ── */}
        <div className="pg-left" style={{
          borderRight: '1px solid var(--border)', overflowY: 'auto', padding: '20px 16px',
          display: 'flex', flexDirection: 'column', gap: 20, background: 'var(--surface)',
        }}>
          {/* style list */}
          <div>
            <p style={{
              fontSize: 9, fontWeight: 600, letterSpacing: '.08em',
              textTransform: 'uppercase', color: 'var(--text-hint)',
              marginBottom: 10, fontFamily: 'var(--font-body)',
            }}>
              {pl ? 'Fryzura' : 'Hairstyle'}
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {styles.map(s => (
                <button key={s.name}
                  onClick={() => { setSelectedStyle(s.name); setResult(null); setTransformation(null) }}
                  style={{
                    padding: '8px 10px', borderRadius: 8, textAlign:  'left', border: 'none',
                    background: selectedStyle === s.name ? 'var(--accent-soft)' : 'transparent',
                    color: selectedStyle === s.name ? 'var(--accent)' : 'var(--text-muted)',
                    fontSize: 13, cursor: 'pointer', fontFamily: 'var(--font-body)',
                    fontWeight: selectedStyle === s.name ? 500 : 300,
                    transition: 'background .15s, color .15s',
                  }}>
                  {s.name}
                </button>
              ))}
            </div>
          </div>

          {/* color */}
          <div>
            <p style={{
              fontSize: 9, fontWeight: 600, letterSpacing: '.08em',
              textTransform: 'uppercase', color: 'var(--text-hint)',
              marginBottom: 10, fontFamily: 'var(--font-body)',
            }}>
              {pl ? 'Kolor włosów' : 'Hair color'}
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {HAIR_COLORS.map(c => (
                <button key={c.id}
                  onClick={() => { setSelectedColor(c.id); setResult(null) }}
                  style={{
                    padding: '5px 12px', borderRadius: 20, fontSize: 12,
                    border: `1px solid ${selectedColor === c.id ? 'var(--accent)' : 'var(--border)'}`,
                    background: selectedColor === c.id ? 'var(--accent-soft)' : 'none',
                    color: selectedColor === c.id ? 'var(--accent)' : 'var(--text-muted)',
                    cursor: 'pointer', fontFamily: 'var(--font-body)', transition: 'all .15s',
                  }}>
                  {pl ? c.label_pl : c.label_en}
                </button>
              ))}
            </div>
          </div>

          {/* generate */}
          <button onClick={handleGenerate} disabled={generating} style={{
            padding: '12px', borderRadius: 10,
            background: generating ? 'var(--border)' : 'var(--accent)',
            color: '#fff', border: 'none', cursor: generating ? 'wait' : 'pointer',
            fontSize: 14, fontWeight: 500, fontFamily: 'var(--font-body)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
            marginTop: 'auto', transition: 'background .2s',
          }}>
            {generating ? (
              <>
                <div style={{
                  width: 14, height: 14, borderRadius: '50%',
                  border: '2px solid rgba(255,255,255,.3)',
                  borderTopColor: '#fff', animation: 'spin .7s linear infinite',
                }} />
                {pl ? 'Generowanie…' : 'Generating…'}
              </>
            ) : `✨ ${pl ? 'Generuj podgląd' : 'Generate preview'}`}
          </button>

          {error && (
            <p style={{
              fontSize: 12, color: '#c0392b', padding: '8px 10px',
              background: '#fef4f2', borderRadius: 8, border: '1px solid #f5c6bc',
            }}>{error}</p>
          )}
        </div>

        {/* ── right panel ── */}
        <div style={{
          overflowY: 'auto', padding: '24px', background: 'var(--surface-2)',
          display: 'flex', flexDirection: 'column', gap: 20, alignItems: 'center',
        }}>

          {/* empty state */}
          {!result && !generating && (
            <div style={{
              flex: 1, display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center',
              color: 'var(--text-hint)', textAlign: 'center',
              padding: 40,
            }}>
              <Scissors size={48} color="var(--border)" strokeWidth={1}
                style={{ marginBottom: 16 }} />
              <p style={{ fontSize: 14, fontWeight: 300, maxWidth: 240 }}>
                {pl ? 'Wybierz fryzurę i kolor, następnie kliknij Generuj'
                    : 'Choose a style and color, then click Generate'}
              </p>
            </div>
          )}

          {/* generating */}
          {generating && (
            <div style={{
              flex: 1, display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center', gap: 16,
            }}>
              <div style={{
                width: 52, height: 52, borderRadius: '50%',
                border: '3px solid var(--border)', borderTopColor: 'var(--accent)',
                animation: 'spin .7s linear infinite',
              }} />
              <p style={{ fontSize: 14, color: 'var(--text-muted)', fontWeight: 300 }}>
                {pl ? 'Generowanie podglądu (~15s)…' : 'Generating preview (~15s)…'}
              </p>
            </div>
          )}

          {/* result */}
          {result && (
            <div style={{ width: '100%', maxWidth: 960, animation: 'fadeIn .3s ease' }}>

              {/* before / after */}
              <div className="pg-imgs" style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16}}>
                {[
                  { src: origUrl, label: pl ? 'Przed' : 'Before' },
                  { src: result,  label: `${pl ? 'Po' : 'After'} · ${selectedStyle}` },
                ].map(({ src, label }) => (
                  <div key={label}>
                    <img src={src} alt={label} style={{
                      width: '100%', aspectRatio: '1 / 1', objectFit: 'cover',
                      objectPosition: 'top', borderRadius: 12,
                      border: '1px solid var(--border)', display: 'block',
                    }} />
                    <p style={{
                      fontSize: 10, color: 'var(--text-hint)',
                      textAlign: 'center', marginTop: 6,
                      fontFamily: 'var(--font-mono)',
                      textTransform: 'uppercase', letterSpacing: '.06em',
                    }}>{label}</p>
                  </div>
                ))}
              </div>

              {/* actions */}
              <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
                <a href={result}
                  download={`stylizzer-${selectedStyle.toLowerCase().replace(/ /g,'-')}.jpg`}
                  style={{
                    flex: 1, textAlign: 'center', padding: '11px',
                    background: 'var(--accent)', color: '#fff',
                    borderRadius: 8, textDecoration: 'none',
                    fontSize: 13, fontFamily: 'var(--font-body)', fontWeight: 500,
                  }}>
                  {pl ? '↓ Pobierz' : '↓ Download'}
                </a>
                <button
                  onClick={() => { setResult(null); setTransformation(null); setPreviewRating(null) }}
                  style={{
                    flex: 1, padding: '11px',
                    border: '1px solid var(--border)', borderRadius: 8,
                    background: 'none', color: 'var(--text-muted)',
                    fontSize: 13, cursor: 'pointer', fontFamily: 'var(--font-body)',
                  }}>
                  {pl ? '↺ Generuj ponownie' : '↺ Regenerate'}
                </button>
              </div>

              {/* feedback */}
              {!previewRating ? (
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 10,
                  padding: '10px 14px', background: 'var(--surface)',
                  borderRadius: 8, border: '1px solid var(--border)',
                  marginBottom: 16,
                }}>
                  <span style={{ fontSize: 12, color: 'var(--text-muted)',
                    fontWeight: 300, flex: 1 }}>
                    {pl ? 'Jak wygląda wynik?' : 'How does it look?'}
                  </span>
                  {[
                    { v: 'good', label: pl ? 'Dobrze' : 'Good',
                      s: { border: '1px solid #b7dfc7', color: '#2d8f4e' } },
                    { v: 'bad',  label: pl ? 'Słabo' : 'Bad',
                      s: { border: '1px solid #f5c6bc', color: '#c0392b' } },
                  ].map(({ v, label, s }) => (
                    <button key={v} onClick={() => handlePreviewRating(v)}
                      style={{
                        ...s, padding: '4px 12px', borderRadius: 20,
                        fontSize: 11, cursor: 'pointer',
                        background: 'var(--surface)', fontFamily: 'var(--font-body)',
                      }}>
                      {label}
                    </button>
                  ))}
                </div>
              ) : (
                <p style={{ fontSize: 11, color: 'var(--text-hint)',
                  textAlign: 'center', marginBottom: 16 }}>
                  {pl ? 'Dziękujemy za ocenę!' : 'Thanks for the feedback!'}
                </p>
              )}

              {/* transformation */}
              {transformation && (
                <div style={{
                  padding: '18px 20px', background: 'var(--surface)',
                  borderRadius: 12, border: '1px solid var(--border)',
                }}>
                  <p style={{
                    fontSize: 9, fontWeight: 600, letterSpacing: '.08em',
                    textTransform: 'uppercase', color: 'var(--text-hint)',
                    marginBottom: 14, fontFamily: 'var(--font-body)',
                  }}>
                    {pl ? 'Szacowana transformacja' : 'Estimated transformation'}
                  </p>

                  <div style={{ display: 'flex', justifyContent: 'center',
                    gap: 32, marginBottom: 14 }}>
                    {[
                      { val: transformation.visits, label: pl ? 'wizyty' : 'visits' },
                      { val: transformation.months, label: pl ? 'miesięcy' : 'months' },
                    ].map(item => (
                      <div key={item.label} style={{ textAlign: 'center' }}>
                        <p style={{ fontFamily: 'var(--font-mono)', fontSize: 32,
                          fontWeight: 700, color: 'var(--accent)', lineHeight: 1 }}>
                          {item.val}
                        </p>
                        <p style={{ fontSize: 11, color: 'var(--text-muted)',
                          marginTop: 4, fontWeight: 300 }}>{item.label}</p>
                      </div>
                    ))}
                    <div style={{ textAlign: 'center' }}>
                      <span style={{
                        display: 'inline-block', padding: '5px 14px',
                        borderRadius: 20, fontSize: 11, fontWeight: 500,
                        background: { easy:'#f0faf4', moderate:'#fef8f0',
                          challenging:'#fef4f2' }[transformation.difficulty],
                        color: { easy:'#2d8f4e', moderate:'#C8975A',
                          challenging:'#c0392b' }[transformation.difficulty],
                        border: `1px solid ${{ easy:'#b7dfc7', moderate:'#f0d5a8',
                          challenging:'#f5c6bc' }[transformation.difficulty]}`,
                      }}>
                        {{ easy: pl?'Łatwa':'Easy',
                           moderate: pl?'Umiarkowana':'Moderate',
                           challenging: pl?'Wymagająca':'Challenging',
                        }[transformation.difficulty]}
                      </span>
                      <p style={{ fontSize: 10, color: 'var(--text-hint)', marginTop: 4 }}>
                        {pl ? 'trudność' : 'difficulty'}
                      </p>
                    </div>
                  </div>

                  <div style={{ borderTop: '1px solid var(--border)', paddingTop: 12 }}>
                    {[
                      { key: pl ? 'note_pl' : 'note_en',   label: null },
                      { key: 'recommendation_pl', key_en: 'recommendation_en', label: pl?'Rekomendacja':'Recommendation'},
                      { key: 'first_step_pl', key_en: 'first_step_en', label: pl?'Pierwszy krok':'First step'},
                      { key: 'maintenance_pl', key_en: 'maintenance_en', label: pl?'Pielęgnacja':'Maintenance'},
                    ].filter(f => transformation[pl ? f.key_pl : f.key_en]).map(f => {
                      const key = pl ? f.key_pl : f.key_en
                      return (
                        <div key={f.key} style={{ marginBottom: 10 }}>
                          {f.label && (
                            <p style={{ fontSize: 9, fontWeight: 600,
                              color: 'var(--accent)', textTransform: 'uppercase',
                              letterSpacing: '.06em', marginBottom: 3,
                              fontFamily: 'var(--font-body)',
                            }}>{f.label}</p>
                          )}
                          <p style={{ fontSize: 12, color: 'var(--text-muted)',
                            fontWeight: 300, lineHeight: 1.65 }}>
                            {transformation[f.key]}
                          </p>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* CSS */}
      <style>{`
        @media (max-width: 640px) {
          .pg-body { grid-template-columns: 1fr !important; }
          .pg-left { border-right: none !important;
                     border-bottom: 1px solid var(--border); }
          .pg-imgs { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  )
}