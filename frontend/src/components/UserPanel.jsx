import { useState, useEffect } from 'react'
import { supabase } from '../lib/supabase'
import { useTranslation } from 'react-i18next'
import { ChevronRight, Sparkles, X } from 'lucide-react'

const BASE = import.meta.env.VITE_API_URL || '/api'

const TRAIT_NAMES_PL = {
  face_ratio: 'Proporcje twarzy',
  jaw_ratio: 'Szerokość szczęki',
  eye_ratio: 'Rozstaw oczu',
  eye_height: 'Otwartość oczu',
  lip_ratio: 'Szerokość ust',
  nose_position: 'Pozycja nosa',
  lower_face_ratio: 'Dolna część twarzy',
  chin_prominence: 'Projekcja brody',
  symmetry: 'Symetria',
  upper_third: 'Czoło',
  middle_third: 'Środkowa część',
  mid_lower_ratio: 'Balans środek/dół',
  hair_type: 'Typ włosów',
  hairline: 'Linia włosów',
  jaw: 'Szczęka',
  eyes: 'Oczy',
  forehead: 'Czoło',
  face_length: 'Długość twarzy',
  facial_thirds: 'Tercje twarzy',
}

const TRAIT_VALUES_PL = {
  narrow: 'wąska',
  wide: 'szeroka',
  high: 'wysoka',
  low: 'niska',
  long: 'długa',
  short: 'krótka',
  prominent: 'wyraźna',
  recessed: 'cofnięta',
  close: 'blisko',
  balanced: 'zbalansowana',
  middle_dominant: 'dominuje środek',
  top_heavy: 'dominuje góra',
  bottom_heavy: 'dominuje dół',
  straight: 'proste',
  wavy: 'falowane',
  curly: 'kręcone',
  coily: 'spiralne',
  receding: 'cofnięta',
  uneven: 'nierówna',
  normal: 'normalna',
}

export function UserPanel({ user, onClose, isPremium, onUpgrade }) {
  const { i18n } = useTranslation()
  const pl = i18n.language === 'pl'
  const [analyses, setAnalyses] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedAnalysis, setSelectedAnalysis] = useState(null)

  useEffect(() => {
    async function load() {
      const { data } = await supabase.auth.getSession()
      const token = data.session?.access_token
      if (!token) { setLoading(false); return }

      try {
        const res = await fetch(`${BASE}/history`, {
          headers: { Authorization: `Bearer ${token}` }
        })
        if (res.ok) {
          const d = await res.json()
          setAnalyses(d.analyses || [])
        }
      } catch (e) {
        console.error('History fetch error:', e)
      }
      setLoading(false)
    }
    load()
  }, [])

  return (
    <>
      <div onClick={onClose} style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,.5)', zIndex: 200,
      }} />

      <div style={{
        position: 'fixed', top: 0, right: 0,
        height: '100vh', width: '100%', maxWidth: 380,
        background: 'var(--surface)', borderLeft: '1px solid var(--border)',
        zIndex: 201, overflowY: 'auto', animation: 'slideIn .25s ease',
        padding: '24px', display: 'flex', flexDirection: 'column',
      }}>

        {/* header */}
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24,
        }}>
          <h2 style={{
            fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 500, color: 'var(--text)',
          }}>
            {pl ? 'Twoje konto' : 'Your account'}
          </h2>
          <button onClick={onClose} style={{
            background: 'none', border: 'none', cursor: 'pointer', 
            color: 'var(--text-hint)', display: 'flex', alignItems: 'center',
          }}>
            <X size={18} strokeWidth={1.5} />
          </button>
        </div>

        {/* plan badge */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '12px 14px', marginBottom: 20,
          background: isPremium ? 'var(--accent-soft)' : 'var(--surface-2)',
          borderRadius: 'var(--radius-md)',
          border: `1px solid ${isPremium ? 'var(--accent)' : 'var(--border)'}`,
        }}>
          <div style={{
            width: 32, height: 32, borderRadius: '50%',
            background: isPremium ? 'var(--accent)' : 'var(--border)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
          }}>
            {isPremium
              ? <Sparkles size={14} color="#fff" strokeWidth={1.5} />
              : <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>
                  {user.email?.[0]?.toUpperCase()}
                </span>
            }
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <p style={{
              fontSize: 12, fontWeight: 500, color: 'var(--text)',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>
              {user.email}
            </p>
            <p style={{
              fontSize: 11, color: isPremium ? 'var(--accent)' : 'var(--text-muted)', fontFamily: 'var(--font-mono)',
            }}>
              {isPremium
                ? (pl ? '✦ Premium' : '✦ Premium plan')
                : (pl ? '○ Free' : '○ Free plan')}
            </p>
          </div>
          {!isPremium && (
            <button onClick={onUpgrade} style={{
              background: 'var(--accent)', border: 'none', borderRadius: 'var(--radius-sm)',
              color: '#fff', padding: '5px 10px', fontSize: 11, cursor: 'pointer',
              fontFamily: 'var(--font-body)', whiteSpace: 'nowrap', flexShrink: 0,
            }}>
              {pl ? 'Ulepsz do Premium' : 'Upgrade to Premium'}
            </button>
          )}
        </div>

        {/* history */}
        <h3 style={{
          fontSize: 10, fontWeight: 600, letterSpacing: '.08em',
          textTransform: 'uppercase', color: 'var(--text-hint)',
          marginBottom: 12, fontFamily: 'var(--font-body)',
        }}>
          {pl ? 'Historia analiz' : 'Analysis history'}
        </h3>

        {loading ? (
          <p style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 300 }}>
            {pl ? 'Ładowanie...' : 'Loading...'}
          </p>
        ) : analyses.length === 0 ? (
          <div style={{
            textAlign: 'center', padding: '32px 16px', background: 'var(--surface-2)', 
            borderRadius: 'var(--radius-md)', border: '1px dashed var(--border)',
          }}>
            <p style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 300 }}>
              {pl ? 'Brak zapisanych analiz.' : 'No analyses yet.'}
            </p>
            <p style={{ fontSize: 11, color: 'var(--text-hint)', marginTop: 6, fontWeight: 300 }}>
              {pl ? 'Wykonaj analizę żeby zobaczyć historię.' : 'Run an analysis to see it here.'}
            </p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {analyses.map(a => (
              <div key={a.id} style={{
                background: 'var(--surface-2)', borderRadius: 'var(--radius-md)',
                border: `1px solid ${selectedAnalysis?.id === a.id ? 'var(--accent)' : 'var(--border)'}`,
                overflow: 'hidden', transition: 'border-color .15s',
              }}>
                {/* card header */}
                <div
                  onClick={() => setSelectedAnalysis(
                    selectedAnalysis?.id === a.id ? null : a
                  )}
                  style={{
                    display: 'flex', justifyContent: 'space-between',
                    alignItems: 'center', padding: '10px 14px', cursor: 'pointer',
                    background: selectedAnalysis?.id === a.id
                      ? 'var(--accent-soft)' : 'none',
                  }}
                >
                  <div>
                    <span style={{
                      fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-hint)', 
                      letterSpacing: '.04em', display: 'block', marginBottom: 4,
                    }}>
                      {new Date(a.created_at).toLocaleDateString(
                        pl ? 'pl-PL' : 'en-GB',
                        { day: 'numeric', month: 'short', year: 'numeric' }
                      )}
                      {' · '}
                      {a.gender || ''}
                    </span>
                    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                      {a.top_styles?.slice(0, 3).map(s => (
                        <span key={s.name} style={{
                          fontSize: 10, padding: '2px 7px', borderRadius: 20,
                          background: 'var(--surface)',
                          border: '1px solid var(--border)', color: 'var(--text-muted)',
                        }}>{s.name}</span>
                      ))}
                    </div>
                  </div>
                  <ChevronRight
                    size={14} color="var(--text-hint)"
                    style={{
                      transform:  selectedAnalysis?.id === a.id
                        ? 'rotate(90deg)' : 'none',
                      transition: 'transform .2s',
                      flexShrink: 0, marginLeft: 8,
                    }}
                  />
                </div>

                {selectedAnalysis?.id === a.id && (
                  <div style={{
                    padding: '12px 14px', borderTop: '1px solid var(--border)',
                  }}>
                    {/* traits */}
                    {a.traits && Object.entries(a.traits)
                      .filter(([_, v]) => v && v !== 'normal' && v !== 'balanced'
                        && v !== 'slight_imbalance' && v !== null)
                      .slice(0, 6)
                      .map(([key, val]) => (
                        <div key={key} style={{
                          display: 'flex', justifyContent: 'space-between', fontSize: 11, 
                          padding: '4px 0', borderBottom: '1px solid var(--border)',
                        }}>
                          <span style={{
                            fontFamily: 'var(--font-mono)', color: 'var(--text-hint)',
                          }}>
                            {pl ? (TRAIT_NAMES_PL[key] || key.replace(/_/g, ' ')) : key.replace(/_/g, ' ')}
                          </span>
                          <span style={{ color: 'var(--text)', fontWeight: 500 }}>
                            {pl ? (TRAIT_VALUES_PL[String(val)] || String(val)) : String(val)}
                          </span>
                        </div>
                      ))
                    }

                    {/* recommended styles */}
                    <p style={{
                      fontSize: 10, fontWeight: 600, letterSpacing: '.06em',
                      textTransform: 'uppercase', color: 'var(--text-hint)',
                      margin: '10px 0 6px', fontFamily: 'var(--font-body)',
                    }}>
                      {pl ? 'Rekomendowane fryzury' : 'Recommended styles'}
                    </p>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                      {a.top_styles?.map(s => (
                        <span key={s.name} style={{
                          fontSize: 10, padding: '3px 8px', borderRadius: 20,
                          background: 'var(--accent-soft)', border: '1px solid var(--accent)',
                          color: 'var(--accent)',
                        }}>{s.name}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  )
}