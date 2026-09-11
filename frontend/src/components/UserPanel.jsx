import { useState, useEffect } from 'react'
import { supabase } from '../lib/supabase'
import { useTranslation } from 'react-i18next'
import { Sparkles, X, Trash } from 'lucide-react'
import { btnAccent } from '../styles/shared'

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
  const [analyses, setAnalyses] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedAnalysis, setSelectedAnalysis] = useState(null)
  const [deleting, setDeleting] = useState(null)
  const { i18n } = useTranslation()
  const pl = i18n.language === 'pl'

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

  async function handleDelete(id) {
    setDeleting(id)
    try {
      const { data } = await supabase.auth.getSession()
      const token = data.session?.access_token
      await fetch(`${BASE}/history/${id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      })
      setAnalyses(prev => prev.filter(a => a.id !== id))
      if (selectedAnalysis?.id === id) setSelectedAnalysis(null)
    } catch (e) { console.error(e) }
    finally { setDeleting(null) }
  }

  return (
    <>
      <div onClick={onClose} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,.55)', zIndex: 200}} />

      <div style={{
        position: 'fixed', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
        zIndex: 201, width: 'calc(100% - 32px)', maxWidth: 720,
        maxHeight: '85vh', background: 'var(--surface)', borderRadius: 'var(--radius-lg)',
        border: '1px solid var(--border)', boxShadow: '0 24px 64px rgba(0,0,0,.25)',
        overflow: 'hidden', display: 'flex', flexDirection: 'column', animation: 'modalIn .2s ease',
      }}>

        {/* header */}
        <div style={{
          padding: '20px 24px', borderBottom: '1px solid var(--border)',display: 'flex', 
          justifyContent: 'space-between', alignItems: 'center', flexShrink: 0,
        }}>
          <div>
            <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 18,
              fontWeight: 500, color: 'var(--text)' }}>
              {pl ? 'Twoje konto' : 'Your account'}
            </h2>
            <p style={{ fontSize: 11, color: isPremium ? 'var(--accent)' : 'var(--text-hint)',
              fontFamily: 'var(--font-mono)', marginTop: 2 }}>
              {user.email} · {isPremium ? '✦ Premium' : '○ Free'}
            </p>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            {!isPremium && (
              <button onClick={onUpgrade} style={{
                ...btnAccent, fontSize: 12, padding: '6px 12px', minWidth: 'auto',
              }}>
                <Sparkles size={12} strokeWidth={1.5} style={{ marginRight: 5 }} />
                {pl ? 'Kup Premium' : 'Upgrade'}
              </button>
            )}
            <button onClick={onClose} style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: 'var(--text-hint)', display: 'flex',
            }}>
              <X size={18} strokeWidth={1.5} />
            </button>
          </div>
        </div>

        {/* body */}
        <div style={{
          display: 'grid', gridTemplateColumns: selectedAnalysis ? '280px 1fr' : '1fr',
          flex: 1, overflow: 'hidden', minHeight: 0,
        }}>

          {/* analysis list */}
          <div style={{ overflowY: 'auto', borderRight: selectedAnalysis ? '1px solid var(--border)' : 'none', padding: '16px' }}>
            <p style={{
              fontSize: 10, fontWeight: 600, letterSpacing: '.08em',
              textTransform: 'uppercase', color: 'var(--text-hint)',
              marginBottom: 12, fontFamily: 'var(--font-body)',
            }}>
              {pl ? `Historia analiz (${analyses.length})` : `Analysis history (${analyses.length})`}
            </p>

            {loading ? (
              <p style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 300 }}>
                {pl ? 'Ładowanie...' : 'Loading...'}
              </p>
            ) : analyses.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '32px 16px',
                background: 'var(--surface-2)', borderRadius: 'var(--radius-md)',
                border: '1px dashed var(--border)' }}>
                <p style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 300 }}>
                  {pl ? 'Brak analiz.' : 'No analyses yet.'}
                </p>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {analyses.map(a => (
                  <div
                    key={a.id}
                    style={{
                      background: selectedAnalysis?.id === a.id ? 'var(--accent-soft)' : 'var(--surface-2)',
                      border: `1px solid ${selectedAnalysis?.id === a.id ? 'var(--accent)' : 'var(--border)'}`,
                      borderRadius: 'var(--radius-sm)', padding: '10px 12px', cursor: 'pointer',
                      transition: 'all .15s', display: 'flex', justifyContent: 'space-between',
                      alignItems: 'flex-start', gap: 8,
                    }}
                    onClick={() => setSelectedAnalysis(
                      selectedAnalysis?.id === a.id ? null : a
                    )}
                  >
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <span style={{
                        fontSize: 10, fontFamily: 'var(--font-mono)',
                        color: 'var(--text-hint)', display: 'block', marginBottom: 4,
                      }}>
                        {new Date(a.created_at).toLocaleDateString(
                          pl ? 'pl-PL' : 'en-GB',
                          { day: 'numeric', month: 'short', year: 'numeric' }
                        )} · {a.gender}
                      </span>
                      <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                        {a.top_styles?.slice(0, 2).map(s => (
                          <span key={s.name} style={{
                            fontSize: 9, padding: '1px 6px', borderRadius: 20,
                            background: 'var(--surface)',
                            border: '1px solid var(--border)',
                            color: 'var(--text-muted)',
                          }}>{s.name}</span>
                        ))}
                      </div>
                    </div>

                    {/* delete button */}
                    <button
                      onClick={e => { e.stopPropagation(); handleDelete(a.id) }}
                      disabled={deleting === a.id}
                      style={{
                        background: 'none', border: 'none', cursor: 'pointer',
                        color: 'var(--text-hint)', padding: 2, flexShrink: 0,
                        opacity: deleting === a.id ? 0.4 : 1,
                        display: 'flex', alignItems: 'center',
                      }}
                      title={pl ? 'Usuń' : 'Delete'}
                    >
                      <Trash size={13} strokeWidth={1.5} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* analysis details */}
          {selectedAnalysis && (
            <div style={{ overflowY: 'auto', padding: '16px 20px' }}>
              <p style={{
                fontSize: 10, fontWeight: 600, letterSpacing: '.08em',
                textTransform: 'uppercase', color: 'var(--text-hint)',
                marginBottom: 16, fontFamily: 'var(--font-body)',
              }}>
                {new Date(selectedAnalysis.created_at).toLocaleDateString(
                  pl ? 'pl-PL' : 'en-GB',
                  { day: 'numeric', month: 'long', year: 'numeric' }
                )}
              </p>

              {/* facial features */}
              <p style={{
                fontSize: 10, fontWeight: 600, letterSpacing: '.06em',
                textTransform: 'uppercase', color: 'var(--text-hint)',
                marginBottom: 8, fontFamily: 'var(--font-body)',
              }}>
                {pl ? 'Cechy twarzy' : 'Facial traits'}
              </p>
              <div style={{ marginBottom: 20 }}>
                {selectedAnalysis.traits && Object.entries(selectedAnalysis.traits)
                  .filter(([_, v]) => v && v !== 'normal' && v !== 'balanced'
                    && v !== 'slight_imbalance')
                  .map(([key, val]) => (
                    <div key={key} style={{
                      display: 'flex', justifyContent: 'space-between',
                      fontSize: 12, padding: '5px 0',
                      borderBottom: '1px solid var(--border)',
                    }}>
                      <span style={{ color: 'var(--text-hint)', fontFamily: 'var(--font-mono)' }}>
                        {pl ? (TRAIT_NAMES_PL[key] || key.replace(/_/g, ' '))
                             : key.replace(/_/g, ' ')}
                      </span>
                      <span style={{ color: 'var(--text)', fontWeight: 500 }}>
                        {pl ? (TRAIT_VALUES_PL[String(val)] || String(val)) : String(val)}
                      </span>
                    </div>
                  ))
                }
              </div>

              {/* recommendations */}
              <p style={{
                fontSize: 10, fontWeight: 600, letterSpacing: '.06em',
                textTransform: 'uppercase', color: 'var(--text-hint)',
                marginBottom: 8, fontFamily: 'var(--font-body)',
              }}>
                {pl ? 'Rekomendowane fryzury' : 'Recommended styles'}
              </p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                {selectedAnalysis.top_styles?.map(s => (
                  <div key={s.name} style={{
                    display: 'flex', alignItems: 'center', gap: 6, fontSize: 11,
                    padding: '5px 10px', borderRadius: 20, background: 'var(--accent-soft)',
                    border: '1px solid var(--accent)', color: 'var(--accent)',
                  }}>
                    <span>{s.name}</span>
                    {s.score && (
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10 }}>
                        {s.score}%
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  )
}