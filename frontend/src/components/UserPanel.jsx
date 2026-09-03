import { useState, useEffect } from 'react'
import { supabase } from '../lib/supabase'
import { useTranslation } from 'react-i18next'

const BASE = import.meta.env.VITE_API_URL || '/api'

export function UserPanel({ user, onClose, isPremium, onUpgrade }) {
  const { i18n } = useTranslation()
  const pl = i18n.language === 'pl'
  const [analyses, setAnalyses] = useState([])
  const [loading, setLoading]  = useState(true)

  useEffect(() => {
    async function load() {
      const { data } = await supabase.auth.getSession()
      const token = data.session?.access_token
      const res = await fetch(`${BASE}/history`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      if (res.ok) {
        const d = await res.json()
        setAnalyses(d.analyses)
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
        position: 'fixed', top: 0, right: 0, height: '100vh', width: '100%', maxWidth: 380,
        background: 'var(--surface)', borderLeft: '1px solid var(--border)',
        zIndex: 201, overflowY: 'auto', animation: 'slideIn .25s ease', padding: '24px',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', 
            alignItems: 'center', marginBottom: 24 }}>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 18, color: 'var(--text)' }}>
            {pl ? 'Twoje konto' : 'Your account'}
          </h2>
          <button onClick={onClose} style={{
            background: 'none', border: 'none', cursor: 'pointer', fontSize: 18, color: 'var(--text-hint)',
          }}>✕</button>
        </div>

        {/* plan badge */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '12px 14px', marginBottom: 20,
          background: isPremium ? 'var(--accent-soft)' : 'var(--surface-2)',
          borderRadius: 'var(--radius-md)',
          border: `1px solid ${isPremium ? 'var(--accent)' : 'var(--border)'}`,
        }}>
          <span style={{ fontSize: 20 }}>{isPremium ? '✨' : '👤'}</span>
          <div>
            <p style={{ fontSize: 12, fontWeight: 500, color: 'var(--text)' }}>
              {user.email}
            </p>
            <p style={{ fontSize: 11, color: isPremium ? 'var(--accent)' : 'var(--text-muted)' }}>
              {isPremium
                ? (pl ? 'Plan Premium' : 'Premium plan')
                : (pl ? 'Plan Free' : 'Free plan')}
            </p>
          </div>
          {!isPremium && (
            <button
              onClick={onUpgrade}
              style={{
                marginLeft: 'auto', background: 'var(--accent)', border: 'none', fontSize: 11,
                borderRadius: 'var(--radius-sm)', color: '#fff',padding: '5px 10px',
                cursor: 'pointer', fontFamily: 'var(--font-body)', whiteSpace: 'nowrap',
              }}
            >
              {pl ? 'Ulepsz do Premium' : 'Upgrade'}
            </button>
          )}
        </div>

        {/* history */}
        <h3 style={{
          fontSize: 12, fontWeight: 600, letterSpacing: '.06em',
          textTransform: 'uppercase', color: 'var(--text-hint)',
          marginBottom: 12, fontFamily: 'var(--font-body)',
        }}>
          {pl ? 'Historia analiz' : 'Analysis history'}
        </h3>

        {loading ? (
          <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
            {pl ? 'Ładowanie...' : 'Loading...'}
          </p>
        ) : analyses.length === 0 ? (
          <p style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 300 }}>
            {pl ? 'Brak zapisanych analiz.' : 'No analyses yet.'}
          </p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {analyses.map(a => (
              <div key={a.id} style={{
                background: 'var(--surface-2)', borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border)', padding: '12px 14px',
              }}>
                <div style={{
                  display: 'flex', justifyContent: 'space-between', marginBottom: 8,
                }}>
                  <span style={{
                    fontSize: 10, fontFamily: 'var(--font-mono)',
                    color: 'var(--text-hint)', letterSpacing: '.04em',
                  }}>
                    {new Date(a.created_at).toLocaleDateString(
                      pl ? 'pl-PL' : 'en-GB',
                      { day: 'numeric', month: 'short', year: 'numeric' }
                    )}
                  </span>
                  <span style={{fontSize: 10, color: 'var(--text-hint)', fontFamily: 'var(--font-mono)'}}>
                    {a.gender}
                  </span>
                </div>
                {a.top_styles?.slice(0, 3).map(s => (
                  <span key={s.name} style={{
                    display: 'inline-block', fontSize: 10, color: 'var(--text-muted)',
                    borderRadius: 20, marginRight: 4, marginBottom: 4, padding: '2px 7px',
                    background: 'var(--surface)', border: '1px solid var(--border)',
                  }}>{s.name}</span>
                ))}
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  )
}