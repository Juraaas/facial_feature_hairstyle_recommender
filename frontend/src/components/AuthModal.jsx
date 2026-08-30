import { useState } from 'react'
import { useAuth } from '../hooks/useAuth'
import { useTranslation } from 'react-i18next'

export function AuthModal({ onClose, onSuccess }) {
  const { signInWithEmail, signUpWithEmail } = useAuth()
  const { i18n } = useTranslation()
  const pl = i18n.language === 'pl'
  const [mode, setMode] = useState('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)

  async function handleSubmit() {
    setError(null)
    setLoading(true)
    try {
      if (mode === 'login') {
        await signInWithEmail(email, password)
        onSuccess?.()
        onClose()
      } else {
        await signUpWithEmail(email, password)
        setSuccess(true)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      {/* overlay */}
      <div
        onClick={onClose}
        style={{position: 'fixed', inset: 0, background: 'rgba(0,0,0,.5)', zIndex: 200}}
      />

      {/* modal */}
      <div style={{
        position: 'fixed', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',zIndex: 201,
        background: 'var(--surface)', borderRadius: 'var(--radius-lg)',
        border: '1px solid var(--border)', padding: '32px 28px',
        width: 'calc(100% - 32px)', maxWidth: 400, 
        boxShadow: '0 24px 64px rgba(0,0,0,.25)', animation: 'modalIn .2s ease',
      }}>
        {/* close */}
        <button
          onClick={onClose}
          style={{
            position: 'absolute', top: 14, right: 14, background: 'none', border: 'none',
            cursor: 'pointer', fontSize: 18, color: 'var(--text-hint)',
          }}
        >✕</button>

        <h2 style={{
          fontFamily: 'var(--font-display)', fontSize: 20, fontWeight: 500,
          color: 'var(--text)', marginBottom: 6,
        }}>
          {mode === 'login'
            ? (pl ? 'Zaloguj się' : 'Sign in')
            : (pl ? 'Utwórz konto' : 'Create account')}
        </h2>
        <p style={{fontSize: 13, color: 'var(--text-muted)', fontWeight: 300, marginBottom: 24}}>
          {mode === 'login'
            ? (pl ? 'Dostęp do historii analiz i funkcji premium' : 'Access your analysis history and premium features')
            : (pl ? 'Bezpłatne' : 'Free')}
        </p>

        {success ? (
          <div style={{
            background: '#f0faf4', border: '1px solid #b7dfc7',
            borderRadius: 'var(--radius-md)', padding: '14px 16px',
            fontSize: 13, color: '#2d8f4e', textAlign: 'center',
          }}>
            {pl
              ? '✓ Sprawdź email aby potwierdzić konto'
              : '✓ Check your email to confirm your account'}
          </div>
        ) : (
          <>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 16 }}>
              <input
                type="email"
                placeholder={pl ? 'Adres email' : 'Email address'}
                value={email}
                onChange={e => setEmail(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSubmit()}
                style={inputStyle}
              />
              <input
                type="password"
                placeholder={pl ? 'Hasło (min. 8 znaków)' : 'Password (min. 8 chars)'}
                value={password}
                onChange={e => setPassword(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSubmit()}
                style={inputStyle}
              />
            </div>

            {error && (
              <p style={{
                fontSize: 12, color: '#c0392b',
                marginBottom: 12, padding: '8px 10px',
                background: '#fef4f2', borderRadius: 'var(--radius-sm)',
                border: '1px solid #f5c6bc',
              }}>{error}</p>
            )}

            <button
              onClick={handleSubmit}
              disabled={loading || !email || !password}
              className="analyse-btn"
              style={{ opacity: (!email || !password) ? 0.5 : 1 }}
            >
              {loading
                ? (pl ? 'Ładowanie...' : 'Loading...')
                : mode === 'login'
                  ? (pl ? 'Zaloguj się' : 'Sign in')
                  : (pl ? 'Utwórz konto' : 'Create account')}
            </button>

            <p style={{
              textAlign: 'center', fontSize: 12,
              color: 'var(--text-muted)', marginTop: 16,
            }}>
              {mode === 'login'
                ? (pl ? 'Nie masz konta? ' : "Don't have an account? ")
                : (pl ? 'Masz już konto? ' : 'Already have an account? ')}
              <button
                onClick={() => { setMode(m => m === 'login' ? 'register' : 'login'); setError(null) }}
                style={{
                  background: 'none', border: 'none', cursor: 'pointer',
                  color: 'var(--accent)', fontSize: 12, padding: 0,
                  fontFamily: 'var(--font-body)',
                }}
              >
                {mode === 'login'
                  ? (pl ? 'Zarejestruj się' : 'Register')
                  : (pl ? 'Zaloguj się' : 'Sign in')}
              </button>
            </p>
          </>
        )}
      </div>
    </>
  )
}

const inputStyle = {
  width: '100%',
  padding: '10px 12px',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-md)',
  background: 'var(--surface-2)',
  color: 'var(--text)',
  fontSize: 13,
  fontFamily: 'var(--font-body)',
  outline: 'none',
}