import { useState } from 'react'
import { useAuth } from '../hooks/useAuth'
import { useTranslation } from 'react-i18next'
import { supabase } from '../lib/supabase'
import { X, Mail, Lock, Eye, EyeOff, ArrowLeft } from 'lucide-react'

export function AuthModal({ onClose, onSuccess }) {
  const { signInWithEmail, signUpWithEmail } = useAuth()
  const { i18n } = useTranslation()
  const pl = i18n.language === 'pl'

  const [mode, setMode] = useState('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPw, setShowPw] = useState(false)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)
  const [showForgot, setShowForgot] = useState(false)
  const [forgotSent, setForgotSent] = useState(false)

  const isRegister = mode === 'register'

  async function handleSubmit() {
    setError(null); setLoading(true)
    try {
      if (mode === 'login') {
        await signInWithEmail(email, password)
        onSuccess?.(); onClose()
      } else {
        await signUpWithEmail(email, password)
        setSuccess(true)
      }
    } catch (e) {setError(e.message) } 
    finally { setLoading(false) }
  }

  async function handleForgot() {
    if (!email) { setError(pl ? 'Wpisz email' : 'Enter your email'); return }
    setLoading(true)
    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/analyse`,
    })
    setLoading(false)
    if (error) setError(error.message)
    else setForgotSent(true)
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
        background: 'var(--surface)', borderRadius: isRegister ? 'var(--radius-lg)' : '24px',
        border: '1px solid var(--border)', padding: isRegister ? '32px 28px' : '28px 28px',
        width: 'calc(100% - 32px)', maxWidth: isRegister ? 420 : 380,
        boxShadow: '0 24px 64px rgba(0,0,0,.28)', animation: 'modalIn .2s ease',
        transition: 'max-width .2s, border-radius .2s',
      }}>
        {/* close */}
        <button
          onClick={onClose}
          style={{
            position: 'absolute', top: 12, right: 12, background: 'none', border: 'none',
            cursor: 'pointer', fontSize: 18, color: 'var(--text-hint)', display: 'flex', padding: 4
          }}
        >
          <X size={16} strokeWidth={1.5} />
        </button>

        <h2 style={{
          fontFamily: 'var(--font-display)', fontSize: isRegister ? 22 : 20, fontWeight: 500,
          color: 'var(--text)', marginBottom: 4,
        }}>
          {showForgot
          ? (pl ? 'Reset hasła' : 'Reset password')
          : isRegister
            ? (pl ? 'Utwórz konto' : 'Create account')
            : (pl ? 'Zaloguj się' : 'Log in')}
        </h2>
        <p style={{fontSize: 13, color: 'var(--text-muted)', fontWeight: 300, marginBottom: 24}}>
          {showForgot
            ? (pl ? 'Wyślemy link do zresetowania hasła' : "We'll send you a reset link")
            : isRegister
              ? (pl ? 'Bezpłatne' : 'Free')
              : (pl ? 'Dostęp do historii i funkcji premium' : 'Access your history and premium features')}
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
              <div style={{ position: 'relative' }}>
                <Mail size={14} strokeWidth={1.5} color="var(--text-hint)" style={{
                  position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)',
                }} />
                <input
                  type="email"
                  placeholder={pl ? 'Adres email' : 'Email address'}
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && !showForgot && handleSubmit()}
                  style={{ ...inputStyle, paddingLeft: 36 }}
                />
              </div>
              {!showForgot && (
                <div style={{ position: 'relative' }}>
                  <Lock size={14} strokeWidth={1.5} color="var(--text-hint)" style={{
                    position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)',
                  }} />
                  <input
                    type={showPw ? 'text' : 'password'}
                    placeholder={isRegister
                      ? (pl ? 'Hasło (min. 8 znaków)' : 'Password (min. 8 chars)')
                      : (pl ? 'Hasło' : 'Password')}
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleSubmit()}
                    style={{ ...inputStyle, paddingLeft: 36, paddingRight: 36 }}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPw(p => !p)}
                    style={{
                      position: 'absolute', right: 10, top: '50%',
                      transform: 'translateY(-50%)',
                      background: 'none', border: 'none', cursor: 'pointer',
                      color: 'var(--text-hint)', display: 'flex', padding: 2,
                    }}
                  >
                    {showPw
                      ? <EyeOff size={14} strokeWidth={1.5} />
                      : <Eye size={14} strokeWidth={1.5} />}
                  </button>
                </div>
              )}
            </div>

            {/* forgot password link */}
            {mode === 'login' && !showForgot && (
              <div style={{ textAlign: 'right', marginBottom: 14, marginTop: -6 }}>
                <button onClick={() => { setShowForgot(true); setError(null) }} style={{
                  background: 'none', border: 'none', cursor: 'pointer',
                  fontSize: 11, color: 'var(--text-hint)', fontFamily: 'var(--font-body)',
                }}>
                  {pl ? 'Zapomniałeś hasła?' : 'Forgot password?'}
                </button>
              </div>
            )}

            {error && (
              <p style={{
                fontSize: 12, color: '#c0392b',
                marginBottom: 12, padding: '8px 10px',
                background: '#fef4f2', borderRadius: 'var(--radius-sm)',
                border: '1px solid #f5c6bc',
              }}>{error}</p>
            )}

            <button
              onClick={showForgot ? handleForgot : handleSubmit}
              disabled={loading || !email || (!showForgot && !password)}
              className="analyse-btn"
              style={{ opacity: (!email || (!showForgot && !password)) ? 0.5 : 1 }}
            >
              {loading ? (pl ? 'Ładowanie...' : 'Loading...')
               : showForgot ? (pl ? 'Wyślij link' : 'Send reset link')
               : isRegister ? (pl ? 'Utwórz konto' : 'Create account')
               : (pl ? 'Zaloguj się' : 'Sign in')}
            </button>

            {/* back from forgot / switch mode */}
            <div style={{ textAlign: 'center', marginTop: 14, fontSize: 12, color: 'var(--text-muted)' }}>
              {showForgot ? (
                <button onClick={() => { setShowForgot(false); setError(null) }} style={{
                  background: 'none', border: 'none', cursor: 'pointer',
                  fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-body)',
                  display: 'inline-flex', alignItems: 'center', gap: 4,
                }}>
                  <ArrowLeft size={12} strokeWidth={1.5} />
                  {pl ? 'Wróć do logowania' : 'Back to sign in'}
                </button>
              ) : (
                <>
                  {isRegister
                    ? (pl ? 'Masz już konto? ' : 'Already have an account? ')
                    : (pl ? 'Nie masz konta? ' : "Don't have an account? ")}
                  <button onClick={() => { setMode(m => m === 'login' ? 'register' : 'login'); setError(null) }} style={{
                    background: 'none', border: 'none', cursor: 'pointer',
                    color: 'var(--accent)', fontSize: 12, padding: 0,
                    fontFamily: 'var(--font-body)',
                  }}>
                    {isRegister ? (pl ? 'Zaloguj się' : 'Sign in') : (pl ? 'Zarejestruj się' : 'Register')}
                  </button>
                </>
              )}
            </div>
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