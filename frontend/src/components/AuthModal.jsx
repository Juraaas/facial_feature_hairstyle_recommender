import { useState, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { supabase } from '../lib/supabase'
import { isPwned, MIN_PASSWORD } from '../lib/pwned'
import { Turnstile } from '@marsidev/react-turnstile'
import { X, Mail, Lock, Eye, EyeOff, ArrowLeft } from 'lucide-react'

const SITE_KEY = import.meta.env.VITE_TURNSTILE_SITE_KEY

export function AuthModal({ onClose, onSuccess }) {
  const { i18n } = useTranslation()
  const pl = i18n.language?.startsWith('pl')

  const [view, setView] = useState('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPw, setShowPw] = useState(false)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [sent, setSent] = useState(null)
  const [code, setCode] = useState('')
  const [captchaToken, setCaptchaToken] = useState(null)
  const captchaRef = useRef(null)

  const isRegister = view === 'register'
  const needsPassword = view === 'login' || view === 'register'

  function go(next) { setView(next); setError(null); setSent(null); setCode('') }

  function resetCaptcha() {
    setCaptchaToken(null)
    captchaRef.current?.reset()
  }

  function friendlyError(e) {
    const c = e?.code || ''
    const m = String(e?.message || '')
    if (/captcha/i.test(m))
      return pl ? 'Weryfikacja antybotowa nie powiodła się. Odśwież stronę i spróbuj ponownie.'
                : 'Bot verification failed. Reload the page and try again.'
    if (/rate limit|too many|security purposes/i.test(`${c} ${m}`))
      return pl ? 'Zbyt wiele prób. Spróbuj za chwilę.' : 'Too many attempts. Please try again shortly.'
    if (c === 'invalid_credentials' || /invalid login credentials/i.test(m))
      return pl ? 'Nieprawidłowy e-mail lub hasło. Jeśli konto powstało bez hasła (np. przez link z e-maila), użyj logowania linkiem.'
                : 'Incorrect email or password. If your account was created without a password (e.g. via an email link), use the sign-in link option.'
    if (c === 'email_not_confirmed' || /not confirmed/i.test(m))
      return pl ? 'Potwierdź adres e-mail: sprawdź skrzynkę i kliknij link z wiadomości.'
                : 'Please confirm your email: check your inbox and click the link.'
    if (c === 'weak_password' || /weak|password should/i.test(`${c} ${m}`))
      return pl ? `Hasło jest zbyt słabe. Użyj co najmniej ${MIN_PASSWORD} znaków.`
                : `The password is too weak. Use at least ${MIN_PASSWORD} characters.`
    if (c === 'otp_expired' || /expired|invalid.*token|token.*invalid/i.test(m))
      return pl ? 'Kod jest nieprawidłowy lub wygasł. Poproś o nowy.' : 'The code is invalid or has expired. Request a new one.'
    return m
  }

  async function submit() {
    setError(null)
    if (isRegister && password.length < MIN_PASSWORD) {
      setError(pl ? `Hasło musi mieć co najmniej ${MIN_PASSWORD} znaków.` : `The password must be at least ${MIN_PASSWORD} characters.`)
      return
    }
    setLoading(true)
    try {
      const withCaptcha = captchaToken ? { captchaToken } : {}
      if (view === 'login') {
        const { error } = await supabase.auth.signInWithPassword({ email, password, options: withCaptcha })
        if (error) throw error
        onSuccess?.(); onClose()
      } else if (view === 'register') {
        if (await isPwned(password)) {
          setError(pl ? 'To hasło pojawiło się w publicznych wyciekach danych. Wybierz inne.'
                      : 'This password has appeared in public data breaches. Choose a different one.')
          return
        }
        const { error } = await supabase.auth.signUp({ email, password, options: withCaptcha })
        if (error) throw error
        setSent('register')
      } else if (view === 'forgot') {
        const { error } = await supabase.auth.resetPasswordForEmail(email, {
          redirectTo: `${window.location.origin}/analyse`, ...withCaptcha,
        })
        if (error) throw error
        setSent('forgot')
      } else {
        const { error } = await supabase.auth.signInWithOtp({
          email,
          options: { shouldCreateUser: false, emailRedirectTo: `${window.location.origin}/analyse`, ...withCaptcha },
        })
        if (error && !/not allowed|otp_disabled|signups/i.test(`${error.code} ${error.message}`)) throw error
        setSent('magic')
      }
    } catch (e) {
      setError(friendlyError(e))
    } finally {
      setLoading(false)
      resetCaptcha()
    }
  }

  async function verifyCode(e) {
    e.preventDefault()
    if (!code.trim() || loading) return
    setLoading(true); setError(null)
    try {
      const { error } = await supabase.auth.verifyOtp({ email, token: code.trim(), type: 'email' })
      if (error) throw error
      onSuccess?.(); onClose()
    } catch (err) {
      setError(friendlyError(err))
    } finally {
      setLoading(false)
    }
  }

  function onFormSubmit(e) { e.preventDefault(); if (!loading) submit() }

  const waitingForCaptcha = !!SITE_KEY && !captchaToken
  const submitDisabled = loading || !email || (needsPassword && !password) || waitingForCaptcha

  const titles = {
    login: pl ? 'Zaloguj się' : 'Log in',
    register: pl ? 'Utwórz konto' : 'Create account',
    forgot: pl ? 'Reset hasła' : 'Reset password',
    magic: pl ? 'Logowanie linkiem' : 'Sign in with a link',
  }
  const subtitles = {
    login: pl ? 'Dostęp do historii analiz i kredytów' : 'Access your analysis history and credits',
    register: pl ? 'Bezpłatne konto' : 'Free account',
    forgot: pl ? 'Wyślemy Ci link do zresetowania hasła' : "We'll send you a reset link",
    magic: pl ? 'Wyślemy Ci link i kod do jednorazowego logowania' : "We'll email you a one-time sign-in link and code",
  }
  const buttonLabels = {
    login: pl ? 'Zaloguj się' : 'Sign in',
    register: pl ? 'Utwórz konto' : 'Create account',
    forgot: pl ? 'Wyślij link' : 'Send reset link',
    magic: pl ? 'Wyślij link do logowania' : 'Email me a sign-in link',
  }
  const sentMessages = {
    register: pl ? '✓ Sprawdź email aby potwierdzić konto' : '✓ Check your email to confirm your account',
    forgot: pl ? '✓ Jeśli konto istnieje, wysłaliśmy link do resetu hasła' : '✓ If the account exists, we sent a password reset link',
    magic: pl ? '✓ Jeśli konto istnieje, wysłaliśmy link do logowania. Kliknij go albo wpisz kod z wiadomości.'
              : '✓ If the account exists, we sent a sign-in link. Click it or enter the code from the email.',
  }

  const linkBtn = {
    background: 'none', border: 'none', cursor: 'pointer', padding: 0,
    fontSize: 11, color: 'var(--text-hint)', fontFamily: 'var(--font-body)',
  }

  return (
    <>
      <div onClick={onClose} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,.5)', zIndex: 200 }} />

      <div style={{
        position: 'fixed', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', zIndex: 201,
        background: 'var(--surface)', borderRadius: isRegister ? 'var(--radius-lg)' : '24px',
        border: '1px solid var(--border)', padding: isRegister ? '32px 28px' : '28px 28px',
        width: 'calc(100% - 32px)', maxWidth: isRegister ? 420 : 380, maxHeight: '92vh', overflowY: 'auto',
        boxShadow: '0 24px 64px rgba(0,0,0,.28)', animation: 'modalIn .2s ease',
        transition: 'max-width .2s, border-radius .2s',
      }}>
        <button type="button" onClick={onClose} aria-label="close" style={{
          position: 'absolute', top: 12, right: 12, background: 'none', border: 'none',
          cursor: 'pointer', color: 'var(--text-hint)', display: 'flex', padding: 4,
        }}>
          <X size={16} strokeWidth={1.5} />
        </button>

        <h2 style={{
          fontFamily: 'var(--font-display)', fontSize: isRegister ? 22 : 20, fontWeight: 500,
          color: 'var(--text)', marginBottom: 4,
        }}>{titles[view]}</h2>
        <p style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 300, marginBottom: 24 }}>{subtitles[view]}</p>

        {sent ? (
          <>
            <div style={{
              background: '#f0faf4', border: '1px solid #b7dfc7', borderRadius: 'var(--radius-md)',
              padding: '14px 16px', fontSize: 13, color: '#2d8f4e', textAlign: 'center', lineHeight: 1.6,
            }}>{sentMessages[sent]}</div>

            {sent === 'magic' && (
              <form onSubmit={verifyCode} noValidate style={{ marginTop: 16 }}>
                <input
                  value={code} onChange={e => setCode(e.target.value.replace(/\s/g, ''))}
                  inputMode="numeric" autoComplete="one-time-code" maxLength={10}
                  aria-label={pl ? 'Kod z e-maila' : 'Code from email'}
                  placeholder={pl ? 'Kod z e-maila' : 'Code from email'}
                  style={{ ...inputStyle, textAlign: 'center', letterSpacing: 4, marginBottom: 10 }}
                />
                {error && (
                  <p style={{
                    fontSize: 12, color: '#c0392b', marginBottom: 10, padding: '8px 10px',
                    background: '#fef4f2', borderRadius: 'var(--radius-sm)', border: '1px solid #f5c6bc',
                  }}>{error}</p>
                )}
                <button type="submit" className="analyse-btn" disabled={!code || loading}
                  style={{ opacity: !code || loading ? 0.5 : 1 }}>
                  {loading ? (pl ? 'Ładowanie...' : 'Loading...') : (pl ? 'Zaloguj kodem' : 'Sign in with code')}
                </button>
              </form>
            )}

            <div style={{ textAlign: 'center', marginTop: 14 }}>
              <button type="button" onClick={() => go('login')} style={{ ...linkBtn, fontSize: 12, color: 'var(--text-muted)' }}>
                <ArrowLeft size={12} strokeWidth={1.5} style={{ verticalAlign: '-2px', marginRight: 4 }} />
                {pl ? 'Wróć do logowania' : 'Back to sign in'}
              </button>
            </div>
          </>
        ) : (
          <form onSubmit={onFormSubmit} noValidate>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 16 }}>
              <div style={{ position: 'relative' }}>
                <Mail size={14} strokeWidth={1.5} color="var(--text-hint)" style={{
                  position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)',
                }} />
                <input
                  type="email" name="email" autoComplete="email"
                  aria-label={pl ? 'Adres email' : 'Email address'}
                  placeholder={pl ? 'Adres email' : 'Email address'}
                  value={email} onChange={e => setEmail(e.target.value)}
                  style={{ ...inputStyle, paddingLeft: 36 }}
                />
              </div>
              {needsPassword && (
                <div style={{ position: 'relative' }}>
                  <Lock size={14} strokeWidth={1.5} color="var(--text-hint)" style={{
                    position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)',
                  }} />
                  <input
                    type={showPw ? 'text' : 'password'} name="password"
                    autoComplete={isRegister ? 'new-password' : 'current-password'}
                    aria-label={pl ? 'Hasło' : 'Password'}
                    placeholder={isRegister
                      ? (pl ? `Hasło (min. ${MIN_PASSWORD} znaków)` : `Password (min. ${MIN_PASSWORD} chars)`)
                      : (pl ? 'Hasło' : 'Password')}
                    value={password} onChange={e => setPassword(e.target.value)}
                    style={{ ...inputStyle, paddingLeft: 36, paddingRight: 36 }}
                  />
                  <button type="button" onClick={() => setShowPw(p => !p)}
                    aria-label={showPw ? 'hide password' : 'show password'}
                    style={{
                      position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)',
                      background: 'none', border: 'none', cursor: 'pointer',
                      color: 'var(--text-hint)', display: 'flex', padding: 2,
                    }}>
                    {showPw ? <EyeOff size={14} strokeWidth={1.5} /> : <Eye size={14} strokeWidth={1.5} />}
                  </button>
                </div>
              )}
            </div>

            {view === 'login' && (
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, marginBottom: 14, marginTop: -6 }}>
                <button type="button" onClick={() => go('magic')} style={linkBtn}>
                  {pl ? 'Zaloguj się linkiem z e-maila' : 'Sign in with an email link'}
                </button>
                <button type="button" onClick={() => go('forgot')} style={linkBtn}>
                  {pl ? 'Zapomniałeś hasła?' : 'Forgot password?'}
                </button>
              </div>
            )}

            {SITE_KEY && (
              <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 12 }}>
                <Turnstile
                  key={view}
                  ref={captchaRef}
                  siteKey={SITE_KEY}
                  onSuccess={setCaptchaToken}
                  onExpire={() => setCaptchaToken(null)}
                  onError={() => setCaptchaToken(null)}
                  options={{ theme: 'auto' }}
                />
              </div>
            )}

            {error && (
              <p style={{
                fontSize: 12, color: '#c0392b', marginBottom: 12, padding: '8px 10px',
                background: '#fef4f2', borderRadius: 'var(--radius-sm)', border: '1px solid #f5c6bc',
              }}>{error}</p>
            )}

            <button type="submit" disabled={submitDisabled} className="analyse-btn"
              style={{ opacity: submitDisabled ? 0.5 : 1 }}>
              {loading ? (pl ? 'Ładowanie...' : 'Loading...')
                : waitingForCaptcha ? (pl ? 'Weryfikacja…' : 'Verifying…')
                : buttonLabels[view]}
            </button>

            <div style={{ textAlign: 'center', marginTop: 14, fontSize: 12, color: 'var(--text-muted)' }}>
              {view === 'forgot' || view === 'magic' ? (
                <button type="button" onClick={() => go('login')} style={{ ...linkBtn, fontSize: 12, color: 'var(--text-muted)' }}>
                  <ArrowLeft size={12} strokeWidth={1.5} style={{ verticalAlign: '-2px', marginRight: 4 }} />
                  {pl ? 'Wróć do logowania' : 'Back to sign in'}
                </button>
              ) : (
                <>
                  {isRegister
                    ? (pl ? 'Masz już konto? ' : 'Already have an account? ')
                    : (pl ? 'Nie masz konta? ' : "Don't have an account? ")}
                  <button type="button" onClick={() => go(isRegister ? 'login' : 'register')}
                    style={{ ...linkBtn, fontSize: 12, color: 'var(--accent)' }}>
                    {isRegister ? (pl ? 'Zaloguj się' : 'Sign in') : (pl ? 'Zarejestruj się' : 'Register')}
                  </button>
                </>
              )}
            </div>
          </form>
        )}
      </div>
    </>
  )
}

const inputStyle = {
  width: '100%', padding: '10px 12px', border: '1px solid var(--border)',
  borderRadius: 'var(--radius-md)', background: 'var(--surface-2)', color: 'var(--text)',
  fontSize: 13, fontFamily: 'var(--font-body)', outline: 'none',
}