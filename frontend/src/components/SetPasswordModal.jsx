import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { supabase } from '../lib/supabase'
import { isPwned, MIN_PASSWORD } from '../lib/pwned'
import { X, Lock, Eye, EyeOff, Check } from 'lucide-react'

export function SetPasswordModal({ mode = 'change', onClose }) {
  const { i18n } = useTranslation()
  const pl = i18n.language?.startsWith('pl')
  const recovery = mode === 'recovery'

  const [pw, setPw] = useState('')
  const [pw2, setPw2] = useState('')
  const [show, setShow] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [done, setDone] = useState(false)

  const tooShort = pw.length > 0 && pw.length < MIN_PASSWORD
  const mismatch = pw2.length > 0 && pw !== pw2
  const valid = pw.length >= MIN_PASSWORD && pw === pw2

  function explain(e) {
    const code = e?.code || ''
    const msg = String(e?.message || '')
    if (code === 'same_password' || /different from the old password/i.test(msg))
      return pl ? 'Nowe hasło musi różnić się od obecnego.' : 'The new password must differ from the current one.'
    if (code === 'weak_password' || /weak|password should/i.test(msg))
      return pl ? `Hasło jest zbyt słabe. Użyj co najmniej ${MIN_PASSWORD} znaków.` : `The password is too weak. Use at least ${MIN_PASSWORD} characters.`
    if (code === 'reauthentication_needed' || /reauthentic/i.test(msg))
      return pl ? 'Ze względów bezpieczeństwa zaloguj się ponownie i spróbuj jeszcze raz.' : 'For security reasons, please sign in again and retry.'
    if (/rate limit|too many/i.test(msg))
      return pl ? 'Zbyt wiele prób. Spróbuj za chwilę.' : 'Too many attempts. Please try again shortly.'
    return msg
  }

  async function submit(e) {
    e.preventDefault()
    if (!valid || busy) return
    setBusy(true); setError(null)
    try {
      if (await isPwned(pw)) {
        setError(pl ? 'To hasło pojawiło się w publicznych wyciekach danych. Wybierz inne.'
                    : 'This password has appeared in public data breaches. Choose a different one.')
        return
      }
      const { error } = await supabase.auth.updateUser({ password: pw })
      if (error) throw error
      await supabase.auth.signOut({ scope: 'others' }).catch(() => {})
      setDone(true)
    } catch (err) {
      setError(explain(err))
    } finally {
      setBusy(false)
    }
  }

  const input = {
    width: '100%', padding: '10px 36px 10px 36px', border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)', background: 'var(--surface-2)', color: 'var(--text)',
    fontSize: 13, fontFamily: 'var(--font-body)', outline: 'none',
  }

  return (
    <>
      <div onClick={onClose} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,.5)', zIndex: 600 }} />
      <div role="dialog" aria-modal="true" style={{
        position: 'fixed', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', zIndex: 601,
        background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius-lg)',
        padding: '28px 26px', width: 'calc(100% - 32px)', maxWidth: 400, maxHeight: '92vh', overflowY: 'auto',
        boxShadow: '0 24px 64px rgba(0,0,0,.28)', animation: 'modalIn .2s ease',
      }}>
        <button type="button" onClick={onClose} aria-label="close" style={{
          position: 'absolute', top: 12, right: 12, background: 'none', border: 'none',
          cursor: 'pointer', color: 'var(--text-hint)', display: 'flex', padding: 4,
        }}><X size={16} strokeWidth={1.5} /></button>

        {done ? (
          <div style={{ textAlign: 'center', padding: '8px 0' }}>
            <div style={{
              width: 44, height: 44, borderRadius: '50%', margin: '0 auto 14px', background: '#f0faf4',
              border: '1.5px solid #2d8f4e', display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}><Check size={20} color="#2d8f4e" strokeWidth={2} /></div>
            <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 500, marginBottom: 8 }}>
              {pl ? 'Hasło zapisane' : 'Password saved'}
            </h2>
            <p style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 300, lineHeight: 1.6, marginBottom: 18 }}>
              {pl ? 'Od teraz możesz logować się hasłem. Pozostałe urządzenia zostały wylogowane.'
                  : 'You can now sign in with your password. Other devices have been signed out.'}
            </p>
            <button className="analyse-btn" onClick={onClose} style={{ maxWidth: 200, margin: '0 auto' }}>
              {pl ? 'Zamknij' : 'Close'}
            </button>
          </div>
        ) : (
          <form onSubmit={submit} noValidate>
            <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 20, fontWeight: 500, marginBottom: 4, marginRight: 24 }}>
              {recovery ? (pl ? 'Ustaw nowe hasło' : 'Set a new password')
                        : (pl ? 'Ustaw lub zmień hasło' : 'Set or change password')}
            </h2>
            <p style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 300, lineHeight: 1.6, marginBottom: 18 }}>
              {pl ? `Minimum ${MIN_PASSWORD} znaków. Po zapisaniu wylogujemy pozostałe urządzenia.`
                  : `At least ${MIN_PASSWORD} characters. Other devices will be signed out after saving.`}
            </p>

            {[[pw, setPw, pl ? 'Nowe hasło' : 'New password', 'new-pw'],
              [pw2, setPw2, pl ? 'Powtórz hasło' : 'Repeat password', 'new-pw2']].map(([val, set, ph, id]) => (
              <div key={id} style={{ position: 'relative', marginBottom: 10 }}>
                <Lock size={14} strokeWidth={1.5} color="var(--text-hint)" style={{
                  position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)',
                }} />
                <input type={show ? 'text' : 'password'} name={id} autoComplete="new-password"
                  aria-label={ph} placeholder={ph} value={val} onChange={e => set(e.target.value)} style={input} />
                {id === 'new-pw' && (
                  <button type="button" onClick={() => setShow(s => !s)} aria-label="toggle"
                    style={{
                      position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)',
                      background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-hint)', display: 'flex', padding: 2,
                    }}>
                    {show ? <EyeOff size={14} strokeWidth={1.5} /> : <Eye size={14} strokeWidth={1.5} />}
                  </button>
                )}
              </div>
            ))}

            {(tooShort || mismatch) && (
              <p style={{ fontSize: 11, color: 'var(--text-hint)', margin: '0 0 10px' }}>
                {tooShort ? (pl ? `Jeszcze ${MIN_PASSWORD - pw.length} znaków.` : `${MIN_PASSWORD - pw.length} more characters needed.`)
                          : (pl ? 'Hasła nie są takie same.' : 'The passwords do not match.')}
              </p>
            )}
            {error && (
              <p style={{
                fontSize: 12, color: '#c0392b', margin: '0 0 12px', padding: '8px 10px',
                background: '#fef4f2', borderRadius: 'var(--radius-sm)', border: '1px solid #f5c6bc',
              }}>{error}</p>
            )}

            <button type="submit" className="analyse-btn" disabled={!valid || busy} style={{ opacity: valid && !busy ? 1 : 0.5 }}>
              {busy ? (pl ? 'Zapisywanie…' : 'Saving…') : (pl ? 'Zapisz hasło' : 'Save password')}
            </button>
          </form>
        )}
      </div>
    </>
  )
}