import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { X, LifeBuoy, Check } from 'lucide-react'
import { createTicket, parseError } from '../api/client'

const CATEGORIES = [
  ['preview_quality', 'Jakość podglądu', 'Preview quality'],
  ['technical', 'Problem techniczny', 'Technical problem'],
  ['billing', 'Płatności i kredyty', 'Payments and credits'],
  ['other', 'Inne', 'Other'],
]

export function TicketModal({ generationId = null, styleLabel = '', onClose, onCreated }) {
  const { i18n } = useTranslation()
  const pl = i18n.language?.startsWith('pl')

  const [category, setCategory] = useState(generationId ? 'preview_quality' : 'technical')
  const [message, setMessage] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState(null)
  const [done, setDone] = useState(false)

  useEffect(() => {
    const onKey = e => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const len = message.trim().length
  const canSend = len >= 10 && !sending

  async function submit(e) {
    e.preventDefault()
    if (!canSend) return
    setSending(true); setError(null)
    try {
      await createTicket({ category, message: message.trim(), generationId })
      setDone(true)
      onCreated?.()
    } catch (err) {
      const code = parseError(err).code
      const msgs = {
        TICKET_EXISTS: pl ? 'Dla tego podglądu masz już otwarte zgłoszenie.' : 'You already have an open request for this preview.',
        TOO_MANY_TICKETS: pl ? 'Wykorzystałeś dzisiejszy limit zgłoszeń. Spróbuj jutro.' : 'You have used today\'s request limit. Please try tomorrow.',
        RATE_LIMITED: pl ? 'Za dużo prób. Spróbuj za chwilę.' : 'Too many attempts. Please try again shortly.',
        INVALID_GENERATION: pl ? 'Nie znaleziono tego podglądu.' : 'This preview was not found.',
        UNAUTHORIZED: pl ? 'Sesja wygasła. Zaloguj się ponownie.' : 'Your session expired. Please sign in again.',
        SERVER_UNAVAILABLE: pl ? 'Serwer jest chwilowo niedostępny. Spróbuj za minutę.' : 'The server is temporarily unavailable. Try again in a minute.',
      }
      setError(msgs[code] ?? (pl ? 'Nie udało się wysłać zgłoszenia. Spróbuj ponownie.' : 'Could not send the request. Please try again.'))
    } finally {
      setSending(false)
    }
  }

  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, zIndex: 500, background: 'rgba(15,15,14,.55)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16, overflowY: 'auto',
    }}>
      <div role="dialog" aria-modal="true" onClick={e => e.stopPropagation()} style={{
        position: 'relative', width: '100%', maxWidth: 440, background: 'var(--bg)',
        border: '1px solid var(--border)', borderRadius: 16, padding: '24px 22px 20px',
      }}>
        <button onClick={onClose} aria-label="close" style={{
          position: 'absolute', top: 12, right: 12, background: 'none', border: 'none',
          cursor: 'pointer', color: 'var(--text-hint)', display: 'flex', padding: 6,
        }}><X size={18} strokeWidth={1.5} /></button>

        {done ? (
          <div style={{ textAlign: 'center', padding: '12px 0 4px' }}>
            <div style={{
              width: 44, height: 44, borderRadius: '50%', margin: '0 auto 14px', background: '#f0faf4',
              border: '1.5px solid #2d8f4e', display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}><Check size={20} color="#2d8f4e" strokeWidth={2} /></div>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: 17, fontWeight: 500, marginBottom: 8 }}>
              {pl ? 'Zgłoszenie przyjęte' : 'Request received'}
            </h3>
            <p style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 300, lineHeight: 1.6, marginBottom: 18 }}>
              {pl ? 'Sprawdzimy to. Status zobaczysz w panelu konta, w zakładce Konto.'
                  : 'We will review it. You can follow the status in your account panel, under Account.'}
            </p>
            <button className="analyse-btn" onClick={onClose} style={{ maxWidth: 200, margin: '0 auto' }}>
              {pl ? 'Zamknij' : 'Close'}
            </button>
          </div>
        ) : (
          <form onSubmit={submit} noValidate>
            <h3 style={{
              fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 500, margin: '0 28px 4px 0',
              display: 'flex', alignItems: 'center', gap: 8,
            }}>
              <LifeBuoy size={17} color="var(--accent)" strokeWidth={1.5} />
              {pl ? 'Zgłoś problem' : 'Report a problem'}
            </h3>
            {generationId && (
              <p style={{ fontSize: 12, color: 'var(--text-hint)', marginBottom: 6 }}>
                {pl ? 'Dotyczy podglądu' : 'About the preview'}{styleLabel ? `: ${styleLabel}` : ''}
              </p>
            )}
            <p style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 300, lineHeight: 1.6, marginBottom: 14 }}>
              {pl
                ? 'Za nieudane podglądy kredyty zwracane są automatycznie. Jeśli podgląd się udał, ale jakość była niezadowalająca, opisz problem. Zgłoszenie sprawdzimy ręcznie i w uzasadnionych przypadkach zwrócimy kredyty.'
                : 'Credits for failed previews are refunded automatically. If a preview completed but the quality is poor, describe the problem. We will review it and, where justified, grant credits.'}
            </p>

            <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
              {pl ? 'Kategoria' : 'Category'}
            </label>
            <select value={category} onChange={e => setCategory(e.target.value)} style={{
              width: '100%', padding: '9px 10px', marginBottom: 12, border: '1px solid var(--border)',
              borderRadius: 'var(--radius-md)', background: 'var(--surface-2)', color: 'var(--text)',
              fontSize: 13, fontFamily: 'var(--font-body)',
            }}>
              {CATEGORIES.map(([id, plL, enL]) => <option key={id} value={id}>{pl ? plL : enL}</option>)}
            </select>

            <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
              {pl ? 'Opis (min. 10 znaków)' : 'Description (min. 10 characters)'}
            </label>
            <textarea value={message} onChange={e => setMessage(e.target.value)} rows={5} maxLength={2000}
              style={{
                width: '100%', padding: '10px 12px', border: '1px solid var(--border)',
                borderRadius: 'var(--radius-md)', background: 'var(--surface-2)', color: 'var(--text)',
                fontSize: 13, resize: 'none', outline: 'none', lineHeight: 1.5, fontFamily: 'var(--font-body)',
              }} />
            <p style={{ fontSize: 10, color: 'var(--text-hint)', textAlign: 'right', margin: '4px 0 12px',
              fontFamily: 'var(--font-mono)' }}>{message.length}/2000</p>

            {error && (
              <p style={{
                fontSize: 12, color: '#c0392b', marginBottom: 12, padding: '8px 10px',
                background: '#fef4f2', borderRadius: 'var(--radius-sm)', border: '1px solid #f5c6bc',
              }}>{error}</p>
            )}

            <button type="submit" className="analyse-btn" disabled={!canSend}
              style={{ opacity: canSend ? 1 : 0.5 }}>
              {sending ? (pl ? 'Wysyłanie…' : 'Sending…') : (pl ? 'Wyślij zgłoszenie' : 'Send request')}
            </button>
          </form>
        )}
      </div>
    </div>
  )
}