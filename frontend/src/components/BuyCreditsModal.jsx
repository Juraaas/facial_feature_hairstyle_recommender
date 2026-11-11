import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { X } from 'lucide-react'
import { creditWord, plural } from '../lib/credits'

export function BuyCreditsModal({ packs, costs, credits, buying, onBuy, onClose }) {
  const { i18n } = useTranslation()
  const pl = i18n.language === 'pl'
  const [agree, setAgree] = useState(false)

  const money = (cents, cur) =>
    new Intl.NumberFormat(pl ? 'pl-PL' : 'en-US',
      { style: 'currency', currency: (cur || 'pln').toUpperCase() }).format(cents / 100)

  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, zIndex: 400, background: 'rgba(15,15,14,.55)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: 16, overflowY: 'auto',
    }}>
      <div onClick={e => e.stopPropagation()} style={{
        width: '100%', maxWidth: 780, background: 'var(--bg)', borderRadius: 16,
        border: '1px solid var(--border)', padding: '26px 20px 20px', position: 'relative',
      }}>
        <button onClick={onClose} aria-label="close" style={{
          position: 'absolute', top: 12, right: 12, background: 'none', border: 'none',
          cursor: 'pointer', color: 'var(--text-hint)', display: 'flex',
        }}><X size={20} strokeWidth={1.5} /></button>

        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 20, fontWeight: 500, marginBottom: 6 }}>
          {pl ? 'Kup kredyty' : 'Buy credits'}
        </h2>
        <p style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 300 }}>
          {pl
            ? `Masz ${credits} ${creditWord(credits, pl)}. Pełna analiza kosztuje ${costs.analysis}, podgląd fryzury ${costs.preview}.`
            : `You have ${credits} ${creditWord(credits, pl)}. A full analysis costs ${costs.analysis}, a hairstyle preview ${costs.preview}.`}
        </p>

        {packs.length === 0 ? (
          <p style={{ fontSize: 13, color: 'var(--text-hint)', margin: '24px 0' }}>…</p>
        ) : (
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: 12, margin: '18px 0',
          }}>
            {packs.map((p, i) => {
              const featured = packs.length === 3 && i === 1
              const previews = Math.floor(p.credits / costs.preview)
              return (
                <div key={p.id} style={{
                  border: `${featured ? 1.5 : 1}px solid ${featured ? 'var(--accent)' : 'var(--border)'}`,
                  borderRadius: 14, padding: '18px 16px', background: 'var(--surface)',
                  display: 'flex', flexDirection: 'column', gap: 6, position: 'relative', textAlign: 'center',
                }}>
                  {featured && (
                    <span style={{
                      position: 'absolute', top: -10, left: '50%', transform: 'translateX(-50%)',
                      background: 'var(--accent)', color: '#fff', fontSize: 9, fontWeight: 600,
                      padding: '3px 10px', borderRadius: 20, letterSpacing: '.06em', textTransform: 'uppercase',
                    }}>{pl ? 'Najpopularniejszy' : 'Most popular'}</span>
                  )}
                  <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>{pl ? p.name_pl : p.name_en}</p>
                  <p style={{ fontFamily: 'var(--font-mono)', fontSize: 30, fontWeight: 700,
                    color: 'var(--accent)', lineHeight: 1 }}>{p.credits}</p>
                  <p style={{ fontSize: 11, color: 'var(--text-hint)' }}>
                    {pl ? `${creditWord(p.credits, pl)} · ok. ${previews} ${plural(previews, 'podgląd', 'podglądy', 'podglądów')}`
                        : `credits · about ${previews} ${previews === 1 ? 'preview' : 'previews'}`}
                  </p>
                  <p style={{ fontSize: 20, fontWeight: 600, margin: '8px 0 0' }}>
                    {money(p.price_cents, p.currency)}
                  </p>
                  <p style={{ fontSize: 11, color: 'var(--text-hint)', marginBottom: 8 }}>
                    {pl ? `ok. ${money(p.price_cents / p.credits * costs.preview, p.currency)} za podgląd`
                        : `about ${money(p.price_cents / p.credits * costs.preview, p.currency)} per preview`}
                  </p>
                  <button className="analyse-btn" disabled={!agree || buying !== null}
                    onClick={() => onBuy(p.id)}
                    style={{ opacity: agree ? 1 : 0.45, marginTop: 'auto' }}>
                    {buying === p.id ? '…' : (pl ? 'Kup' : 'Buy')}
                  </button>
                </div>
              )
            })}
          </div>
        )}

        {/* TODO zgoda + (natychmiastowe dostarczenie, odstąpienie) */}
        <label style={{ display: 'flex', gap: 10, alignItems: 'flex-start', cursor: 'pointer' }}>
          <input type="checkbox" checked={agree} onChange={e => setAgree(e.target.checked)}
            style={{ marginTop: 3 }} />
          <span style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6, fontWeight: 300 }}>
            {pl ? (<>Akceptuję <a href="/terms" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent)' }}>Regulamin</a> i <a href="/privacy" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent)' }}>Politykę Prywatności</a>. Zgadzam się na natychmiastowe dostarczenie kredytów i przyjmuję do wiadomości, że po ich wykorzystaniu nie przysługuje mi prawo odstąpienia od umowy.</>)
                : (<>I accept the <a href="/terms" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent)' }}>Terms</a> and <a href="/privacy" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent)' }}>Privacy Policy</a>. I agree to immediate delivery of the credits and acknowledge that once they are used I lose my right of withdrawal.</>)}
          </span>
        </label>
        <p style={{ fontSize: 10, color: 'var(--text-hint)', marginTop: 10, textAlign: 'center' }}>
          {pl ? 'Płatność obsługuje Stripe. Dane karty nie trafiają do nas.' : 'Payments are processed by Stripe. We never see your card details.'}
        </p>
      </div>
    </div>
  )
}